"""MCP server for MIRA-DB and MIRA.

Run locally over stdio (e.g. for Claude Code)::

    claude mcp add miradb -- miradb-mcp

or as a shared service over Streamable HTTP::

    miradb-mcp --transport streamable-http --host 0.0.0.0 --port 8004

The database is selected with ``--db`` (or the ``MCP_MIRADB_DB``
environment variable), naming a section of ``~/.config/mira/db_config.ini``
or ``primary`` for the ``MIRADB_DB_*`` environment variables. Pointing it at
a role with only ``SELECT`` grants is recommended.
"""

import functools
import logging
import os
from importlib.resources import files
from typing import Annotated, Literal

import anyio
import click
from mcp.server.fastmcp import FastMCP, Image
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from miradb.db.client import MiraDatabaseClient, get_client
from miradb.mcp import compare, db_tools, docs, mira_tools, model_edit, simulate as sim
from miradb.mcp.model_store import ModelStore
from miradb.mcp.refinement import dkg_closure

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tools for MIRA-DB, a database of compartmental epidemiology models extracted
from papers as grounded MIRA TemplateModels, and for working with those
models.

Models are passed by handle (model_ref), never as JSON: 'db:<id>' is a
stored model (mira_template_models.id); 'ws:<id>' is a model you derived
in this session. Tools that change a model return a new ws: handle and
leave the input untouched.

Typical workflow:
1. db_search_publications -> db_get_publication(pmid) to find model_refs
   and scenario_ids.
2. mira_get_model / mira_get_equations to inspect a model.
3. simulate_model(model_ref, t_end, scenario_id=...) to simulate. Stored
   models rarely have parameter values or initials; scenarios provide them.
   If the result has status 'missing_values', supply the listed values via
   parameters/initials and call again.

Editing (mira_add_* / mira_remove_* / mira_update_* / mira_set_*): each
call returns a new ws: model_ref, so chain edits by passing the previous
result's model_ref. Unnamed transitions are referred to as t1, t2, ...
(see mira_get_model); these IDs shift when transitions are removed.

Comparing: mira_compare_models (similarity scores), mira_model_delta (text
diff), mira_draw_model_delta (diff image), mira_compare_concepts,
mira_compare_context. By default concepts are matched using the DKG
ontology, so a more specific grounding counts as a refinement.

For questions the curated tools don't answer, read the miradb://schema
resource (or db_describe_schema) and use db_run_sql (read-only).
"""

# Get allowed hosts and origins environment variables
_allowed_hosts_env = os.environ["MCP_ALLOWED_HOSTS"]
_allowed_origins_env = os.environ["MCP_ALLOWED_ORIGINS"]
_allowed_hosts = [h.strip() for h in _allowed_hosts_env.split(",") if h.strip()]
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]


mcp = FastMCP(
    "miradb",
    instructions=INSTRUCTIONS,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
)

_db_label = os.environ.get("MCP_MIRADB_DB", "primary")


@functools.lru_cache(maxsize=1)
def _client() -> MiraDatabaseClient:
    return get_client(_db_label)


store = ModelStore(_client)
sims = sim.SimulationStore()


async def _run(fn, *args, **kwargs):
    """Run blocking DB/MIRA work off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


ModelRef = Annotated[
    str, Field(description="Model handle: 'db:<id>' or 'ws:<id>'.")
]
Pmid = Annotated[str, Field(description="PubMed ID of the paper.")]


# ---------------------------------------------------------------------------
# Custom routes (streamable-http only)
# ---------------------------------------------------------------------------

_TEMPLATES = files("miradb.mcp").joinpath("templates")
_LANDING_PAGE = _TEMPLATES.joinpath("landing_page.html").read_text()
_DOCS_PAGE = _TEMPLATES.joinpath("docs.html").read_text()


@mcp.custom_route("/", methods=["GET"])
async def landing_page(request: Request) -> HTMLResponse:
    """Serve the HTML landing page at the root path."""
    return HTMLResponse(_LANDING_PAGE)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint for monitoring."""
    return JSONResponse({"status": "healthy", "service": "miradb"})


@mcp.custom_route("/docs", methods=["GET"])
async def docs_page(request: Request) -> HTMLResponse:
    """Serve the tool reference page; it renders /docs.json."""
    return HTMLResponse(_DOCS_PAGE)


@mcp.custom_route("/docs.json", methods=["GET"])
async def docs_json(request: Request) -> JSONResponse:
    """Every tool (with its input schema) and resource, generated from
    the registered definitions."""
    return JSONResponse(await docs.docs_payload(mcp))


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@mcp.tool()
def db_describe_schema() -> str:
    """Describe MIRA-DB's tables, columns, foreign keys and JSON column
    layouts. Read this before writing SQL for db_run_sql."""
    return db_tools.describe_schema()


@mcp.tool()
async def db_search_publications(
    query: Annotated[str | None, Field(
        description="Case-insensitive substring match on PMID, title, "
                    "authors, year and grounded concepts: variable names, "
                    "ontology prefixes or IDs (search 'ido' or '0000511', "
                    "not 'ido:0000511'), context values. Omit to list all.")] = None,
    limit: Annotated[int, Field(ge=1, le=2000)] = 25,
) -> dict:
    """Search papers that have at least one converted model, ordered by
    number of models."""
    return await _run(db_tools.search_publications, _client(), query, limit)


@mcp.tool()
async def db_get_publication(pmid: Pmid) -> dict:
    """Get a paper's metadata plus its models (as model_refs, one per
    extraction method) and its scenarios (as scenario_ids)."""
    result = await _run(db_tools.get_publication, _client(), pmid)
    if result is None:
        raise ValueError(f"No publication with PMID {pmid} in MIRA-DB.")
    return result


@mcp.tool()
async def db_get_odes(pmid: Pmid) -> list[dict]:
    """Get the SymPy ODE source extracted from a paper, per extraction
    method, including any corrected version."""
    return await _run(db_tools.get_odes, _client(), pmid)


@mcp.tool()
async def db_get_scenarios(pmid: Pmid) -> list[dict]:
    """Get all scenarios of a paper with their parameter values, units and
    distributions, grouped by extraction method."""
    return await _run(db_tools.get_scenarios, _client(), pmid)


@mcp.tool()
async def db_get_scenario(scenario_id: int) -> dict:
    """Get one scenario and its parameters by scenario ID."""
    result = await _run(db_tools.get_scenario, _client(), scenario_id)
    if result is None:
        raise ValueError(f"No scenario with id {scenario_id}.")
    return result


@mcp.tool()
async def db_run_sql(
    sql: Annotated[str, Field(
        description="A single read-only PostgreSQL statement (SELECT/WITH/"
                    "EXPLAIN). Use :name placeholders for params.")],
    params: Annotated[dict | None, Field(
        description="Values for :name placeholders in the statement.")] = None,
    limit: Annotated[int, Field(ge=1, le=2000)] = db_tools.DEFAULT_ROW_LIMIT,
) -> dict:
    """Run a read-only SQL query against MIRA-DB. Runs in a READ ONLY
    transaction with a 15 s timeout; long cell values are truncated. See
    db_describe_schema for tables and columns."""
    return await _run(db_tools.run_sql, _client(), sql, params, limit=limit)


# ---------------------------------------------------------------------------
# MIRA models
# ---------------------------------------------------------------------------

@mcp.tool()
async def mira_get_model(model_ref: ModelRef) -> dict:
    """Summarize a model: compartments with groundings, transitions with
    rate laws, parameters (and which lack values), initials (and which
    compartments lack them), observables."""
    return await _run(mira_tools.get_model, store, model_ref)


@mcp.tool()
async def mira_get_equations(model_ref: ModelRef) -> dict:
    """Get the model's ODE system as plain-text equations."""
    return await _run(mira_tools.get_equations, store, model_ref)


@mcp.tool()
async def mira_stratify(
    model_ref: ModelRef,
    key: Annotated[str, Field(description="Name of the stratification, e.g. 'age' or 'city'.")],
    strata: Annotated[list[str], Field(description="Strata values, e.g. ['young', 'old'].")],
    structure: Annotated[list[list[str]] | None, Field(
        description="Pairs [from, to] of strata between which transitions are "
                    "added. Omit for all pairs; [] for none.")] = None,
    directed: bool = False,
    cartesian_control: Annotated[bool, Field(
        description="If true, every stratum of a controller (e.g. infected "
                    "in each age group) acts on every stratum of the subject.")] = False,
    concepts_to_stratify: list[str] | None = None,
    concepts_to_preserve: list[str] | None = None,
    params_to_stratify: list[str] | None = None,
    params_to_preserve: list[str] | None = None,
) -> dict:
    """Stratify a model (MIRA's stratify), e.g. split each compartment by
    age group. Returns the summary of the new workspace model."""
    return await _run(
        mira_tools.stratify_model, store, model_ref, key, strata,
        structure=structure, directed=directed,
        cartesian_control=cartesian_control,
        concepts_to_stratify=concepts_to_stratify,
        concepts_to_preserve=concepts_to_preserve,
        params_to_stratify=params_to_stratify,
        params_to_preserve=params_to_preserve,
    )


@mcp.tool()
async def mira_simplify_rate_laws(model_ref: ModelRef) -> dict:
    """Split rate laws that combine several processes into simpler
    templates (MIRA's simplify_rate_laws). Returns the new model."""
    return await _run(mira_tools.simplify_model, store, model_ref)


@mcp.tool()
async def mira_set_values(
    model_ref: ModelRef,
    parameters: Annotated[dict[str, float] | None, Field(
        description="Parameter name -> value.")] = None,
    initials: Annotated[dict[str, float | str] | None, Field(
        description="Compartment name -> initial value or expression in "
                    "parameters, e.g. 'N - I0'.")] = None,
) -> dict:
    """Create a workspace copy of a model with parameter values and initial
    conditions set, e.g. to keep a fixed parameterization for several
    simulations."""
    return await _run(
        mira_tools.set_model_values, store, model_ref, parameters, initials
    )


@mcp.tool()
async def mira_export_model(
    model_ref: ModelRef,
    format: Literal["template_model", "petrinet_amr"] = "template_model",
) -> dict:
    """Get a model's full JSON, as a MIRA TemplateModel or an ASKEM Petri
    net AMR. This can be large; prefer mira_get_model for inspection."""
    return await _run(mira_tools.export_model, store, model_ref, format)


@mcp.tool()
def mira_list_workspace() -> list[dict]:
    """List the workspace models derived in this session and the operation
    that produced each."""
    return store.list_workspace()


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

Refinement = Annotated[
    Literal["dkg", "exact"],
    Field(description="'dkg': a concept whose grounding is an ontological "
                      "child of another's counts as a refinement (uses the "
                      "MIRA DKG). 'exact': only identical groundings match."),
]
ImageFormat = Annotated[
    Literal["png", "svg"],
    Field(description="png is shown inline; svg is only saved to disk."),
]


def _image_response(data: bytes | None, info: dict):
    return [Image(data=data, format="png"), info] if data else info


@mcp.tool()
async def mira_compare_models(
    model_refs: Annotated[list[str], Field(min_length=2)],
    refinement: Refinement = "dkg",
    include_graph: Annotated[bool, Field(
        description="Also return the full comparison graph (large).")] = False,
) -> dict:
    """Pairwise structural similarity (0-1) of two or more models, based on
    their transitions and grounded concepts."""
    return await _run(compare.compare_models, store, model_refs,
                      refinement, include_graph)


@mcp.tool()
async def mira_compare_concepts(
    model_ref_1: ModelRef,
    model_ref_2: ModelRef,
    refinement: Refinement = "dkg",
    name_only: Annotated[bool, Field(
        description="Label concepts by name only, without grounding and "
                    "context.")] = False,
) -> dict:
    """Relate each concept (state) of one model to each concept of another:
    is_equal, refinement_of or refined_by, plus unmatched concepts."""
    return await _run(compare.compare_concepts, store, model_ref_1,
                      model_ref_2, refinement, name_only)


@mcp.tool()
async def mira_compare_context(
    model_refs: Annotated[list[str], Field(min_length=2)],
) -> dict:
    """For each context key used by the models' states (e.g. age, location,
    severity), count its distinct values per model."""
    return await _run(compare.compare_context, store, model_refs)


@mcp.tool()
async def mira_model_delta(
    model_ref_1: ModelRef,
    model_ref_2: ModelRef,
    refinement: Refinement = "dkg",
    concepts_only: bool = False,
) -> dict:
    """Text diff of two models: concepts and transitions that match
    (is_equal / refinement), and those only in one model."""
    return await _run(compare.model_delta, store, model_ref_1, model_ref_2,
                      refinement, concepts_only)


@mcp.tool()
async def mira_draw_model_delta(
    model_ref_1: ModelRef,
    model_ref_2: ModelRef,
    refinement: Refinement = "dkg",
    concepts_only: bool = False,
    format: ImageFormat = "png",
):
    """Draw the difference of two models as a graph: blue = only in model 1,
    green = only in model 2, orange = in both; red edges link equal or
    refined elements. Also saved to disk (image_path)."""
    data, info = await _run(compare.draw_model_delta, store, model_ref_1,
                            model_ref_2, refinement, concepts_only, format)
    return _image_response(data, info)


@mcp.tool()
async def mira_draw_model(
    model_ref: ModelRef,
    use_display_name: bool = False,
    format: ImageFormat = "png",
):
    """Draw a model's states and transitions as a graph. Also saved to disk
    (image_path)."""
    data, info = await _run(compare.draw_model, store, model_ref,
                            use_display_name, format)
    return _image_response(data, info)


# ---------------------------------------------------------------------------
# Editing models: every tool returns a new ws: model_ref
# ---------------------------------------------------------------------------

Units = Annotated[str | None, Field(
    description="Unit expression, e.g. 'person', '1/day'.")]
Distribution = Annotated[dict | None, Field(
    description="e.g. {'type': 'Uniform1', 'parameters': {'minimum': 0.1, "
                "'maximum': 0.5}}")]
Grounding = Annotated[dict[str, str] | None, Field(
    description="Ontology identifiers, prefix -> id, e.g. {'ido': '0000511'}. "
                "Replaces all existing identifiers.")]
Context = Annotated[dict[str, str] | None, Field(
    description="Context, e.g. {'age': 'young'}. Replaces the existing "
                "context.")]
TransitionRef = Annotated[str, Field(
    description="Transition name, or 't<n>' (1-based position) for unnamed "
                "transitions, as shown by mira_get_model.")]
NewParameters = Annotated[dict[str, dict] | None, Field(
    description="Attributes for parameters the rate law introduces: name -> "
                "{value, units, distribution, display_name, description}. "
                "Undeclared symbols without an entry are added without a "
                "value.")]


@mcp.tool()
async def mira_add_state(
    model_ref: ModelRef,
    name: str,
    display_name: str | None = None,
    description: str | None = None,
    identifiers: Grounding = None,
    context: Context = None,
    units: Units = None,
    initial: Annotated[float | str | None, Field(
        description="Initial value or expression in parameters.")] = None,
) -> dict:
    """Add a state (compartment). It is not connected to anything until you
    add transitions with mira_add_transition."""
    return await _run(model_edit.add_state, store, model_ref, name,
                      display_name, description, identifiers, context, units,
                      initial)


@mcp.tool()
async def mira_remove_state(model_ref: ModelRef, name: str) -> dict:
    """Remove a state together with every transition involving it and its
    initial; the state is replaced by 0 in observables."""
    return await _run(model_edit.remove_state, store, model_ref, name)


@mcp.tool()
async def mira_update_state(
    model_ref: ModelRef,
    name: str,
    new_name: Annotated[str | None, Field(
        description="Rename the state everywhere, including rate laws, "
                    "initials and observables.")] = None,
    display_name: str | None = None,
    description: str | None = None,
    identifiers: Grounding = None,
    context: Context = None,
    units: Units = None,
) -> dict:
    """Edit a state (concept): rename it, change its grounding, context,
    units or labels. Applied to every copy of the concept in the model."""
    return await _run(model_edit.update_state, store, model_ref, name,
                      new_name, display_name, description, identifiers,
                      context, units)


@mcp.tool()
async def mira_set_initial(
    model_ref: ModelRef,
    state: str,
    expression: Annotated[float | str, Field(
        description="Number or expression in parameters, e.g. 'N - I0'.")],
) -> dict:
    """Add or replace a state's initial condition."""
    return await _run(model_edit.set_initial, store, model_ref, state,
                      expression)


@mcp.tool()
async def mira_remove_initial(model_ref: ModelRef, state: str) -> dict:
    """Remove a state's initial condition."""
    return await _run(model_edit.remove_initial, store, model_ref, state)


@mcp.tool()
async def mira_add_transition(
    model_ref: ModelRef,
    name: str | None = None,
    subject: Annotated[str | None, Field(
        description="State the flow leaves; omit for a production.")] = None,
    outcome: Annotated[str | None, Field(
        description="State the flow enters; omit for a degradation.")] = None,
    controllers: Annotated[list[str] | None, Field(
        description="States that drive the transition without being "
                    "consumed, e.g. ['I'] for infection of S.")] = None,
    rate_law: Annotated[str | None, Field(
        description="e.g. 'beta*S*I/N'. Omit for mass action with "
                    "mass_action_parameter.")] = None,
    mass_action_parameter: Annotated[str | None, Field(
        description="Rate constant for the mass-action rate law; default "
                    "'k_<name>'.")] = None,
    parameters: NewParameters = None,
) -> dict:
    """Add a transition. Its type follows from the states given:
    subject+outcome = conversion, outcome only = production, subject only =
    degradation; with controllers, the controlled variant. States must
    already exist (see mira_add_state)."""
    return await _run(model_edit.add_transition, store, model_ref, name,
                      subject, outcome, controllers, rate_law,
                      mass_action_parameter, parameters)


@mcp.tool()
async def mira_remove_transition(
    model_ref: ModelRef, transition: TransitionRef
) -> dict:
    """Remove a transition."""
    return await _run(model_edit.remove_transition, store, model_ref,
                      transition)


@mcp.tool()
async def mira_update_transition(
    model_ref: ModelRef,
    transition: TransitionRef,
    new_name: str | None = None,
    rate_law: str | None = None,
    display_name: str | None = None,
    parameters: NewParameters = None,
) -> dict:
    """Rename a transition and/or replace its rate law."""
    return await _run(model_edit.update_transition, store, model_ref,
                      transition, new_name, rate_law, display_name, parameters)


@mcp.tool()
async def mira_add_parameter(
    model_ref: ModelRef,
    name: str,
    value: float | None = None,
    distribution: Distribution = None,
    units: Units = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Add a parameter."""
    return await _run(model_edit.add_parameter, store, model_ref, name,
                      value, distribution, units, display_name, description)


@mcp.tool()
async def mira_remove_parameter(
    model_ref: ModelRef,
    name: str,
    replacement_value: Annotated[float | None, Field(
        description="Value substituted wherever the parameter is used; "
                    "default 0, which switches off transitions that use "
                    "it as a factor.")] = None,
) -> dict:
    """Remove a parameter, substituting a value for it in rate laws,
    initials and observables."""
    return await _run(model_edit.remove_parameter, store, model_ref, name,
                      replacement_value)


@mcp.tool()
async def mira_update_parameter(
    model_ref: ModelRef,
    name: str,
    new_name: Annotated[str | None, Field(
        description="Rename the parameter everywhere it is used.")] = None,
    value: float | None = None,
    distribution: Distribution = None,
    clear_distribution: bool = False,
    units: Units = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Edit any attribute of a parameter, or rename it. For setting many
    values at once, use mira_set_values."""
    return await _run(model_edit.update_parameter, store, model_ref, name,
                      new_name, value, distribution, clear_distribution,
                      units, display_name, description)


@mcp.tool()
async def mira_eliminate_unused_parameters(model_ref: ModelRef) -> dict:
    """Remove parameters that no rate law uses."""
    return await _run(model_edit.eliminate_unused_parameters, store,
                      model_ref)


@mcp.tool()
async def mira_merge_duplicate_parameters(
    model_ref: ModelRef, redundant: str, keep: str
) -> dict:
    """Replace parameter `redundant` by `keep` everywhere and remove it."""
    return await _run(model_edit.merge_duplicate_parameters, store, model_ref,
                      redundant, keep)


@mcp.tool()
async def mira_aggregate_parameters(model_ref: ModelRef) -> dict:
    """Combine products of parameters in mass-action rate laws (e.g.
    beta*c) into single parameters with the product's value. All
    parameters need values first."""
    return await _run(model_edit.aggregate_model_parameters, store, model_ref)


@mcp.tool()
async def mira_counts_to_dimensionless(
    model_ref: ModelRef,
    counts_unit: Annotated[str, Field(description="e.g. 'person'.")],
    norm_factor: Annotated[float, Field(
        description="Total population to divide counts by.")],
) -> dict:
    """Convert quantities measured in counts to fractions of norm_factor."""
    return await _run(model_edit.to_dimensionless, store, model_ref,
                      counts_unit, norm_factor)


@mcp.tool()
async def mira_add_observable(
    model_ref: ModelRef,
    name: str,
    expression: Annotated[str, Field(description="e.g. 'I + H'.")],
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Add an observable: an expression in states and parameters that
    simulations also report."""
    return await _run(model_edit.add_observable, store, model_ref, name,
                      expression, display_name, description)


@mcp.tool()
async def mira_remove_observable(model_ref: ModelRef, name: str) -> dict:
    """Remove an observable."""
    return await _run(model_edit.remove_observable, store, model_ref, name)


@mcp.tool()
async def mira_update_observable(
    model_ref: ModelRef,
    name: str,
    new_name: str | None = None,
    expression: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Change an observable's expression or labels, or rename it."""
    return await _run(model_edit.update_observable, store, model_ref, name,
                      new_name, expression, display_name, description)


@mcp.tool()
async def mira_set_annotations(
    model_ref: ModelRef,
    name: str | None = None,
    description: str | None = None,
    license: Annotated[str | None, Field(description="SPDX id, e.g. 'CC0'.")] = None,
    references: Annotated[list[str] | None, Field(
        description="CURIEs, e.g. ['pubmed:32616574'].")] = None,
    time_scale: Annotated[str | None, Field(description="e.g. 'day'.")] = None,
    time_start: Annotated[str | None, Field(
        description="ISO date or datetime, e.g. '2020-03-01'.")] = None,
    time_end: Annotated[str | None, Field(description="ISO date or datetime.")] = None,
    locations: Annotated[list[str] | None, Field(
        description="CURIEs, e.g. ['geonames:5128581'].")] = None,
    pathogens: Annotated[list[str] | None, Field(
        description="NCBI Taxonomy CURIEs, e.g. ['ncbitaxon:2697049'].")] = None,
    diseases: Annotated[list[str] | None, Field(
        description="e.g. ['doid:0080600'].")] = None,
    hosts: Annotated[list[str] | None, Field(
        description="e.g. ['ncbitaxon:9606'].")] = None,
    model_types: Annotated[list[str] | None, Field(
        description="MAMO CURIEs, e.g. ['mamo:0000028'].")] = None,
) -> dict:
    """Set model-level annotations. Fields you pass replace the old value
    (lists included); fields you omit are kept. Use mira_add_author /
    mira_remove_author for authors."""
    return await _run(
        model_edit.set_annotations, store, model_ref, name=name,
        description=description, license=license, references=references,
        time_scale=time_scale, time_start=time_start, time_end=time_end,
        locations=locations, pathogens=pathogens, diseases=diseases,
        hosts=hosts, model_types=model_types,
    )


@mcp.tool()
async def mira_add_author(model_ref: ModelRef, name: str) -> dict:
    """Add an author to the model's annotations."""
    return await _run(model_edit.add_author, store, model_ref, name)


@mcp.tool()
async def mira_remove_author(model_ref: ModelRef, name: str) -> dict:
    """Remove an author from the model's annotations."""
    return await _run(model_edit.remove_author, store, model_ref, name)


@mcp.tool()
async def mira_set_time(
    model_ref: ModelRef,
    name: Annotated[str | None, Field(
        description="New name of the time variable; renamed in rate laws "
                    "too.")] = None,
    units: Units = None,
) -> dict:
    """Set the model's time variable name and/or units."""
    return await _run(model_edit.set_time, store, model_ref, name, units)


@mcp.tool()
async def mira_merge_models(
    model_ref: ModelRef, other_model_ref: ModelRef
) -> dict:
    """Combine two models: add the second model's transitions, parameters,
    initials and observables to the first. States with the same name are
    treated as the same state; on other name clashes the first model's
    definitions win."""
    return await _run(model_edit.merge_models, store, model_ref,
                      other_model_ref)


@mcp.tool()
async def mira_apply_scenario(model_ref: ModelRef, scenario_id: int) -> dict:
    """Bake a DB scenario into a model: parameter values, and initials from
    scenario parameters such as I_0. To just simulate a scenario, pass
    scenario_id to simulate_model instead."""
    scenario = await _run(db_tools.get_scenario, _client(), scenario_id)
    if scenario is None:
        raise ValueError(f"No scenario with id {scenario_id}.")
    return await _run(model_edit.apply_db_scenario, store, model_ref, scenario)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@mcp.tool()
async def simulate_model(
    model_ref: ModelRef,
    t_end: Annotated[float, Field(gt=0, description="End time, in the model's time units (usually days).")],
    scenario_id: Annotated[int | None, Field(
        description="Apply a DB scenario's parameters and initial "
                    "conditions (e.g. E_0, I_0) before simulating.")] = None,
    parameters: Annotated[dict[str, float] | None, Field(
        description="Parameter overrides; take precedence over the scenario.")] = None,
    initials: Annotated[dict[str, float | str] | None, Field(
        description="Initial condition overrides by compartment name; "
                    "numbers or expressions in parameters.")] = None,
    t_start: float = 0.0,
    n_points: Annotated[int, Field(ge=2, le=sim.MAX_POINTS)] = 101,
    method: Literal["LSODA", "RK45", "BDF", "Radau", "DOP853"] = "LSODA",
) -> dict:
    """Simulate a model's ODEs deterministically.

    Returns status 'ok' with per-compartment stats (peak and time of peak,
    final value), a short preview, and a sim_id for get_simulation; or
    status 'missing_values' listing the parameters/initials to supply; or
    'solver_failed'."""
    scenario = None
    if scenario_id is not None:
        scenario = await _run(db_tools.get_scenario, _client(), scenario_id)
        if scenario is None:
            raise ValueError(f"No scenario with id {scenario_id}.")
    return await _run(
        sim.simulate, store, sims, model_ref, t_end,
        t_start=t_start, n_points=n_points, parameters=parameters,
        initials=initials, scenario=scenario, method=method,
    )


@mcp.tool()
def get_simulation(
    sim_id: str,
    variables: Annotated[list[str] | None, Field(
        description="Compartments/observables to return; omit for all.")] = None,
    n_points: Annotated[int, Field(ge=2, le=sim.MAX_POINTS)] = 50,
) -> dict:
    """Get a downsampled trajectory of a previous simulate_model run."""
    return sim.get_trajectory(sims, sim_id, variables, n_points)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("miradb://schema", mime_type="text/markdown")
def schema_resource() -> str:
    """MIRA-DB tables, columns and JSON layouts."""
    return db_tools.describe_schema()


@mcp.resource("miradb://model/{model_ref}", mime_type="application/json")
def model_resource(model_ref: str) -> dict:
    """Full TemplateModel JSON of a model handle."""
    return store.get_json(model_ref)


@mcp.resource("miradb://model/{model_ref}/graph", mime_type="application/json")
def model_graph_resource(model_ref: str) -> dict:
    """Node-link JSON of a model's state/transition graph."""
    return compare.model_graph_json(store, model_ref)


@mcp.resource("miradb://simulation/{sim_id}.csv", mime_type="text/csv")
def simulation_resource(sim_id: str) -> str:
    """Full trajectory of a simulation as CSV."""
    return sim.trajectory_csv(sims, sim_id)


@click.command()
@click.option("--transport", type=click.Choice(["stdio", "streamable-http"]),
              default="stdio", show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address for streamable-http.")
@click.option("--port", default=8004, show_default=True,
              help="Port for streamable-http.")
@click.option("--db", "db_label", default=None,
              help="Database label from db_config.ini or 'primary' "
                   "[default: $MCP_MIRADB_DB or primary].")
def main(transport, host, port, db_label):
    """Run the MIRA-DB MCP server."""
    global _db_label
    if db_label:
        _db_label = db_label
        _client.cache_clear()
    # stdout carries the protocol under stdio, so log to stderr only
    logging.basicConfig(level=logging.INFO)
    mcp.settings.host = host
    mcp.settings.port = port
    # Comparisons default to DKG refinement; load the ontology up front
    dkg_closure.prefetch()
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
