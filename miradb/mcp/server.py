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
from typing import Annotated, Literal

import anyio
import click
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

from miradb.db.client import MiraDatabaseClient, get_client
from miradb.mcp import db_tools, mira_tools, simulate as sim
from miradb.mcp.model_store import ModelStore

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
async def mira_compare_models(
    model_refs: Annotated[list[str], Field(min_length=2)],
) -> dict:
    """Pairwise structural similarity of models, based on their transitions
    and grounded concepts (exact grounding match, no ontology refinement)."""
    return await _run(mira_tools.compare_models, store, model_refs)


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
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
