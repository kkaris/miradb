"""Read-only MIRA-DB access for the MCP server.

Curated lookups wrap the functions in ``miradb.db.queries``; ``run_sql`` is
an escape hatch for anything they do not cover. ``run_sql`` runs inside a
``READ ONLY`` transaction with a statement timeout, so even a statement that
slips past the syntax check cannot modify the database. For defense in depth,
point the server at a Postgres role that only has ``SELECT`` grants (see
the ``--db`` option of ``miradb-mcp``).
"""

import datetime
import decimal
import json
import math
import re

from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from miradb.db import queries
from miradb.db.client import MiraDatabaseClient
from miradb.db.schema import (
    Base,
    ExtractionMethod,
    MiraModel,
    ODEs,
    Scenario,
    ScenarioParameter,
    TextContent,
    TextRef,
)

DEFAULT_ROW_LIMIT = 200
MAX_ROW_LIMIT = 2000
DEFAULT_STATEMENT_TIMEOUT_MS = 15_000
DEFAULT_MAX_CELL_CHARS = 2000

_READ_ONLY_START = re.compile(
    r"^\s*(select|with|explain|values|table|show)\b", re.IGNORECASE
)

SCHEMA_NOTES = """\
## Provenance chain

text_references (one paper, keyed by pmid)
  -> text_contents (one per PDF/XML extraction method run on the paper)
    -> ode_expressions (SymPy source of the ODEs extracted from that content)
      -> mira_template_models (grounded MIRA TemplateModel built from the ODEs)
    -> scenarios (named parameterizations reported in the paper)
      -> scenario_parameters (one value/distribution per parameter)

scenarios.mira_model_ref points at the template model whose parameter names
were used when the scenario was extracted, so its parameters line up with
that model's parameters.

## JSON columns

- mira_template_models.mira_template_model: TemplateModel JSON with keys
  templates, parameters, initials, observables, annotations, time. Each
  template has type (e.g. ControlledConversion), rate_law, and concepts
  (subject/outcome/controller) with name, identifiers, context.
- mira_template_models.grounded_concepts: {variable_name: {identifiers,
  context}}; sometimes wrapped in a one-element list.
- scenario_parameters.distribution: e.g.
  {"type": "Uniform1", "parameters": {"minimum": 0.1, "maximum": 0.5}}.

Initial conditions are usually stored as scenario parameters named after the
compartment, e.g. E_0 or I0, not in the template model's initials.

## Query tips

- Cast JSON columns to jsonb for operators: mira_template_model::jsonb.
- Prefer the curated tools for common lookups; use SQL for aggregates and
  cross-paper questions.
"""


def describe_schema() -> str:
    """Return a Markdown description of the MIRA-DB tables and columns."""
    lines = ["# MIRA-DB schema", ""]
    for table in Base.metadata.sorted_tables:
        lines.append(f"## {table.name}")
        for col in table.columns:
            flags = []
            if col.primary_key:
                flags.append("PK")
            for fk in col.foreign_keys:
                flags.append(f"FK -> {fk.target_fullname}")
            if not col.nullable and not col.primary_key:
                flags.append("NOT NULL")
            suffix = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"- {col.name}: {col.type}{suffix}")
        lines.append("")
    lines.append(SCHEMA_NOTES)
    return "\n".join(lines)


def _jsonable(value, max_chars: int):
    """Convert a DB value to something JSON-serializable, truncating long
    values so a single cell cannot flood the agent's context."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple)):
        dumped = json.dumps(value, default=str)
        if len(dumped) <= max_chars:
            return value
        return dumped[:max_chars] + f"... [truncated, {len(dumped)} chars]"
    value = str(value)
    if len(value) > max_chars:
        return value[:max_chars] + f"... [truncated, {len(value)} chars]"
    return value


def check_read_only_sql(sql: str) -> str:
    """Validate that ``sql`` is a single read-only statement.

    This is a usability check that gives the agent a clear error message;
    the READ ONLY transaction in ``run_sql`` is what enforces it.

    Returns
    -------
    :
        The statement with surrounding whitespace and a trailing semicolon
        removed.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("Empty SQL statement.")
    if ";" in stripped:
        raise ValueError(
            "Only a single statement is allowed (found ';' inside the query)."
        )
    if not _READ_ONLY_START.match(stripped):
        raise ValueError(
            "Only read-only statements are allowed (SELECT, WITH, EXPLAIN, "
            "VALUES, TABLE, SHOW)."
        )
    return stripped


def run_sql(
    client: MiraDatabaseClient,
    sql: str,
    params: dict | None = None,
    *,
    limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    max_cell_chars: int = DEFAULT_MAX_CELL_CHARS,
) -> dict:
    """Run a read-only SQL statement and return at most ``limit`` rows.

    Returns
    -------
    :
        A dict with keys ``columns``, ``rows`` (list of dicts), ``row_count``
        and ``truncated`` (True when more than ``limit`` rows matched).
    """
    sql = check_read_only_sql(sql)
    limit = max(1, min(int(limit), MAX_ROW_LIMIT))
    timeout_ms = max(100, int(timeout_ms))

    try:
        with client.session() as sess:
            # Both SETs must run before the query, in the same transaction.
            sess.execute(text("SET TRANSACTION READ ONLY"))
            sess.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
            result = sess.execute(text(sql), params or {})
            if not result.returns_rows:
                return {"columns": [], "rows": [], "row_count": 0,
                        "truncated": False}
            columns = list(result.keys())
            fetched = result.mappings().fetchmany(limit + 1)
            sess.rollback()
    except DBAPIError as err:
        # Hand the agent Postgres' own message (syntax error, unknown
        # column, timeout, ...) so it can fix the query.
        raise ValueError(f"Query failed: {str(err.orig).strip()}") from None

    truncated = len(fetched) > limit
    rows = [
        {k: _jsonable(v, max_cell_chars) for k, v in row.items()}
        for row in fetched[:limit]
    ]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def search_publications(
    client: MiraDatabaseClient, query: str | None = None, limit: int = 25
) -> dict:
    """Search papers that have at least one converted model.

    With no ``query``, list papers ordered by model count.
    """
    if query and query.strip():
        results = queries.search_publication_summaries(client, query.strip())
    else:
        results = queries.list_publication_summaries(client)
    limit = max(1, min(int(limit), MAX_ROW_LIMIT))
    return {
        "total": len(results),
        "results": results[:limit],
    }


def get_publication(client: MiraDatabaseClient, pmid: str) -> dict | None:
    """Return a paper's metadata, its models and its scenarios.

    Models are listed by ``model_ref`` so they can be passed straight to the
    MIRA and simulation tools.
    """
    pmid = str(pmid).strip()
    ref = queries.get_text_ref(client, pmid)
    if ref is None:
        return None

    model_rows = client.query(
        select(
            MiraModel.id,
            ODEs.id.label("ode_id"),
            ExtractionMethod.extraction_method,
            func.json_array_length(
                MiraModel.mira_template_model.op("->")("templates")
            ).label("template_count"),
        )
        .join(ODEs, MiraModel.ode_ref == ODEs.id)
        .join(TextContent, ODEs.txt_content_ref == TextContent.id)
        .join(ExtractionMethod, ODEs.extraction_method_id == ExtractionMethod.id)
        .where(
            TextContent.text_ref == ref["id"],
            MiraModel.mira_template_model.isnot(None),
        )
    )
    models = sorted(
        (
            {
                "model_ref": f"db:{row['id']}",
                "ode_id": row["ode_id"],
                "extraction_method": row["extraction_method"],
                "template_count": row["template_count"],
            }
            for row in model_rows
        ),
        key=lambda m: queries._priority_index(m["extraction_method"]),
    )

    scenario_rows = client.query(
        select(
            Scenario.id,
            Scenario.name,
            Scenario.mira_model_ref,
            ExtractionMethod.extraction_method,
            func.count(ScenarioParameter.id).label("parameter_count"),
        )
        .join(TextContent, Scenario.txt_content_ref == TextContent.id)
        .join(ExtractionMethod, Scenario.extraction_method_id == ExtractionMethod.id)
        .outerjoin(ScenarioParameter, ScenarioParameter.scenario_ref == Scenario.id)
        .where(TextContent.text_ref == ref["id"])
        .group_by(Scenario.id, ExtractionMethod.extraction_method)
        .order_by(Scenario.id)
    )
    scenarios = [
        {
            "scenario_id": row["id"],
            "name": row["name"],
            "extraction_method": row["extraction_method"],
            "model_ref": (
                f"db:{row['mira_model_ref']}" if row["mira_model_ref"] else None
            ),
            "parameter_count": row["parameter_count"],
        }
        for row in scenario_rows
    ]

    return {
        "pmid": ref["pmid"],
        "doi": ref["doi"],
        "pmcid": ref["pmcid"],
        "title": ref["title"],
        "authors": ref["authors"],
        "journal": ref["journal"],
        "year": ref["year"],
        "keywords": ref["keywords"],
        "models": models,
        "scenarios": scenarios,
    }


def get_odes(client: MiraDatabaseClient, pmid: str) -> list[dict]:
    """Return the extracted SymPy ODE source for a paper, per extraction."""
    methods = {
        m["id"]: m["extraction_method"]
        for m in queries.get_all_extraction_methods(client)
    }
    odes = queries.list_odes_for_pmid(client, str(pmid).strip())
    for ode in odes:
        ode["extraction_method"] = methods.get(ode["extraction_method_id"])
    return odes


def get_scenarios(client: MiraDatabaseClient, pmid: str) -> list[dict]:
    """Return a paper's scenarios with all parameter values."""
    groups = queries.list_scenarios_for_pmid(client, str(pmid).strip())
    for group in groups:
        for scenario in group["scenarios"]:
            model_id = scenario.pop("mira_model_ref", None)
            scenario["model_ref"] = f"db:{model_id}" if model_id else None
    return groups


def get_scenario(client: MiraDatabaseClient, scenario_id: int) -> dict | None:
    """Return one scenario and its parameters by scenario ID."""
    rows = client.query(
        select(
            Scenario.id,
            Scenario.name,
            Scenario.description,
            Scenario.mira_model_ref,
            ExtractionMethod.extraction_method,
            TextRef.pmid,
        )
        .join(TextContent, Scenario.txt_content_ref == TextContent.id)
        .join(TextRef, TextContent.text_ref == TextRef.id)
        .join(ExtractionMethod, Scenario.extraction_method_id == ExtractionMethod.id)
        .where(Scenario.id == int(scenario_id))
    )
    if not rows:
        return None
    row = rows[0]
    params = client.query(
        select(
            ScenarioParameter.name,
            ScenarioParameter.display_name,
            ScenarioParameter.description,
            ScenarioParameter.value,
            ScenarioParameter.value_expression,
            ScenarioParameter.units,
            ScenarioParameter.distribution,
        )
        .where(ScenarioParameter.scenario_ref == row["id"])
        .order_by(ScenarioParameter.id)
    )
    return {
        "scenario_id": row["id"],
        "pmid": row["pmid"],
        "name": row["name"],
        "description": row["description"],
        "extraction_method": row["extraction_method"],
        "model_ref": (
            f"db:{row['mira_model_ref']}" if row["mira_model_ref"] else None
        ),
        "parameters": [
            {k: _jsonable(v, DEFAULT_MAX_CELL_CHARS) for k, v in p.items()}
            for p in params
        ],
    }
