import logging
import re
import math

from flask import Blueprint, jsonify, render_template, request, send_file, abort
import json
import io
from miradb.db.client import get_client
from miradb.db import queries
from mira.modeling import Model
from mira.modeling.ode import OdeModel
from mira.metamodel.template_model import Time
from mira.modeling.sbml import template_model_to_sbml_string


logger = logging.getLogger(__name__)

explorer_blueprint = Blueprint("explorer", __name__, url_prefix="/explorer")
explorer_blueprint.template_folder = "templates"

client = get_client("primary")
symbol_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


@explorer_blueprint.route("/")
def index():
    """Serve the explorer SPA shell."""
    return render_template("explorer.html")


@explorer_blueprint.route("/api/search")
def search_pmids():
    """Search across text_references metadata and grounded_concepts JSON.

    Searches:
      - text_references: pmid, title, author_list, pub_year
      - mira_template_models.grounded_concepts JSON. Covers variable names,
        ontology IDs, context keys and values

    Response format:
    [
        {
            "pmid": "...",
            "title": "...",
            "author_list": "...",
            "pub_year": ...,
            "model_count": N
        },
        ...
    ]

    Returns empty list if no query is provided.
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    return jsonify(queries.search_publication_summaries(client, q))


@explorer_blueprint.route("/api/pmids")
def get_all_pmids():
    """Return all text_references and its associated count of ode_expressions

    Response format:
    [
        {
            "pmid": "33451107",
            "title": "An SEIR model of influenza",
            "author_list": "Zhang et al.",
            "pub_year":   2021,
            "model_count": 3
        },
        ...
    ]
    """
    rows = queries.list_publication_summaries(client)
    return jsonify(rows)


@explorer_blueprint.route("/api/pmids/<pmid>/models")
def get_models_for_pmid(pmid: str):
    """Return all ode_expressions for a PMID, with LaTeX-rendered equations.

    Response format:
    [
        {
            "id": 1,
            "extraction_method_id": 0,
            "method_label": "Multi-Agent Pipeline",
            "latex": ["\\frac{dS}{dt} = ...", ...]
            "grounded_concepts": {}
        },
      ...
    ]
    """
    results = queries.list_models_for_pmid(client, pmid)
    return jsonify(results)


@explorer_blueprint.route("/api/models/<int:ode_id>/download/json")
def download_json(ode_id: int):
    """Export the given model as a MIRA TemplateModel JSON"""
    tm = queries.get_template_model_by_ode_id(client, ode_id)
    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    json_bytes = json.dumps(tm.model_dump(), indent=2).encode("utf-8")

    return send_file(
        io.BytesIO(json_bytes),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"model_{ode_id}.json",
    )


@explorer_blueprint.route("/api/models/<int:ode_id>/download/sbml")
def download_sbml(ode_id: int):
    """Export the given model as SBML"""
    tm = queries.get_template_model_by_ode_id(client, ode_id)
    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    try:
        # SBML does not allow empty/nan parameter values
        for param in tm.parameters.values():
            if param.value is None or (
                isinstance(param.value, float) and math.isnan(param.value)
            ):
                param.value = 0.0
            elif not isinstance(param.value, (int, float)):
                try:
                    param.value = float(param.value)
                except (TypeError, ValueError):
                    param.value = 0.0
        sbml_str = template_model_to_sbml_string(tm)
    except Exception as e:
        logger.error(f"SBML export failed for ode id {ode_id}")
        logger.exception(e)
        abort(500, description=f"SBML export failed for ode id {ode_id}")

    return send_file(
        io.BytesIO(sbml_str.encode()),
        mimetype="application/xml",
        as_attachment=True,
        download_name=f"model_{ode_id}.xml",
    )


@explorer_blueprint.route("/api/models/<int:ode_id>/download/sympy")
def download_sympy(ode_id: int):
    """Export ODEs as a SymPy .py file"""
    tm = queries.get_template_model_by_ode_id(client, ode_id)

    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    try:
        tm.time = Time(name="t", units=None)
        om = OdeModel(model=Model(template_model=tm), initialized=False)
        kinetics = om.get_interpretable_kinetics()

        # Collect (lhs_str, rhs_str) pairs
        ode_pairs = []
        if hasattr(kinetics, "tolist"):
            for row in kinetics.tolist():
                if len(row) == 3:
                    lhs, _, rhs = row
                    ode_pairs.append((str(lhs), str(rhs)))
                elif len(row) == 2:
                    lhs, rhs = row
                    ode_pairs.append((str(lhs), str(rhs)))
        elif isinstance(kinetics, (list, tuple)):
            for expr in kinetics:
                ode_pairs.append((str(expr), ""))
        else:
            ode_pairs.append((str(kinetics), ""))

    except Exception:
        logger.exception(f"SymPy export failed for ode_id={ode_id}")
        abort(500, description="SymPy ODE export failed - see server logs.")

    lines = [
        "from sympy import *",
        "",
    ]

    # Declare all symbols that appear across lhs + rhs
    all_symbols: set[str] = set()
    skip = {"Derivative", "Function", "Symbol", "symbols", "t"}
    for lhs, rhs in ode_pairs:
        for token in symbol_re.findall(lhs + " " + rhs):
            if token not in skip:
                all_symbols.add(token)

    if all_symbols:
        lines.append("# Declare symbols")
        lines.append(
            f"{', '.join(sorted(all_symbols))} = symbols('{' '.join(sorted(all_symbols))}')"
        )
        lines.append("")

    for lhs, rhs in ode_pairs:
        if rhs:
            lines.append(f"# {lhs} = {rhs}")
        else:
            lines.append(f"# {lhs}")

    lines += [
        "",
        "odes = {",
    ]
    for lhs, rhs in ode_pairs:
        if rhs:
            lines.append(f"    {lhs!r}: {rhs},")
    lines.append("}")

    py_str = "\n".join(lines) + "\n"

    return send_file(
        io.BytesIO(py_str.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"model_{ode_id}.py",
    )
