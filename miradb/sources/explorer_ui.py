import logging

from flask import Blueprint, jsonify, render_template, request, send_file, abort
from sqlalchemy import select, Table, MetaData, func, or_, cast, Text, literal
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import text as sa_text
import json
import io
from miradb.db.manager import get_db
from miradb.db.client import get_client
from miradb.db import queries
from mira.modeling import Model
from mira.modeling.ode import OdeModel
from mira.metamodel import TemplateModel
from mira.metamodel.template_model import Time
from mira.modeling.sbml import template_model_to_sbml_string


logger = logging.getLogger(__name__)

explorer_blueprint = Blueprint("explorer", __name__, url_prefix="/explorer")
explorer_blueprint.template_folder = "templates"

# ── DB setup ──────────────

client = get_client('primary')

db = get_db('primary')
engine = db.engine

metadata = MetaData()
text_references      = Table("text_references",      metadata, autoload_with=engine)
text_contents        = Table("text_contents",        metadata, autoload_with=engine)
ode_expressions      = Table("ode_expressions",      metadata, autoload_with=engine)
mira_template_models = Table("mira_template_models", metadata, autoload_with=engine)

Session = sessionmaker(bind=engine)


# ── Routes ───────────────────────────────────────────────────────────────────

@explorer_blueprint.route("/")
def index():
    """Serve the explorer SPA shell."""
    return render_template("explorer.html")


@explorer_blueprint.route("/api/search")
def search_pmids():
    """
    Server-side search across text_references metadata and grounded_concepts JSON.
    Returns empty list if no query is provided.

    Query param: q (string)

    Searches:
      - text_references: pmid, title, author_list (cast to text), pub_year (cast to text)
      - mira_template_models.grounded_concepts JSON (cast to text, ILIKE match)
        covers variable names, ontology IDs, context keys and values

    Response shape: same as /api/pmids
    [{"pmid": "...", "title": "...", "author_list": "...", "pub_year": ..., "model_count": N}]
    """
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    pattern = f"%{q.lower()}%"

    with Session() as session:
        # Subquery: count ode_expressions per text_reference
        ode_count_sq = (
            select(
                text_contents.c.text_ref,
                func.count(ode_expressions.c.id).label("ode_count"),
            )
            .join(ode_expressions, ode_expressions.c.txt_content_ref == text_contents.c.id)
            .group_by(text_contents.c.text_ref)
            .subquery()
        )

        # Define the lateral subquery properly
        gc_lateral = (
            select(
                sa_text("var_key"),
                sa_text("var_val"),
            )
            .select_from(
                sa_text("""
                    jsonb_each(
                        CASE
                            WHEN jsonb_typeof(mira_template_models.grounded_concepts::jsonb) = 'array'
                                AND jsonb_typeof(mira_template_models.grounded_concepts::jsonb -> 0) = 'object'
                            THEN mira_template_models.grounded_concepts::jsonb -> 0
                            WHEN jsonb_typeof(mira_template_models.grounded_concepts::jsonb) = 'object'
                            THEN mira_template_models.grounded_concepts::jsonb
                            ELSE '{}'::jsonb
                        END
                    ) AS gc(var_key, var_val)
                """)
            )
            .lateral("gc_rows")
        )

        gc_pmid_sq = (
            select(text_references.c.pmid)
            .join(text_contents, text_contents.c.text_ref == text_references.c.id)
            .join(ode_expressions, ode_expressions.c.txt_content_ref == text_contents.c.id)
            .join(mira_template_models, mira_template_models.c.ode_ref == ode_expressions.c.id)
            .join(gc_lateral, literal(True))
            .where(
                or_(
                    sa_text("gc_rows.var_key ILIKE :pattern"),
                    sa_text("""
                        EXISTS (
                            SELECT 1 FROM jsonb_each_text(gc_rows.var_val -> 'identifiers') AS id(k, v)
                            WHERE id.k ILIKE :pattern OR id.v ILIKE :pattern
                        )
                    """),
                    sa_text("""
                        EXISTS (
                            SELECT 1 FROM jsonb_each_text(gc_rows.var_val -> 'context') AS ctx(k, v)
                            WHERE ctx.k ILIKE :pattern OR ctx.v ILIKE :pattern
                        )
                    """),
                )
            )
            .distinct()
            .subquery()
        )

        rows = session.execute(
            select(
                text_references.c.pmid,
                text_references.c.title,
                text_references.c.authors,
                text_references.c.year,
                func.coalesce(ode_count_sq.c.ode_count, 0).label("model_count"),
            )
            .outerjoin(ode_count_sq, ode_count_sq.c.text_ref == text_references.c.id)
            .where(
                or_(
                    text_references.c.pmid.ilike(pattern),
                    text_references.c.title.ilike(pattern),
                    cast(text_references.c.authors, Text).ilike(pattern),
                    cast(text_references.c.year,   Text).ilike(pattern),
                    text_references.c.pmid.in_(gc_pmid_sq),
                )
            )
            .order_by(func.coalesce(ode_count_sq.c.ode_count, 0).desc()),
            {"pattern": pattern}
        ).mappings().all()

    return jsonify([
        {
            "pmid":        r["pmid"],
            "title":       r["title"] or "",
            "author_list": ", ".join(r["authors"]) if r["authors"] else "",
            "pub_year":    r["year"],
            "model_count": r["model_count"],
        }
        for r in rows
    ])


@explorer_blueprint.route("/api/pmids")
def get_all_pmids():
    """Return every text_reference with a count of its associated ode_expressions.

    Response shape (list):
    [
      {
        "pmid":       "33451107",
        "title":      "A SEIR model …",
        "author_list": "Zhang et al.",
        "pub_year":   2021,
        "model_count": 3
      },
      …
    ]
    """
    rows = queries.list_publication_summaries(client)
    return jsonify(rows)


@explorer_blueprint.route("/api/pmids/<pmid>/models")
def get_models_for_pmid(pmid: str):
    """
    Returns all ode_expressions for a given PMID, with LaTeX-rendered equations.

    Response shape (list):
    [
      {
        "id":                1,
        "extraction_method_id": 0,
        "method_label":      "Multi-Agent Pipeline",
        "latex":             ["\\frac{dS}{dt} = …", …]
        "grounded_concepts": {}
      },
      …
    ]
    """
    results = queries.list_models_for_pmid(client, pmid)
    return jsonify(results)


@explorer_blueprint.route("/api/models/<int:ode_id>/download/json")
def download_json(ode_id: int):
    """Export the MIRA TemplateModel as JSON."""
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
    """Export the model as SBML via MIRA."""
    tm = queries.get_template_model_by_ode_id(client, ode_id)
    if tm is None:
        abort(404, description=f"No TemplateModel found for ode id {ode_id}")

    try:
        tm_clean = queries.sanitize_tm_for_sbml(tm)
        sbml_str = template_model_to_sbml_string(tm_clean)
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
    """
    Export ODEs as a SymPy .py file derived from a MIRA TemplateModel.

    Parameters
    ----------
    ode_id : int
        Identifier of the ODE model to export.

    Returns
    -------
    flask.Response
        Downloadable SymPy .py file for the ODE model.
    """
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
        logger.exception("SymPy export failed for ode_id=%s", ode_id)
        abort(500, description="SymPy ODE export failed — see server logs.")

    lines = [
        "from sympy import *",
        "",
    ]

    # Declare all symbols that appear across lhs + rhs
    all_symbols: set[str] = set()
    import re
    symbol_re = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
    skip = {"Derivative", "Function", "Symbol", "symbols", "t"}
    for lhs, rhs in ode_pairs:
        for token in symbol_re.findall(lhs + " " + rhs):
            if token not in skip:
                all_symbols.add(token)

    if all_symbols:
        lines.append("# Declare symbols")
        lines.append(f"{', '.join(sorted(all_symbols))} = symbols('{' '.join(sorted(all_symbols))}')")
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

