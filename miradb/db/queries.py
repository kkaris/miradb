import json
import logging
import math
import copy

from sqlalchemy import Text, cast, func, literal, or_, select
from sqlalchemy.sql.expression import text as sa_text
from sympy import Derivative, latex

from miradb.db.client import MiraDatabaseClient
from miradb.db.schema import MiraModel, ODEs, TextContent, TextRef
from mira.metamodel import TemplateModel
from mira.metamodel.template_model import Time
from mira.modeling import Model
from mira.modeling.ode import OdeModel


logger = logging.getLogger(__name__)

EXTRACTION_METHOD_LABELS = {
    1: "Marker Extraction",
    2: "MinerU Image Pipeline",
    3: "MinerU Text Extraction",
    4: "XML Extraction",
}


def _derivative_to_latex(expr) -> str:
    """Render Derivative(X, t) as \\frac{dX}{dt} instead of \\frac{d}{dt} X."""
    if isinstance(expr, Derivative) and len(expr.args) == 2:
        var, (wrt, _) = expr.args
        return r"\frac{d" + latex(var) + r"}{d" + latex(wrt) + r"}"
    return latex(expr)


def _template_to_latex_lines(tm, ode_id) -> tuple[list[str], dict | None]:
    if not tm or not tm.get("mira_template_model"):
        return [], []

    try:
        raw = tm["mira_template_model"]
        if isinstance(raw, str):
            raw = json.loads(raw)

        loaded_model = TemplateModel.from_json(raw)
        loaded_model.time = Time(name='t', units=None)

        om = OdeModel(
            model=Model(template_model=loaded_model),
            initialized=False,
        )

        kinetics = om.get_interpretable_kinetics()
        latex_lines = []

        if hasattr(kinetics, 'tolist'):
            rows = kinetics.tolist()
            for row in rows:
                if len(row) == 3:
                    lhs, _, rhs = row
                    latex_lines.append(_derivative_to_latex(lhs) + " = " + latex(rhs))
                elif len(row) == 2:
                    lhs, rhs = row
                    latex_lines.append(_derivative_to_latex(lhs) + " = " + latex(rhs))
                else:
                    for expr in row:
                        latex_lines.append(latex(expr))

        elif isinstance(kinetics, (list, tuple)):
            for expr in kinetics:
                latex_lines.append(latex(expr))

        else:
            latex_lines.append(latex(kinetics))

        raw_gc = tm.get("grounded_concepts")
        if isinstance(raw_gc, str):
            try:
                raw_gc = json.loads(raw_gc)
            except Exception:
                raw_gc = None

        return latex_lines, raw_gc

    except Exception:
        logger.exception("Failed to render LaTeX for ode id=%s", ode_id)
        return [], []


def _ode_count_subquery(*, outer_join_odes: bool = True):
    join_kwargs = {"isouter": True} if outer_join_odes else {}
    return (
        select(
            TextContent.text_ref,
            func.count(ODEs.id).label("ode_count"),
        )
        .join(ODEs, ODEs.txt_content_ref == TextContent.id, **join_kwargs)
        .group_by(TextContent.text_ref)
        .subquery()
    )


def _format_publication_summary(row) -> dict:
    return {
        "pmid": row["pmid"],
        "title": row["title"] or "",
        "author_list": ", ".join(row["authors"]) if row["authors"] else "",
        "pub_year": int(row["year"]),
        "model_count": int(row["model_count"]),
    }


def _publication_summary_select(ode_count_sq):
    return (
        select(
            TextRef.pmid,
            TextRef.title,
            TextRef.authors,
            TextRef.year,
            func.coalesce(ode_count_sq.c.ode_count, 0).label("model_count"),
        )
        .outerjoin(ode_count_sq, ode_count_sq.c.text_ref == TextRef.id)
    )


def _pmids_matching_grounded_concepts_subquery():
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
    return (
        select(TextRef.pmid)
        .join(TextContent, TextContent.text_ref == TextRef.id)
        .join(ODEs, ODEs.txt_content_ref == TextContent.id)
        .join(MiraModel, MiraModel.ode_ref == ODEs.id)
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


def list_publication_summaries(client: MiraDatabaseClient) -> list[dict]:
    """Return every text_reference with a count of associated ode_expressions.

    Parameters
    ----------
    client :
        An instance of MiraDatabaseClient.

    Returns
    -------
    :
        A list of dictionaries, each containing a publication summary.
        The dictionary keys are:
        - pmid: The PubMed ID of the publication.
        - title: The title of the publication.
        - author_list: The list of authors of the publication.
        - pub_year: The year of the publication.
        - model_count: The number of ODE expressions associated with the publication.

    Example
    -------
    [
        {
            "pmid": "1234567890",
            "title": "A study of the effects of caffeine on productivity",
            "author_list": "John Doe, Jane Doe",
            "pub_year": 2020,
            "model_count": 2,
        },
        ...
    ]
    """
    ode_count_sq = _ode_count_subquery(outer_join_odes=True)
    stmt = (
        _publication_summary_select(ode_count_sq)
        .order_by(func.coalesce(ode_count_sq.c.ode_count, 0).desc())
    )
    rows = client.query(stmt)
    return [_format_publication_summary(row) for row in rows]


def search_publication_summaries(client: MiraDatabaseClient, q: str) -> list[dict]:
    """Search text_references by metadata and grounded_concepts JSON.

    Parameters
    ----------
    client :
        An instance of MiraDatabaseClient.
    q :
        Search string (non-empty; callers should validate).

    Returns
    -------
    :
        A list of publication summary dicts. Matches against pmid, title,
        authors, year, and grounded_concepts (variable names, ontology IDs,
        context keys and values).
    """
    pattern = f"%{q.lower()}%"
    ode_count_sq = _ode_count_subquery(outer_join_odes=False)
    gc_pmid_sq = _pmids_matching_grounded_concepts_subquery()
    stmt = (
        _publication_summary_select(ode_count_sq)
        .where(
            or_(
                TextRef.pmid.ilike(pattern),
                TextRef.title.ilike(pattern),
                cast(TextRef.authors, Text).ilike(pattern),
                cast(TextRef.year, Text).ilike(pattern),
                TextRef.pmid.in_(gc_pmid_sq),
            )
        )
        .order_by(func.coalesce(ode_count_sq.c.ode_count, 0).desc())
    )
    rows = client.query(stmt, {"pattern": pattern})
    return [_format_publication_summary(row) for row in rows]


def list_models_for_pmid(client: MiraDatabaseClient, pmid: str) -> list[dict]:
    """Return all ode_expressions for a PMID with LaTeX-rendered equations.

    Parameters
    ----------
    client :
        An instance of MiraDatabaseClient.
    pmid :
        The PubMed ID of the publication.

    Returns
    -------
    :
        A list of dictionaries, each describing one extracted model.
        The dictionary keys are:
        - id: The ode_expressions row ID.
        - extraction_method: 0-based extraction method index for the frontend.
        - method_label: Human-readable extraction method name.
        - latex: List of LaTeX equation strings.
        - grounded_concepts: Grounded concept metadata for the model.
    """
    pmid = str(pmid)
    text_ref = client.query_one(
        select(TextRef.id).where(TextRef.pmid == pmid)
    )
    if not text_ref:
        return []

    stmt = (
        select(
            ODEs.id,
            ODEs.extraction_method_id,
            ODEs.ode,
            ODEs.corrected_ode,
            MiraModel.mira_template_model,
            MiraModel.grounded_concepts,
        )
        .join(TextContent, ODEs.txt_content_ref == TextContent.id)
        .outerjoin(MiraModel, MiraModel.ode_ref == ODEs.id)
        .where(TextContent.text_ref == text_ref["id"])
    )
    rows = client.query(stmt)

    results = []
    for row in rows:
        tm = None
        if row["mira_template_model"] is not None:
            tm = {
                "mira_template_model": row["mira_template_model"],
                "grounded_concepts": row["grounded_concepts"],
            }

        latex_lines, grounded_concepts = _template_to_latex_lines(tm, row["id"])
        if not latex_lines:
            latex_lines = [row.get("corrected_ode") or row.get("ode")]

        method = row["extraction_method_id"]
        results.append({
            "id": row["id"],
            "extraction_method": method - 1,
            "method_label": EXTRACTION_METHOD_LABELS.get(method, f"Method {method}"),
            "latex": latex_lines,
            "grounded_concepts": grounded_concepts or {},
        })

    results.sort(key=lambda r: r["extraction_method"])
    return results


def get_template_model_by_ode_id(client: MiraDatabaseClient, ode_id: int) -> TemplateModel | None:
    """Look up the mira_template_models row whose ode_ref == ode_id and
    deserialize it into a TemplateModel.  Returns None if not found.

    Parameters
    ----------
    client :
        An instance of MiraDatabaseClient.
    ode_id :
        The ID of the ODE.

    Returns
    -------
    :
        A TemplateModel if found, otherwise None.
    """
    row = client.query_one(
        select(MiraModel.mira_template_model, MiraModel.grounded_concepts).where(
            MiraModel.ode_ref == ode_id
        )
    )

    if not row or not row.get("mira_template_model"):
        return None

    raw = row["mira_template_model"]

    # Todo: should be removed once sure that the database is not returning strings
    if isinstance(raw, str):
        raw = json.loads(raw)

    loaded_model = TemplateModel.from_json(raw)
    loaded_model.time = Time(name='t', units=None)

    return loaded_model


def sanitize_tm_for_sbml(tm: TemplateModel) -> TemplateModel:
    """Replace None/non-numeric parameter values with 0.0 for SBML export.

    Parameters
    ----------
    tm :
        The template model to sanitize.

    Returns
    -------
    :
        The sanitized template model.
    """
    sanitized_tm = copy.deepcopy(tm)
    for param in sanitized_tm.parameters.values():
        if param.value is None or (isinstance(param.value, float) and math.isnan(param.value)):
            param.value = 0.0
        elif not isinstance(param.value, (int, float)):
            try:
                param.value = float(param.value)
            except (TypeError, ValueError):
                param.value = 0.0
    return sanitized_tm
