import json
import logging

from sqlalchemy import func, select
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
    ode_count_sq = (
        select(
            TextContent.text_ref,
            func.count(ODEs.id).label("ode_count"),
        )
        .join(ODEs, ODEs.txt_content_ref == TextContent.id, isouter=True)
        .group_by(TextContent.text_ref)
        .subquery()
    )
    stmt = (
        select(
            TextRef.pmid,
            TextRef.title,
            TextRef.authors,
            TextRef.year,
            func.coalesce(ode_count_sq.c.ode_count, 0).label("model_count"),
        )
        .outerjoin(ode_count_sq, ode_count_sq.c.text_ref == TextRef.id)
        .order_by(func.coalesce(ode_count_sq.c.ode_count, 0).desc())
    )
    rows = client.query(stmt)
    return [
        {
            "pmid": row["pmid"],
            "title": row["title"] or "",
            "author_list": ", ".join(row["authors"]) if row["authors"] else "",
            "pub_year": int(row["year"]),
            "model_count": int(row["model_count"]),
        }
        for row in rows
    ]


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
