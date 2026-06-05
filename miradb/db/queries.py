import json
import logging
import math
import copy

from sqlalchemy import Text, cast, func, literal, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.expression import text as sa_text
from sympy import Derivative, latex

from miradb.db.client import MiraDatabaseClient
from miradb.db.schema import ExtractionMethod, MiraModel, ODEs, TextContent, TextRef
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


def list_odes_for_pmid(client: MiraDatabaseClient, pmid: str) -> list[dict]:
    """Return ode_expressions for a PMID as SymPy source strings.

    Parameters
    ----------
    client
        Database client.
    pmid
        PubMed ID of the publication.

    Returns
    -------
    list of dict
        Each dict has keys ``id``, ``extraction_method_id``, ``ode``,
        and ``corrected_ode``.
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
        )
        .join(TextContent, ODEs.txt_content_ref == TextContent.id)
        .where(TextContent.text_ref == text_ref["id"])
    )
    return [
        {
            "id": row["id"],
            "extraction_method_id": row["extraction_method_id"],
            "ode": row["ode"],
            "corrected_ode": row["corrected_ode"],
        }
        for row in client.query(stmt)
    ]


def add_extraction_method(
    client: MiraDatabaseClient,
    extraction_method: str,
    extraction_method_desc: str = None,
):
    """Add an extraction method to the registry.

    Parameters
    ----------
    client :
        The database client to use.
    extraction_method :
        The short name of the extraction method (e.g., ``mineru_text``,
        ``mineru_image``, ``marker``).
    extraction_method_desc :
        A human-readable description of the extraction pipeline.

    Returns
    -------
    :
        The ID of the extraction method entry that was added.
    """
    def _write(sess):
        row = ExtractionMethod(
            extraction_method=extraction_method,
            extraction_method_desc=extraction_method_desc,
        )
        sess.add(row)
        sess.flush()
        sess.refresh(row)
        logger.info("Registered extraction method '%s'.", extraction_method)
        return row.id
    return client.mutate(_write)


def get_extraction_method_id(
    client: MiraDatabaseClient,
    extraction_method: str,
) -> int | None:
    """Look up an extraction method's ID by its short name.

    Parameters
    ----------
    client :
        The database client to use.
    extraction_method :
        The short name of the extraction method (e.g., ``mineru_text``,
        ``xml_extraction``).

    Returns
    -------
    :
        The ID of the matching extraction method entry, or None if not found.
    """
    stmt = select(ExtractionMethod.id).where(
        ExtractionMethod.extraction_method == extraction_method
    )
    row = client.query_one(stmt)
    return row["id"] if row else None


def get_all_extraction_methods(client: MiraDatabaseClient) -> list[dict]:
    """Retrieve all registered extraction methods.

    Parameters
    ----------
    client :
        The database client to use.

    Returns
    -------
    :
        A list of dictionaries, each containing an extraction method entry.
        The dictionary keys are:
            - id
            - extraction_method
            - extraction_method_desc
    """
    with client.session() as sess:
        rows = sess.query(ExtractionMethod).order_by(ExtractionMethod.id).all()
        return [row.to_dict() for row in rows]


def add_text_ref(
    client: MiraDatabaseClient,
    pmid: str,
    pmcid: str = None,
    doi: str = None,
    authors: str = None,
    title: str = None,
    journal: str = None,
    year: int = None,
    keywords: list = None,
):
    """Add a paper to the database, returning the new paper's ID.

    Parameters
    ----------
    client :
        The database client to use.
    pmid :
        The PubMed ID of the paper.
    pmcid :
        The PubMed Central ID of the paper.
    doi :
        The DOI of the paper.
    authors : str, optional
        The authors of the paper.
    title : str, optional
        The title of the paper.
    journal : str, optional
        The journal of the paper.
    year : int, optional
        The publication year of the paper.
    keywords : list, optional
        A list of keywords associated with the paper.

    Returns
    -------
    int
        The ID of the paper that was added.
    """
    try:
        def _write(sess):
            p = TextRef(
                pmid=pmid, pmcid=pmcid, doi=doi, authors=authors, title=title,
                journal=journal, year=year, keywords=keywords,
            )
            sess.add(p)
            sess.flush()
            sess.refresh(p)
            logger.info("Registered paper '%s'.", pmid)
            return p.id
        return client.mutate(_write)
    except IntegrityError:
        logger.warning("A paper with pmid %s already exists.", pmid)
        return None


def update_text_ref(
    client: MiraDatabaseClient,
    pmid: str,
    pmcid: str = None,
    doi: str = None,
    authors: str = None,
    title: str = None,
    journal: str = None,
    year: int = None,
    keywords: list = None,
) -> bool:
    """Update a paper's identifiers in the database.

    Parameters
    ----------
    client :
        The database client to use.
    pmid :
        The PubMed ID of the paper to update.
    pmcid :
        The new PubMed Central ID of the paper.
    doi :
        The new DOI of the paper.
    authors :
        The new authors of the paper.
    title :
        The new title of the paper.
    journal :
        The new journal of the paper.
    year :
        The new publication year of the paper.
    keywords :
        A list of keywords to update associated with the paper.

    Returns
    -------
    bool
        True if the paper was found and updated, False if the paper was not found.
    """
    def _write(sess):
        p = sess.query(TextRef).filter_by(pmid=pmid).first()
        if not p:
            logger.warning("Paper '%s' not found.", pmid)
            return False
        if pmcid is not None:
            p.pmcid = pmcid
        if doi is not None:
            p.doi = doi
        if authors is not None:
            p.authors = authors
        if title is not None:
            p.title = title
        if journal is not None:
            p.journal = journal
        if year is not None:
            p.year = year
        if keywords is not None:
            p.keywords = keywords
        logger.info("Updated paper '%s'.", pmid)
        return True
    return client.mutate(_write)


def get_text_ref(client: MiraDatabaseClient, pmid: str) -> dict | None:
    """Retrieve a paper's text identifiers by PubMed ID

    Parameters
    ----------
    client :
        The database client to use.
    pmid :
        The PubMed ID of the paper to retrieve.

    Returns
    -------
    :
        A dictionary containing the paper's information if found, or None if not found.
        Dictionary keys: 'id', 'pmid', 'doi', 'pmcid', 'created_at', 'updated_at', 'authors', 'title', 'journal', 'year', 'keywords'
    """
    with client.session() as sess:
        paper = sess.query(TextRef).filter_by(pmid=pmid).first()
        return paper.to_dict() if paper else None


def get_all_text_refs(client: MiraDatabaseClient) -> list[dict]:
    """Retrieve all papers' text identifiers

    Parameters
    ----------
    client :
        The database client to use.

    Returns
    -------
    :
        A list of dictionaries, each containing a paper's information.
        The dictionary keys are:
            - id
            - pmid
            - doi
            - pmcid
            - created_at
            - updated_at
            - authors
            - title
            - journal
            - year
            - keywords
    """
    with client.session() as sess:
        papers = sess.query(TextRef).order_by(TextRef.created_at.desc()).all()
        return [p.to_dict() for p in papers]


def remove_text_ref(client: MiraDatabaseClient, pmid: str) -> bool:
    """Delete a paper and all FK-linked rows

    Parameters
    ----------
    client :
        The database client to use.
    pmid : str
        The PubMed ID of the paper to delete.

    Returns
    -------
    :
        True if the paper was found and deleted, False if the paper was not found.
    """
    def _write(sess):
        p = sess.query(TextRef).filter_by(pmid=pmid).first()
        if not p:
            logger.warning("Paper '%s' not found.", pmid)
            return False
        sess.delete(p)
        logger.info("Deleted paper '%s' and all linked rows.", pmid)
        return True
    return client.mutate(_write)


def add_text_content(
    client: MiraDatabaseClient,
    text_ref: int,
    folder_path: str,
    extraction_method_id: int,
    extracted_info_path: str,
):
    """Add a paper's source locations to the database

    Parameters
    ----------
    client :
        The database client to use.
    text_ref :
        The ID of the text reference that this source information is linked to.
    folder_path :
        The relative file path to the folder containing the paper's source files (e.g., PDFs, images).
    extraction_method_id :
        The ID of the extraction method used to extract the ODE string e.g., 0 for MinerU image extraction.
    extracted_info_path :
        The relative file path to the extracted information for the paper.

    Returns
    -------
    :
        The ID of the new text content entry that was added.
    """
    try:
        def _write(sess):
            row = TextContent(
                text_ref=text_ref,
                folder_path=folder_path,
                extraction_method_id=extraction_method_id,
                extracted_info_path=extracted_info_path,
            )
            sess.add(row)
            sess.flush()
            sess.refresh(row)
            logger.info("Registered paper source '%s'.", text_ref)
            return row.id
        return client.mutate(_write)
    except IntegrityError:
        logger.warning("A paper source with text_ref %s already exists.", text_ref)
        return None


def update_text_content(
    client: MiraDatabaseClient,
    id: int,
    folder_path: str = None,
    extraction_method_id: int = None,
    extracted_info_path: str = None,
) -> bool:
    """Update a paper source's information in the database.

    Parameters
    ----------
    client :
        The database client to use.
    id :
        The ID of the text content entry to update.
    folder_path :
        The relative file path to the folder containing the paper's source files (e.g., PDFs, images).
    extraction_method_id :
        The ID of the extraction method used to extract the ODE string.
    extracted_info_path :
        The relative file path to the extracted information for the paper.

    Returns
    -------
    bool
        True if the paper source was found and updated, False if the paper source was not found.
    """
    def _write(sess):
        p_source = sess.query(TextContent).filter_by(id=id).first()
        if not p_source:
            logger.warning("Paper source '%s' not found.", id)
            return False
        if folder_path is not None:
            p_source.folder_path = folder_path
        if extraction_method_id is not None:
            p_source.extraction_method_id = extraction_method_id
        if extracted_info_path is not None:
            p_source.extracted_info_path = extracted_info_path
        logger.info("Updated paper source '%s'.", id)
        return True
    return client.mutate(_write)


def get_text_content(client: MiraDatabaseClient, text_ref: int):
    """Retrieve a paper's source information by text reference ID.

    Parameters
    ----------
    client :
        The database client to use.
    text_ref :
        The ID of the text reference that this source information is linked to.

    Returns
    -------
    :
        A list of dictionaries containing paper source information if found,
        or None if not found. Each dictionary has keys:
            - id
            - text_ref
            - folder_path
            - extraction_method_id
            - extracted_info_path
            - created_at
            - updated_at
    """
    with client.session() as sess:
        p_source = sess.query(TextContent).filter_by(text_ref=text_ref).all()
        return [p.to_dict() for p in p_source] if p_source else None


def get_all_text_contents(client: MiraDatabaseClient) -> list[dict]:
    """Retrieve all papers' source information.

    Parameters
    ----------
    client :
        The database client to use.

    Returns
    -------
    :
        A list of dictionaries, each containing a paper source's information.
        The dictionary keys are:
            - id
            - text_ref
            - folder_path
            - extraction_method_id
            - extracted_info_path
            - created_at
            - updated_at
    """
    with client.session() as sess:
        p_sources = sess.query(TextContent).order_by(TextContent.created_at.desc()).all()
        return [p.to_dict() for p in p_sources]


def remove_text_content(client: MiraDatabaseClient, text_ref: int) -> bool:
    """Delete text content rows and all FK-linked rows.

    Parameters
    ----------
    client :
        The database client to use.
    text_ref :
        The ID of the text reference that this source information is linked to.

    Returns
    -------
    :
        True if the paper source was found and deleted, False if the paper source was not found.
    """
    def _write(sess):
        rows = sess.query(TextContent).filter_by(text_ref=text_ref).all()
        if not rows:
            logger.warning("Paper source '%s' not found.", text_ref)
            return False
        for item in rows:
            sess.delete(item)
        logger.info("Deleted paper source '%s' and all linked rows.", text_ref)
        return True
    return client.mutate(_write)


def add_odes(
    client: MiraDatabaseClient,
    txt_content_ref: int,
    extraction_method_id: int,
    ode: str,
    corrected_ode: str = None,
):
    """Add a paper's ODE equations to the database.

    Parameters
    ----------
    client :
        The database client to use.
    txt_content_ref :
        The ID of the text content that this ODE information is linked to.
    extraction_method_id :
        The ID of the extraction method used to extract the ODE string.
    ode :
        The ODE string extracted for this paper.
    corrected_ode :
        A corrected version of the ODE string, if it exists. Can be the same as
        ``ode`` if no correction was needed.

    Returns
    -------
    int
        The ID of the ODE entry that was added.
    """
    try:
        def _write(sess):
            p = ODEs(
                txt_content_ref=txt_content_ref,
                ode=ode,
                corrected_ode=corrected_ode,
                extraction_method_id=extraction_method_id,
            )
            sess.add(p)
            sess.flush()
            sess.refresh(p)
            logger.info("Registered ODEs for txt_content_ref '%s'.", txt_content_ref)
            return p.id
        return client.mutate(_write)
    except IntegrityError as e:
        if 'ode' in str(e.orig):
            logger.warning(
                "ODE already exists for txt_content_ref %s, skipping.",
                txt_content_ref,
            )
        else:
            logger.warning(
                "An ODE entry for txt_content_ref %s already exists.",
                txt_content_ref,
            )
        return None


def update_odes(
    client: MiraDatabaseClient,
    txt_content_ref: int,
    extraction_method_id: int,
    ode: str = None,
    corrected_ode: str = None,
) -> bool:
    """Update a paper's ODE equations in the database.

    Parameters
    ----------
    client :
        The database client to use.
    txt_content_ref :
        The ID of the text content that this ODE information is linked to.
    extraction_method_id :
        The ID of the extraction method used to extract the ODE string.
    ode :
        The new ODE string extracted for this paper.
    corrected_ode :
        A new corrected version of the ODE string, if it exists. Can be the same
        as ``ode`` if no correction was needed.

    Returns
    -------
    bool
        True if the ODE entry was found and updated, False if the ODE entry was not found.
    """
    def _write(sess):
        odes = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
        if not odes:
            logger.warning("ODEs for txt_content_ref '%s' not found.", txt_content_ref)
            return False
        if ode is not None:
            odes.ode = ode
        if corrected_ode is not None:
            odes.corrected_ode = corrected_ode
        if extraction_method_id is not None:
            odes.extraction_method_id = extraction_method_id
        logger.info("Updated ODEs for txt_content_ref '%s'.", txt_content_ref)
        return True
    return client.mutate(_write)


def get_odes(client: MiraDatabaseClient, txt_content_ref: int):
    """Retrieve a paper's ODE equations by text content ID.

    Parameters
    ----------
    client :
        The database client to use.
    txt_content_ref :
        The ID of the text content that this ODE information is linked to.

    Returns
    -------
    :
        A dictionary containing the ODE information if found, or None if not found.
        The dictionary keys are:
            - id
            - txt_content_ref
            - ode
            - corrected_ode
            - extraction_method_id
            - created_at
            - updated_at
    """
    with client.session() as sess:
        odes = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
        return odes.to_dict() if odes else None


def get_all_odes(client: MiraDatabaseClient) -> list[dict]:
    """Retrieve all papers' ODE equations.

    Parameters
    ----------
    client :
        The database client to use.

    Returns
    -------
    :
        A list of dictionaries, each containing a paper's ODE information.
        The dictionary keys are:
            - id
            - txt_content_ref
            - ode
            - corrected_ode
            - extraction_method_id
            - created_at
            - updated_at
    """
    with client.session() as sess:
        odes = sess.query(ODEs).order_by(ODEs.created_at.desc()).all()
        return [o.to_dict() for o in odes]


def remove_odes(client: MiraDatabaseClient, txt_content_ref: int) -> bool:
    """Delete ODEs for a paper and all FK-linked rows.

    Parameters
    ----------
    client :
        The database client to use.
    txt_content_ref :
        The ID of the text content that this ODE information is linked to.

    Returns
    -------
    :
        True if the ODE entry was found and deleted, False if the ODE entry was not found.
    """
    def _write(sess):
        p = sess.query(ODEs).filter_by(txt_content_ref=txt_content_ref).first()
        if not p:
            logger.warning("ODEs for txt_content_ref '%s' not found.", txt_content_ref)
            return False
        sess.delete(p)
        logger.info("Deleted ODEs for txt_content_ref '%s' and all linked rows.", txt_content_ref)
        return True
    return client.mutate(_write)


def add_tm(
    client: MiraDatabaseClient,
    ode_ref: int,
    grounded_concepts: dict,
    mira_template_model: dict = None,
):
    """Add a MIRA template model extracted from a paper to the database.

    Parameters
    ----------
    client :
        The database client to use.
    ode_ref :
        The ID of the ODE that this template model information is linked to.
    grounded_concepts :
        A dictionary (JSON-convertible) containing the grounded concepts extracted.
    mira_template_model :
        A dictionary (JSON-convertible) containing the MIRA template model extracted
        from the paper.

    Returns
    -------
    int
        The ID of the template model entry that was added.
    """
    try:
        def _write(sess):
            p = MiraModel(
                ode_ref=ode_ref,
                grounded_concepts=grounded_concepts,
                mira_template_model=mira_template_model,
            )
            sess.add(p)
            sess.flush()
            sess.refresh(p)
            logger.info("Registered MiraModel for ode_ref '%s'.", ode_ref)
            return p.id
        return client.mutate(_write)
    except IntegrityError:
        logger.warning("A MiraModel with ode_ref %s already exists.", ode_ref)
        return None


def update_tm(
    client: MiraDatabaseClient,
    ode_ref: int,
    grounded_concepts: dict = None,
    mira_template_model: dict = None,
) -> bool:
    """Update a MIRA template model's information in the database.

    Parameters
    ----------
    client :
        The database client to use.
    ode_ref :
        The ID of the ODE that this template model information is linked to.
    grounded_concepts :
        A dictionary (JSON-convertible) containing the grounded concepts extracted
        from the paper.
    mira_template_model :
        A dictionary (JSON-convertible) containing the MIRA template model extracted
        from the paper.

    Returns
    -------
    bool
        True if the template model entry was found and updated, False if the
        template model entry was not found.
    """
    def _write(sess):
        tm = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
        if not tm:
            logger.warning("MiraModel for ode_ref '%s' not found.", ode_ref)
            return False
        if grounded_concepts is not None:
            tm.grounded_concepts = grounded_concepts
        if mira_template_model is not None:
            tm.mira_template_model = mira_template_model
        logger.info("Updated MiraModel for ode_ref '%s'.", ode_ref)
        return True
    return client.mutate(_write)


def get_tm(client: MiraDatabaseClient, ode_ref: int):
    """Retrieve an ODE's MIRA template model information by ODE ID.

    Parameters
    ----------
    client :
        The database client to use.
    ode_ref :
        The ID of the ODE that this template model information is linked to.

    Returns
    -------
    :
        A dictionary containing the template model information if found, or None
        if not found. Dictionary keys: 'id', 'ode_ref', 'grounded_concepts',
        'mira_template_model', 'created_at', 'updated_at'
    """
    with client.session() as sess:
        tm = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
        return tm.to_dict() if tm else None


def get_all_tms(client: MiraDatabaseClient) -> list[dict]:
    """Retrieve all MIRA template model information.

    Parameters
    ----------
    client :
        The database client to use.

    Returns
    -------
    :
        A list of dictionaries, each containing a paper's MIRA template model
        information. The dictionary keys are:
            - id
            - ode_ref
            - grounded_concepts
            - mira_template_model
            - created_at
            - updated_at
    """
    with client.session() as sess:
        tms = sess.query(MiraModel).order_by(MiraModel.created_at.desc()).all()
        return [tm.to_dict() for tm in tms]


def remove_tm(client: MiraDatabaseClient, ode_ref: int) -> bool:
    """Delete a MIRA template model and all FK-linked rows.

    Parameters
    ----------
    client :
        The database client to use.
    ode_ref :
        The ID of the ODE that this template model information is linked to.

    Returns
    -------
    :
        True if the MIRA template model was deleted, False if not found.
    """
    def _write(sess):
        p = sess.query(MiraModel).filter_by(ode_ref=ode_ref).first()
        if not p:
            logger.warning("MiraModel for ode_ref '%s' not found.", ode_ref)
            return False
        sess.delete(p)
        logger.info("Deleted MiraModel for ode_ref '%s' and all linked rows.", ode_ref)
        return True
    return client.mutate(_write)


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
