import os
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from miradb.db.manager import MiraModelManager, get_db
from miradb.db.schema import ExtractionMethod

logger = logging.getLogger(__name__)

# Modify path as needed to point to the root directory containing
# per-PMID subfolders with TemplateModel outputs.
MODEL_PATH = Path.home() / ".data" / "mira" / "paper_extraction"

def add_extraction_methods(mira_db: MiraModelManager) -> None:
    """Populate the ExtractionMethod table with supported extraction methods.

    Parameters
    ----------
    mira_db : MiraModelManager
        MiraModelManager instance bound to the target database.

    Notes
    -----
    Safe to call on a fresh database; skips insertion for existing rows.
    """

    # Define supported extraction methods and their descriptions
    # Can add or update methods and descriptions here as needed
    methods = [
        (
            "marker",
            "LLM Pipeline, Text, GPU, Marker",
        ),
        (
            "mineru_image",
            "LLM Pipeline, Image, CPU, MinerU",
        ),
        (
            "mineru_text",
            "LLM Pipeline, Text, CPU, MinerU",
        ),
        (
            "xml_extraction",
            "LLM Pipeline, Text, CPU, pubmed xml file",
        ),
    ]

    Session = sessionmaker(bind=mira_db.engine)
    with Session() as session:
        existing = {
            row.extraction_method
            for row in session.execute(
                select(ExtractionMethod.extraction_method)
            ).scalars().all()
        }
        for method, desc in methods:
            if method not in existing:
                session.add(
                    ExtractionMethod(
                        extraction_method=method,
                        extraction_method_desc=desc,
                    )
                )
        session.commit()

def get_folder_names(model_path: Path) -> list[str]:
    """Return the names of all subdirectories under model_path.

    Parameters
    ----------
    model_path : Path
        Path to the root directory containing PMID subfolders.

    Returns
    -------
    list[str]
        Names of all subdirectories under ``model_path``.
    """
    return [f.name for f in model_path.iterdir() if f.is_dir()]


def get_extraction_id(session: Session, method: str) -> int:
    """Return the primary key for an extraction method.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session used to execute the query.
    method : str
        Extraction method name to look up.

    Returns
    -------
    int
        Database primary key for the requested extraction method.
    """
    return session.execute(
        select(ExtractionMethod.id).where(ExtractionMethod.extraction_method == method)
    ).scalar()


def get_xml_text(el) -> str | None:
    """Recursively concatenate all text inside an XML element.

    Parameters
    ----------
    el : xml.etree.ElementTree.Element | None
        XML element to extract text from.

    Returns
    -------
    str | None
        Stripped text content if present, otherwise ``None``.
    """
    return " ".join(el.itertext()).strip() if el is not None else None

def ingest_extraction_method(
    mira_db: MiraModelManager,
    session: Session,
    folder_names: list[str],
    method: str,
    model_path: Path = MODEL_PATH,
) -> None:
    """Ingest TemplateModel outputs from per-PMID folders into the database.

    Parameters
    ----------
    mira_db : MiraModelManager
        MiraDB instance with add_*/get_* helpers.
    session : Session
        Active SQLAlchemy session used only to fetch extraction_id.
    folder_names : list[str]
        PMID folder names discovered under ``model_path``.
    method : str
        One of ``"xml_extraction"``, ``"marker"``, ``"mineru_image"``, or
        ``"mineru_text"``.
    model_path : Path, optional
        Root directory that contains per-PMID subfolders, by default
        ``MODEL_PATH``.
    """
    extraction_id = get_extraction_id(session, method)

    for pmid in tqdm(folder_names, desc=f"Ingesting [{method}]"):
        try:
            # Load TemplateModel JSON
            model_file = model_path / pmid / "tm" / method / f"{pmid}.json"
            with open(model_file) as f:
                data = json.load(f)

            # Load intermediates JSON
            intermediates_file = model_path / pmid / "tm" / method / f"{pmid}_intermediates.json"
            with open(intermediates_file) as f:
                intermediates = json.load(f)

            # Resolve or create TextRef
            text_ref = mira_db.get_text_ref(pmid=pmid)
            if text_ref is None:
                text_ref = mira_db.add_text_ref(pmid=pmid)
            else:
                text_ref = text_ref["id"]

            # Locate the PMC subfolder (name starts with "P")
            base = model_path / str(pmid)
            pmc_folder = next(
                (p for p in base.iterdir() if p.is_dir() and p.name.startswith("P")),
                None,
            )
            folder_path = str(pmc_folder.relative_to(base.parent)) if pmc_folder else None

            # Determine extracted_info_path per method
            extracted_info_path = intermediates["ode"]["extraction_file"]

            if method == "marker":
                extracted_info_path = f"{base.stem}/{method}"

            elif "mineru" in method:
                if not pmc_folder:
                    logger.warning(f"[{pmid}] No PMC folder found — skipping.")
                    continue
                nxml_file = next(
                    (f for f in pmc_folder.iterdir() if f.suffix == ".nxml"),
                    None,
                )
                if not nxml_file:
                    logger.warning(
                        f"[{pmid}] No .nxml file in PMC folder — skipping."
                    )
                    continue
                extracted_info_path = f"{base.stem}/{nxml_file.stem}"

            # Persist
            context_ref = mira_db.add_text_content(
                text_ref=text_ref,
                folder_path=folder_path,
                extraction_method_id=extraction_id,
                extracted_info_path=extracted_info_path,
            )
            ode_id = mira_db.add_odes(
                txt_content_ref=int(context_ref),
                extraction_method_id=extraction_id,
                ode=intermediates["ode"]["ode_str"],
                corrected_ode=intermediates["ode"]["corrected_ode_str"],
            )
            mira_db.add_tm(
                ode_ref=ode_id,
                grounded_concepts=intermediates["ode"]["concepts"],
                mira_template_model=data,
            )

        except Exception as e:
            logger.error(f"[{pmid}] Error during ingestion: {e}")
            continue


def update_metadata_from_nxml(
    mira_db: MiraModelManager,
    folder_names: list[str],
) -> None:
    """Parse article NXML files and update bibliographic metadata.

    Parameters
    ----------
    mira_db : MiraModelManager
        MiraDB instance with an ``update_text_ref()`` helper.
    folder_names : list[str]
        PMID folder names to process.
    """
    for pmid in tqdm(folder_names, desc="Updating metadata"):
        base_path = MODEL_PATH / str(pmid)

        pmc_folder = next(
            (
                p
                for p in base_path.iterdir()
                if p.is_dir() and p.name.startswith("P")
            ),
            None,
        )
        if not pmc_folder:
            logger.warning(f"[{pmid}] No PMC folder found — skipping metadata update.")
            continue

        nxml_file = next(
            (f for f in os.listdir(pmc_folder) if f.endswith(".nxml")),
            None,
        )
        if not nxml_file:
            logger.warning(f"[{pmid}] No .nxml file found — skipping metadata update.")
            continue

        tree = ET.parse(os.path.join(pmc_folder, nxml_file))
        root = tree.getroot()

        # Title
        title = get_xml_text(root.find(".//title-group/article-title"))

        # Authors
        authors = []
        for contrib in root.findall('.//contrib[@contrib-type="author"]'):
            surname = get_xml_text(contrib.find("name/surname"))
            given = get_xml_text(contrib.find("name/given-names"))
            if surname and given:
                authors.append(f"{given} {surname}")
            elif surname:
                authors.append(surname)

        # Publication year
        pub_date = (
            root.find('.//pub-date[@pub-type="epub"]')
            or root.find('.//pub-date[@pub-type="ppub"]')
            or root.find(".//pub-date")
        )
        year = get_xml_text(pub_date.find("year")) if pub_date is not None else None

        # Journal
        journal = get_xml_text(root.find(".//journal-title"))

        # DOI and PMC ID
        doi = pmc_id = None
        for aid in root.findall(".//article-id"):
            pub_id_type = aid.attrib.get("pub-id-type")
            if pub_id_type == "doi":
                doi = aid.text
            elif pub_id_type == "pmc":
                pmc_id = aid.text

        # Keywords
        keywords = [get_xml_text(kw) for kw in root.findall(".//kwd")]

        mira_db.update_text_ref(
            pmid=pmid,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            pmcid=pmc_id,
            keywords=keywords,
        )


def main():

    db = get_db("primary")
    
    db.create_tables()
    mira_db = MiraModelManager(db.host)
    Session = sessionmaker(bind=db.engine)
    add_extraction_methods(mira_db)
    folder_names = get_folder_names(MODEL_PATH)

    with Session() as session:
        for method in (
            "xml_extraction",
            "marker",
            "mineru_image",
            "mineru_text",
        ):
            ingest_extraction_method(mira_db, session, folder_names, method)

    update_metadata_from_nxml(mira_db, folder_names)


if __name__ == "__main__":
    main()