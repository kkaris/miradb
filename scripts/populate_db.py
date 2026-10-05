import os
import json
import math
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pystow
from tqdm import tqdm

from miradb.db.client import MiraDatabaseClient, get_client
from miradb.db.extraction_methods import EXTRACTION_METHODS_INFO
from miradb.db import queries


logger = logging.getLogger(__name__)

# Modify path as needed to point to the root directory containing
# per-PMID subfolders with TemplateModel outputs.
MODEL_PATH = pystow.module("mira", "paper_extraction").base


def add_extraction_methods(client: MiraDatabaseClient, extraction_methods: list[dict]) -> None:
    """Populate the ExtractionMethod table with supported extraction methods.

    Parameters
    ----------
    client :
        MiraDatabaseClient instance.
    extraction_methods :
        List of dictionaries containing extraction method information.

    Notes
    -----
    Safe to call on a fresh database; skips insertion for existing rows.
    """

    for method in extraction_methods:
        queries.add_extraction_method(
            client=client,
            extraction_method=method["extraction_method"],
            extraction_method_desc=method["description"],
        )


def get_xml_text(el) -> str | None:
    """Recursively concatenate all text inside an XML element.

    Parameters
    ----------
    el : xml.etree.ElementTree.Element | None
        XML element to extract text from.

    Returns
    -------
    :
        Stripped text content if present, otherwise ``None``.
    """
    return " ".join(el.itertext()).strip() if el is not None else None


def ingest_extraction_method(
    client: MiraDatabaseClient,
    folder_names: list[str],
    method: str,
    model_path: Path = MODEL_PATH,
):
    """Ingest TemplateModel outputs from per-PMID folders into the database.

    Parameters
    ----------
    client :
        MiraDatabaseClient instance.
    folder_names : list[str]
        PMID folder names discovered under ``model_path``.
    method :
        One of ``"xml_extraction"``, ``"marker"``, ``"mineru_image"``, or
        ``"mineru_text"``.
    model_path :
        Root directory that contains per-PMID subfolders, by default
        ``MODEL_PATH``.
    """
    extraction_id = queries.get_extraction_method_id(client, method)

    for pmid in tqdm(folder_names, desc=f"Ingesting [{method}]"):
        try:
            # Load TemplateModel JSON
            model_file = model_path / pmid / "tm" / method / f"{pmid}.json"
            with open(model_file) as f:
                data = json.load(f)

            # Load intermediates JSON
            intermediates_file = (
                model_path / pmid / "tm" / method / f"{pmid}_intermediates.json"
            )
            with open(intermediates_file) as f:
                intermediates = json.load(f)

            # Resolve or create TextRef
            text_ref = queries.get_text_ref(client, pmid)
            if text_ref is None:
                text_ref = queries.add_text_ref(client, pmid)
            else:
                text_ref = text_ref["id"]

            # Locate the PMC subfolder (name starts with "P")
            base = model_path / str(pmid)
            pmc_folder = next(
                (
                    p for p in base.iterdir()
                    if p.is_dir() and p.name.startswith("P")
                ), None
            )
            folder_path = (
                str(pmc_folder.relative_to(base.parent)) if pmc_folder else None
            )

            # Determine extracted_info_path per method
            extracted_info_path = intermediates["ode"]["extraction_file"]

            if method == "marker":
                extracted_info_path = f"{base.stem}/{method}"
                assert pmc_folder is not None, \
                    f"[{pmid}] No PMC folder found for marker method."

            elif "mineru" in method:
                if not pmc_folder:
                    logger.warning(f"[{pmid}] No PMC folder found - skipping.")
                    continue
                nxml_file = next(
                    (f for f in pmc_folder.iterdir() if f.suffix == ".nxml"),
                    None,
                )
                if not nxml_file:
                    logger.warning(
                        f"[{pmid}] No .nxml file in PMC folder - skipping."
                    )
                    continue
                extracted_info_path = f"{base.stem}/{nxml_file.stem}"

            # Persist
            context_ref = queries.add_text_content(
                client=client,
                text_ref=text_ref,
                folder_path=folder_path,
                extraction_method_id=extraction_id,
                extracted_info_path=extracted_info_path,
            )
            ode_id = queries.add_odes(
                client=client,
                txt_content_ref=int(context_ref),
                extraction_method_id=extraction_id,
                ode=intermediates["ode"]["ode_str"],
                corrected_ode=intermediates["ode"]["corrected_ode_str"],
            )
            _ = queries.add_tm(
                client=client,
                ode_ref=ode_id,
                grounded_concepts=intermediates["ode"]["concepts"],
                mira_template_model=data,
            )

        except FileNotFoundError:
            # No output for this PMID under this method, which is expected when
            # the paper was extracted with a different method.
            logger.debug(f"[{pmid}] No {method} output found, skipping.")
            continue
        except Exception:
            logger.exception(f"[{pmid}] Error during ingestion")
            continue


def scenario_parameter_fields(parameter: dict) -> dict | None:
    """Map a MIRA Parameter JSON to ``add_scenario_parameter`` arguments.

    Parameters
    ----------
    parameter :
        One entry of a scenario's ``parameters`` list, as written by
        ``Parameter.to_json()``.

    Returns
    -------
    :
        Keyword arguments for ``queries.add_scenario_parameter``, or ``None``
        if the parameter has no value, value expression or distribution.
    """
    value = parameter.get("value")
    value_expression = None
    if isinstance(value, str):
        # Unevaluated expressions such as "1/5.2" are saved as strings
        value_expression, value = value, None
    elif isinstance(value, float) and math.isnan(value):
        value = None

    distribution = parameter.get("distribution") or None
    if value is None and value_expression is None and distribution is None:
        return None

    units = parameter.get("units")
    return {
        "name": parameter["name"],
        "value": value,
        "value_expression": value_expression,
        "display_name": parameter.get("display_name"),
        "description": parameter.get("description"),
        "units": units.get("expression") if isinstance(units, dict) else None,
        "distribution": distribution,
    }


def ingest_scenarios(
    client: MiraDatabaseClient,
    folder_names: list[str],
    method: str,
    model_path: Path = MODEL_PATH,
):
    """Ingest the scenarios saved by generate_template_models.py.

    Reads ``<pmid>/tm/<method>/<pmid>_scenarios.json`` for each PMID. Each
    scenario is linked to the paper's text content for this extraction method,
    so ``ingest_extraction_method`` must have been run for the method first.
    Scenarios that are already in the database are skipped, so this is safe
    to re-run.

    Parameters
    ----------
    client :
        MiraDatabaseClient instance.
    folder_names :
        PMID folder names discovered under ``model_path``.
    method :
        One of ``"xml_extraction"``, ``"marker"``, ``"mineru_image"``, or
        ``"mineru_text"``.
    model_path :
        Root directory that contains per-PMID subfolders, by default
        ``MODEL_PATH``.
    """
    extraction_id = queries.get_extraction_method_id(client, method)

    for pmid in tqdm(folder_names, desc=f"Ingesting scenarios [{method}]"):
        try:
            scenarios_file = (
                model_path / pmid / "tm" / method / f"{pmid}_scenarios.json"
            )
            with open(scenarios_file) as f:
                scenarios = json.load(f)
            if not scenarios:
                continue

            # Find the text content this extraction method produced
            text_ref = queries.get_text_ref(client, pmid)
            text_contents = (
                queries.get_text_content(client, text_ref["id"]) or []
                if text_ref else []
            )
            text_content = next(
                (
                    tc for tc in text_contents
                    if tc["extraction_method_id"] == extraction_id
                ), None
            )
            if text_content is None:
                logger.warning(
                    f"[{pmid}] No {method} text content found - skipping "
                    f"scenarios. Run ingest_extraction_method first."
                )
                continue

            # Link to the template model the scenarios were named after
            odes = queries.get_odes(client, text_content["id"])
            tm = queries.get_tm(client, odes["id"]) if odes else None

            for scenario in scenarios:
                name = scenario["name"]
                if queries.get_scenario(
                    client, text_content["id"], extraction_id, name
                ):
                    logger.debug(f"[{pmid}] Scenario '{name}' exists, skipping.")
                    continue

                scenario_id = queries.add_scenario(
                    client=client,
                    txt_content_ref=text_content["id"],
                    extraction_method_id=extraction_id,
                    name=name,
                    description=scenario.get("description"),
                    mira_model_ref=tm["id"] if tm else None,
                    scenarios_path=str(scenarios_file),
                )
                if scenario_id is None:
                    continue

                for parameter in scenario.get("parameters", []):
                    fields = scenario_parameter_fields(parameter)
                    if fields is None:
                        logger.debug(
                            f"[{pmid}] Parameter '{parameter.get('name')}' in "
                            f"scenario '{name}' has no value - skipping."
                        )
                        continue
                    queries.add_scenario_parameter(
                        client=client,
                        scenario_ref=scenario_id,
                        extraction_method_id=extraction_id,
                        **fields,
                    )

        except FileNotFoundError:
            # No scenarios saved for this PMID under this method
            logger.debug(f"[{pmid}] No {method} scenarios found, skipping.")
            continue
        except Exception:
            logger.exception(f"[{pmid}] Error during scenario ingestion")
            continue


def update_metadata_from_nxml(
    client: MiraDatabaseClient,
    folder_names: list[str],
):
    """Parse article NXML files and update bibliographic metadata.

    Parameters
    ----------
    client :
        MiraDatabaseClient instance.
    folder_names :
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
            logger.warning(
                f"[{pmid}] No PMC folder found - skipping metadata update."
            )
            continue

        nxml_file = next(
            (f for f in os.listdir(pmc_folder) if f.endswith(".nxml")),
            None,
        )
        if not nxml_file:
            logger.warning(
                f"[{pmid}] No .nxml file found - skipping metadata update."
            )
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
        year = get_xml_text(pub_date.find("year")) if pub_date else None
        year = int(year) if year and year.isdigit() else None

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
        keywords = []
        for kw in root.findall(".//kwd"):
            xml_kw = get_xml_text(kw)
            if xml_kw:
                keywords.append(xml_kw)

        success = queries.update_text_ref(
            client=client,
            pmid=pmid,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            pmcid=pmc_id,
            keywords=keywords,
        )
        if not success:
            logger.warning(f"Failed to update metadata for pmid: {pmid}")
            continue


def main():
    client = get_client()

    client.create_tables()

    add_extraction_methods(client=client, extraction_methods=EXTRACTION_METHODS_INFO)
    folder_names = [f.name for f in MODEL_PATH.iterdir() if f.is_dir()]

    for method in EXTRACTION_METHODS_INFO:
        ingest_extraction_method(client, folder_names, method["extraction_method"])
        ingest_scenarios(client, folder_names, method["extraction_method"])

    update_metadata_from_nxml(client, folder_names)


if __name__ == "__main__":
    main()
