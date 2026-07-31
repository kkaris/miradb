import os
import gc
import json
import logging
from pathlib import Path

import pandas as pd
import pystow
import tqdm
import traceback
from datetime import datetime

from mira.openai_utility import OpenAIClient
from mira.sources.sympy_ode.paper_extraction import \
    get_template_model_from_pmid, get_pmid_pmc_download_mapping
from mira.modeling import Model
from mira.modeling.ode import OdeModel

# Path to the relevance-ranker output listing the PMIDs to extract. Override
# with the MIRADB_POSITIVE_PAPERS environment variable.
POSITIVE_PAPERS_PATH = Path(
    os.environ.get("MIRADB_POSITIVE_PAPERS",
                   Path.home() / "mira_data" / "positive_papers.tsv")
)

BASE = pystow.module("mira", "paper_extraction")

now = datetime.now()

logdump_file = f'app_{now.strftime("%Y-%m-%d-%H-%M-%S")}.log'
logging.basicConfig(filename=logdump_file,
                    filemode='a', 
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    force=True)

logger = logging.getLogger(__name__)

logger.info("Handlers:", logging.root.handlers)
for h in logging.root.handlers:
    if hasattr(h, 'baseFilename'):
        logger.info("Log file path:", h.baseFilename)

def result_to_intermediates(result):
    """Map a mira PipelineResult to the flat intermediates dict we persist.

    The grounded concepts (mira Concept models) are dumped to plain dicts so
    the result is JSON-serializable.
    """
    grounding = result.grounding
    concepts = None
    try:
        if grounding is not None and grounding.concepts:
            concepts = {
                k: v if isinstance(v, dict) else v.to_json()
                if hasattr(v, "to_json") else ValueError(f"Unexpected \
                                                         concept type: {type(v)}")
                for k, v in grounding.concepts.items()
            }

    except Exception as e:
        logger.warning(f"Error processing grounding concepts: {e}")
        logger.debug(traceback.format_exc())

    return {
        "ode_str": result.extraction.ode_str if result.extraction else None,
        "corrected_ode_str":
            result.correction.ode_str if result.correction else None,
        "concepts": concepts,
        "extraction_file": result.extraction_file,
    }


def save_with_intermediates(template_model, result, pmid, folder_name):
    """Save both intermediate extraction results and the final model.

    Parameters
    ----------
    template_model :
        The extracted template model.
    result :
        The mira PipelineResult from get_template_model_from_pmid.
    pmid :
        PubMed ID.
    folder_name :
        Name of the folder where the extractions will be stored.
    """
    paper_base = BASE.join(pmid)
    out_dir = paper_base / "tm" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediates = {"ode": result_to_intermediates(result)}
    with open(out_dir / f"{pmid}_intermediates.json", 'w') as f:
        json.dump(intermediates, f, indent=2)
    with open(out_dir / f"{pmid}.json", 'w') as f:
        json.dump(template_model.to_json(), f, indent=2)


def main():
    # Load the list of PMIDs for papers that were classified as relevant
    # (positive) by the trained model.
    df = pd.read_csv(POSITIVE_PAPERS_PATH, sep='\t')

    # Initialize the OpenAI client with the desired model and temperature. 
    # Temperature is set to 0.0 for deterministic outputs.
    client = OpenAIClient(model="gpt-5.4-mini", temperature=0.0)

    # modify based on preferred settings
    extractor = "xml"  # options: "mineru" or "marker" or "xml"
    extraction_method = "text"  # options: "text" or "image"

    # Canonical extraction-method folder name that populate_db.py reads (see
    # miradb/db/extraction_methods.json). Marker and XML are text-only, so they
    # use a fixed name rather than the "{extractor}_{method}" form.
    fixed_names = {"marker": "marker", "xml": "xml_extraction"}
    folder_name = fixed_names.get(extractor, f"{extractor}_{extraction_method}")

    # Track progress - append to CSV after each success
    output_directory = BASE.join(folder_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    progress_file = output_directory / "extraction_progress.csv"
    logger.info(f"Saving progress to {progress_file}")

    pmid_to_download_mapping = get_pmid_pmc_download_mapping()

    processed_pmids = set()
    if progress_file.exists():
        progress_df = pd.read_csv(progress_file, header=None,
                                  names=['pmid', 'status', 'error'],
                                  quotechar='"', on_bad_lines='skip', sep=";")
        processed_pmids = set(progress_df['pmid'].astype(str))
        logger.info(f"Found {len(processed_pmids)} already processed PMIDs")

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        pmid = str(row["PMID"])
        # Skip if already processed
        if pmid in processed_pmids:
            logger.info(f"\n\n#{idx} PMID {pmid} already attempted, skipping...\n")
            continue
        try:
            logger.info(f"#{idx} - Processing PMID {pmid}...")
            tm, result = get_template_model_from_pmid(
                pmid=pmid, ode_extraction_method=extraction_method,
                extractor=extractor,
                pmid_to_download_mapping=pmid_to_download_mapping,
                client=client)
    
            if tm is None:
                logger.warning(f"PMID {pmid} - No template model extracted.")
                with open(progress_file, 'a') as f:
                    f.write(f"{pmid};failed;no_model_found\n")
                continue

            # logger.info(f"PMID {pmid} ODE:\n{result.final_ode_str}\n")
            
            # Create OdeModel only for validation, then release
            om = OdeModel(Model(tm), initialized=True)
            save_with_intermediates(tm, result, pmid, folder_name)
            # Explicitly cleanup. Memory usage gets high if there are many papers.
            del om, tm

            with open(progress_file, 'a') as f:
                f.write(f"{pmid};success;\n")

        except Exception as e:
            logger.warning(f"Failed to extract model for PMID {pmid}: \
                           {type(e).__name__}")
            with open(progress_file, 'a') as f:
                f.write(f"{pmid};failed;{type(e).__name__}:{str(e)}\n")
            continue
        finally:
            gc.collect()

    ls_filename = f"llm_settings_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    llm_settings_file = output_directory / ls_filename

    client.save_client_usage(filepath=llm_settings_file)
    with open(llm_settings_file, "a") as f:
        f.write(f"\n Log dump filename: {logdump_file}")

    logger.info(f"Saved LLM usage settings to {llm_settings_file}")

if __name__ == "__main__":
    main()
