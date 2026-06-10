import logging
from pathlib import Path

import pandas as pd
import numpy as np

from miradb.db.client import get_client
from miradb.db import queries
from miradb.compare.equation import compare_models


logger = logging.getLogger('benchmark.compare')


def generate_report(report: dict, e1: int, pmid: str):
    """Generate, append a detailed model comparison report to the progress file

    Parameters
    ----------
    report : dict
        Output of compare_models().
    e1 : int
        Extraction method ID.
    pmid : str
        PubMed ID.
    """
    cj = report["compartment_jaccard"]
    report_cj = f"\n[Layer 1] Compartment Jaccard: {cj['jaccard']:.3f}"
    report_cj += f" ⚠ mismatch" if cj["compartment_mismatch"] else ""
    report_cj+= f"  Shared:    {cj['shared']}"
    if cj["only_in_1"]: report_cj+= f"  Only in 1: {cj['only_in_1']}"
    if cj["only_in_2"]: report_cj+= f"  Only in 2: {cj['only_in_2']}"

    tj = report["term_jaccard"]
    report_tj = f"\n[Layer 2] Term-Set Jaccard (aggregate): {tj['aggregate']:.3f}"
    for role, info in tj["per_compartment"].items():
        flag = " ⚠ no shared terms" if info["no_shared_terms"] else ""
        report_tj += f"  d({role})/dt  jaccard={info['jaccard']:.3f}{flag}"
        if info["only_in_1"]: report_tj += f"    only_in_1: {info['only_in_1']}"
        if info["only_in_2"]: report_tj += f"    only_in_2: {info['only_in_2']}"

    ted = report["ted"]
    report_ted = f"\n[Layer 3] Tree Edit Distance"

    agg = ted["aggregate_per_compartment"]
    wm  = ted["whole_model"]
    report_ted += f" agg(normalized)={agg['normalized']:.4f}  "
    report_ted += f" whole_model raw={wm['raw']}, normalized={wm['normalized']:.4f}"
    for role, scores in ted["per_compartment"].items():
        report_ted += f"    d({role})/dt  raw={scores['raw']}, normalized={scores['normalized']:.4f}"

    with open(progress_file, 'a') as f:
        f.write(f"{pmid};{e1};{report_cj};{report_tj};{report_ted}\n")

def generate_score_only_report(report: dict, e1: int, pmid: str):
    """
    Generate a score-only report for a model comparison and append it to the progress file.

    Parameters
    ----------
    report : dict
        Output of compare_models().
    e1 : int
        Extraction method ID.
    pmid : str
        PubMed ID.
    """
    cj = report["compartment_jaccard"]
    report_cj = f"{cj['jaccard']:.3f}"

    tj = report["term_jaccard"]
    report_tj = f"{tj['aggregate']:.3f}"

    ted = report["ted"]
    agg = ted["aggregate_per_compartment"]
    report_ted = f"{1 - agg['normalized']:.4f} "

    with open(progress_file, 'a') as f:
        f.write(f"{pmid};{e1};{report_cj};{report_tj};{report_ted}\n")


if __name__ == "__main__":
    progress_file = Path("results/report_score.csv")
    print(f"Saving progress to {progress_file}")

    client = get_client("primary")
    gold_standard = pd.read_csv("resources/eqs_list.tsv", sep="\t")

    for idx in range(len(gold_standard)):
        gold_pmid = gold_standard.iloc[idx]["pmid"]
        if np.isnan(gold_pmid):
            print(f"Skipping row with missing PMID.")
            continue

        gold_standard_odes = gold_standard.iloc[idx]["corrected_sympy"]
        if gold_standard_odes == "":
            print(
                f"No gold standard ODEs provided for PMID {gold_pmid}. Skipping."
            )
            continue

        ode_rows = queries.list_odes_for_pmid(client, str(int(gold_pmid)))
        if not ode_rows:
            print(f"PMID {gold_pmid} not found in text_references table.")
            continue

        for row in ode_rows:
            sympy_src = row["corrected_ode"] or row["ode"]
            if not sympy_src:
                continue
            try:
                comparison_report = compare_models(
                    gold_standard_odes, sympy_src
                )
            except Exception as e:
                print(
                    f"Error occurred while comparing models for PMID {gold_pmid}: {e}"
                )
                continue
            generate_score_only_report(
                comparison_report, row["extraction_method_id"], gold_pmid
            )
            # OR - For detailed report:
            # generate_report(report, row["extraction_method_id"], pmid)
