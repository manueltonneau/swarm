"""Retrieval selection bias: are extraction failures biased toward pro-Kremlin sources?

A sampled URL enters the labelled set only if a clean document can be extracted
from it. If extraction fails more often for Russian state-affiliated outlets
(paywalls, bot protection, or hosts that do not resolve from our collection
setup), the retained set under-counts the most overtly propaganda-supporting
sources, a conservative bias for any prevalence estimate. This script quantifies
that by comparing the genuine extraction failures against the retained documents
on two markers of pro-Kremlin provenance: membership of the Russian-propaganda
blocklist (data/propaganda_domains_bundle.csv, built by build_domain_blocklist.py)
and a .ru registered domain.

Substantiates the "Retrieval Selection Bias" appendix of the SWARM paper.

Inputs:
  data/annotation/sampled_not_annotated_recovery.csv  (sampled URLs, with an
      extraction_status column; rows that extracted OK are excluded)
  data/predictions/eval_9lang/gold_eval_set.parquet   (retained original_ids)
  data/categorized_domains/dataset_domain_categories.csv (original_id -> domain)
  data/propaganda_domains_bundle.csv                  (the blocklist)

Outputs: a printed comparison table + the top failing domains, and
  data/predictions/eval_9lang/retrieval_bias.csv
"""
import sys
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
try:
    from .build_domain_blocklist import reg_domain
except ImportError:  # noqa: E402
    from build_domain_blocklist import reg_domain

AUX = ROOT / "data_aux"
EVAL = Path(os.environ.get("SWARM_EVAL_DIR", ROOT / "data" / "eval"))
# Not in the public release: the sampled URLs that never reached annotation.
FAILURES = ROOT / "data" / "annotation" / "sampled_not_annotated_recovery.csv"
GOLD = EVAL / "gold_eval_set.parquet"
DOMAINS = AUX / "domain_categories.csv"
BUNDLE = AUX / "propaganda_domains_bundle.csv"
OUT = EVAL / "retrieval_bias.csv"

# extraction_status values that mean the page WAS retrieved cleanly (so the item
# was dropped for another reason, e.g. annotators could not assess it) -- these
# are not retrieval failures and are excluded from the comparison.
OK_STATUS = {"html_ok", "pdf_ok"}


def failures() -> pd.DataFrame:
    """Genuine extraction failures with a normalised registered domain."""
    df = pd.read_csv(FAILURES)
    df = df[~df["extraction_status"].astype(str).isin(OK_STATUS)].copy()
    df["rd"] = df["domain"].map(reg_domain)
    return df


def retained() -> pd.DataFrame:
    """Retained (labelled) documents with a normalised registered domain."""
    gold = pd.read_parquet(GOLD)[["original_id"]]
    dom = pd.read_csv(DOMAINS)[["original_id", "domain"]]
    df = gold.merge(dom, on="original_id", how="left")
    df["rd"] = df["domain"].map(reg_domain)
    return df


def markers(rd: pd.Series, block: set) -> dict:
    return {
        "on_blocklist_%": round(100 * rd.isin(block).mean(), 1),
        "dot_ru_%": round(100 * rd.astype(str).str.endswith(".ru").mean(), 1),
        "n": int(len(rd)),
    }


def _require_failures():
    if not FAILURES.exists():
        raise SystemExit(
            "This analysis needs the list of sampled URLs that never reached annotation, "
            "which is not part of the public release (see README). "
            f"Expected at {FAILURES}.")


def main():
    _require_failures()
    block = set(pd.read_csv(BUNDLE)["domain"].dropna().astype(str))
    fail, ret = failures(), retained()
    fm, rm = markers(fail["rd"], block), markers(ret["rd"], block)

    tab = pd.DataFrame(
        [{"source_marker": "on Russian-propaganda blocklist",
          "failures_%": fm["on_blocklist_%"], "retained_%": rm["on_blocklist_%"]},
         {"source_marker": ".ru registered domain",
          "failures_%": fm["dot_ru_%"], "retained_%": rm["dot_ru_%"]}]
    )
    print(f"genuine extraction failures: {fm['n']}   retained documents: {rm['n']}\n")
    print(tab.to_string(index=False))
    for _, r in tab.iterrows():
        ratio = r["failures_%"] / r["retained_%"] if r["retained_%"] else float("nan")
        print(f"  {r['source_marker']}: {ratio:.1f}x over-represented among failures")

    print("\ntop failing domains (* = on blocklist or .ru):")
    for d, c in fail["rd"].value_counts().head(12).items():
        mark = "*" if (d in block or str(d).endswith(".ru")) else " "
        print(f"  {mark} {d:28s} {c}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tab.assign(n_failures=fm["n"], n_retained=rm["n"]).to_csv(OUT, index=False)
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
