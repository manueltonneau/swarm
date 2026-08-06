"""Final per-language dataset statistics for the SWARM paper (tab:perlang).

Reports, per language and pooled: gold N, positive count/rate, mean annotators
per item, and Krippendorff's alpha (nominal, "cannot assess" excluded). Uses the
same merged annotation pool as the gold builder (base + additional exports, with
the mananeau live-page pass dropped for PL only; mananeau counts as a valid human
vote everywhere else). Alpha is over items with >= 2 valid votes.

Nominal Krippendorff alpha is computed from the coincidence matrix, so no
external dependency is needed.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotation"
GOLD = ROOT / "data" / "predictions" / "eval_9lang" / "gold_eval_set.parquet"
BASE = ANN / "non_en_ru" / "algo-audit-2nd-binarized-rest-non-enru-at-2026-06-19-08-41-938c77f5.csv"
BATCH = ANN / "last_batch" / "algo-audit-extra-batch-1-at-2026-07-01-12-59-2fd5dc95.csv"
LABMAP = {"Does not support or mention propaganda claim (0)": 0,
          "Supports propaganda claim (1)": 1,
          "Cannot assess (article unavailable)": "na"}
SEED = os.environ.get("SWARM_LIVE_PAGE_ANNOTATOR", "live-page-annotator")


def krippendorff_alpha_nominal(units):
    """units: list of lists of category values (per item, >=1 rating). Nominal alpha."""
    cats = sorted({v for u in units for v in u})
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    O = np.zeros((K, K))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    O[idx[u[i]], idx[u[j]]] += 1.0 / (m - 1)
    n = O.sum()
    if n == 0 or K < 2:
        return float("nan")
    nc = O.sum(axis=1)
    Do = n * (1 - np.trace(O) / n)                       # off-diagonal mass, weighted
    Do = O.sum() - np.trace(O)
    De = (nc @ nc - (nc ** 2).sum()) / (n - 1)           # sum_{c!=k} nc*nk / (n-1)
    De = (nc.sum() ** 2 - (nc ** 2).sum()) / (n - 1)
    return 1 - Do / De if De else float("nan")


def merged_votes():
    b = pd.read_csv(BASE, usecols=["original_id", "label", "annotator", "source_language"])
    b["_r"] = 1
    a = pd.read_csv(BATCH, usecols=["original_id", "label", "annotator", "source_language"])
    a["_r"] = 0
    df = pd.concat([b, a], ignore_index=True)
    df = df[~((df.annotator == SEED) & (df.source_language == "pl"))]        # PL round2 drop
    df = df.sort_values("_r").drop_duplicates(["original_id", "annotator"], keep="first")
    df["lab"] = df["label"].map(LABMAP)
    return df[df["lab"].isin([0, 1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "data" / "predictions" / "eval_9lang" / "final_perlang_stats.csv"))
    args = ap.parse_args()

    gold = pd.read_parquet(GOLD)
    votes = merged_votes()

    rows = []
    for lang in sorted(gold["lang"].unique()):
        g = gold[gold["lang"] == lang]
        n, pos = len(g), int((g["final"] == 1).sum())
        v = votes[votes["source_language"] == lang]
        units = [list(x) for x in v.groupby("original_id")["lab"].apply(list) if len(x) >= 1]
        n_units = [u for u in units if len(u) >= 2]
        alpha = krippendorff_alpha_nominal(units) if units else float("nan")
        mean_ann = np.mean([len(u) for u in units]) if units else float("nan")
        rows.append({"lang": lang, "N": n, "pos": pos,
                     "pos_rate_%": round(100 * pos / n, 1),
                     "mean_coders": round(mean_ann, 2) if units else None,
                     "n_with_alpha": len(n_units),
                     "alpha": round(alpha, 3) if not np.isnan(alpha) else None})

    # pooled alpha over all non-frozen languages present in the exports
    all_units = [list(x) for x in votes.groupby(["source_language", "original_id"])["lab"].apply(list)]
    pooled = krippendorff_alpha_nominal(all_units)
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print(f"\nPooled alpha (langs with per-annotator data): {pooled:.3f}")
    print("Note: EN/RU alpha here covers only the NEW extension items; the frozen "
          "released EN/RU benchmark's per-annotator labels are not in these exports.")
    tab.to_csv(args.out, index=False)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()
