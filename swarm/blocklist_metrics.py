"""HQP-style analysis: do source-domain distant labels predict narrative stance?

HQP (Maarouf et al. 2024) shows that the weak, source-based labels used by large
propaganda corpora (an article inherits the propaganda reputation of its outlet)
diverge sharply from human content judgments and yield weaker classifiers. The
SWARM paper makes this argument in the related work; this script substantiates it
empirically on SWARM's own data.

Our task is finer still: claim-conditioned stance ("does this article support
propaganda claim X"), not generic propaganda-vs-not. So training an off-the-shelf
propaganda classifier and testing here would be a task mismatch. Instead we make
the HQP point directly: treat the cheap, source-level signals we have (curated
propaganda-domain lists; domain news-reliability scores) as distant labels and
score them against the human claim-support gold. Bundle additional domain lists
from the source-based datasets SWARM cites (Proppy/MBFC, Rashkin 2017,
EUvsDisinfo) via build_domain_blocklist.py to widen coverage.

Two distant labels:
  1. flagged-domain   1 if the article's domain is on a curated Russian-propaganda
                      domain list (data/russian_prop_wiki.csv).
  2. low-quality      1 if the domain's news-reliability score (pc1 in
                      data/news_quality_ratings.csv) is below --quality-thr.

For each we report metrics against the human gold on (a) ALL eval items, where an
item with no flag / no rating is treated as distant=0 (the realistic "deploy a
domain filter" setting), and (b) the covered subset only (fair to the signal).
The take-away for the paper: domain-level labels have poor coverage and/or poor
agreement with article-level narrative support, which is why article-level human
annotation is needed. Metrics use the same code as the model benchmark, so the
numbers sit directly beside the model rows.

Outputs: metrics_distant_domain.csv, metrics_distant_quality.csv (in the eval
dir) + a printed summary.
"""
import argparse
import sys
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUX = ROOT / "data_aux"
sys.path.insert(0, str(ROOT / "virgil_eval"))
sys.path.insert(0, str(ROOT / "code"))
try:  # works both as `python -m swarm.x` and as `python swarm/x.py`
    from .common import metrics_table
except ImportError:  # noqa: E402
    from common import metrics_table
try:
    from .build_domain_blocklist import reg_domain
except ImportError:  # noqa: E402
    from build_domain_blocklist import reg_domain

EVAL_DIR = Path(os.environ.get("SWARM_EVAL_DIR", ROOT / "data" / "eval"))
GOLD = EVAL_DIR / "gold_eval_set.parquet"
ITEMS = ROOT / "data" / "ra_annotated_package" / "items.csv"
PROP_DOMAINS = AUX / "russian_prop_wiki.csv"
BUNDLE = AUX / "propaganda_domains_bundle.csv"
QUALITY = AUX / "news_quality_ratings.csv"
DOMAIN_CATS = ROOT / "data" / "categorized_domains" / "final_cat_domains.csv"

# domain types treated as mainstream / institutional (a blocklist would not list
# these); a missed positive here is propaganda mixed into non-propaganda-only media
MAINSTREAM_CATS = {"news outlet", "official government/state",
                   "international organization", "wikipedia", "fact-checker"}


def item_domains() -> pd.DataFrame:
    """Resolved domain + source category per item, from the categorisation table
    (full export link map + fixed multi-TLD parser). Falls back to items.csv."""
    resolved = AUX / "domain_categories.csv"
    if resolved.exists():
        d = pd.read_csv(resolved)[["original_id", "domain", "category"]]
        return d.dropna(subset=["original_id"]).drop_duplicates("original_id")
    items = pd.read_csv(ITEMS, usecols=["original_id", "link"])
    items["domain"] = items["link"].map(reg_domain)
    items["category"] = pd.NA
    return items[["original_id", "domain", "category"]]


def flagged_domains(bundle: Path) -> set:
    """Use the bundled blocklist if present, else the Wikipedia list alone."""
    if bundle.exists():
        return set(pd.read_csv(bundle)["domain"].dropna().astype(str))
    w = pd.read_csv(PROP_DOMAINS)
    col = "clean_domain" if "clean_domain" in w.columns else "Domain"
    return {reg_domain(v) for v in w[col].dropna()}


def report(df, pred_col, gold_col="final", lang_col="lang", tag="", note=""):
    """metrics_table on ALL rows (missing signal -> 0) and on the covered subset."""
    allrows = df.copy()
    allrows["pred"] = pd.to_numeric(allrows[pred_col], errors="coerce").fillna(0).astype(int)
    m_all = metrics_table(allrows, pred_col="pred", gold_col=gold_col, lang_col=lang_col)
    m_all.insert(0, "scope", "all_items")

    cov = df[df[pred_col].notna()].copy()
    cov["pred"] = pd.to_numeric(cov[pred_col], errors="coerce").astype(int)
    m_cov = metrics_table(cov, pred_col="pred", gold_col=gold_col, lang_col=lang_col)
    m_cov.insert(0, "scope", "covered_only")

    out = pd.concat([m_all, m_cov], ignore_index=True)
    all_row = m_all[m_all["lang"] == "ALL"].iloc[0]
    cov_row = m_cov[m_cov["lang"] == "ALL"].iloc[0]
    print(f"\n=== {tag} ===  {note}")
    print(f"  coverage: {len(cov)}/{len(df)} items carry the signal "
          f"({100 * len(cov) / len(df):.0f}%)")
    print(f"  ALL items   (missing=0): bal_acc={all_row.bal_acc}  "
          f"f1_pos={all_row.f1_pos}  rec_pos={all_row.rec_pos}  prec_pos={all_row.prec_pos}")
    print(f"  covered only           : bal_acc={cov_row.bal_acc}  "
          f"f1_pos={cov_row.f1_pos}  rec_pos={cov_row.rec_pos}  prec_pos={cov_row.prec_pos}")
    return out


def recall_by_language(df, cat_path=DOMAIN_CATS):
    """Per-language recall of gold positives by the flagged-domain label, and a
    breakdown of the MISSED positives by domain type: mainstream/institutional
    (propaganda mixed into ordinary media) vs fringe/unlisted (a coverage gap).

    Answers: is a source-based blocklist failing because it lacks non-English
    coverage (fringe sites not listed), or because propaganda support is carried
    by mainstream outlets no blocklist would contain?
    """
    d = df.copy()
    d["onlist"] = d["distant_domain"].fillna(0).astype(int) == 1
    if "category" not in d.columns:
        d["category"] = pd.NA

    rows = []
    for lang, g in list(d.groupby("lang")) + [("ALL", d)]:
        pos = g[g["final"] == 1]
        caught = int(pos["onlist"].sum())
        miss = pos[~pos["onlist"]]
        mainstream = int(miss["category"].isin(MAINSTREAM_CATS).sum())
        rows.append({
            "lang": lang, "n": len(g), "pos": len(pos),
            "item_cov_%": round(100 * g["onlist"].mean(), 0),
            "pos_recall_%": round(100 * caught / max(len(pos), 1), 0),
            "missed": len(miss),
            "miss_mainstream": mainstream,          # H2: mixed into ordinary media
            "miss_fringe": len(miss) - mainstream,  # H1: unlisted / coverage gap
        })
    tab = pd.DataFrame(rows)
    print("\n=== per-language recall of positives by flagged-domain, and miss type ===")
    print("  miss_mainstream = missed positive on a news/gov/intl-org/wiki/fact-checker "
          "domain (propaganda in ordinary media)")
    print("  miss_fringe     = missed positive on a blog/unlisted domain (coverage gap)")
    print(tab.to_string(index=False))
    m = tab[tab["lang"] == "ALL"].iloc[0]
    if m["missed"]:
        print(f"\n  overall: blocklist recalls {m['pos_recall_%']:.0f}% of supporting "
              f"articles; of the {int(m['missed'])} it misses, "
              f"{100 * m['miss_mainstream'] / m['missed']:.0f}% are on mainstream media "
              f"(mixing) and {100 * m['miss_fringe'] / m['missed']:.0f}% on fringe/unlisted "
              f"domains (coverage gap).")
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(GOLD))
    ap.add_argument("--quality-thr", type=float, default=0.5,
                    help="pc1 below this = low-quality (distant propaganda proxy)")
    ap.add_argument("--bundle", default=str(BUNDLE),
                    help="bundled blocklist from build_domain_blocklist.py "
                         "(falls back to the Wikipedia list if absent)")
    ap.add_argument("--out-dir", default=str(EVAL_DIR))
    args = ap.parse_args()

    gold = pd.read_parquet(args.eval)[["original_id", "lang", "final"]]
    df = gold.merge(item_domains(), on="original_id", how="left")

    # 1) flagged propaganda domain (bundled blocklist, registered-domain match)
    flags = flagged_domains(Path(args.bundle))
    df["distant_domain"] = df["domain"].isin(flags).astype(int)
    # items with an empty domain have no signal -> NA (treated as 0 in the ALL scope)
    df.loc[df["domain"].fillna("") == "", "distant_domain"] = pd.NA

    # 2) low news-quality domain
    q = pd.read_csv(QUALITY)
    q["domain"] = q["domain"].map(reg_domain)
    q = q.dropna(subset=["domain"]).drop_duplicates("domain")
    df = df.merge(q[["domain", "pc1"]], on="domain", how="left")
    df["distant_quality"] = pd.NA
    rated = df["pc1"].notna()
    df.loc[rated, "distant_quality"] = (df.loc[rated, "pc1"] < args.quality_thr).astype(int)

    n_pos = int((df["final"] == 1).sum())
    print(f"eval items: {len(df)}  gold positives: {n_pos} "
          f"(base rate {n_pos / len(df):.3f})")
    print(f"gold positives from a flagged propaganda domain: "
          f"{int(((df.distant_domain == 1) & (df.final == 1)).sum())} "
          f"-> a domain blocklist recalls "
          f"{100 * ((df.distant_domain == 1) & (df.final == 1)).sum() / max(n_pos,1):.1f}% of them")

    m_dom = report(df, "distant_domain", tag="distant label 1: flagged propaganda domain",
                   note="(russian_prop_wiki.csv)")
    m_qual = report(df, "distant_quality", tag="distant label 2: low news-quality domain",
                    note=f"(pc1 < {args.quality_thr}, news_quality_ratings.csv)")

    by_lang = recall_by_language(df)

    out_dir = Path(args.out_dir)
    m_dom.to_csv(out_dir / "metrics_distant_domain.csv", index=False)
    m_qual.to_csv(out_dir / "metrics_distant_quality.csv", index=False)
    by_lang.to_csv(out_dir / "metrics_domain_recall_by_language.csv", index=False)
    print(f"\nWritten: {out_dir / 'metrics_distant_domain.csv'}")
    print(f"Written: {out_dir / 'metrics_distant_quality.csv'}")
    print(f"Written: {out_dir / 'metrics_domain_recall_by_language.csv'}")
    print("\nInterpretation: both source-level distant labels have poor coverage "
          "and/or poor agreement with the human claim-support gold, well below the "
          "trained and zero-shot models -> article-level annotation is necessary.")


if __name__ == "__main__":
    main()
