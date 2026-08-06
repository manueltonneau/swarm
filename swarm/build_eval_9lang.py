"""Build the authoritative 9-language gold evaluation set.

Gold label = majority vote of all annotators (live-page + text-based);
an item is kept only if its extracted (English-translated) text is usable,
i.e. at least one text-based annotator could assess it. This reproduces the
released EN/RU 332-item benchmark exactly and extends it to the seven other
languages.

Output columns (gold_eval_set.parquet): original_id, lang, question,
art_trunc, final, content_hash
  - art_trunc: English-translated article text, truncated to MAX_CHARS
  - final: binary gold label (1 = supports propaganda claim, 0 = does not)
  - content_hash: sha1(question + art_trunc); lets the zero-shot runners score
    only new-or-changed items and reuse prior predictions for the rest.

A companion gold_native_text.parquet (original_id, lang, original_text) is
written from the same merged export so build_native_eval.py picks up the
source-language text of any re-scraped item.

EN/RU rows come from the released eval set. Their labels and questions are
taken verbatim, but for RU the released art_trunc is the *source* Russian text
(that study ran Russian natively), so it is replaced with the English
translation from the EN/RU annotation export, matching the other seven
languages. Pass --legacy-frozen-ru to reproduce the earlier, uncorrected build.

Merging additional annotation
-----------------------------
When a new round of labels lands (the PL reannotation + recovered items that
had not been scraped/translated properly), pass it as one or more combined
export CSVs:

    python build_eval_9lang.py \
        --additional-export data/annotation/<new-combined-export>.csv

Merge semantics:
  * rows from base + additional are pooled; on a (original_id, annotator)
    collision the additional (newer/corrected) row wins;
  * for PL, the mananeau live-page pass is dropped entirely (it over-flagged PL
    and is the reason PL alpha was 0.14 -> the round2 reannotation replaces it);
  * item-level fields (question, EN text, source text) are taken from the
    additional row when present, so re-scraped items get their corrected text;
  * recovered items simply enter the pool and pass the usable-text filter.

With no --additional-export the output is byte-identical to the released build.
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotation"
RELEASED = ROOT / "data" / "predictions" / "eval_257301_gpt41_vs_gpt5nano" / "eval_set.parquet"
OUT_DIR = ROOT / "data" / "predictions" / "eval_9lang"
MAX_CHARS = 12_000
# Annotator id of the live-page coding pass, as it appears in the Label Studio
# exports. Set SWARM_LIVE_PAGE_ANNOTATOR to the id used in your own exports.
# This coder's votes count like any other, except for Polish, where the pass
# was discarded and re-coded after it was found to over-flag the positive class.
SEED = os.environ.get("SWARM_LIVE_PAGE_ANNOTATOR", "live-page-annotator")
DROP_PL_ANNOTATOR = SEED

LABMAP = {
    "Does not support or mention propaganda claim (0)": 0,
    "Supports propaganda claim (1)": 1,
    "Cannot assess (article unavailable)": "na",
}
BASE_EXPORT = ANN / "non_en_ru" / "algo-audit-2nd-binarized-rest-non-enru-at-2026-06-19-08-41-938c77f5.csv"
EN_RU_EXPORT = ANN / "en_ru" / "export_257301_project-257301-at-2026-05-12-05-28-7ab77b95.csv"
REFIX = ANN / "translation_refix_results.csv"

EXPORT_COLS = ["original_id", "label", "annotator", "source_language",
               "text", "original_text", "question"]


def en_ru_english_text() -> dict[int, str]:
    """original_id -> English translation, from the EN/RU annotation export.

    The released EN/RU benchmark was built for a study that ran Russian
    *natively*, so its art_trunc holds the source Russian text. Inheriting it
    verbatim puts Russian in the English-input column for RU. The export's
    `text` column is the English translation annotators actually read, which is
    what the other seven languages use, so we take art_trunc from there.
    """
    csv.field_size_limit(10**9)
    out: dict[int, str] = {}
    with open(EN_RU_EXPORT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source_language") != "ru":
                continue
            oid, text = int(row["original_id"]), (row.get("text") or "")
            if len(text) > len(out.get(oid, "")):
                out[oid] = text
    return out


def refix_overrides() -> dict[int, str]:
    """original_id -> corrected English text from the chunked re-translation pass.

    The original translation call fed up to 60k chars at once and hit the output
    cap, so long documents came back as a meta-response ("shall I translate in
    parts?") instead of a translation. refix_translations.py re-ran those in
    chunks; these results were never propagated into the gold set.
    """
    if not REFIX.exists():
        return {}
    df = pd.read_csv(REFIX, dtype=str)
    df = df[df["refusal"].str.lower().isin(["false", "0"])]
    df = df[df["english"].fillna("").str.len() > 200]
    return {int(r.original_id): r.english for r in df.itertuples()}


def content_hash(question: str, art_trunc: str) -> str:
    h = hashlib.sha1()
    h.update(str(question).encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(art_trunc).encode("utf-8"))
    return h.hexdigest()


def majority(labels):
    usable = [x for x in labels if x in (0, 1)]
    if not usable:
        return None
    n1, n0 = usable.count(1), usable.count(0)
    if n1 > n0:
        return 1
    if n0 > n1:
        return 0
    return None  # tie -> dropped


def load_export(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in EXPORT_COLS)
    for c in EXPORT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df["_src"] = source
    df["_rank"] = 0 if source == "additional" else 1  # additional wins ties
    return df[EXPORT_COLS + ["_src", "_rank"]]


def merge_exports(base: Path, additional: list[Path], drop_pl_mananeau: bool) -> pd.DataFrame:
    frames = [load_export(base, "base")]
    for p in additional:
        frames.append(load_export(Path(p), "additional"))
    df = pd.concat(frames, ignore_index=True)

    # Drop the mananeau live-page pass for PL only. This belongs with the round2
    # reannotation (which replaces it with fresh PL labels); dropping it without
    # those new labels would just delete votes, so it is gated on the merge.
    if drop_pl_mananeau:
        mask_drop = (df["annotator"] == DROP_PL_ANNOTATOR) & (df["source_language"] == "pl")
        df = df[~mask_drop].copy()

    # on a (item, annotator) collision keep the additional (newer) label
    df = df.sort_values("_rank").drop_duplicates(
        ["original_id", "annotator"], keep="first")
    return df


def build_new_languages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lab"] = df["label"].map(LABMAP)
    rows = []
    for (lang, oid), g in df.groupby(["source_language", "original_id"]):
        # usable text-based votes = any non-SEED annotator who could assess it.
        # (For PL, SEED/mananeau is already dropped, so this is all PL votes.)
        scrape_usable = [x for x in g.loc[g.annotator != SEED, "lab"] if x in (0, 1)]
        if not scrape_usable:                       # extracted text not usable -> excluded
            continue
        gold = majority([x for x in g["lab"] if x in (0, 1)])
        if gold is None:                            # tie -> dropped
            continue
        # item-level fields from the newest (additional-preferred) row
        first = g.sort_values("_rank").iloc[0]
        text = "" if pd.isna(first["text"]) else str(first["text"])
        orig = "" if pd.isna(first["original_text"]) else str(first["original_text"])
        rows.append({
            "original_id": int(oid),
            "lang": lang,
            "question": first["question"],
            "art_trunc": text[:MAX_CHARS],
            "original_text": orig[:MAX_CHARS],
            "final": float(gold),
        })
    return pd.DataFrame(rows)


def diff_report(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Print what changed vs the previously written gold set."""
    if old is None:
        print("\n(no existing gold set to diff against)")
        return
    ok = old.set_index(["original_id", "lang"])["final"]
    nk = new.set_index(["original_id", "lang"])["final"]
    added = nk.index.difference(ok.index)
    removed = ok.index.difference(nk.index)
    common = ok.index.intersection(nk.index)
    flips = [(i, ok[i], nk[i]) for i in common if ok[i] != nk[i]]

    print("\n=== DIFF vs previous gold_eval_set.parquet ===")
    print(f"  items: {len(ok)} -> {len(nk)}  (+{len(added)} / -{len(removed)})")
    if len(added):
        by = pd.Series([l for _, l in added]).value_counts().to_dict()
        print(f"  added by lang:   {by}")
    if len(removed):
        by = pd.Series([l for _, l in removed]).value_counts().to_dict()
        print(f"  removed by lang: {by}")
    print(f"  gold-label flips on retained items: {len(flips)}")
    if flips:
        by = pd.Series([l for (_, l), _, _ in flips]).value_counts().to_dict()
        print(f"    flips by lang: {by}")
        for (oid, lang), a, b in flips[:15]:
            print(f"      {lang} id={oid}: {int(a)} -> {int(b)}")
        if len(flips) > 15:
            print(f"      ... and {len(flips) - 15} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-export", default=str(BASE_EXPORT),
                    help="base Label Studio export CSV (non-EN/RU annotations)")
    ap.add_argument("--additional-export", nargs="*", default=[],
                    help="one or more combined export CSVs with new labels to merge")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--legacy-frozen-ru", action="store_true",
                    help="reproduce the pre-correction build, where RU rows are "
                         "inherited verbatim from the released EN/RU benchmark "
                         "and so carry Russian text in the English-input column")
    ap.add_argument("--pl-drop-mananeau", choices=["auto", "yes", "no"], default="auto",
                    help="drop the mananeau live-page pass for PL. 'auto' (default) "
                         "= drop only when an --additional-export (the round2 PL "
                         "reannotation) is supplied, so the no-additional run "
                         "reproduces the released set exactly")
    args = ap.parse_args()

    if args.pl_drop_mananeau == "auto":
        drop_pl = bool(args.additional_export)
    else:
        drop_pl = args.pl_drop_mananeau == "yes"

    out_dir = Path(args.out_dir)
    gold_path = out_dir / "gold_eval_set.parquet"
    prev = pd.read_parquet(gold_path) if gold_path.exists() else None

    merged = merge_exports(Path(args.base_export), args.additional_export, drop_pl)
    new = build_new_languages(merged)

    released = pd.read_parquet(
        RELEASED, columns=["original_id", "lang", "question", "art_trunc", "final"]
    ).copy()
    released["original_text"] = released["art_trunc"]  # EN source == EN text

    if not args.legacy_frozen_ru:
        # For RU the released art_trunc is the *source* text, not a translation
        # (see en_ru_english_text). Swap in the English the annotators read;
        # original_text keeps the Russian for the native ablation.
        eng = en_ru_english_text()
        is_ru = released["lang"] == "ru"
        english = released.loc[is_ru, "original_id"].map(lambda i: eng.get(int(i), ""))
        usable = english.str.len() > 200
        released.loc[english.index[usable], "art_trunc"] = english[usable]
        print(f"RU rows given their English translation: {int(usable.sum())} "
              f"of {int(is_ru.sum())}")

    full = pd.concat([released, new[released.columns]], ignore_index=True)

    if not args.legacy_frozen_ru:
        # Long documents whose first-pass translation returned a meta-response
        # were re-translated in chunks; apply those corrections everywhere.
        fixes = refix_overrides()
        hit = full["original_id"].isin(fixes)
        full.loc[hit, "art_trunc"] = full.loc[hit, "original_id"].map(fixes)
        print(f"re-translated documents applied: {int(hit.sum())}")

    full["art_trunc"] = full["art_trunc"].str.slice(0, MAX_CHARS)
    full["content_hash"] = [content_hash(q, t) for q, t
                            in zip(full["question"], full["art_trunc"])]

    out_dir.mkdir(parents=True, exist_ok=True)
    gold_cols = ["original_id", "lang", "question", "art_trunc", "final", "content_hash"]
    full[gold_cols].to_parquet(gold_path, index=False)
    # companion source-language text for the native ablation builder
    full[["original_id", "lang", "original_text"]].to_parquet(
        out_dir / "gold_native_text.parquet", index=False)

    summary = (
        full.assign(pos=full["final"] == 1)
        .groupby("lang")
        .agg(n=("final", "size"), pos=("pos", "sum"))
    )
    summary["rate"] = (summary["pos"] / summary["n"] * 100).round(0)
    print(summary)
    print(f"\nTotal: {len(full)} items, {int((full['final'] == 1).sum())} positive")
    if args.additional_export:
        print(f"Merged additional exports: {args.additional_export}")
    diff_report(prev, full)
    print(f"\nWritten: {gold_path}")
    print(f"Written: {out_dir / 'gold_native_text.parquet'}")


if __name__ == "__main__":
    main()
