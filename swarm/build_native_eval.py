"""Build the native-language variant of the gold eval set.

Same items, gold labels, and (English) question as gold_eval_set.parquet, but the
article text is the SOURCE language: original_text for non-English items, and the
English text for English items (whose original already is English). Isolates the
article-language variable for a translate-vs-native ablation.

Source text is taken from the companion gold_native_text.parquet that
build_eval_9lang.py writes from the merged export (so re-scraped items get their
corrected source text), falling back to the frozen items.csv for any item not in
that file (e.g. EN/RU or older items).

Adds a content_hash column (over question + native art_trunc) so the zero-shot
runners can reuse predictions for unchanged native items.
"""
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "predictions" / "eval_9lang"
GOLD = EVAL_DIR / "gold_eval_set.parquet"
NATIVE_TEXT = EVAL_DIR / "gold_native_text.parquet"          # written by build_eval_9lang
ITEMS = ROOT / "data" / "ra_annotated_package" / "items.csv"  # fallback source
OUT = EVAL_DIR / "gold_eval_set_native.parquet"
MAX_CHARS = 12_000


def content_hash(question: str, art_trunc: str) -> str:
    h = hashlib.sha1()
    h.update(str(question).encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(art_trunc).encode("utf-8"))
    return h.hexdigest()


def main():
    gold = pd.read_parquet(GOLD)  # original_id, lang, question, art_trunc(EN), final, content_hash

    # Preferred source text: the merged-export companion, keyed by (original_id, lang).
    src = {}
    if NATIVE_TEXT.exists():
        nt = pd.read_parquet(NATIVE_TEXT)
        src = {(int(r.original_id), r.lang): ("" if pd.isna(r.original_text) else str(r.original_text))
               for r in nt.itertuples()}
    # Fallback: frozen items.csv, keyed by original_id.
    items = pd.read_csv(ITEMS, usecols=["original_id", "original_text"])
    fallback = items.set_index("original_id")["original_text"].to_dict()

    def native(oid, lang, en_text):
        orig = src.get((int(oid), lang))
        if orig is None:
            orig = fallback.get(int(oid))
        orig = "" if (orig is None or (isinstance(orig, float) and pd.isna(orig))) else str(orig)
        return (orig if len(orig) > 50 else str(en_text))[:MAX_CHARS]

    gold["art_trunc"] = [native(r.original_id, r.lang, r.art_trunc) for r in gold.itertuples()]
    gold["content_hash"] = [content_hash(q, t) for q, t in zip(gold["question"], gold["art_trunc"])]
    gold.to_parquet(OUT, index=False)

    # report how many actually switched to a non-English source
    en = (gold["lang"] == "en").sum()
    print(f"native eval set: {len(gold)} items ({en} EN kept as English, "
          f"{len(gold) - en} non-EN in source language)")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
