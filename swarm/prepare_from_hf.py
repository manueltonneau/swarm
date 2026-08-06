"""Turn the released SWARM dataset into the eval parquets the runners expect.

This is the entry point for replication. The gold-set builders in this repo
(build_eval_9lang.py, build_native_eval.py) operate on the raw Label Studio
annotation exports, which are not released, so they are included for
transparency rather than for you to run. This script reproduces their output
from the public release instead.

Usage:
    huggingface-cli download manueltonneau/SWARM --repo-type dataset \
        --local-dir data/swarm
    python -m swarm.prepare_from_hf --src data/swarm --out data/eval

Writes gold_eval_set.parquet (English input) and gold_eval_set_native.parquet
(source-language input), the two files every model runner reads.

Note on text_en: the release ships a null English text for the 10 documents
whose translation pass returned a refusal or a request to shorten the input
rather than a translation. Those rows are kept, since the item count must stay
at 2,129 to match the paper, and their English text becomes an empty string
here. All 10 carry the negative label.

This is the one place where replication is not bit-exact. In the original runs
the models were shown the refusal string itself, not an empty document, so the
content_hash of those 10 rows differs from ours and predictions on them are not
strictly comparable. Every model assigned them the negative class, which is
also their gold label, so the effect on the reported metrics is nil, but the
difference is real and is noted here rather than papered over.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

N_EXPECTED = 2129
POS_EXPECTED = 384


def content_hash(question: str, text: str) -> str:
    """Must match build_eval_9lang.content_hash so --reuse works across runs."""
    h = hashlib.sha1()
    h.update(str(question).encode("utf-8"))
    h.update(b"\x1f")
    h.update(str(text).encode("utf-8"))
    return h.hexdigest()


def load(src: Path) -> pd.DataFrame:
    if src.is_file():
        return pd.read_parquet(src)
    files = sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no .parquet found under {src}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def to_eval_frame(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "original_id": df["id"].astype("int64"),
        "lang": df["lang"].astype(str),
        "question": df["narrative_question"].astype(str),
        "art_trunc": df[text_col].fillna("").astype(str),
        "final": df["label"].astype(float),
    })
    out["content_hash"] = [content_hash(q, t) for q, t
                           in zip(out["question"], out["art_trunc"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="directory or parquet file of the released SWARM dataset")
    ap.add_argument("--out", default=Path("data/eval"), type=Path)
    args = ap.parse_args()

    df = load(args.src)
    missing = {"id", "lang", "narrative_question", "label", "text_en", "text_native"} - set(df.columns)
    if missing:
        raise SystemExit(f"release is missing expected columns: {sorted(missing)}")

    if len(df) != N_EXPECTED:
        print(f"WARNING: expected {N_EXPECTED} rows, found {len(df)}. "
              "Numbers will not match the paper.")
    n_pos = int((df["label"] == 1).sum())
    if n_pos != POS_EXPECTED:
        print(f"WARNING: expected {POS_EXPECTED} positives, found {n_pos}.")

    args.out.mkdir(parents=True, exist_ok=True)
    en = to_eval_frame(df, "text_en")
    nat = to_eval_frame(df, "text_native")
    en.to_parquet(args.out / "gold_eval_set.parquet", index=False)
    nat.to_parquet(args.out / "gold_eval_set_native.parquet", index=False)

    n_null_en = int(df["text_en"].isna().sum())
    print(f"rows: {len(df)}  positives: {n_pos} ({100 * n_pos / len(df):.0f}%)")
    print(f"documents with no usable English text: {n_null_en}")
    print("per language:")
    print(df.groupby("lang").agg(n=("label", "size"), pos=("label", "sum")).to_string())
    print(f"\nWritten: {args.out / 'gold_eval_set.parquet'}")
    print(f"Written: {args.out / 'gold_eval_set_native.parquet'}")


if __name__ == "__main__":
    main()
