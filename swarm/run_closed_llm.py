#!/usr/bin/env python3
"""Zero-shot GPT-5-nano benchmark over the 9-language gold eval set (Batch API).

Subcommands:
  build         Build batch input JSONL from gold_eval_set.parquet
  submit        Upload + submit the batch (saves resumable state)
  collect       Poll, download, parse, write predictions + per-language metrics

Prompt template is reconstructed byte-for-byte from the released EN/RU run.
Uses the OpenAI Batch API (Responses endpoint) with the liza key.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_utils import (  # noqa: E402
    submit_batch, poll_batch, download_results, parse_batch_results,
    build_request, write_batch_input,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "predictions" / "eval_9lang"
GOLD = OUT / "gold_eval_set.parquet"
MODEL = "gpt-5-nano"

PREFIX = (
    "You are tasked with analyzing a given text to determine its stance on a "
    "specific claim.\nYour goal is to classify the text based on whether it "
    "supports the claim or not.\n"
)
SUFFIX = (
    "\nInstructions: Read the text carefully and identify any statements that "
    "relate to the claim.\nDetermine the overall sentiment of the text towards "
    "the claim:\n- If the text supports the claim, respond with 1.\n- If the "
    "text rejects the claim, offers a mixed view on the claim, or does not "
    "contain the claim at all, respond with 0.\nIf the beginning of the text "
    "expresses a different stance on the claim than the end or the middle of "
    "the text, then respond with 0.\nRepeat the assessment nine times and "
    "choose the response occurring more often.\nExample Response: If the text "
    "contains evidence that contradicts the idea of Russia as a defender of "
    "traditional values, such as references to high abortion rates, divorce "
    "rates, and the political motivations behind the anti-gay campaign, you "
    "would respond with 0.\nApply this analytical logic to other claims and "
    "texts.\nPlease provide your classification based on the analysis of the "
    "text and respond only with a numeric label (0 or 1)."
)


def make_prompt(question: str, art_trunc: str) -> str:
    return PREFIX + f"Claim: {question}.\nText: {art_trunc}." + SUFFIX


def _hash_series(df):
    """content_hash column if present, else recomputed from question+art_trunc."""
    import hashlib
    if "content_hash" in df.columns and df["content_hash"].notna().all():
        return df["content_hash"].astype(str)

    def h(q, t):
        m = hashlib.sha1()
        m.update(str(q).encode("utf-8"))
        m.update(b"\x1f")
        m.update(str(t).encode("utf-8"))
        return m.hexdigest()
    return pd.Series([h(q, t) for q, t in zip(df["question"], df["art_trunc"])],
                     index=df.index)


def load_reuse(paths):
    """Map content_hash -> raw_pred from prior prediction parquet(s)."""
    reuse = {}
    for p in paths or []:
        d = pd.read_parquet(p)
        if "raw_pred" not in d.columns:
            continue
        for hsh, raw in zip(_hash_series(d), d["raw_pred"]):
            reuse[hsh] = raw
    return reuse


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "Set OPENAI_API_KEY in the environment before running the closed-model steps.")
    return OpenAI(api_key=key)


def cmd_build(args):
    df = pd.read_parquet(args.eval)
    if args.lang:
        df = df[df["lang"] == args.lang]
    if args.n:
        df = df.head(args.n)
    if args.reuse:
        reuse = load_reuse(args.reuse)
        h = _hash_series(df)
        keep = ~h.map(lambda x: x in reuse)
        print(f"incremental: {int((~keep).sum())} items reused, "
              f"{int(keep.sum())} to score", file=sys.stderr)
        df = df[keep]
    records = [
        build_request(f"item_{int(r.original_id)}",
                      make_prompt(str(r.question), str(r.art_trunc)),
                      model=args.model, effort=args.effort)
        for r in df.itertuples()
    ]
    print(f"model={args.model} effort={args.effort} items={len(records)}", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    write_batch_input(records, args.out)


def cmd_submit(args):
    bid = submit_batch(client(), args.input, args.state)
    print(bid)


def _to_label(text):
    if not isinstance(text, str):
        return None
    for ch in text.strip():
        if ch in "01":
            return int(ch)
    return None


def cmd_collect(args):
    cl = client()
    state = json.loads(Path(args.state).read_text())
    job = poll_batch(cl, state["batch_id"])
    download_results(cl, job, args.output, args.errors)
    raw = parse_batch_results(args.output)

    gold = pd.read_parquet(args.eval)
    gold["cid"] = "item_" + gold["original_id"].astype(int).astype(str)
    gold["raw_pred"] = gold["cid"].map(raw)
    if args.reuse:
        # fill items that were not re-scored (unchanged) from prior predictions
        reuse = load_reuse(args.reuse)
        h = _hash_series(gold)
        filled = gold["raw_pred"].isna() & h.map(lambda x: x in reuse)
        gold.loc[filled, "raw_pred"] = h[filled].map(reuse)
        print(f"reused {int(filled.sum())} prior predictions", file=sys.stderr)
    gold["pred"] = gold["raw_pred"].map(_to_label)

    n_missing = gold["pred"].isna().sum()
    print(f"Parsed {len(gold) - n_missing}/{len(gold)} predictions "
          f"({n_missing} missing/unparseable)", file=sys.stderr)

    gold.to_parquet(OUT / f"predictions_{args.tag}.parquet", index=False)

    ev = gold.dropna(subset=["pred"]).copy()
    ev["pred"] = ev["pred"].astype(int)
    ev["y"] = ev["final"].astype(int)
    rows = []
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 cohen_kappa_score, f1_score,
                                 precision_score, recall_score, matthews_corrcoef)
    for lang, g in list(ev.groupby("lang")) + [("ALL", ev)]:
        y, p = g["y"], g["pred"]
        rows.append({
            "lang": lang, "n": len(g), "pos": int(y.sum()),
            "acc": round(accuracy_score(y, p), 3),
            "bal_acc": round(balanced_accuracy_score(y, p), 3),
            "kappa": round(cohen_kappa_score(y, p), 3) if y.nunique() > 1 else None,
            "mcc": round(matthews_corrcoef(y, p), 3) if y.nunique() > 1 else None,
            "f1_macro": round(f1_score(y, p, average="macro", zero_division=0), 3),
            "prec_pos": round(precision_score(y, p, pos_label=1, zero_division=0), 3),
            "rec_pos": round(recall_score(y, p, pos_label=1, zero_division=0), 3),
            "f1_pos": round(f1_score(y, p, pos_label=1, zero_division=0), 3),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / f"metrics_{args.tag}.csv", index=False)
    print(metrics.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--eval", default=str(GOLD), help="eval-set parquet")
    b.add_argument("--out", default=str(OUT / "batch_input_gpt5nano.jsonl"))
    b.add_argument("--effort", default="low",
                   choices=["minimal", "low", "medium", "high"],
                   help="reasoning effort")
    b.add_argument("--model", default=MODEL, help="OpenAI model id")
    b.add_argument("--lang", default=None, help="restrict to one language (dry run)")
    b.add_argument("--n", type=int, default=0, help="limit to first N items (0=all)")
    b.add_argument("--reuse", nargs="*", default=[],
                   help="prior predictions_*.parquet: items whose content_hash "
                        "matches are skipped (only new/changed items are sent to the API)")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("submit")
    s.add_argument("--input", default=str(OUT / "batch_input_gpt5nano.jsonl"))
    s.add_argument("--state", default=str(OUT / "batch_state_gpt5nano.json"))
    s.set_defaults(func=cmd_submit)

    c = sub.add_parser("collect")
    c.add_argument("--eval", default=str(GOLD), help="eval-set parquet (for gold join)")
    c.add_argument("--tag", default="gpt5nano", help="output filename tag")
    c.add_argument("--state", default=str(OUT / "batch_state_gpt5nano.json"))
    c.add_argument("--output", default=str(OUT / "batch_output_gpt5nano.jsonl"))
    c.add_argument("--errors", default=str(OUT / "batch_errors_gpt5nano.jsonl"))
    c.add_argument("--reuse", nargs="*", default=[],
                   help="prior predictions_*.parquet to fill items that were not "
                        "re-scored this round (must match the --reuse used at build)")
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
