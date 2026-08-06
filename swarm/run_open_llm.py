#!/usr/bin/env python3
"""Zero-shot open-LLM benchmark on the 9-language gold eval set (multi-GPU).

Uses vLLM if available (fast batched inference), else transformers. Greedy
decoding, identical prompt to the GPT-5-nano run, predictions parsed to 0/1.

Usage:
  # 1. see which instruct models are already cached
  python run_open_llm.py --list-cached
  # 2. run (model = HF id or local path); tp = number of GPUs to shard across
  python run_open_llm.py --model Qwen/Qwen2.5-72B-Instruct --tp 4
  python run_open_llm.py --model meta-llama/Llama-3.1-8B-Instruct --tp 1

Outputs (tagged by a sanitised model name):
  predictions_<tag>.parquet   one row per item with raw_pred + pred
  metrics_<tag>.csv           per-language + overall metrics with CIs
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_prompt, to_label, metrics_table  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold_eval_set.parquet"


def _hash_of(df):
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
    for p in paths:
        d = pd.read_parquet(p)
        if "raw_pred" not in d.columns:      # e.g. supervised preds have no raw text
            continue
        for hsh, raw in zip(_hash_of(d), d["raw_pred"]):
            reuse[hsh] = raw
    return reuse


def list_cached():
    roots = [
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("HF_HOME", ""),
        os.path.expanduser("~/.cache/huggingface"),
    ]
    seen = set()
    for r in roots:
        if not r:
            continue
        # models--* may live directly in the cache root or under hub/
        for pat in (os.path.join(r, "models--*"), os.path.join(r, "hub", "models--*")):
            for p in glob.glob(pat):
                name = os.path.basename(p).replace("models--", "").replace("--", "/")
                if name not in seen:
                    seen.add(name)
                    print(f"  {name}")
    if not seen:
        print("  (no cached models found; set HF_HOME or check the cache path)")


def tag_of(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model.strip("/").split("/")[-1]).lower()


def build_prompts(df, tokenizer):
    """Apply the model's chat template to each (claim, document) prompt."""
    texts = []
    for r in df.itertuples():
        msg = [{"role": "user", "content": make_prompt(str(r.question), str(r.art_trunc))}]
        try:
            t = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        except Exception:
            t = make_prompt(str(r.question), str(r.art_trunc))  # no chat template
        texts.append(t)
    return texts


def run_vllm(model, df, tp, max_len):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    prompts = build_prompts(df, tok)
    kw = dict(model=model, tensor_parallel_size=tp, dtype="bfloat16",
              max_model_len=max_len, gpu_memory_utilization=0.90, trust_remote_code=True)
    # allow forcing a quantization kernel (e.g. VLLM_QUANT=awq to avoid the
    # awq_marlin kernel whose PTX is too new for an older CUDA driver)
    quant = os.environ.get("VLLM_QUANT")
    if quant:
        kw["quantization"] = quant
        if quant == "awq":
            kw["dtype"] = "float16"  # the awq (non-marlin) GEMM kernel is fp16-only
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outs]


def run_hf(model, df, max_len):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    mdl.eval()
    raws = []
    for i, r in enumerate(df.itertuples()):
        msg = [{"role": "user", "content": make_prompt(str(r.question), str(r.art_trunc))}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True,
                                      return_tensors="pt", truncation=True,
                                      max_length=max_len).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=8, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        raws.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}", file=sys.stderr)
    return raws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF id or local path of the instruct model")
    ap.add_argument("--list-cached", action="store_true")
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (vLLM)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--eval", default=str(GOLD),
                    help="eval-set parquet (gold_eval_set.parquet [English] or "
                         "gold_eval_set_native.parquet [source language])")
    ap.add_argument("--reuse", nargs="*", default=[],
                    help="prior predictions_*.parquet to reuse: items whose "
                         "content_hash matches are NOT re-scored (incremental rerun)")
    args = ap.parse_args()

    if args.list_cached or not args.model:
        print("Cached models:")
        list_cached()
        if not args.model:
            print("\nRe-run with --model <id-or-path>.")
            return

    df = pd.read_parquet(args.eval)
    suffix = "_native" if "native" in Path(args.eval).stem else ""
    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except Exception:
            backend = "hf"
    # Incremental: reuse prior predictions for unchanged items (matched by hash),
    # score only new-or-changed rows with the model.
    df["_hash"] = _hash_of(df)
    reuse = load_reuse(args.reuse) if args.reuse else {}
    known = df["_hash"].map(lambda h: h in reuse) if reuse else pd.Series(False, index=df.index)
    to_run = df[~known].copy()
    print(f"Backend: {backend} | model: {args.model} | items: {len(df)} "
          f"(reuse {int(known.sum())}, run {len(to_run)})", file=sys.stderr)

    if len(to_run):
        raws_run = run_vllm(args.model, to_run, args.tp, args.max_len) if backend == "vllm" \
            else run_hf(args.model, to_run, args.max_len)
    else:
        raws_run = []
    df["raw_pred"] = df["_hash"].map(reuse)
    df.loc[~known, "raw_pred"] = raws_run
    df = df.drop(columns="_hash")
    df["pred"] = df["raw_pred"].map(to_label)
    n_bad = df["pred"].isna().sum()
    print(f"Parsed {len(df) - n_bad}/{len(df)} ({n_bad} unparseable)", file=sys.stderr)

    tag = tag_of(args.model) + suffix
    df.to_parquet(HERE / f"predictions_{tag}.parquet", index=False)
    m = metrics_table(df)
    m.to_csv(HERE / f"metrics_{tag}.csv", index=False)
    print(m.to_string(index=False))
    print(f"\nWrote predictions_{tag}.parquet and metrics_{tag}.csv")


if __name__ == "__main__":
    main()
