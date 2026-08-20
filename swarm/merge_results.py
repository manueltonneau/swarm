"""Merge all benchmark predictions into one results table.

Recomputes every method's metrics from its predictions_*.parquet with identical
code, so all methods (majority baseline, GPT-5-nano, Qwen, SetFit) x both input
variants (English / native) are strictly comparable on the same 1,702 items.

Scans for predictions_*.parquet in:
  data/predictions/eval_9lang/   (GPT-5-nano)
  results/                       (open-LLM and supervised runs, if kept separate)

Outputs: prints an ALL-row summary and a per-language table; writes
data/predictions/eval_9lang/all_results_summary.csv and a LaTeX snippet.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent.parent
DIRS = [Path(os.environ.get("SWARM_EVAL_DIR", ROOT / "data" / "eval")),
        ROOT / "results"]
EVAL = Path(os.environ.get("SWARM_EVAL_DIR", ROOT / "data" / "eval"))
GOLD = EVAL / "gold_eval_set.parquet"
OUTDIR = EVAL

# pretty names for filename stems (after stripping predictions_/ _native)
NAMES = {
    "gpt5nano": "GPT-5-nano (closed, 0-shot)",
    "qwen2.5-72b-instruct-awq": "Qwen2.5-72B (open, 0-shot)",
    "qwen2.5-7b-instruct": "Qwen2.5-7B (open, 0-shot)",
    "setfit": "SetFit e5-large (supervised)",
    "gpt54": "GPT-5.4 (closed, 0-shot)",
    "embed_lr": "e5-large + LR (supervised)",
    "xlmr": "XLM-R fine-tuned (supervised)",
}
ORDER = ["majority", "xlmr", "embed_lr", "setfit", "qwen2.5-7b-instruct",
         "qwen2.5-72b-instruct-awq", "gpt5nano", "gpt54"]


def boot_ci(y, p, fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p, n, out = np.asarray(y), np.asarray(p), len(y), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            out.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    return tuple(np.percentile(out, [2.5, 97.5])) if out else (np.nan, np.nan)


def posf1(y, p):
    return f1_score(y, p, pos_label=1, zero_division=0)


def metrics_row(y, p):
    return {
        "acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
        "kappa": cohen_kappa_score(y, p) if len(set(y)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y, p) if len(set(y)) > 1 else np.nan,
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "prec_pos": precision_score(y, p, pos_label=1, zero_division=0),
        "rec_pos": recall_score(y, p, pos_label=1, zero_division=0),
        "f1_pos": posf1(y, p),
    }


def collect():
    """Return list of (stem, variant, predictions_df)."""
    runs = {}
    for d in DIRS:
        for f in glob.glob(str(d / "predictions_*.parquet")):
            stem = Path(f).stem[len("predictions_"):]
            variant = "native" if stem.endswith("_native") else "english"
            stem = stem[:-len("_native")] if variant == "native" else stem
            df = pd.read_parquet(f)
            df = df.dropna(subset=["pred"]).copy()
            df["pred"] = df["pred"].astype(int)
            df["y"] = df["final"].astype(int)
            runs[(stem, variant)] = df[["original_id", "lang", "y", "pred"]]
    return runs


def main():
    gold = pd.read_parquet(GOLD)
    gold["y"] = gold["final"].astype(int)
    runs = collect()

    # majority baseline (predict 0); same for both variants
    maj = gold[["original_id", "lang", "y"]].copy()
    maj["pred"] = 0
    runs[("majority", "english")] = maj

    rows, per_lang = [], []
    for (stem, variant), df in runs.items():
        m = metrics_row(df["y"], df["pred"])
        bl = boot_ci(df["y"], df["pred"], balanced_accuracy_score)
        fl = boot_ci(df["y"], df["pred"], posf1)
        rows.append({"method": NAMES.get(stem, stem), "stem": stem,
                     "variant": variant, "n": len(df),
                     **{k: round(v, 3) for k, v in m.items()},
                     "bal_acc_ci": f"[{bl[0]:.2f},{bl[1]:.2f}]",
                     "f1_pos_ci": f"[{fl[0]:.2f},{fl[1]:.2f}]"})
        for lang, g in df.groupby("lang"):
            lm = metrics_row(g["y"], g["pred"])
            per_lang.append({"method": NAMES.get(stem, stem), "variant": variant,
                             "lang": lang, **{k: round(v, 3) for k, v in lm.items()}})

    summ = pd.DataFrame(rows)
    summ["ord"] = summ["stem"].map({s: i for i, s in enumerate(ORDER)}).fillna(99)
    summ = summ.sort_values(["ord", "variant"]).drop(columns="ord")
    summ.to_csv(OUTDIR / "all_results_summary.csv", index=False)
    pd.DataFrame(per_lang).to_csv(OUTDIR / "all_results_per_language.csv", index=False)

    show = ["method", "variant", "n", "acc", "bal_acc", "kappa", "mcc",
            "f1_macro", "prec_pos", "rec_pos", "f1_pos"]
    print("=== ALL-row summary (every method x variant) ===")
    print(summ[show].to_string(index=False))
    print(f"\nWrote {OUTDIR/'all_results_summary.csv'} and all_results_per_language.csv")


if __name__ == "__main__":
    main()
