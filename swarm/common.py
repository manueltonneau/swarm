"""Shared prompt, parsing, and metrics for the SWARM benchmark runs.

The prompt is byte-for-byte the one used for the GPT-5-nano benchmark, so all
models are evaluated under identical inputs on the same 1,702-item gold set.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

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


def to_label(text):
    """Extract the first 0/1 from a model's raw output; None if neither."""
    if text is None:
        return None
    for ch in str(text):
        if ch in "01":
            return int(ch)
    return None


def _boot_ci(y, p, fn, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            out.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    if not out:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(out, [2.5, 97.5]))


def _posf1(y, p):
    return f1_score(y, p, pos_label=1, zero_division=0)


def metrics_table(df, pred_col="pred", gold_col="final", lang_col="lang"):
    """Per-language + overall metrics with bootstrap CIs on bal-acc and pos-F1.

    df must contain pred_col (0/1, non-null), gold_col, lang_col.
    Returns a DataFrame.
    """
    d = df.dropna(subset=[pred_col]).copy()
    d["_p"] = d[pred_col].astype(int)
    d["_y"] = d[gold_col].astype(int)
    rows = []
    for lang, g in list(d.groupby(lang_col)) + [("ALL", d)]:
        y, p = g["_y"].values, g["_p"].values
        ba_lo, ba_hi = _boot_ci(y, p, balanced_accuracy_score)
        pf_lo, pf_hi = _boot_ci(y, p, _posf1)
        rows.append({
            "lang": lang, "n": len(g), "pos": int(y.sum()),
            "acc": round(accuracy_score(y, p), 3),
            "bal_acc": round(balanced_accuracy_score(y, p), 3),
            "bal_acc_lo": round(ba_lo, 3), "bal_acc_hi": round(ba_hi, 3),
            "kappa": round(cohen_kappa_score(y, p), 3) if len(set(y)) > 1 else None,
            "mcc": round(matthews_corrcoef(y, p), 3) if len(set(y)) > 1 else None,
            "f1_macro": round(f1_score(y, p, average="macro", zero_division=0), 3),
            "prec_pos": round(precision_score(y, p, pos_label=1, zero_division=0), 3),
            "rec_pos": round(recall_score(y, p, pos_label=1, zero_division=0), 3),
            "f1_pos": round(_posf1(y, p), 3),
            "f1_pos_lo": round(pf_lo, 3), "f1_pos_hi": round(pf_hi, 3),
        })
    return pd.DataFrame(rows)
