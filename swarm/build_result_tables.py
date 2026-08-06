"""Generate the paper's result tables from the merged metrics CSVs.

Writes latex/tables/full_results.tex, latex/tables/per_lang_posf1.tex and
latex/tables/per_lang_balacc.tex directly, so the paper cannot drift from the
pipeline the way it did before (the .tex files were previously hand-maintained).

Run after ./run_rerun.sh merge:
    python3 code/build_result_tables.py
"""
import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path(os.environ.get("SWARM_EVAL_DIR", ROOT / "data" / "eval"))
TABLES = Path(os.environ.get("SWARM_TABLES_DIR", ROOT / "tables"))
LANGS = ["AR", "DE", "EN", "ES", "HI", "PL", "PT", "RU", "UK"]

# display name -> (short label, family) in the order the paper presents them
ORDER = [
    ("XLM-R fine-tuned (supervised)", "XLM-R (sup.)", "XLM-R", "sup"),
    ("e5-large + LR (supervised)", "e5-LR (sup.)", "e5-LR", "sup"),
    ("SetFit e5-large (supervised)", "SetFit (sup.)", "SetFit", "sup"),
    ("Qwen2.5-7B (open, 0-shot)", "Qwen2.5-7B", "Qwen-7B", "zs"),
    ("Qwen2.5-72B (open, 0-shot)", "Qwen2.5-72B", "Qwen-72B", "zs"),
    ("GPT-5-nano (closed, 0-shot)", "GPT-5-nano", "GPT-5-nano", "zs"),
    ("GPT-5.4 (closed, 0-shot)", "GPT-5.4", "GPT-5.4", "zs"),
]


def f2(x):
    return f"{x:.2f}"


def blocklist_overall():
    d = pd.read_csv(EVAL / "metrics_distant_domain.csv")
    r = d[(d.scope == "all_items") & (d.lang == "ALL")]
    if r.empty:                      # merge writes per-language rows only
        r = d[d.scope == "all_items"]
        n, pos = r.n.sum(), r.pos.sum()
        return None
    return r.iloc[0]


def full_results():
    s = pd.read_csv(EVAL / "all_results_summary.csv")
    s = s[s.stem != "majority"]
    rows = []
    bl = blocklist_overall()
    if bl is not None:
        rows.append(
            f"Domain blocklist & --- & {f2(bl.bal_acc)} ({f2(bl.bal_acc_lo)}--{f2(bl.bal_acc_hi)}) "
            f"& {f2(bl.acc)} & {f2(bl.prec_pos)} & {f2(bl.rec_pos)} "
            f"& {f2(bl.f1_pos)} ({f2(bl.f1_pos_lo)}--{f2(bl.f1_pos_hi)}) \\\\")
        rows.append("\\midrule")
    prev_family = None
    for disp, label, _short, family in ORDER:
        if prev_family and family != prev_family:
            rows.append("\\midrule")
        prev_family = family
        for variant, vlabel in (("english", "en"), ("native", "native")):
            r = s[(s.method == disp) & (s.variant == variant)]
            if r.empty:
                continue
            r = r.iloc[0]
            blo, bhi = eval(r.bal_acc_ci.replace(" ", ","))
            flo, fhi = eval(r.f1_pos_ci.replace(" ", ","))
            rows.append(
                f"{label} & {vlabel} & {f2(r.bal_acc)} ({f2(blo)}--{f2(bhi)}) "
                f"& {f2(r.acc)} & {f2(r.prec_pos)} & {f2(r.rec_pos)} "
                f"& {f2(r.f1_pos)} ({f2(flo)}--{f2(fhi)}) \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table*}}[t]
\\centering
\\footnotesize
\\begin{{tabular}}{{llccccc}}
\\toprule
Method & Input & Bal.\\ acc.\\ (95\\% CI) & Acc. & Prec. & Rec. & F1 (95\\% CI) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Results for every method and input variant, overall, on the 2{{,}}129-document labelled set: precision, recall, and F1 for the positive (support) class, with balanced accuracy as a prevalence-robust summary and raw accuracy for reference. Bootstrap 95\\% confidence intervals (2{{,}}000 resamples) are shown for balanced accuracy and F1. \\emph{{en}} is the English-translated input and \\emph{{native}} the source-language input. \\emph{{Domain blocklist}} is a non-learned source-based baseline with no input variant (\\S\\ref{{sec:experiments}}).}}
\\label{{tab:full_results}}
\\end{{table*}}
"""


def per_lang(metric, label, caption, tag):
    p = pd.read_csv(EVAL / "all_results_per_language.csv")
    p = p[(p.variant == "english") & (p.method != "majority")]
    dom = pd.read_csv(EVAL / "metrics_distant_domain.csv")
    dom = dom[dom.scope == "all_items"].set_index("lang")

    cols = [short for _, _, short, _ in ORDER]
    lines = []
    for lg in LANGS:
        vals, cells = [], []
        d = dom.loc[lg.lower()][metric] if lg.lower() in dom.index else float("nan")
        for disp, _, _short, _ in ORDER:
            r = p[(p.method == disp) & (p.lang == lg.lower())]
            v = float(r.iloc[0][metric]) if not r.empty else float("nan")
            vals.append(v)
            cells.append(v)
        best = max(vals)
        out = [f"\\textbf{{{f2(v)}}}" if v == best else f2(v) for v in cells]
        lines.append(f"{lg} & {f2(d)} & " + " & ".join(out) + f" & {f2(best)} \\\\")

    s = pd.read_csv(EVAL / "all_results_summary.csv")
    s = s[(s.variant == "english") & (s.stem != "majority")]
    allvals = [float(s[s.method == disp].iloc[0][metric]) for disp, _, _, _ in ORDER]
    bl = blocklist_overall()
    blv = f2(bl[metric]) if bl is not None else "---"
    bestall = max(allvals)
    allcells = [f"\\textbf{{{f2(v)}}}" if v == bestall else f2(v) for v in allvals]
    oracle = sum(max(float(p[(p.method == disp) & (p.lang == lg.lower())].iloc[0][metric])
                     for disp, _, _, _ in ORDER) for lg in LANGS) / len(LANGS)
    lines.append("\\midrule")
    lines.append(f"\\textbf{{All}} & {blv} & " + " & ".join(allcells) + f" & {f2(oracle)} \\\\")

    header = " & ".join(cols)
    body = "\n".join(lines)
    return f"""\\begin{{table*}}[t]
\\centering
\\small
\\begin{{tabular}}{{l c @{{\\hspace{{1.6em}}}} ccc @{{\\hspace{{1.6em}}}} cccc @{{\\hspace{{1.6em}}}} c}}
\\toprule
 & Rule-based & \\multicolumn{{3}}{{c}}{{Supervised}} & \\multicolumn{{4}}{{c}}{{Zero-shot}} & \\\\
\\cmidrule(lr){{2-2}}\\cmidrule(lr){{3-5}}\\cmidrule(lr){{6-9}}
Lang & Domain & {header} & Best \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:{tag}}}
\\end{{table*}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=EVAL,
                    help="directory holding the merged metrics CSVs")
    ap.add_argument("--out", type=Path, default=TABLES, help="where to write the .tex files")
    args = ap.parse_args()
    # the table builders read these at call time
    globals()["EVAL"] = args.eval_dir
    globals()["TABLES"] = args.out
    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "full_results.tex").write_text(full_results())
    (args.out / "per_lang_posf1.tex").write_text(per_lang(
        "f1_pos", "F1",
        "Positive-class F1 per language and overall, English input, on the 2{,}129-document labelled set, best per row in bold. "
        "\\emph{Domain} is the non-learned source-based blocklist, XLM-R, e5-LR, and SetFit are the supervised baselines, and the "
        "remaining columns are the zero-shot LLMs. \\emph{Best} is the highest F1 any method reaches for that language. Its "
        "\\emph{All} entry is the \\emph{oracle}, the mean of these per-language ceilings.",
        "per_lang_posf1"))
    (args.out / "per_lang_balacc.tex").write_text(per_lang(
        "bal_acc", "Balanced accuracy",
        "Balanced accuracy per language and overall, English input, best per row in bold. Balanced accuracy is prevalence-robust "
        "and so comparable across languages with very different positive rates (Table~\\ref{tab:perlang}), unlike positive-class "
        "F1 in Table~\\ref{tab:per_lang_posf1}.",
        "per_lang_balacc"))
    for f in ("full_results.tex", "per_lang_posf1.tex", "per_lang_balacc.tex"):
        print("Written:", args.out / f)


if __name__ == "__main__":
    main()
