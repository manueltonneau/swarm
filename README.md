# SWARM replication

Code to reproduce the benchmark in *SWARM: A Multilingual Human-Annotated
Dataset for Russian Propaganda Detection in Search Engine Results*.

SWARM is 2,129 search engine results across nine languages, each labelled by
trained coders for whether the linked document supports one of twenty recurring
pro-Kremlin narratives. This repo contains the evaluation pipeline: the
source-based blocklist baseline, three supervised classifiers, four zero-shot
LLMs, and the analysis that produces every table in the paper.

## Data

The dataset is released separately under gated, research-use-only access:

    https://huggingface.co/datasets/manueltonneau/SWARM

No data ships with this repo. Request access, then:

    pip install -r requirements.txt
    huggingface-cli download manueltonneau/SWARM --repo-type dataset --local-dir data/swarm
    python -m swarm.prepare_from_hf --src data/swarm --out data/eval

That writes `gold_eval_set.parquet` (English input) and
`gold_eval_set_native.parquet` (source-language input), which every runner below
reads. It should report 2,129 rows and 384 positives; if it does not, the
numbers will not match the paper.

## Reproducing the results

Each command writes `metrics_<tag>.csv` and `predictions_<tag>.parquet`.

**Source-based blocklist** (Table 10, first row; Table 5). The assembled blocklist ships in
`data_aux/propaganda_domains_bundle.csv` (2,962 registered domains merged from a scraped Wikipedia
list of Russian disinformation sites, EUvsDisinfo, Proppy/MBFC and Rashkin et al.), so this baseline
runs without rebuilding it:

    python -m swarm.merge_results

To rebuild the blocklist from the upstream sources instead:

    python -m swarm.build_domain_blocklist --extra euvsdisinfo=<csv> proppy=<csv> rashkin=<csv>

**Supervised baselines** (XLM-R, e5-large + logistic regression, SetFit).
5-fold cross-validation stratified by language and label, out-of-fold
predictions, first 512 tokens only. Needs a GPU.

    for m in xlmr embed_lr setfit; do
      python -m swarm.train_supervised --method $m
      python -m swarm.train_supervised --method $m --eval data/eval/gold_eval_set_native.parquet
    done

**Open zero-shot LLMs** (Qwen2.5-7B, Qwen2.5-72B-AWQ). Greedy decoding,
vLLM. The 72B checkpoint needs two GPUs.

    python -m swarm.run_open_llm --model Qwen/Qwen2.5-7B-Instruct --tp 1
    python -m swarm.run_open_llm --model Qwen/Qwen2.5-72B-Instruct-AWQ --tp 2
    # add --eval data/eval/gold_eval_set_native.parquet for the native variant

**Closed zero-shot LLMs** (GPT-5-nano, GPT-5.4) via the OpenAI Batch API at
reasoning effort `high`. Costs money.

    export OPENAI_API_KEY=sk-...
    python -m swarm.run_closed_llm build   --model gpt-5-nano --effort high --out batch.jsonl
    python -m swarm.run_closed_llm submit  --input batch.jsonl --state state.json
    python -m swarm.run_closed_llm collect --tag gpt5nano --state state.json

Pass `--reuse <predictions.parquet>` at `build` and `collect` to score only
items whose `content_hash` changed and reuse the rest.

**Ablations reported in the appendices**

Prompt ablation (Appendix I), which removes the "repeat the assessment nine times" sentence and
changes nothing else:

    SWARM_NO_SELFCONSISTENCY=1 python -m swarm.run_open_llm --model Qwen/Qwen2.5-7B-Instruct --tp 1

Matched-context comparison (Appendix J), which gives the LLMs the same 512-token budget as the
supervised encoders:

    python -m swarm.run_open_llm --model Qwen/Qwen2.5-7B-Instruct --tp 1 --max-doc-tokens 512

Both write to a distinct tag (`_nosc`, `_doc512`) so they do not overwrite the main runs.

**Tables and statistics**

    python -m swarm.merge_results        # aggregates every metrics_*.csv
    python -m swarm.final_stats          # per-language counts, positive rates, Krippendorff alpha
    python -m swarm.retrieval_bias       # retrieval selection-bias analysis (Appendix H)
    python -m swarm.build_result_tables --out tables/   # emits the LaTeX tables

## Which script produces which table

| Paper | Script |
| --- | --- |
| Table 2 (per-language composition, alpha) | `final_stats.py` |
| Table 4 (per-language positive-class F1) | `build_result_tables.py` |
| Table 10 (overall results, all methods) | `merge_results.py` then `build_result_tables.py` |
| Table 5 (blocklist recall by language) | `blocklist_metrics.py` |
| Table 6 (per-language balanced accuracy) | `build_result_tables.py` |
| Table 11 (composition and support by source type) | `blocklist_metrics.py` |
| Table 12, Appendix H (retrieval selection bias) | `retrieval_bias.py` |
| Appendix I (prompt ablation) | `run_open_llm.py` with `SWARM_NO_SELFCONSISTENCY=1` |
| Appendix J (matched context) | `run_open_llm.py --max-doc-tokens 512` |

## The zero-shot prompt

Used verbatim for every item and every model, in `prompt.txt`. Two properties
are worth knowing before reusing it. It ends with a worked example, so this is
instruction-with-exemplar rather than a bare zero-shot prompt. It also asks the
model to repeat the assessment nine times and take the majority, which cannot
happen under greedy decoding with an eight-token output budget. Removing that
sentence changes under 2% of predictions and moves F1 by less than 0.01 in
either direction (Appendix I of the paper, reproducible with the command below). The prompt was fixed in
advance and never tuned on the reported items.

## What reproduces, and what does not

Runs end to end from the released dataset: the gold-set preparation, all three supervised baselines,
all four zero-shot models, both appendix ablations, the source-based blocklist, the aggregation and the
result tables. Verified: `build_result_tables.py` reproduces the paper's Tables 4, 6 and 10 byte for byte.

Three analyses cannot be reproduced from the public release, because they depend on inputs that are not
released:

- `final_stats.py` computes per-language Krippendorff's alpha from the per-annotator labels, which are
  not published. The aggregate figures are in the paper (Table 2).
- `retrieval_bias.py` compares retained against non-retained URLs (Appendix H). The non-retained URLs
  are not part of the release, since they were never annotated.
- `build_eval_9lang.py` and `build_native_eval.py` build the gold set from the raw annotation
  exports, which are not released because they carry per-annotator labels. They ship so the
  aggregation logic is inspectable: majority vote over usable votes, ties dropped, and an item
  kept only if at least one coder could assess its extracted text. Use `prepare_from_hf.py`.
- The source-composition breakdown needs each document's domain category. `data_aux/domain_categories.csv`
  gives the domain-to-category mapping for the 871 domains in the corpus, without item ids, labels or
  URLs, so the mapping can be inspected and reused without circumventing the dataset gating.

## Notes

- Open-model runs are deterministic (greedy, temperature 0). Closed reasoning
  models are not, so re-running flips a small fraction of predictions.
- Documents are truncated to 12,000 characters for the LLMs and to 512 tokens
  for the encoders, as in the paper.
- 10 documents have no usable English text, because the translation pass
  returned a refusal or a request to shorten the input. They are kept so the
  item count matches the paper, and all carry the negative label. These are the
  one place replication is not bit-exact: the original runs showed the models
  the refusal string, whereas the release nulls it and this repo substitutes an
  empty string. Every model assigned all 10 the negative class, which is their
  gold label, so the reported metrics are unaffected.
- Reproducing the paper exactly means running the English-input variant. The
  released `text_native` for English items is the English text, by construction.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`).
The auxiliary domain lists under `data_aux/` are derived from public sources and are
redistributed for replication.

The **dataset is licensed separately** and is not covered by the MIT licence: SWARM is
distributed under gated, research-use-only access on the Hugging Face Hub, with its own
terms and a takedown procedure.

## Citation

Citation details will be added on publication.
