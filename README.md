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

**Source-based blocklist** (Table 8, first row; Table 9)

    python -m swarm.build_domain_blocklist --extra euvsdisinfo=<csv> proppy=<csv> rashkin=<csv>
    python -m swarm.merge_results

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
| Table 8 (overall results, all methods) | `merge_results.py` then `build_result_tables.py` |
| Table 9 (blocklist recall by language) | `merge_results.py` |
| Table 10 (per-language balanced accuracy) | `build_result_tables.py` |
| Appendix H (retrieval selection bias) | `retrieval_bias.py` |

## The zero-shot prompt

Used verbatim for every item and every model, in `prompt.txt`. Two properties
are worth knowing before reusing it. It ends with a worked example, so this is
instruction-with-exemplar rather than a bare zero-shot prompt. It also asks the
model to repeat the assessment nine times and take the majority, which cannot
take effect under greedy decoding with a single-token output budget and is
therefore inert. Both are documented in the paper. The prompt was fixed in
advance and never tuned on the reported items.

## Not runnable from the public release

`build_eval_9lang.py` and `build_native_eval.py` construct the gold set from the
raw Label Studio annotation exports, which are not released (they contain
per-annotator labels). They are included so the label-aggregation logic is
inspectable: majority vote over usable votes, ties dropped, and an item retained
only if at least one text-based coder could assess it. Use
`prepare_from_hf.py` instead.

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

## Citation

Citation details will be added on publication.
