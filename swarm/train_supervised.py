#!/usr/bin/env python3
"""Supervised baseline with stratified k-fold cross-validation.

Designed for the label-scarce regime (281 positives / 1,702 items): instead of
fine-tuning a large encoder, we use methods that are stable with few labels.

  --method setfit    SetFit (contrastive few-shot fine-tuning of a multilingual
                     sentence transformer) + logistic head.   [default]
  --method embed_lr  Frozen multilingual sentence embeddings + class-weighted
                     logistic regression.

Both run 5-fold CV stratified by language x label and produce out-of-fold
predictions for ALL 1,702 items (each predicted by a model that did not train on
it), so results are on the SAME common eval set as the zero-shot LLMs.

Inputs to the model: question + article text. Sentence encoders truncate to
~512 tokens, so only the article's opening is seen (a noted limitation).

Note: the embedding model is downloaded if not cached; unset HF_HUB_OFFLINE for
this step or pre-download `intfloat/multilingual-e5-large`.

Outputs: predictions_<method>.parquet, metrics_<method>.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import metrics_table  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold_eval_set.parquet"
DEFAULT_ST = "intfloat/multilingual-e5-large"


def load(path=GOLD):
    df = pd.read_parquet(path).reset_index(drop=True)
    df["y"] = df["final"].astype(int)
    # e5 convention: prefix inputs with "query: "; question then article body
    df["text"] = ["query: " + f"{q}\n\n{t}"
                  for q, t in zip(df["question"].astype(str), df["art_trunc"].astype(str))]
    df["strat"] = df["lang"].astype(str) + "_" + df["y"].astype(str)
    return df


def run_embed_lr(df, st_model, folds, seed, max_seq):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(st_model)
    enc.max_seq_length = max_seq
    print(f"Encoding {len(df)} docs with {st_model} ...", file=sys.stderr)
    X = enc.encode(df["text"].tolist(), batch_size=32, show_progress_bar=True,
                   normalize_embeddings=True)
    y = df["y"].values
    oof = np.full(len(df), -1, int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(X, df["strat"])):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
        print(f"  fold {k + 1}/{folds} done", file=sys.stderr)
    return oof


def run_xlmr(df, model_name, folds, seed, max_seq, epochs, lr, batch):
    """Fine-tune XLM-R for binary sequence classification, 5-fold CV.

    Class-weighted cross-entropy compensates for the ~16% positive rate. A fresh
    model is trained per fold; out-of-fold predictions cover all items.
    """
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    # Raw question + article (no e5 "query:" prefix); XLM-R reads the pair plainly.
    texts = [f"{q}\n\n{t}" for q, t in
             zip(df["question"].astype(str), df["art_trunc"].astype(str))]
    y = df["y"].values
    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch_texts):
        return tok(batch_texts, truncation=True, max_length=max_seq, padding=False)

    class WeightedTrainer(Trainer):
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw)
            self._cw = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=self._cw.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    oof = np.full(len(df), -1, int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(df.index, df["strat"])):
        print(f"\n===== XLM-R fold {k + 1}/{folds} "
              f"(train {len(tr)}, test {len(te)}) =====", file=sys.stderr)
        n_pos = max(int(y[tr].sum()), 1)
        n_neg = max(len(tr) - n_pos, 1)
        cw = torch.tensor([len(tr) / (2 * n_neg), len(tr) / (2 * n_pos)],
                          dtype=torch.float)
        ds_tr = Dataset.from_dict({"text": [texts[i] for i in tr],
                                   "labels": y[tr].tolist()}).map(
            lambda b: tokenize(b["text"]), batched=True, remove_columns=["text"])
        ds_te = Dataset.from_dict({"text": [texts[i] for i in te]}).map(
            lambda b: tokenize(b["text"]), batched=True, remove_columns=["text"])
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)
        targs = TrainingArguments(
            output_dir=f"/tmp/xlmr_fold{k}", num_train_epochs=epochs,
            per_device_train_batch_size=batch, per_device_eval_batch_size=64,
            learning_rate=lr, weight_decay=0.01, warmup_ratio=0.1,
            logging_steps=50, save_strategy="no", report_to=[],
            seed=seed, fp16=torch.cuda.is_available())
        trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr,
                                  data_collator=DataCollatorWithPadding(tok),
                                  class_weights=cw)
        trainer.train()
        logits = trainer.predict(ds_te).predictions
        oof[te] = logits.argmax(-1)
    return oof


def run_setfit(df, st_model, folds, seed):
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset
    y = df["y"].values
    oof = np.full(len(df), -1, int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(df.index, df["strat"])):
        print(f"\n===== SetFit fold {k + 1}/{folds} "
              f"(train {len(tr)}, test {len(te)}) =====", file=sys.stderr)
        model = SetFitModel.from_pretrained(st_model)
        ds = Dataset.from_dict({"text": df["text"].iloc[tr].tolist(),
                                "label": y[tr].tolist()})
        args = TrainingArguments(batch_size=16, num_epochs=1, seed=seed,
                                 num_iterations=20)
        Trainer(model=model, args=args, train_dataset=ds).train()
        oof[te] = np.array(model.predict(df["text"].iloc[te].tolist())).astype(int)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["setfit", "embed_lr", "xlmr"],
                    default="setfit")
    ap.add_argument("--st-model", default=DEFAULT_ST,
                    help="sentence-transformer body / embedding model")
    ap.add_argument("--xlmr-model", default="xlm-roberta-base",
                    help="HF model id for --method xlmr")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=4.0, help="xlmr epochs")
    ap.add_argument("--lr", type=float, default=2e-5, help="xlmr learning rate")
    ap.add_argument("--batch", type=int, default=16, help="xlmr train batch size")
    ap.add_argument("--eval", default=str(GOLD),
                    help="eval-set parquet (English or _native)")
    args = ap.parse_args()

    df = load(args.eval)
    suffix = "_native" if "native" in Path(args.eval).stem else ""
    if args.method == "embed_lr":
        df["pred"] = run_embed_lr(df, args.st_model, args.folds, args.seed, args.max_seq)
    elif args.method == "xlmr":
        df["pred"] = run_xlmr(df, args.xlmr_model, args.folds, args.seed,
                              args.max_seq, args.epochs, args.lr, args.batch)
    else:
        df["pred"] = run_setfit(df, args.st_model, args.folds, args.seed)

    tag = args.method + suffix
    out = df[["original_id", "lang", "question", "final", "pred"]].copy()
    out.to_parquet(HERE / f"predictions_{tag}.parquet", index=False)
    m = metrics_table(df)
    m.to_csv(HERE / f"metrics_{tag}.csv", index=False)
    print(m.to_string(index=False))
    print(f"\nWrote predictions_{tag}.parquet and metrics_{tag}.csv")


if __name__ == "__main__":
    main()
