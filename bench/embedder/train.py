"""Fine-tune a small sentence embedder on the generated pairs.

    pip install sentence-transformers
    python -m bench.embedder.train --pairs pairs/ --out tuned/

Uses the Mac GPU (MPS) or CUDA when there is one, the CPU otherwise.
Each question is pulled towards the turn that answers it and pushed
away from the wrong turns listed with it and from every other turn in
the batch. Prints how often the answering turn is the top result among
the validation turns before and after, and saves the model to ``--out``
in sentence-transformers format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASE_MODEL = "BAAI/bge-small-en-v1.5"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recall(model: Any, rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Share of questions whose answering turn is in the top k, searching
    every turn of the same conversation that appears in ``rows``."""
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_conversation.setdefault(row["conversation"], []).append(row)
    hits = dict.fromkeys(ks, 0)
    for group in by_conversation.values():
        turns = sorted({t for row in group for t in [row["positive"], *row["negatives"]]})
        index = {turn: i for i, turn in enumerate(turns)}
        turn_vectors = model.encode(turns, normalize_embeddings=True, convert_to_tensor=True)
        queries = model.encode(
            [row["query"] for row in group], normalize_embeddings=True, convert_to_tensor=True
        )
        order = (queries @ turn_vectors.T).argsort(dim=1, descending=True).tolist()
        for row, ranked in zip(group, order, strict=True):
            for k in ks:
                hits[k] += index[row["positive"]] in ranked[:k]
    return {k: 100 * hits[k] / max(len(rows), 1) for k in ks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", required=True, help="directory with train.jsonl and val.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--rate", type=float, default=2e-5)
    parser.add_argument("--negatives", type=int, default=3, help="wrong turns used per question")
    parser.add_argument("--limit", type=int, default=0, help="use only this many training pairs")
    args = parser.parse_args(argv)

    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    train = load(Path(args.pairs) / "train.jsonl")
    val = load(Path(args.pairs) / "val.jsonl")
    if args.limit:
        train = train[: args.limit]
    model = SentenceTransformer(args.base)
    print(f"device: {model.device}; {len(train)} training pairs, {len(val)} validation pairs")
    before = recall(model, val)

    columns: dict[str, list[str]] = {"anchor": [], "positive": []}
    for n in range(args.negatives):
        columns[f"negative_{n + 1}"] = []
    for row in train:
        if len(row["negatives"]) < args.negatives:
            continue
        columns["anchor"].append(row["query"])
        columns["positive"].append(row["positive"])
        for n in range(args.negatives):
            columns[f"negative_{n + 1}"].append(row["negatives"][n])

    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(Path(args.out) / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=args.rate,
            warmup_steps=max(1, int(0.1 * args.epochs * len(columns["anchor"]) / args.batch)),
            # The same question or turn twice in a batch would be counted
            # as a wrong answer to itself.
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no",
            logging_steps=50,
            report_to="none",
        ),
        train_dataset=Dataset.from_dict(columns),
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    after = recall(model, val)
    model.save(args.out)
    for k in before:
        print(f"validation, answering turn in top {k}: {before[k]:.1f}% -> {after[k]:.1f}%")
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
