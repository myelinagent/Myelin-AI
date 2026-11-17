"""Turn generated conversations into training pairs.

    python -m bench.embedder.pairs bench/embedder/data/conversations.jsonl.gz --out pairs/

Each input is a directory of conversation files or a ``.jsonl.gz`` with
one conversation per line; several can be given.

Writes ``train.jsonl`` and ``val.jsonl``. Each line is one question
with one turn that answers it and some turns from the same conversation
that do not::

    {"query": "...", "positive": "Mei: ...", "negatives": ["Omar: ...", ...],
     "conversation": "c0001", "kind": "fact"}

A turn is written as ``Speaker: text``, which is how the benchmark
stores one; in a conversation of kind ``notes`` it is the note alone. The wrong turns are the ones the current static embedder
ranks highest for the question, since those are the mistakes a better
embedder has to stop making. Turns next to an evidence turn are left
out of them: a neighbour often half-answers the question, and teaching
the model that it does not would be teaching it something false.

Whole conversations go to one side of the split or the other, so the
validation questions are about people the model has not seen.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from bench.embedder.validate import problems


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def pairs_from(conversation: dict[str, Any], embedder: Any, negatives: int) -> list[dict[str, Any]]:
    ids: list[str] = []
    texts: list[str] = []
    for session in conversation["sessions"]:
        for turn in session["turns"]:
            ids.append(turn["id"])
            # An agent's saved note is stored as written; a message in a
            # conversation is stored with who said it.
            notes = conversation.get("kind") == "notes"
            texts.append(turn["text"] if notes else f"{turn['speaker']}: {turn['text']}")
    position = {turn_id: index for index, turn_id in enumerate(ids)}
    vectors = embedder.encode_batch(texts)
    asked = [q["question"] for q in conversation["questions"]]
    out = []
    for question, vector in zip(conversation["questions"], embedder.encode_batch(asked), strict=True):
        evidence = [position[e] for e in dict.fromkeys(question["evidence"])]
        near = {i + step for i in evidence for step in (-1, 0, 1)}
        ranked = sorted(
            (i for i in range(len(texts)) if i not in near),
            key=lambda i: -_dot(vector, vectors[i]),
        )
        wrong = [texts[i] for i in ranked[:negatives]]
        for index in evidence:
            out.append({
                "query": question["question"],
                "positive": texts[index],
                "negatives": wrong,
                "conversation": conversation["id"],
                "kind": question["kind"],
            })
    return out


def read_conversations(source: str) -> tuple[list[dict[str, Any]], int]:
    """The valid conversations in a directory or a ``.jsonl.gz``, and how
    many were left out for failing ``bench.embedder.validate``."""
    path = Path(source)
    if path.is_dir():
        raw = [file.read_text() for file in sorted(path.glob("c*.json"))]
    else:
        with gzip.open(path, "rt") as handle:
            raw = [line for line in handle if line.strip()]
    kept, skipped = [], 0
    for text in raw:
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if problems(loaded):
            skipped += 1
            continue
        kept.append(loaded)
    return kept, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "sources", nargs="+", help="directories of c*.json files, or .jsonl.gz files"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from myelin.embedders import StaticEmbedder

    embedder = StaticEmbedder()
    conversations, skipped = [], 0
    for source in args.sources:
        kept, left_out = read_conversations(source)
        conversations += kept
        skipped += left_out
    random.Random(args.seed).shuffle(conversations)
    held = max(1, round(len(conversations) * args.val_share))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", conversations[:held]), ("train", conversations[held:])):
        rows = [row for c in part for row in pairs_from(c, embedder, args.negatives)]
        with (out / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(part)} conversations, {len(rows)} pairs")
    if skipped:
        print(f"skipped {skipped} conversations that do not pass bench.embedder.validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
