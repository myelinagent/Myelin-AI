"""Turn generated conversations into training pairs.

    python -m bench.embedder.pairs CONVERSATIONS_DIR --out pairs/

Writes ``train.jsonl`` and ``val.jsonl``. Each line is one question
with one turn that answers it and some turns from the same conversation
that do not::

    {"query": "...", "positive": "Mei: ...", "negatives": ["Omar: ...", ...],
     "conversation": "c0001", "kind": "fact"}

A turn is written as ``Speaker: text``, which is how the benchmark
stores one. The wrong turns are the ones the current static embedder
ranks highest for the question, since those are the mistakes a better
embedder has to stop making. Turns next to an evidence turn are left
out of them: a neighbour often half-answers the question, and teaching
the model that it does not would be teaching it something false.

Whole conversations go to one side of the split or the other, so the
validation questions are about people the model has not seen.
"""

from __future__ import annotations

import argparse
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
            texts.append(f"{turn['speaker']}: {turn['text']}")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--val-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from myelin.embedders import StaticEmbedder

    embedder = StaticEmbedder()
    conversations, skipped = [], 0
    for file in sorted(Path(args.directory).glob("c*.json")):
        try:
            loaded = json.loads(file.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        if problems(loaded):
            skipped += 1
            continue
        conversations.append(loaded)
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
        print(f"skipped {skipped} files that do not pass bench.embedder.validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
