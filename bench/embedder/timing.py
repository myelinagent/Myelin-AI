"""Time an embedder: texts per second in batches, and one query at a time.

    python -m bench.embedder.timing bench/embedder/data/conversations2.jsonl.gz \
        --neural myelinagent/myelin-embed-small-en-v1

Short texts are single turns from the given conversations; long texts
are eight consecutive turns joined, about the length of an assistant
reply. Run it on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import statistics
import time

from bench.embedder.pairs import read_conversations
from myelin.embedders import DEFAULT_STATIC_MODEL, NeuralEmbedder, StaticEmbedder


def texts_from(source: str, count: int) -> tuple[list[str], list[str], list[str]]:
    turns: list[str] = []
    questions: list[str] = []
    for conversation in read_conversations(source)[0]:
        for session in conversation["sessions"]:
            turns.extend(turn["text"] for turn in session["turns"])
        questions.extend(q["question"] for q in conversation["questions"])
        if len(turns) >= count * 8:
            break
    short = turns[:count]
    long = [" ".join(turns[i : i + 8]) for i in range(0, count * 8, 8)][: count // 4]
    return short, long, questions[:200]


def measure(embedder, short: list[str], long: list[str], questions: list[str]) -> dict:
    embedder.encode_batch(short[:16])  # load and warm up
    out = {}
    for name, texts in (("short", short), ("long", long)):
        start = time.perf_counter()
        for i in range(0, len(texts), 32):
            embedder.encode_batch(texts[i : i + 32])
        out[name] = len(texts) / (time.perf_counter() - start)
    laps = []
    for question in questions:
        start = time.perf_counter()
        embedder.encode_batch([question])
        laps.append((time.perf_counter() - start) * 1000)
    out["query_ms"] = statistics.median(laps)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source")
    parser.add_argument("--neural", required=True)
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    short, long, questions = texts_from(args.source, args.count)
    for name, texts in (("short", short), ("long", long)):
        mean = sum(map(len, texts)) / len(texts)
        print(f"{name}: {len(texts)} texts, {mean:.0f} characters on average")
    print("| Embedder | short texts/s | long texts/s | one query, ms |")
    print("|---|---:|---:|---:|")
    for label, embedder in (
        (DEFAULT_STATIC_MODEL, StaticEmbedder()),
        (args.neural, NeuralEmbedder(args.neural)),
    ):
        r = measure(embedder, short, long, questions)
        print(
            f"| `{label}` | {r['short']:,.0f} | {r['long']:,.0f} | {r['query_ms']:.2f} |"
        )


if __name__ == "__main__":
    main()
