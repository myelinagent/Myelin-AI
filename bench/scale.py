"""Write throughput and recall latency as a store grows. No model, no data.

    python -m bench.scale --sizes 5000 50000 200000
    python -m bench.scale --sizes 5000 50000 --embedder minishlab/potion-retrieval-32M

Memories are synthetic sentences drawn from a skewed vocabulary, so a
query matches a few rare words and many common ones, as real text does.
All memories belong to one user, which is the worst case: search never
gets to skip another user's rows.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from typing import Any

from myelin import MemoryClient

BATCH = 5000


def vocabulary(size: int = 20000) -> list[str]:
    rng = random.Random(7)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(letters) for _ in range(rng.randint(3, 9))) for _ in range(size)]


def sentence(rng: random.Random, words: list[str], length: int) -> str:
    # Rank-skewed draw: low ranks (common words) come up far more often.
    return " ".join(words[int(len(words) * rng.random() ** 3)] for _ in range(length))


def measure(size: int, embedder: Any = None, queries: int = 50) -> dict[str, float]:
    rng = random.Random(size)
    words = vocabulary()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "scale.db")
        client = MemoryClient(db_path=path, user_id="scale", embedder=embedder)
        try:
            started = time.monotonic()
            for start in range(0, size, BATCH):
                count = min(BATCH, size - start)
                client.add_many([sentence(rng, words, rng.randint(8, 40)) for _ in range(count)])
            add_seconds = time.monotonic() - started
            latencies = []
            for _ in range(queries):
                query = sentence(rng, words, 9)
                started = time.monotonic()
                client.recall(query, token_budget=6000)
                latencies.append(time.monotonic() - started)
        finally:
            client.close()
        latencies.sort()
        return {
            "memories": size,
            "adds_per_second": size / add_seconds,
            "recall_p50_ms": 1000 * latencies[len(latencies) // 2],
            "recall_p95_ms": 1000 * latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "db_mb": os.path.getsize(path) / 1e6,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.scale", description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 50000])
    parser.add_argument("--embedder", default="", help="static model id; omit for lexical only")
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args(argv)
    embedder = None
    if args.embedder:
        from myelin.embedders import StaticEmbedder

        embedder = StaticEmbedder(args.embedder)
    print("| Memories | Adds per second | Recall p50 (ms) | Recall p95 (ms) | Database (MB) |")
    print("|---:|---:|---:|---:|---:|")
    for size in args.sizes:
        row = measure(size, embedder, args.queries)
        print(
            f"| {size:,} | {row['adds_per_second']:,.0f} | {row['recall_p50_ms']:.0f} "
            f"| {row['recall_p95_ms']:.0f} | {row['db_mb']:.0f} |",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
