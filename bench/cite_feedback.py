"""Does cite feedback improve ranking on questions it has not seen? No model.

    python -m bench.cite_feedback --data locomo10.json --embedder minishlab/potion-retrieval-32M

Uses LoCoMo, where one store answers about 150 questions. For each
conversation, half of the questions (chosen at random) give feedback:
each is searched, and every memory in its top results is cited as
useful or not. The other half are then scored by whether all of their
evidence is in the top k, with cite-informed ranking off and on.

Two kinds of feedback are simulated:

* ``per-question`` — what an application can do without labels: every
  memory shown for a question gets the same verdict, correct when the
  question's evidence was all in view (a stand-in for "the answer was
  right").
* ``per-memory`` — an upper bound: a memory is cited as useful only if
  it is evidence for that question.

Evidence utterances are shared between questions about the same events,
which is how feedback on one question can help another.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench import locomo
from myelin import MemoryClient

ARMS = ("off", "per-question", "per-memory")
KS = (5, 10, 20, 50)
FEEDBACK_DEPTH = 20


def run_split(
    samples: list[dict[str, Any]], seed: int, embedder: Any = None
) -> dict[str, dict[int, float]]:
    """Share of held-out questions with all evidence in the top k, per arm."""
    hits: dict[str, dict[int, list[bool]]] = {arm: defaultdict(list) for arm in ARMS}
    for sample in samples:
        turns = locomo.conversation_turns(sample)
        known = {t.dia_id for t in turns}
        asked = [q for q in locomo.questions(sample) if set(q.get("evidence", [])) & known]
        random.Random(seed).shuffle(asked)
        feedback, held_out = asked[: len(asked) // 2], asked[len(asked) // 2 :]
        for arm in ARMS:
            client = MemoryClient(user_id="bench", embedder=embedder, cite_ranking=arm != "off")
            try:
                stored = client.add_many(
                    [{"content": t.text, "t_valid": t.session_date} for t in turns]
                )
                dia = {m.id: t.dia_id for m, t in zip(stored, turns, strict=True)}
                if arm != "off":
                    for qa in feedback:
                        shown = client.search(qa["question"], top_k=FEEDBACK_DEPTH)
                        evidence = set(qa["evidence"]) & known
                        answered = evidence <= {dia[h.memory.id] for h in shown}
                        for hit in shown:
                            useful = (
                                answered if arm == "per-question"
                                else dia[hit.memory.id] in evidence
                            )
                            client.cite(hit.memory.id, answer_correct=useful, query=qa["question"])
                for qa in held_out:
                    evidence = set(qa["evidence"]) & known
                    ranked = [dia[h.memory.id] for h in client.search(qa["question"], top_k=max(KS))]
                    for k in KS:
                        hits[arm][k].append(evidence <= set(ranked[:k]))
            finally:
                client.close()
    return {arm: {k: sum(v) / len(v) for k, v in by_k.items()} for arm, by_k in hits.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.cite_feedback", description=__doc__)
    parser.add_argument("--data", required=True, help="path to locomo10.json")
    parser.add_argument("--embedder", default="", help="static model id; omit for lexical only")
    parser.add_argument("--seeds", type=int, default=5, help="random splits to average over")
    parser.add_argument("--limit", type=int, default=0, help="only the first N conversations")
    args = parser.parse_args(argv)
    # Cite learning is off unless switched on; this benchmark is about it.
    os.environ["MYELIN_CITE_LEARNING"] = "1"
    samples = json.loads(Path(args.data).read_text())
    if args.limit:
        samples = samples[: args.limit]
    embedder = None
    if args.embedder:
        from myelin.embedders import StaticEmbedder

        embedder = StaticEmbedder(args.embedder)
    runs = [run_split(samples, seed, embedder) for seed in range(args.seeds)]
    print("| Feedback | " + " | ".join(f"All evidence in top {k}" for k in KS) + " |")
    print("|---|" + "---:|" * len(KS))
    for arm in ARMS:
        cells = []
        for k in KS:
            values = [100 * run[arm][k] for run in runs]
            spread = f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else ""
            cells.append(f"{statistics.mean(values):.1f}{spread}")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
