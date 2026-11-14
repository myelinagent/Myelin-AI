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

* ``model-named`` (with ``--model``) — what an application can do with
  an answering model: the model is shown the numbered results, answers
  the question and names the ones it used; those are cited as useful
  and the rest as not.

Evidence utterances are shared between questions about the same events,
which is how feedback on one question can help another.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench import locomo
from bench.llm import ChatClient, LLMError
from myelin import MemoryClient

ARMS = ("off", "per-question", "per-memory")
MODEL_ARM = "model-named"

USED_PROMPT = """\
Below are numbered messages from past conversations, then a question.

{messages}

Question: {question}

Answer the question in one sentence. Then, on a new line, write "Used:" followed by \
the numbers of the messages your answer relies on, separated by commas. If none of \
them helps, write "Used: none"."""


def named_by_model(client: ChatClient, question: str, texts: list[str]) -> set[int] | None:
    """Positions (from 0) of the messages the model says it used, or None
    when the call fails or the reply has no ``Used:`` line."""
    messages = "\n".join(f"[{n}] {text}" for n, text in enumerate(texts, start=1))
    try:
        reply = client.complete(
            USED_PROMPT.format(messages=messages, question=question), max_tokens=200
        ).text
    except LLMError:
        return None
    found = re.findall(r"(?im)^\s*used:\s*(.*)$", reply)
    if not found:
        return None
    return {int(n) - 1 for n in re.findall(r"\d+", found[-1]) if 0 < int(n) <= len(texts)}
KS = (5, 10, 20, 50)
FEEDBACK_DEPTH = 20


def run_split(
    samples: list[dict[str, Any]],
    seed: int,
    embedder: Any = None,
    model: ChatClient | None = None,
) -> dict[str, dict[int, float]]:
    """Share of held-out questions with all evidence in the top k, per arm."""
    arms = (*ARMS, MODEL_ARM) if model is not None else ARMS
    hits: dict[str, dict[int, list[bool]]] = {arm: defaultdict(list) for arm in arms}
    for sample in samples:
        turns = locomo.conversation_turns(sample)
        known = {t.dia_id for t in turns}
        asked = [q for q in locomo.questions(sample) if set(q.get("evidence", [])) & known]
        random.Random(seed).shuffle(asked)
        feedback, held_out = asked[: len(asked) // 2], asked[len(asked) // 2 :]
        for arm in arms:
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
                        named: set[int] | None = None
                        if arm == MODEL_ARM:
                            assert model is not None
                            named = named_by_model(
                                model, qa["question"], [h.memory.content for h in shown]
                            )
                            if named is None:
                                continue
                        for position, hit in enumerate(shown):
                            if named is not None:
                                useful = position in named
                            elif arm == "per-question":
                                useful = answered
                            else:
                                useful = dia[hit.memory.id] in evidence
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
    parser.add_argument(
        "--model", default="",
        help="model id; adds an arm where the model names the messages it used",
    )
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
    model = ChatClient(args.model) if args.model else None
    runs = [run_split(samples, seed, embedder, model) for seed in range(args.seeds)]
    print("| Feedback | " + " | ".join(f"All evidence in top {k}" for k in KS) + " |")
    print("|---|" + "---:|" * len(KS))
    for arm in (*ARMS, MODEL_ARM) if model is not None else ARMS:
        cells = []
        for k in KS:
            values = [100 * run[arm][k] for run in runs]
            spread = f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else ""
            cells.append(f"{statistics.mean(values):.1f}{spread}")
        print(f"| {arm} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
