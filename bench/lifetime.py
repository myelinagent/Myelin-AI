"""Memory over time: what forgetting and cite feedback do as a store ages.

    python -m bench.lifetime --data locomo10.json --embedder minishlab/potion-retrieval-32M

A LoCoMo conversation runs over months. This replays each one in order
on a simulated clock: a session is stored on its date, and each question
is asked some sessions after the last message that answers it (a random
lag, so memories have had time to age). No model is involved.

Four ways of running the same timeline are compared:

* ``static`` — store and search; nothing else.
* ``decay`` — ``prune()`` after every session, so memories whose health
  has fallen (age, mostly) are quarantined and leave search.
* ``feedback`` — after each question, ``cite_used`` reports which of the
  shown memories were its evidence.
* ``decay + feedback`` — both: use keeps a memory healthy, and unused
  memories fade.

Reported per arm: how often all of a question's evidence is in the top
results, how often decay had already hidden a memory the question
needed, and how much of the store is still searchable at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from bench import locomo
from myelin import MemoryClient

ARMS = {
    "static": (False, False),
    "decay": (True, False),
    "feedback": (False, True),
    "decay + feedback": (True, True),
}
KS = (5, 10, 20)
SHOWN = 20


def replay(
    sample: dict[str, Any],
    seed: int,
    *,
    decay: bool,
    feedback: bool,
    max_lag: int = 6,
    embedder: Any = None,
) -> dict[str, Any]:
    """Run one conversation's timeline; return per-question outcomes and
    the state of the store at the end."""
    turns = locomo.conversation_turns(sample)
    sessions: dict[int, list[Any]] = defaultdict(list)
    for turn in turns:
        sessions[int(turn.session_id.split()[-1])].append(turn)
    order = sorted(sessions)
    session_of = {t.dia_id: int(t.session_id.split()[-1]) for t in turns}

    rng = random.Random(seed)
    due: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for qa in locomo.questions(sample):
        evidence = [e for e in qa.get("evidence", []) if e in session_of]
        if not evidence:
            continue
        last = order.index(max(session_of[e] for e in evidence))
        ask_at = min(len(order) - 1, last + rng.randint(0, max_lag))
        due[ask_at].append({**qa, "evidence": evidence, "lag": ask_at - last})

    now = [sessions[order[0]][0].session_date]
    client = MemoryClient(
        user_id="bench", embedder=embedder, clock=lambda: now[0], cite_learning=True
    )
    id_of: dict[str, str] = {}
    outcomes: list[dict[str, Any]] = []
    try:
        for step, number in enumerate(order):
            batch = sessions[number]
            now[0] = batch[0].session_date
            stored = client.add_many(
                [{"content": t.text, "t_valid": t.session_date} for t in batch]
            )
            id_of.update({t.dia_id: m.id for t, m in zip(batch, stored, strict=True)})
            # Questions come a little after the session they follow.
            now[0] = batch[0].session_date + timedelta(hours=1)
            if decay:
                client.prune(now=now[0])
            for qa in due.get(step, []):
                wanted = {id_of[e] for e in qa["evidence"]}
                shown = client.search(qa["question"], top_k=SHOWN)
                ranked = [h.memory.id for h in shown]
                hidden = any(
                    (client.get(i) is None or client.get(i).health_state != "healthy")
                    for i in wanted
                )
                outcomes.append(
                    {
                        "lag": qa["lag"],
                        "hidden": hidden,
                        **{f"top{k}": wanted <= set(ranked[:k]) for k in KS},
                    }
                )
                if feedback:
                    used = [n for n, i in enumerate(ranked, start=1) if i in wanted]
                    client.cite_used(shown, used, query=qa["question"])
        searchable = sum(
            1 for i in id_of.values()
            if (m := client.get(i)) is not None and m.health_state == "healthy"
        )
    finally:
        client.close()
    return {"outcomes": outcomes, "stored": len(id_of), "searchable": searchable}


def summarize(runs: list[dict[str, Any]]) -> dict[str, float]:
    outcomes = [o for run in runs for o in run["outcomes"]]
    late = [o for o in outcomes if o["lag"] >= 3]

    def share(rows: list[dict[str, Any]], key: str) -> float:
        return 100 * sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0

    return {
        **{f"top{k}": share(outcomes, f"top{k}") for k in KS},
        "top10_late": share(late, "top10"),
        "hidden": share(outcomes, "hidden"),
        "searchable": 100 * sum(r["searchable"] for r in runs) / sum(r["stored"] for r in runs),
        "questions": float(len(outcomes)),
    }


COLUMNS = (
    ("top5", "All evidence in top 5"),
    ("top10", "top 10"),
    ("top20", "top 20"),
    ("top10_late", "top 10, asked 3+ sessions later"),
    ("hidden", "Needed a memory decay had hidden"),
    ("searchable", "Store still searchable at the end"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.lifetime", description=__doc__)
    parser.add_argument("--data", required=True, help="path to locomo10.json")
    parser.add_argument("--embedder", default="", help="static model id; omit for lexical only")
    parser.add_argument("--seeds", type=int, default=3, help="random question schedules")
    parser.add_argument("--max-lag", type=int, default=6, help="sessions a question may wait")
    parser.add_argument("--limit", type=int, default=0, help="only the first N conversations")
    args = parser.parse_args(argv)
    samples = json.loads(Path(args.data).read_text())
    if args.limit:
        samples = samples[: args.limit]
    embedder = None
    if args.embedder:
        from myelin.embedders import StaticEmbedder

        embedder = StaticEmbedder(args.embedder)

    print("| | " + " | ".join(title for _, title in COLUMNS) + " |")
    print("|---|" + "---:|" * len(COLUMNS))
    for arm, (decay, feedback) in ARMS.items():
        per_seed = [
            summarize([
                replay(s, seed, decay=decay, feedback=feedback,
                       max_lag=args.max_lag, embedder=embedder)
                for s in samples
            ])
            for seed in range(args.seeds)
        ]
        cells = []
        for key, _ in COLUMNS:
            values = [run[key] for run in per_seed]
            spread = f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else ""
            cells.append(f"{statistics.mean(values):.1f}{spread}")
        print(f"| {arm} | " + " | ".join(cells) + " |", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
