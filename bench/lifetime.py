"""LongHAUL (Long-horizon Assessment of Use and Loss): memory over time.

What forgetting and cite feedback do as a store ages.

    python -m bench.lifetime --data locomo10.json --embedder minishlab/potion-retrieval-32M

A LoCoMo conversation runs over months. This replays each one in order
on a simulated clock: a session is stored on its date, and each question
is asked some sessions after the last message that answers it (a random
lag, so memories have had time to age). No model is involved unless
``--model`` is given.

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

LoCoMo asks each question once. ``--revisits N`` asks each one again N
more times at later sessions, as a user returning to the same facts
would, and reports those repeat askings separately: that is where a
memory's earlier use can pay off.

``--model ID`` adds two arms, ``feedback (model)`` and ``decay +
feedback (model)``, in which the used memories are not the question's
labelled evidence but what an answering model says it used: it is shown
the numbered results, answers, and names the ones it relied on. An
asking whose call fails, or whose reply names nothing readable, gives
no feedback. Endpoint and key are configured as in ``bench.llm``.

``--dataset longmemeval`` replays LongMemEval-S instead. There each
entry is one question with its own dated history, so each entry is a
timeline with a single question, and feedback can only carry from one
asking of that question to the next (``--revisits``). Most of those
histories cover under two weeks, too short for decay to act;
``--min-span-days N`` keeps the timelines that span at least N days.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any

from bench import locomo, longmemeval
from bench.cite_feedback import named_by_model
from bench.llm import ChatClient
from bench.systems import Turn
from myelin import MemoryClient
from myelin.context import excerpt

ARMS = {
    "static": (False, False),
    "decay": (True, False),
    "feedback": (False, True),
    "decay + feedback": (True, True),
}
MODEL_ARMS = {
    "feedback (model)": (False, True),
    "decay + feedback (model)": (True, True),
}
KS = (5, 10, 20)
SHOWN = 20
# A message longer than this is shown to the model as an excerpt about
# the question. LoCoMo utterances are shorter; LongMemEval has assistant
# turns of several thousand tokens, and twenty of them are shown.
MODEL_TURN_TOKENS = 200


def locomo_timeline(
    sample: dict[str, Any],
) -> tuple[list[list[Turn]], list[list[str]], list[dict[str, Any]]]:
    """A LoCoMo conversation as sessions in order, an id for every turn,
    and its questions with the ids of their evidence."""
    turns = locomo.conversation_turns(sample)
    by_number: dict[int, list[Turn]] = defaultdict(list)
    for turn in turns:
        by_number[int(turn.session_id.split()[-1])].append(turn)
    sessions = [by_number[number] for number in sorted(by_number)]
    ids = [[t.dia_id for t in batch] for batch in sessions]
    known = {i for batch in ids for i in batch}
    asked = [
        {**qa, "evidence": [e for e in qa.get("evidence", []) if e in known]}
        for qa in locomo.questions(sample)
    ]
    return sessions, ids, asked


def longmemeval_timeline(
    entry: dict[str, Any],
) -> tuple[list[list[Turn]], list[list[str]], list[dict[str, Any]]]:
    """A LongMemEval entry as sessions in date order and its one
    question, whose evidence is the turns flagged as holding the answer.

    Turn ids are positions in the timeline: a history can hold the same
    session id twice.
    """
    by_session: dict[int, list[Turn]] = defaultdict(list)
    position = -1
    for turn in longmemeval.history_turns(entry):
        position += turn.index == 0  # a session's first turn
        by_session[position].append(turn)
    # Stable, so sessions on the same date keep the order they came in.
    sessions = sorted(by_session.values(), key=lambda batch: batch[0].session_date)
    ids = [[f"{n}:{t.index}" for t in batch] for n, batch in enumerate(sessions)]
    evidence = [
        i for batch, names in zip(sessions, ids, strict=True)
        for t, i in zip(batch, names, strict=True) if t.evidence
    ]
    return sessions, ids, [{"question": entry["question"], "evidence": evidence}]


def span_days(sessions: list[list[Turn]]) -> int:
    """Whole days from a timeline's first session to its last."""
    return (sessions[-1][0].session_date - sessions[0][0].session_date).days


TIMELINES = {"locomo": locomo_timeline, "longmemeval": longmemeval_timeline}


def replay(
    sample: dict[str, Any],
    seed: int,
    *,
    decay: bool,
    feedback: bool,
    max_lag: int = 6,
    embedder: Any = None,
    revisits: int = 0,
    model: Any = None,
    dataset: str = "locomo",
) -> dict[str, Any]:
    """Run one timeline; return per-question outcomes and the state of
    the store at the end.

    With ``feedback`` and a ``model`` (anything with ``ChatClient``'s
    ``complete``), the used memories are the ones the model names.
    """
    sessions, ids, asked = TIMELINES[dataset](sample)
    order = range(len(sessions))
    session_of = {i: step for step, names in enumerate(ids) for i in names}

    rng = random.Random(seed)
    due: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for qa in asked:
        evidence = qa["evidence"]
        if not evidence:
            continue
        last = max(session_of[e] for e in evidence)
        ask_at = min(len(order) - 1, last + rng.randint(0, max_lag))
        due[ask_at].append({**qa, "evidence": evidence, "lag": ask_at - last, "visit": 0})
        for visit in range(1, revisits + 1):
            ask_at = min(len(order) - 1, ask_at + rng.randint(1, max_lag))
            due[ask_at].append(
                {**qa, "evidence": evidence, "lag": ask_at - last, "visit": visit}
            )

    now = [sessions[0][0].session_date]
    client = MemoryClient(
        user_id="bench", embedder=embedder, clock=lambda: now[0], cite_learning=True
    )
    id_of: dict[str, str] = {}
    outcomes: list[dict[str, Any]] = []
    try:
        for step, batch in enumerate(sessions):
            now[0] = batch[0].session_date
            stored = client.add_many(
                [
                    {
                        "content": t.text,
                        "t_valid": t.session_date,
                        "metadata": {"session_id": t.session_id, "turn": t.index},
                    }
                    for t in batch
                ]
            )
            id_of.update({i: m.id for i, m in zip(ids[step], stored, strict=True)})
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
                outcome = {
                    "lag": qa["lag"],
                    "visit": qa["visit"],
                    "hidden": hidden,
                    **{f"top{k}": wanted <= set(ranked[:k]) for k in KS},
                }
                outcomes.append(outcome)
                if not feedback:
                    continue
                used = [n for n, i in enumerate(ranked, start=1) if i in wanted]
                if model is not None:
                    named = named_by_model(
                        model,
                        qa["question"],
                        [excerpt(h.memory.content, qa["question"], MODEL_TURN_TOKENS)
                         for h in shown],
                    )
                    # How the model's choice compares with the labels.
                    outcome["replied"] = named is not None
                    if named is None:
                        continue
                    labelled = set(used)
                    used = sorted(position + 1 for position in named)
                    outcome["named"] = len(used)
                    outcome["named_evidence"] = len(labelled & set(used))
                    outcome["evidence_shown"] = len(labelled)
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
    repeats = [o for o in outcomes if o["visit"] > 0]

    def share(rows: list[dict[str, Any]], key: str) -> float:
        return 100 * sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0

    return {
        **{f"top{k}": share(outcomes, f"top{k}") for k in KS},
        "top10_late": share(late, "top10"),
        "top5_repeat": share(repeats, "top5"),
        "top10_repeat": share(repeats, "top10"),
        "hidden_repeat": share(repeats, "hidden"),
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


REPEAT_COLUMNS = (
    ("top5_repeat", "Repeat askings: top 5"),
    ("top10_repeat", "top 10"),
    ("hidden_repeat", "needed a hidden memory"),
)


class _Metered:
    """Passes calls to a chat client and adds up what they used."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def complete(self, prompt: str, **options: Any) -> Any:
        completion = self._client.complete(prompt, **options)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += getattr(completion, "prompt_tokens", 0)
            self.completion_tokens += getattr(completion, "completion_tokens", 0)
        return completion


def model_agreement(runs: list[dict[str, Any]]) -> str:
    """One line on how the model's named memories compare with the labels."""
    asked = [o for run in runs for o in run["outcomes"] if "replied" in o]
    named = [o for o in asked if o["replied"]]
    total = sum(o["named"] for o in named)
    hits = sum(o["named_evidence"] for o in named)
    shown = sum(o["evidence_shown"] for o in named)
    return (
        f"{len(asked)} askings, {len(named)} with a readable reply; "
        f"{total / len(named) if named else 0.0:.2f} memories named per reply, "
        f"{100 * hits / total if total else 0.0:.1f}% of them labelled evidence; "
        f"{100 * hits / shown if shown else 0.0:.1f}% of the labelled evidence shown was named"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.lifetime", description=__doc__)
    parser.add_argument(
        "--data", required=True, help="path to locomo10.json or longmemeval_s_cleaned.json"
    )
    parser.add_argument(
        "--dataset", choices=sorted(TIMELINES), default="locomo", help="what --data holds"
    )
    parser.add_argument("--embedder", default="", help="static model id; omit for lexical only")
    parser.add_argument("--seeds", type=int, default=3, help="random question schedules")
    parser.add_argument("--max-lag", type=int, default=6, help="sessions a question may wait")
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N conversations (or entries)"
    )
    parser.add_argument(
        "--min-span-days", type=int, default=0,
        help="of those, only timelines whose sessions span at least this many days "
             "(decay acts on age; most LongMemEval-S histories cover under two weeks)",
    )
    parser.add_argument(
        "--revisits", type=int, default=0,
        help="ask each question this many more times at later sessions",
    )
    parser.add_argument(
        "--model", default="",
        help="model id; adds feedback arms where the model names the memories it used",
    )
    parser.add_argument(
        "--model-only", action="store_true", help="with --model, run only the model arms"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="timelines replayed at once (threads; useful while waiting on a model)",
    )
    args = parser.parse_args(argv)
    samples = json.loads(Path(args.data).read_text())
    if args.limit:
        samples = samples[: args.limit]
    if args.min_span_days:
        samples = [
            s for s in samples
            if span_days(TIMELINES[args.dataset](s)[0]) >= args.min_span_days
        ]
    print(f"{len(samples)} timelines", file=sys.stderr)
    embedder = None
    if args.embedder:
        from myelin.embedders import StaticEmbedder

        embedder = StaticEmbedder(args.embedder)
    model = _Metered(ChatClient(args.model)) if args.model else None
    arms = [(arm, flags, None) for arm, flags in ARMS.items()]
    if model is not None:
        arms = [] if args.model_only else arms
        arms += [(arm, flags, model) for arm, flags in MODEL_ARMS.items()]

    columns = COLUMNS + (REPEAT_COLUMNS if args.revisits else ())
    print("| | " + " | ".join(title for _, title in columns) + " |")
    print("|---|" + "---:|" * len(columns))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for arm, (decay, feedback), named_by in arms:
            jobs = [(seed, s) for seed in range(args.seeds) for s in samples]
            # Each timeline has its own store, used only by its thread;
            # results come back in the order the jobs were listed.
            runs = list(pool.map(
                lambda job, decay=decay, feedback=feedback, named_by=named_by: replay(
                    job[1], job[0], decay=decay, feedback=feedback, max_lag=args.max_lag,
                    embedder=embedder, revisits=args.revisits, model=named_by,
                    dataset=args.dataset,
                ),
                jobs,
            ))
            per_seed = [
                summarize(runs[seed * len(samples):(seed + 1) * len(samples)])
                for seed in range(args.seeds)
            ]
            cells = []
            for key, _ in columns:
                values = [run[key] for run in per_seed]
                spread = f" ± {statistics.stdev(values):.1f}" if len(values) > 1 else ""
                cells.append(f"{statistics.mean(values):.1f}{spread}")
            print(f"| {arm} | " + " | ".join(cells) + " |", flush=True)
            if named_by is not None:
                print(f"{arm}: {model_agreement(runs)}", file=sys.stderr, flush=True)
    if model is not None:
        print(
            f"model: {model.calls} calls, {model.prompt_tokens} prompt tokens, "
            f"{model.completion_tokens} completion tokens; "
            f"{int(per_seed[0]['questions'])} askings per schedule",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
