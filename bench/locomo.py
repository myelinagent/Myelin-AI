"""Run a memory system through LoCoMo, headless and resumable.

    python -m bench.locomo run --system myelin --data locomo10.json \\
        --out bench/runs/locomo --answerer gpt-4o-mini --judge gpt-4o-mini
    python -m bench.locomo report bench/runs/locomo

LoCoMo is ten long two-person conversations with about 150 questions
each. Each conversation is ingested once into a fresh store and all of
its questions are asked against it. Category 5 (adversarial) questions
have no gold answer and are skipped, as in every published comparison.

The dataset's own scripts score by token overlap; published
"LLM-judge" scores use the prompt below, taken from Mem0's evaluation
code (Apache-2.0), with ``gpt-4o-mini`` as the judge. The prompt asks
the judge to be generous, and an audit found about 6% of the answer key
wrong, so read these scores as comparable with other LoCoMo numbers
and not as a measure of correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bench.longmemeval as _lme
from bench.extract_facts import load_cache
from bench.llm import ChatClient
from bench.longmemeval import (
    BUDGET_DEPTH,
    JsonlLog,
    _git_commit,
    _mean,
    _pct,
    _pool,
    _sha256,
    final_answer,
    render_history,
    within_budget,
)
from bench.systems import SYSTEMS, Turn
from myelin.context import estimate_tokens
from myelin.enrich import build_extraction_prompt, parse_facts
from myelin.prompts import ANSWER_STYLES, build_answer_prompt

# Names as commonly used for the dataset's numeric categories.
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
SKIPPED_CATEGORY = 5

JUDGE_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""  # kept byte-for-byte as published


def parse_date(text: str) -> datetime:
    """LoCoMo dates look like ``1:56 pm on 8 May, 2023``."""
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=UTC)


# Facts per session from ``python -m bench.locomo facts``, keyed by
# ``session_key``; filled by ``run --facts``.
FACTS: dict[str, dict[int, list[str]]] = {}


def session_key(sample_id: str, number: int) -> str:
    return f"{sample_id}|session {number}"


def conversation_turns(sample: dict[str, Any]) -> list[Turn]:
    """Every utterance of a conversation, with its session's date.

    A shared image is represented by its caption, the only form of it in
    the dataset's text. ``Turn.session_id`` holds the dialogue id
    (``D3:7``), which is what the questions' evidence refers to.
    """
    conversation = sample["conversation"]
    turns: list[Turn] = []
    number = 1
    while f"session_{number}" in conversation:
        when = parse_date(conversation[f"session_{number}_date_time"])
        facts = FACTS.get(session_key(sample["sample_id"], number), {})
        for index, utterance in enumerate(conversation[f"session_{number}"]):
            text = utterance["text"]
            caption = utterance.get("blip_caption")
            if caption:
                text = f"{text} [shares a photo: {caption}]"
            turns.append(
                Turn(
                    session_id=f"session {number}",
                    session_date=when,
                    index=index,
                    role=utterance["speaker"],
                    content=text,
                    dia_id=utterance["dia_id"],
                    facts=tuple(facts.get(index, ())),
                )
            )
        number += 1
    return turns


def questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for index, qa in enumerate(sample["qa"]):
        if qa["category"] == SKIPPED_CATEGORY or "answer" not in qa:
            continue
        out.append({**qa, "question_id": f"{sample['sample_id']}:{index}"})
    return out


def judged_correct(reply: str) -> bool:
    """Read the judge's verdict from its JSON reply, or from the text if
    the JSON is malformed."""
    start, end = reply.find("{"), reply.rfind("}")
    if 0 <= start < end:
        try:
            label = json.loads(reply[start : end + 1]).get("label")
            if isinstance(label, str):
                return label.strip().upper() == "CORRECT"
        except ValueError:
            pass
    upper = reply.upper()
    return "CORRECT" in upper and "WRONG" not in upper


def _run_conversation(
    sample: dict[str, Any],
    todo: list[dict[str, Any]],
    args: argparse.Namespace,
    answers: JsonlLog,
    answerer: ChatClient | None,
) -> int:
    """Ingest one conversation and ask its outstanding questions."""
    turns = conversation_turns(sample)
    by_dia = {t.dia_id: t for t in turns}
    speakers = sorted({t.role for t in turns})
    asked_at = max(t.session_date for t in turns)
    system = SYSTEMS[args.system]()
    system.reset()
    started = time.monotonic()
    system.ingest(turns)
    ingest_seconds = time.monotonic() - started
    retrieve_lock = threading.Lock()

    def answer(qa: dict[str, Any]) -> None:
        row: dict[str, Any] = {
            "question_id": qa["question_id"],
            "category": qa["category"],
            "ingest_seconds": ingest_seconds,
        }
        with retrieve_lock:  # one store per conversation, shared by its questions
            started = time.monotonic()
            ranked = system.retrieve(qa["question"], BUDGET_DEPTH)
            row["retrieve_seconds"] = time.monotonic() - started
        context = within_budget(
            ranked, args.token_budget, qa["question"],
            args.max_turn_tokens, args.whole_turns,
        )
        evidence = {d for d in qa.get("evidence", []) if d in by_dia}
        if evidence:
            shown = {t.dia_id for t in context}
            row["evidence_all"] = 1.0 if evidence <= shown else 0.0
            row["evidence_share"] = len(evidence & shown) / len(evidence)
        row["context_turns"] = len(context)
        row["context_tokens_estimate"] = sum(estimate_tokens(t.text) for t in context)
        if answerer is None:
            row["hypothesis"] = ""
        else:
            history = (
                f"(These are conversations between {' and '.join(speakers)}.)\n\n"
                + render_history(context)
            )
            prompt = build_answer_prompt(
                history, qa["question"],
                now=asked_at.strftime("%Y-%m-%d (%a)"), style=args.prompt,
            )
            completion = answerer.complete(prompt, max_tokens=args.max_answer_tokens)
            row.update(
                hypothesis=final_answer(completion.text),
                response=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        answers.append(row)

    try:
        return _pool(
            args.workers,
            {q["question_id"]: (lambda q=q: answer(q)) for q in todo},
            f"{sample['sample_id']} answered",
        )
    finally:
        system.close()


def cmd_run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    samples = json.loads(data_path.read_text())
    if args.limit:
        samples = samples[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "benchmark": "LoCoMo",
        "data_file": data_path.name,
        "data_sha256": _sha256(data_path),
        "conversations": len(samples),
        "questions": sum(len(questions(s)) for s in samples),
        "system": args.system,
        "granularity": "utterance",
        "token_budget": args.token_budget,
        "max_turn_tokens": args.max_turn_tokens,
        "whole_turns": args.whole_turns,
        "embedder": args.embedder,
        "facts": Path(args.facts).name if args.facts else "",
        "show_facts": args.show_facts,
        "max_answer_tokens": args.max_answer_tokens,
        "answerer": args.answerer,
        "judge": args.judge,
        "prompt_version": f"myelin-{args.prompt}-v2",
        "temperature": 0.0,
        "git_commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        pinned = (
            "data_sha256", "system", "token_budget", "max_turn_tokens", "whole_turns",
            "embedder", "facts", "show_facts", "answerer", "prompt_version",
        )
        changed = [k for k in pinned if (previous.get(k) or 0) != (manifest[k] or 0)]
        if changed:
            print(
                f"{out} holds a run with different settings ({', '.join(changed)}); "
                "use a new --out directory.",
                file=sys.stderr,
            )
            return 2
        manifest["started_at"] = previous.get("started_at", manifest["started_at"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if args.embedder:
        if args.system != "myelin":
            print("--embedder applies to --system myelin only", file=sys.stderr)
            return 2
        from myelin.embedders import StaticEmbedder

        SYSTEMS["myelin"].embedder = StaticEmbedder(args.embedder)

    FACTS.clear()
    if args.facts:
        FACTS.update(load_cache(Path(args.facts)))
        if not FACTS:
            print(f"--facts {args.facts}: no facts found", file=sys.stderr)
            return 2
    _lme.SHOW_FACTS = bool(args.show_facts)

    answers = JsonlLog(out / "answers.jsonl")
    answerer = ChatClient(args.answerer) if args.answerer else None
    failed = 0
    for sample in samples:
        todo = [q for q in questions(sample) if q["question_id"] not in answers.rows]
        if not todo:
            continue
        failed += _run_conversation(sample, todo, args, answers, answerer)

    if args.judge and answerer is not None:
        failed += judge_run(out, samples, args.judge, args.workers)
    summarize(out)
    if failed:
        print(f"{failed} calls failed; rerun the same command to retry them.", file=sys.stderr)
    return 1 if failed else 0


def cmd_facts(args: argparse.Namespace) -> int:
    """Extract dated facts for every session, once, to a resumable cache."""
    out = Path(args.out)
    done = load_cache(out)
    FACTS.clear()
    jobs: dict[str, tuple[list[str], list[Turn]]] = {}
    for sample in json.loads(Path(args.data).read_text()):
        turns = conversation_turns(sample)
        people = sorted({t.role for t in turns})
        for turn in turns:
            number = int(turn.session_id.split()[-1])
            key = session_key(sample["sample_id"], number)
            jobs.setdefault(key, (people, []))[1].append(turn)
    todo = [key for key in jobs if key not in done]
    print(f"{len(jobs)} sessions, {len(todo)} to extract", file=sys.stderr)
    client = ChatClient(args.model)
    lock = threading.Lock()

    def work(key: str) -> None:
        people, turns = jobs[key]
        prompt = build_extraction_prompt(
            [(t.role, t.content) for t in turns], turns[0].session_date, people
        )
        facts = parse_facts(client.complete(prompt, max_tokens=4000).text, len(turns))
        with lock, out.open("a") as handle:
            handle.write(json.dumps({"key": key, "facts": facts}) + "\n")

    failed = _pool(args.workers, {key: (lambda key=key: work(key)) for key in todo}, "extracted")
    return 1 if failed else 0


def judge_run(out: Path, samples: list[dict[str, Any]], model: str, workers: int) -> int:
    answers = JsonlLog(out / "answers.jsonl")
    labels = JsonlLog(out / "judged.jsonl")
    judge = ChatClient(model)
    by_id = {q["question_id"]: q for s in samples for q in questions(s)}
    todo = [qid for qid in answers.rows if qid in by_id and qid not in labels.rows]
    print(f"judging {len(todo)} answers with {model}", file=sys.stderr)

    def grade(qid: str) -> None:
        qa = by_id[qid]
        verdict = judge.complete(
            JUDGE_PROMPT.format(
                question=qa["question"],
                gold_answer=qa["answer"],
                generated_answer=answers.rows[qid]["hypothesis"],
            ),
            max_tokens=200,
        )
        labels.append({"question_id": qid, "label": judged_correct(verdict.text)})

    return _pool(workers, {qid: (lambda qid=qid: grade(qid)) for qid in todo}, "judged")


def summarize(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text())
    answers = JsonlLog(out / "answers.jsonl").rows
    labels = JsonlLog(out / "judged.jsonl").rows
    summary: dict[str, Any] = {"manifest": manifest, "answered": len(answers)}
    graded = {q: 1.0 if row["label"] else 0.0 for q, row in labels.items() if q in answers}
    summary["judged"] = len(graded)
    if graded:
        by_category: dict[int, list[float]] = defaultdict(list)
        for qid, score in graded.items():
            by_category[answers[qid]["category"]].append(score)
        summary["accuracy"] = {
            "overall": _mean(list(graded.values())),
            "by_category": {CATEGORIES[c]: _mean(by_category[c]) for c in sorted(by_category)},
            "counts": {CATEGORIES[c]: len(by_category[c]) for c in sorted(by_category)},
        }

    def column(name: str) -> list[float]:
        return [float(row[name]) for row in answers.values() if name in row]

    summary["retrieval"] = {
        "evidence_all": _mean(column("evidence_all")),
        "evidence_share": _mean(column("evidence_share")),
        "questions_with_evidence": len(column("evidence_all")),
    }
    summary["cost"] = {
        "answerer_prompt_tokens": int(sum(column("prompt_tokens"))),
        "answerer_completion_tokens": int(sum(column("completion_tokens"))),
        "mean_context_tokens": _mean(column("prompt_tokens")),
        "mean_context_turns": _mean(column("context_turns")),
        "mean_retrieve_seconds": _mean(column("retrieve_seconds")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def render_report(summaries: list[dict[str, Any]]) -> str:
    def line(label: str, pick: Any) -> str:
        return f"| {label} | " + " | ".join(pick(s) for s in summaries) + " |"

    def category(name: str) -> Any:
        return lambda s: _pct(s.get("accuracy", {}).get("by_category", {}).get(name))

    def number(section: str, key: str, fmt: str, scale: float = 1.0) -> Any:
        def pick(s: dict[str, Any]) -> str:
            value = s[section].get(key)
            return "—" if value is None else fmt.format(value * scale)
        return pick

    rows = [
        "| | " + " | ".join(s["manifest"]["system"] for s in summaries) + " |",
        "|---|" + "---:|" * len(summaries),
        line("Questions", lambda s: str(s["answered"])),
    ]
    if any(s["judged"] for s in summaries):
        rows += [
            line("**Judged correct (overall)**", lambda s: _pct(s.get("accuracy", {}).get("overall"))),
            *[line(f"  {name}", category(name)) for name in CATEGORIES.values()],
            line("Context tokens per question", number("cost", "mean_context_tokens", "{:,.0f}")),
        ]
    rows += [
        line("All evidence utterances in context", number("retrieval", "evidence_all", "{:.1f}", 100)),
        line("Share of evidence utterances in context",
             number("retrieval", "evidence_share", "{:.1f}", 100)),
        line("Utterances in context", number("cost", "mean_context_turns", "{:.1f}")),
        line("Retrieve per question (ms)", number("cost", "mean_retrieve_seconds", "{:.0f}", 1000)),
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.locomo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="ingest, retrieve, answer and optionally judge")
    run.add_argument("--system", required=True, choices=list(SYSTEMS))
    run.add_argument("--data", required=True, help="path to locomo10.json")
    run.add_argument("--out", required=True, help="run directory (created; resumable)")
    run.add_argument("--answerer", default="", help="model id; omit for retrieval metrics only")
    run.add_argument("--judge", default="", help="model id (published scores use gpt-4o-mini)")
    run.add_argument("--token-budget", type=int, default=6000)
    run.add_argument("--max-turn-tokens", type=int, default=300)
    run.add_argument("--whole-turns", type=int, default=3)
    run.add_argument("--embedder", default="")
    run.add_argument(
        "--facts", default="",
        help="cache from `python -m bench.locomo facts`; facts are indexed with each utterance",
    )
    run.add_argument(
        "--show-facts", action="store_true",
        help="with --facts: also print an utterance's facts under it in the answerer's context",
    )
    run.add_argument("--prompt", default="reasoned", choices=list(ANSWER_STYLES))
    run.add_argument("--max-answer-tokens", type=int, default=800)
    run.add_argument("--limit", type=int, default=0, help="only the first N conversations")
    run.add_argument("--workers", type=int, default=8)
    run.set_defaults(func=cmd_run)
    facts = sub.add_parser("facts", help="extract dated facts per session to a cache")
    facts.add_argument("--data", required=True)
    facts.add_argument("--out", required=True, help="cache file (JSONL, resumable)")
    facts.add_argument("--model", required=True)
    facts.add_argument("--workers", type=int, default=16)
    facts.set_defaults(func=cmd_facts)
    report = sub.add_parser("report", help="print a comparison table for one or more runs")
    report.add_argument("runs", nargs="+")
    report.set_defaults(
        func=lambda a: print(render_report([summarize(Path(r)) for r in a.runs])) or 0
    )
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
