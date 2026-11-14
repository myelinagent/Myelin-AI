"""Run a memory system through LongMemEval, headless and resumable.

    python -m bench.longmemeval run --system myelin --data longmemeval_s_cleaned.json \\
        --out bench/runs/myelin --answerer gpt-4o-mini --judge gpt-4o
    python -m bench.longmemeval report bench/runs/myelin bench/runs/bm25

``run`` ingests each question's history into a fresh store, retrieves,
answers with a pinned prompt, and (with ``--judge``) grades with the
benchmark's own judge prompts. Every question is checkpointed, so an
interrupted run continues where it stopped. ``hypotheses.jsonl`` is in
the format the official ``evaluate_qa.py`` reads.

Judge prompts are from LongMemEval (Wu et al., MIT License),
https://github.com/xiaowu0162/LongMemEval — ``src/evaluation/evaluate_qa.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.llm import ChatClient, LLMError
from bench.extract_facts import load_cache, session_key
from bench.systems import SYSTEMS, Turn
from myelin.context import estimate_tokens, excerpt
from myelin.prompts import ANSWER_STYLES, build_answer_prompt

# Answer prompts. "v1" is the plain prompt the first baselines were run
# with, kept so they stay reproducible; the others are Myelin's own.
PROMPT_VERSIONS = {"v1": "lme-answer-v1", "reasoned": "myelin-reasoned-v2", "direct": "myelin-direct-v2"}
RECALL_DEPTH = 50
# How far down the ranking a token budget may reach.
BUDGET_DEPTH = 200
RECALL_AT = (5, 10, 50)
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

ANSWER_PROMPT = """I will give you excerpts from past conversations between a user and an assistant, grouped by session and dated. Answer the question using only this history. If the history does not contain the information needed, say that you do not know.

History:

{history}

Current date: {question_date}
Question: {question}
Answer:"""

_JUDGE_COMMON = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
)
_JUDGE_TAIL = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_JUDGE_STANDARD = (
    _JUDGE_COMMON
    + "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    + _JUDGE_TAIL
)
_JUDGE_TEMPORAL = (
    _JUDGE_COMMON
    + "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still correct. "
    + _JUDGE_TAIL
)
_JUDGE_UPDATE = (
    _JUDGE_COMMON
    + "If the response contains some previous information along with an updated answer, the "
    "response should be considered as correct as long as the updated answer is the required "
    "answer." + _JUDGE_TAIL
)
_JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given "
    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
    "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def is_abstention(question_id: str) -> bool:
    return "_abs" in question_id


def judge_prompt(entry: dict[str, Any], hypothesis: str) -> str:
    args = (entry["question"], entry["answer"], hypothesis)
    if is_abstention(entry["question_id"]):
        return _JUDGE_ABSTENTION.format(*args)
    qtype = entry["question_type"]
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return _JUDGE_STANDARD.format(*args)
    if qtype == "temporal-reasoning":
        return _JUDGE_TEMPORAL.format(*args)
    if qtype == "knowledge-update":
        return _JUDGE_UPDATE.format(*args)
    if qtype == "single-session-preference":
        return _JUDGE_PREFERENCE.format(*args)
    raise ValueError(f"unknown question_type {qtype!r}")


# ---- data -------------------------------------------------------------


def parse_date(text: str) -> datetime:
    """LongMemEval dates look like ``2023/05/20 (Sat) 02:21``."""
    day, _, clock = text.partition(" (")
    clock = clock.partition(") ")[2] or "00:00"
    return datetime.strptime(f"{day} {clock}", "%Y/%m/%d %H:%M").replace(tzinfo=UTC)


# With ``--show-facts``, a turn's extracted facts are printed under it in
# the answerer's context, so resolved dates are visible to the reader.
SHOW_FACTS = False

# Facts extracted per session by ``python -m bench.extract_facts``, keyed
# by session; filled by ``--facts``.
FACTS: dict[str, dict[int, list[str]]] = {}


def history_turns(entry: dict[str, Any]) -> list[Turn]:
    turns: list[Turn] = []
    for session_id, date, session in zip(
        entry["haystack_session_ids"],
        entry["haystack_dates"],
        entry["haystack_sessions"],
        strict=True,
    ):
        when = parse_date(date)
        facts = FACTS.get(session_key(date, session), {}) if FACTS else {}
        for index, turn in enumerate(session):
            turns.append(
                Turn(
                    session_id, when, index, turn["role"], turn["content"],
                    evidence=bool(turn.get("has_answer")),
                    facts=tuple(facts.get(index, ())),
                )
            )
    return turns


def render_history(turns: list[Turn]) -> str:
    """Group turns by session, oldest session first, turns in spoken order."""
    by_session: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        by_session[turn.session_id].append(turn)
    sessions = sorted(by_session.values(), key=lambda ts: (ts[0].session_date, ts[0].session_id))
    blocks = []
    for number, session in enumerate(sessions, start=1):
        session.sort(key=lambda t: t.index)
        date = session[0].session_date.strftime("%Y/%m/%d (%a) %H:%M")
        lines = "\n".join(_shown(t) for t in session)
        blocks.append(f"### Session {number} — {date}\n{lines}")
    return "\n\n".join(blocks) if blocks else "(no relevant history found)"


def within_budget(
    ranked: list[Turn],
    token_budget: int,
    question: str = "",
    max_turn_tokens: int = 0,
    whole_turns: int = 0,
) -> list[Turn]:
    """The best-ranked turns that fit the budget (always at least one).

    With ``max_turn_tokens``, a longer turn is replaced by a
    query-focused excerpt of about that size before it is counted,
    except the ``whole_turns`` best-ranked, which stay complete.
    """
    kept: list[Turn] = []
    used = 0
    for rank, turn in enumerate(ranked):
        if (
            max_turn_tokens
            and rank >= whole_turns
            and estimate_tokens(turn.content) > max_turn_tokens
        ):
            turn = dataclasses.replace(
                turn, content=excerpt(turn.content, question, max_turn_tokens)
            )
        cost = estimate_tokens(turn.text)
        if kept and used + cost > token_budget:
            break
        kept.append(turn)
        used += cost
    return kept


def _shown(turn: Turn) -> str:
    if SHOW_FACTS and turn.facts:
        return f"{turn.text}\n  (noted: {' '.join(turn.facts)})"
    return turn.text


def session_recall(ranked_sessions: list[str], gold: list[str]) -> dict[str, float]:
    """Did the top-k retrieved turns reach any / all of the evidence sessions?"""
    out: dict[str, float] = {}
    wanted = set(gold)
    for k in RECALL_AT:
        found = wanted & set(ranked_sessions[:k])
        out[f"any@{k}"] = 1.0 if found else 0.0
        out[f"all@{k}"] = 1.0 if found == wanted else 0.0
    return out


# ---- checkpointed files ----------------------------------------------


class JsonlLog:
    """Append-only JSONL keyed by ``question_id``; safe across threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["question_id"]] = row

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows[row["question_id"]] = row
            with self.path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def final_answer(response: str) -> str:
    """The part of a response that is graded: what follows the last
    "Answer:" line, so working notes cannot earn credit on their own.
    A response without that marker is graded whole."""
    _, marker, tail = response.rpartition("Answer:")
    tail = tail.strip().lstrip("*").strip()  # models often write "**Answer:** …"
    return tail if marker and tail else response.strip()


# ---- run --------------------------------------------------------------


def answer_one(
    entry: dict[str, Any],
    system_name: str,
    top_k: int,
    answerer: ChatClient | None,
    token_budget: int = 0,
    max_turn_tokens: int = 0,
    prompt_style: str = "reasoned",
    whole_turns: int = 0,
    max_answer_tokens: int = 800,
) -> dict[str, Any]:
    all_turns = history_turns(entry)
    gold = list(entry["answer_session_ids"])
    row: dict[str, Any] = {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "n_turns": len(all_turns),
    }
    if system_name == "oracle":
        # Control: hand the reader exactly the evidence sessions.
        context = [t for t in all_turns if t.session_id in set(gold)]
    else:
        system = SYSTEMS[system_name]()
        try:
            system.reset()
            started = time.monotonic()
            system.ingest(all_turns)
            row["ingest_seconds"] = time.monotonic() - started
            started = time.monotonic()
            depth = max(top_k, RECALL_DEPTH, BUDGET_DEPTH if token_budget else 0)
            ranked = system.retrieve(entry["question"], depth)
            row["retrieve_seconds"] = time.monotonic() - started
        finally:
            system.close()
        row["recall"] = session_recall([t.session_id for t in ranked], gold)
        if token_budget:
            context = within_budget(
                ranked, token_budget, entry["question"], max_turn_tokens, whole_turns
            )
        else:
            context = ranked[:top_k]
        # Did what the answerer is actually shown reach every evidence
        # session, and every turn the benchmark marks as holding the answer?
        row["recall"]["all@context"] = (
            1.0 if set(gold) <= {t.session_id for t in context} else 0.0
        )
        evidence = {(t.session_id, t.index) for t in all_turns if t.evidence}
        if evidence:
            shown = {(t.session_id, t.index) for t in context}
            row["recall"]["evidence_turns_all@context"] = 1.0 if evidence <= shown else 0.0
            row["recall"]["evidence_turns_share@context"] = len(evidence & shown) / len(evidence)
    row["context_turns"] = len(context)
    row["context_tokens_estimate"] = sum(estimate_tokens(t.text) for t in context)
    if answerer is None:
        row["hypothesis"] = ""
        return row
    history = render_history(context)
    if prompt_style == "v1":
        prompt = ANSWER_PROMPT.format(
            history=history,
            question_date=entry["question_date"],
            question=entry["question"],
        )
    else:
        prompt = build_answer_prompt(
            history, entry["question"], now=entry["question_date"], style=prompt_style
        )
    completion = answerer.complete(prompt, max_tokens=max_answer_tokens)
    row.update(
        hypothesis=final_answer(completion.text),
        response=completion.text,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        answer_seconds=completion.seconds,
    )
    return row


def _pool(workers: int, jobs: dict[str, Any], describe: str) -> int:
    """Run ``jobs`` (id -> zero-arg callable); return how many failed."""
    failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job): qid for qid, job in jobs.items()}
        for future in as_completed(futures):
            done += 1
            try:
                future.result()
            except LLMError as exc:
                failed += 1
                print(f"  {futures[future]}: {exc}", file=sys.stderr)
            if done % 25 == 0 or done == len(futures):
                print(f"  {describe}: {done}/{len(futures)}", file=sys.stderr, flush=True)
    return failed


def cmd_run(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    entries = json.loads(data_path.read_text())
    if args.limit:
        entries = entries[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "benchmark": "LongMemEval",
        "data_file": data_path.name,
        "data_sha256": _sha256(data_path),
        "questions": len(entries),
        "system": args.system,
        "granularity": "turn",
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "max_turn_tokens": args.max_turn_tokens,
        "whole_turns": args.whole_turns,
        "embedder": args.embedder,
        "facts": Path(args.facts).name if args.facts else "",
        "show_facts": args.show_facts,
        "max_answer_tokens": args.max_answer_tokens,
        "answerer": args.answerer,
        "judge": args.judge,
        "prompt_version": PROMPT_VERSIONS[args.prompt],
        "temperature": 0.0,
        "git_commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        pinned = (
            "data_sha256", "system", "top_k", "token_budget", "max_turn_tokens",
            "whole_turns", "embedder", "facts", "show_facts", "answerer",
            "prompt_version",
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

    if args.facts:
        FACTS.update(load_cache(Path(args.facts)))
        if not FACTS:
            print(f"--facts {args.facts}: no facts found", file=sys.stderr)
            return 2
    global SHOW_FACTS
    SHOW_FACTS = bool(args.show_facts)

    if args.embedder:
        if args.system != "myelin":
            print("--embedder applies to --system myelin only", file=sys.stderr)
            return 2
        from myelin.embedders import StaticEmbedder

        SYSTEMS["myelin"].embedder = StaticEmbedder(args.embedder)

    answers = JsonlLog(out / "answers.jsonl")
    answerer = ChatClient(args.answerer) if args.answerer else None
    todo = [e for e in entries if e["question_id"] not in answers.rows]
    print(f"answering {len(todo)} of {len(entries)} questions", file=sys.stderr)
    jobs = {
        e["question_id"]: (
            lambda e=e: answers.append(
                answer_one(
                    e, args.system, args.top_k, answerer,
                    args.token_budget, args.max_turn_tokens, args.prompt,
                    args.whole_turns, args.max_answer_tokens,
                )
            )
        )
        for e in todo
    }
    failed = _pool(args.workers, jobs, "answered")

    with (out / "hypotheses.jsonl").open("w") as handle:
        for entry in entries:
            row = answers.rows.get(entry["question_id"])
            if row is not None:
                handle.write(
                    json.dumps(
                        {"question_id": row["question_id"], "hypothesis": row["hypothesis"]}
                    )
                    + "\n"
                )

    if args.judge and answerer is not None:
        failed += judge_run(out, entries, args.judge, args.workers)
    summarize(out)
    if failed:
        print(f"{failed} calls failed; rerun the same command to retry them.", file=sys.stderr)
    return 1 if failed else 0


def judge_run(out: Path, entries: list[dict[str, Any]], model: str, workers: int) -> int:
    answers = JsonlLog(out / "answers.jsonl")
    labels = JsonlLog(out / "judged.jsonl")
    judge = ChatClient(model)
    by_id = {e["question_id"]: e for e in entries}
    todo = [qid for qid in answers.rows if qid in by_id and qid not in labels.rows]
    print(f"judging {len(todo)} answers with {model}", file=sys.stderr)

    def grade(qid: str) -> None:
        hypothesis = answers.rows[qid]["hypothesis"]
        verdict = judge.complete(judge_prompt(by_id[qid], hypothesis), max_tokens=10)
        labels.append(
            {
                "question_id": qid,
                "hypothesis": hypothesis,
                "autoeval_label": {"model": model, "label": "yes" in verdict.text.lower()},
            }
        )

    return _pool(workers, {qid: (lambda qid=qid: grade(qid)) for qid in todo}, "judged")


def cmd_judge(args: argparse.Namespace) -> int:
    out = Path(args.run)
    entries = json.loads(Path(args.data).read_text())
    failed = judge_run(out, entries, args.judge, args.workers)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["judge"] = args.judge
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summarize(out)
    return 1 if failed else 0


# ---- report -----------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text())
    answers = JsonlLog(out / "answers.jsonl").rows
    labels = JsonlLog(out / "judged.jsonl").rows
    summary: dict[str, Any] = {"manifest": manifest, "answered": len(answers)}

    graded = {
        qid: 1.0 if row["autoeval_label"]["label"] else 0.0
        for qid, row in labels.items()
        if qid in answers
    }
    summary["judged"] = len(graded)
    if graded:
        by_type: dict[str, list[float]] = defaultdict(list)
        for qid, score in graded.items():
            by_type[answers[qid]["question_type"]].append(score)
        per_type = {t: _mean(by_type[t]) for t in QUESTION_TYPES if by_type[t]}
        summary["accuracy"] = {
            "overall": _mean(list(graded.values())),
            "task_averaged": _mean([v for v in per_type.values() if v is not None]),
            "abstention": _mean([s for q, s in graded.items() if is_abstention(q)]),
            "by_type": per_type,
            "counts": {t: len(by_type[t]) for t in QUESTION_TYPES if by_type[t]},
        }

    # Abstention questions have no evidence to find, so they are left out
    # of retrieval recall, as in the benchmark's own retrieval metrics.
    recalls = [
        row["recall"] for qid, row in answers.items()
        if "recall" in row and not is_abstention(qid)
    ]
    if recalls:
        summary["session_recall"] = {
            key: _mean([r[key] for r in recalls if key in r])
            for key in {k for r in recalls for k in r}
        }
        summary["session_recall"]["questions"] = len(recalls)
        summary["session_recall"]["all@10_by_type"] = {
            qtype: _mean(
                [
                    row["recall"]["all@10"]
                    for qid, row in answers.items()
                    if "recall" in row
                    and not is_abstention(qid)
                    and row["question_type"] == qtype
                ]
            )
            for qtype in QUESTION_TYPES
        }

    def column(name: str) -> list[float]:
        return [float(row[name]) for row in answers.values() if name in row]

    summary["cost"] = {
        "answerer_prompt_tokens": int(sum(column("prompt_tokens"))),
        "answerer_completion_tokens": int(sum(column("completion_tokens"))),
        "mean_context_tokens": _mean(column("prompt_tokens")),
        "mean_context_tokens_estimate": _mean(column("context_tokens_estimate")),
        "mean_context_turns": _mean(column("context_turns")),
        "mean_ingest_seconds": _mean(column("ingest_seconds")),
        "mean_retrieve_seconds": _mean(column("retrieve_seconds")),
        "mean_answer_seconds": _mean(column("answer_seconds")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def render_report(summaries: list[dict[str, Any]]) -> str:
    names = [s["manifest"]["system"] for s in summaries]
    header = "| | " + " | ".join(names) + " |"
    rule = "|---|" + "---:|" * len(names)

    def line(label: str, pick: Any) -> str:
        return f"| {label} | " + " | ".join(pick(s) for s in summaries) + " |"

    def acc(key: str) -> Any:
        return lambda s: _pct(s.get("accuracy", {}).get(key))

    def by_type(qtype: str) -> Any:
        return lambda s: _pct(s.get("accuracy", {}).get("by_type", {}).get(qtype))

    def recall(key: str) -> Any:
        return lambda s: _pct(s.get("session_recall", {}).get(key))

    def cost(key: str, fmt: str, scale: float = 1.0) -> Any:
        def pick(s: dict[str, Any]) -> str:
            value = s["cost"].get(key)
            return "—" if value is None else fmt.format(value * scale)
        return pick

    def recall_type(qtype: str) -> Any:
        return lambda s: _pct(
            s.get("session_recall", {}).get("all@10_by_type", {}).get(qtype)
        )

    rows = [header, rule, line("Questions", lambda s: str(s["answered"]))]
    if any(s["judged"] for s in summaries):
        rows += [
            line("Questions judged", lambda s: str(s["judged"])),
            line("**QA accuracy (overall)**", acc("overall")),
            line("QA accuracy (task-averaged)", acc("task_averaged")),
            *[line(f"  {t}", by_type(t)) for t in QUESTION_TYPES],
            line("  abstention subset", acc("abstention")),
            line("Context tokens per question", cost("mean_context_tokens", "{:,.0f}")),
        ]
    if any("session_recall" in s for s in summaries):
        rows += [
            *[line(f"Recall, any evidence session @{k}", recall(f"any@{k}")) for k in RECALL_AT],
            *[line(f"Recall, all evidence sessions @{k}", recall(f"all@{k}")) for k in RECALL_AT],
            *[line(f"  all @10: {t}", recall_type(t)) for t in QUESTION_TYPES],
            line("**All evidence turns in the answerer's context**",
                 recall("evidence_turns_all@context")),
            line("Share of evidence turns in context", recall("evidence_turns_share@context")),
            line("All evidence sessions in the answerer's context", recall("all@context")),
            line("Turns in context", cost("mean_context_turns", "{:.1f}")),
            line("Context size (est. tokens)", cost("mean_context_tokens_estimate", "{:,.0f}")),
            line("Ingest per question (s)", cost("mean_ingest_seconds", "{:.2f}")),
            line("Retrieve per question (ms)", cost("mean_retrieve_seconds", "{:.0f}", 1000.0)),
        ]
    return "\n".join(rows)


def cmd_report(args: argparse.Namespace) -> int:
    print(render_report([summarize(Path(run)) for run in args.runs]))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.longmemeval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ingest, retrieve, answer and optionally judge")
    run.add_argument("--system", required=True, choices=[*SYSTEMS, "oracle"])
    run.add_argument("--data", required=True, help="path to a LongMemEval JSON file")
    run.add_argument("--out", required=True, help="run directory (created; resumable)")
    run.add_argument("--answerer", default="", help="model id; omit for retrieval metrics only")
    run.add_argument("--judge", default="", help="model id for grading (official: gpt-4o)")
    run.add_argument("--top-k", type=int, default=10, help="turns given to the answerer")
    run.add_argument(
        "--token-budget", type=int, default=0,
        help="give the answerer as many top turns as fit this many tokens (overrides --top-k)",
    )
    run.add_argument(
        "--max-turn-tokens", type=int, default=0,
        help="with --token-budget: shorten longer turns to a query-focused excerpt",
    )
    run.add_argument(
        "--whole-turns", type=int, default=0,
        help="with --max-turn-tokens: never shorten this many best-ranked turns",
    )
    run.add_argument(
        "--prompt", default="reasoned", choices=["v1", *ANSWER_STYLES],
        help="answer prompt: Myelin's 'reasoned' or 'direct', or 'v1' (first baselines)",
    )
    run.add_argument(
        "--embedder", default="",
        help="static embedding model for hybrid search, e.g. minishlab/potion-retrieval-32M",
    )
    run.add_argument(
        "--max-answer-tokens", type=int, default=800,
        help="output cap for the answerer; raise it for models that spend "
        "output tokens on hidden reasoning",
    )
    run.add_argument(
        "--facts", default="",
        help="cache from `python -m bench.extract_facts`; facts are indexed with each turn",
    )
    run.add_argument(
        "--show-facts", action="store_true",
        help="with --facts: also print a turn's facts under it in the answerer's context",
    )
    run.add_argument("--limit", type=int, default=0, help="only the first N questions")
    run.add_argument("--workers", type=int, default=8)
    run.set_defaults(func=cmd_run)

    judge = sub.add_parser("judge", help="grade an existing run")
    judge.add_argument("--run", required=True)
    judge.add_argument("--data", required=True)
    judge.add_argument("--judge", required=True)
    judge.add_argument("--workers", type=int, default=8)
    judge.set_defaults(func=cmd_judge)

    report = sub.add_parser("report", help="print a comparison table for one or more runs")
    report.add_argument("runs", nargs="+")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
