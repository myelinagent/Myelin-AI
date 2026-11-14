"""Extract facts for every session of a LongMemEval file, once, to a cache.

    python -m bench.extract_facts --data longmemeval_s_cleaned.json \\
        --out facts.jsonl --model gpt-4o-mini

Sessions repeat across questions, so each distinct session is sent to the
model once. The cache is append-only and resumable; pass it to
``python -m bench.longmemeval run --facts facts.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bench.llm import ChatClient, LLMError
from myelin.enrich import build_extraction_prompt, parse_facts


def session_key(date: str, session: list[dict[str, Any]]) -> str:
    """Identity of a session: its date and its messages."""
    payload = json.dumps([date, [(t["role"], t["content"]) for t in session]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_cache(path: Path) -> dict[str, dict[int, list[str]]]:
    cache: dict[str, dict[int, list[str]]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = {int(k): v for k, v in row["facts"].items()}
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.extract_facts")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, help="cache file (JSONL, resumable)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="only the first N sessions")
    args = parser.parse_args(argv)

    out = Path(args.out)
    done = load_cache(out)
    sessions: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for entry in json.loads(Path(args.data).read_text()):
        for date, session in zip(
            entry["haystack_dates"], entry["haystack_sessions"], strict=True
        ):
            sessions.setdefault(session_key(date, session), (date, session))
    todo = [k for k in sessions if k not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(sessions)} distinct sessions, {len(todo)} to extract", file=sys.stderr)

    client = ChatClient(args.model)
    lock = threading.Lock()
    usage = {"prompt": 0, "completion": 0}

    def work(key: str) -> None:
        date, session = sessions[key]
        turns = [(t["role"], t["content"]) for t in session]
        if not any(role == "user" and text.strip() for role, text in turns):
            facts: dict[int, list[str]] = {}
        else:
            reply = client.complete(build_extraction_prompt(turns, date), max_tokens=900)
            facts = parse_facts(reply.text, len(turns))
            with lock:
                usage["prompt"] += reply.prompt_tokens
                usage["completion"] += reply.completion_tokens
        with lock, out.open("a") as handle:
            handle.write(json.dumps({"key": key, "facts": facts}) + "\n")

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, key) for key in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except LLMError as exc:
                failed += 1
                print(f"  {exc}", file=sys.stderr)
            if count % 500 == 0 or count == len(futures):
                print(f"  extracted {count}/{len(futures)}", file=sys.stderr, flush=True)
    print(
        f"tokens: {usage['prompt']:,} in, {usage['completion']:,} out; {failed} failed",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
