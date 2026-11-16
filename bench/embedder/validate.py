"""Check generated conversations against the format the trainer reads.

    python -m bench.embedder.validate DIR [--briefs briefs/batch00.json]

A conversation file is one JSON object:

    {"id": "c0001",
     "sessions": [{"date": "2023-04-02",
                   "turns": [{"id": "s1t1", "speaker": "Mei", "text": "..."}]}],
     "questions": [{"question": "...", "answer": "...", "evidence": ["s2t5"],
                    "kind": "fact|multi|time|inference"}]}

Prints one line per problem and exits 1 if there are any.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

KINDS = {"fact", "multi", "time", "inference"}
TURN_ID = re.compile(r"s(\d+)t(\d+)")


def problems(conversation: Any, brief: dict | None = None) -> list[str]:
    found: list[str] = []
    if not isinstance(conversation, dict):
        return ["not a JSON object"]
    sessions = conversation.get("sessions")
    questions = conversation.get("questions")
    if not isinstance(sessions, list) or len(sessions) < 3:
        return ["needs at least 3 sessions"]
    if not isinstance(questions, list) or len(questions) < 10:
        return ["needs at least 10 questions"]
    texts: dict[str, str] = {}
    for number, session in enumerate(sessions, start=1):
        turns = session.get("turns") if isinstance(session, dict) else None
        if not isinstance(turns, list) or len(turns) < 6:
            found.append(f"session {number}: needs at least 6 turns")
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(session.get("date", ""))):
            found.append(f"session {number}: date must be YYYY-MM-DD")
        for position, turn in enumerate(turns, start=1):
            expected = f"s{number}t{position}"
            if not isinstance(turn, dict) or turn.get("id") != expected:
                found.append(f"session {number}: turn {position} must have id {expected}")
                continue
            text = turn.get("text")
            if not isinstance(text, str) or len(text.split()) < 2 or not turn.get("speaker"):
                found.append(f"{expected}: needs a speaker and a text of two words or more")
                continue
            texts[expected] = text
    if brief is not None:
        if conversation.get("id") != brief["id"]:
            found.append(f"id must be {brief['id']}")
        if [s.get("date") for s in sessions] != brief["session_dates"]:
            found.append("session dates must be the brief's, in order")
        names = {p["name"] for p in brief["people"]}
        spoken = {t.get("speaker") for s in sessions for t in s.get("turns", [])}
        if spoken - names:
            found.append(f"unknown speakers: {sorted(map(str, spoken - names))}")
    seen: set[str] = set()
    for number, question in enumerate(questions, start=1):
        label = f"question {number}"
        if not isinstance(question, dict):
            found.append(f"{label}: must be an object")
            continue
        asked, evidence = question.get("question"), question.get("evidence")
        if not isinstance(asked, str) or len(asked.split()) < 4:
            found.append(f"{label}: question too short")
        elif asked in seen:
            found.append(f"{label}: repeated question")
        else:
            seen.add(asked)
        if not isinstance(question.get("answer"), str) or not question["answer"].strip():
            found.append(f"{label}: needs an answer")
        if question.get("kind") not in KINDS:
            found.append(f"{label}: kind must be one of {sorted(KINDS)}")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4:
            found.append(f"{label}: evidence must list 1 to 4 turn ids")
            continue
        missing = [e for e in evidence if e not in texts]
        if missing:
            found.append(f"{label}: evidence names turns that do not exist: {missing}")
        if question.get("kind") == "multi" and len(set(evidence)) < 2:
            found.append(f"{label}: a multi question needs two or more evidence turns")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--briefs", default="", help="a briefs file these were written from")
    args = parser.parse_args(argv)
    briefs = {}
    if args.briefs:
        briefs = {b["id"]: b for b in json.loads(Path(args.briefs).read_text())}
    bad = 0
    files = sorted(Path(args.directory).glob("c*.json"))
    for file in files:
        try:
            loaded = json.loads(file.read_text())
        except json.JSONDecodeError as exc:
            found = [f"not valid JSON: {exc}"]
        else:
            found = problems(loaded, briefs.get(file.stem))
        for line in found:
            print(f"{file.name}: {line}")
        bad += bool(found)
    absent = sorted(set(briefs) - {f.stem for f in files})
    if absent:
        print(f"not written yet: {' '.join(absent)}")
    print(f"{len(files) - bad} of {len(files)} files are valid", file=sys.stderr)
    return 1 if bad or absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
