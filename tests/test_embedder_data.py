"""The synthetic-data tools for embedder training: briefs and the format check."""

import random

from bench.embedder.briefs import brief
from bench.embedder.validate import problems


def conversation() -> dict:
    sessions = []
    for number in (1, 2, 3):
        turns = [
            {"id": f"s{number}t{t}", "speaker": "Mei" if t % 2 else "Omar",
             "text": f"Message {t} of session {number}."}
            for t in range(1, 7)
        ]
        sessions.append({"date": f"2023-04-0{number}", "turns": turns})
    questions = [
        {"question": f"What happened in message number {n}?", "answer": "Something",
         "evidence": [f"s1t{1 + n % 6}"], "kind": "fact"}
        for n in range(10)
    ]
    return {"id": "c0001", "sessions": sessions, "questions": questions}


def test_a_well_formed_conversation_passes():
    assert problems(conversation()) == []


def test_evidence_must_name_a_real_turn():
    broken = conversation()
    broken["questions"][0]["evidence"] = ["s9t9"]
    assert any("do not exist" in line for line in problems(broken))


def test_turn_ids_must_count_from_one():
    broken = conversation()
    broken["sessions"][1]["turns"][0]["id"] = "s2t7"
    assert any("must have id s2t1" in line for line in problems(broken))


def test_a_multi_question_needs_two_turns():
    broken = conversation()
    broken["questions"][0]["kind"] = "multi"
    assert any("two or more" in line for line in problems(broken))


def test_briefs_are_the_same_for_the_same_seed():
    first = [brief(i, random.Random(3)) for i in range(2)]
    again = [brief(i, random.Random(3)) for i in range(2)]
    assert first == again
    assert first[0]["people"][0]["name"] != first[0]["people"][1]["name"]
