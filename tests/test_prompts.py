"""The answer-prompt builder."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from myelin import MemoryClient, build_answer_prompt, format_context
from myelin.prompts import ANSWER_STYLES


def test_prompt_contains_history_question_and_date():
    prompt = build_answer_prompt(
        "### 2023-05-01 10:00\nuser: I adopted a greyhound.",
        "  What dog do I have?  ",
        now=datetime(2023, 6, 1, 9, 0, tzinfo=UTC),
    )
    assert "user: I adopted a greyhound." in prompt
    assert "Question: What dog do I have?\n" in prompt
    assert "Current date: 2023-06-01 (Thu) 09:00" in prompt


def test_reasoned_style_asks_for_notes_then_a_marked_answer():
    prompt = build_answer_prompt("h", "q", now="2023/06/01")
    assert 'starting with "Answer:"' in prompt
    assert prompt.rstrip().endswith('"Answer:".')


def test_direct_style_ends_ready_for_the_answer():
    prompt = build_answer_prompt("h", "q", now="2023/06/01", style="direct")
    assert prompt.endswith("Question: q\nAnswer:")
    assert "short notes" not in prompt


@pytest.mark.parametrize("style", sorted(ANSWER_STYLES))
def test_every_style_carries_the_rules(style):
    prompt = build_answer_prompt("h", "q", now="today", style=style)
    for phrase in (
        "most recent conversation holds the current value",
        "list every relevant item",
        "relative to that conversation's date",
        "Never refuse such a request",
        "say that it was not mentioned",
    ):
        assert phrase in prompt


def test_now_accepts_text_and_defaults_to_the_present():
    assert "Current date: 2023/05/30 (Tue) 22:28" in build_answer_prompt(
        "h", "q", now="2023/05/30 (Tue) 22:28"
    )
    assert f"Current date: {datetime.now().astimezone():%Y-%m-%d}" in build_answer_prompt("h", "q")


def test_empty_history_is_stated_not_left_blank():
    assert "(no relevant history found)" in build_answer_prompt("  ", "q", now="today")


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="unknown style 'chatty'"):
        build_answer_prompt("h", "q", style="chatty")


def test_braces_in_history_and_question_are_left_alone():
    prompt = build_answer_prompt('{"a": 1} {history}', "what is {x}?", now="today")
    assert '{"a": 1} {history}' in prompt
    assert "what is {x}?" in prompt


def test_recall_format_and_prompt_compose():
    c = MemoryClient()
    try:
        c.add("Scott lived in Paris", t_valid=datetime(2023, 3, 1, tzinfo=UTC))
        c.add("Scott moved to Berlin", t_valid=datetime(2024, 3, 1, tzinfo=UTC))
        prompt = build_answer_prompt(
            format_context(c.recall("Where does Scott live?")),
            "Where does Scott live?",
            now=datetime(2024, 6, 1, tzinfo=UTC),
        )
        assert prompt.index("Paris") < prompt.index("Berlin")
    finally:
        c.close()
