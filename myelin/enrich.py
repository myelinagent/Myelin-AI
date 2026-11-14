"""Extract facts from a conversation when it is written.

A question is often worded unlike the conversation that answers it, and
a remark such as "I went last Saturday" names no date. Extracting short,
standalone, dated facts from a conversation and indexing them *alongside*
each message closes both gaps: the message becomes findable by what it
means, and its relative dates are resolved against the conversation's
date while that date is known.

The facts are extra index text, not replacements. Each message is still
stored and returned whole (``add(..., index_text=...)``); indexing facts
in place of the original text is known to lose answers.

This costs one model call per conversation, so it is opt-in. Bring any
callable that takes a prompt and returns text::

    from myelin import MemoryClient
    from myelin.enrich import extract_facts

    turns = [("user", "I adopted a greyhound last Saturday."),
             ("assistant", "Congratulations!")]
    facts = extract_facts(turns, when=session_date, llm=my_llm)
    client.add_many(
        {"content": f"{role}: {text}", "t_valid": session_date,
         "index_text": " ".join(facts.get(i, []))}
        for i, (role, text) in enumerate(turns)
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime

# Replies from the other party are context for reading the user's
# messages; they are shortened to keep the call cheap.
_OTHER_PARTY_CHARS = 300
_MAX_FACTS_PER_TURN = 6
_MAX_FACT_CHARS = 300

_PROMPT = """\
Below is one conversation between a user and an assistant. It took place on {when}.

For each numbered USER message that says something about the user, write short \
standalone facts. Cover what they did or plan to do, what they bought or own, their \
preferences and opinions, people and places in their life, amounts, counts, durations, \
and anything that changed.

Rules:
- Write in the third person: "The user ...".
- Each fact must make sense on its own, without the conversation.
- Keep names, numbers, amounts and titles exactly as stated.
- Turn relative dates ("yesterday", "last week", "two months ago") into calendar dates, \
using the conversation date above. Say the date in the fact.
- Do not add anything the message does not say. Skip messages that say nothing about \
the user.

Reply with JSON only, mapping message numbers to lists of facts, for example:
{{"0": ["The user adopted a greyhound on 2023-05-13."], "4": ["The user prefers tea to coffee."]}}
If no message says anything about the user, reply {{}}.

Conversation:
{conversation}"""


_PEOPLE_PROMPT = """\
Below is one conversation between {names}. It took place on {when}.

For each numbered message that says something about any of them, write short \
standalone facts. Cover what they did or plan to do, what they bought or own, their \
preferences and opinions, people and places in their life, amounts, counts, durations, \
and anything that changed.

Rules:
- Name the person each fact is about: "{first} ...".
- Each fact must make sense on its own, without the conversation.
- Keep names, numbers, amounts and titles exactly as stated.
- Turn relative dates ("yesterday", "last week", "two months ago") into calendar dates, \
using the conversation date above. Say the date in the fact.
- Do not add anything the message does not say. Skip messages that say nothing about \
any of them.

Reply with JSON only, mapping message numbers to lists of facts, for example:
{{"0": ["{first} adopted a greyhound on 2023-05-13."], "4": ["{first} prefers tea to coffee."]}}
If no message says anything about them, reply {{}}.

Conversation:
{conversation}"""


def build_extraction_prompt(
    turns: Sequence[tuple[str, str]],
    when: datetime | str,
    people: Sequence[str] | None = None,
) -> str:
    """Prompt asking a model for facts stated in ``turns``.

    ``turns`` is ``(role, text)`` pairs in order; ``when`` is the date
    the conversation took place. By default the conversation is between
    a ``user`` and an assistant and the facts are about the user. For a
    conversation between people, pass their names as ``people`` and use
    those names as the roles; facts are then taken from every message
    and name the person they are about.
    """
    if isinstance(when, datetime):
        when = when.strftime("%A %Y-%m-%d")
    if people:
        names = list(people)
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        conversation = "\n".join(
            f"[{number}] {role}: {text}" for number, (role, text) in enumerate(turns)
        )
        return _PEOPLE_PROMPT.format(
            names=joined, first=names[0], when=when, conversation=conversation
        )
    lines = []
    for number, (role, text) in enumerate(turns):
        if role != "user" and len(text) > _OTHER_PARTY_CHARS:
            text = text[:_OTHER_PARTY_CHARS].rstrip() + " …"
        lines.append(f"[{number}] {role.upper()}: {text}")
    return _PROMPT.format(when=when, conversation="\n".join(lines))


def parse_facts(response: str, n_turns: int | None = None) -> dict[int, list[str]]:
    """Read a model's reply into ``{message number: [facts]}``.

    Tolerant of text around the JSON. Anything that is not a list of
    non-empty strings under a valid message number is dropped; an
    unreadable reply yields no facts.
    """
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(response[start : end + 1])
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    facts: dict[int, list[str]] = {}
    for key, value in data.items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        if number < 0 or (n_turns is not None and number >= n_turns):
            continue
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        kept = [
            v.strip()[:_MAX_FACT_CHARS]
            for v in value
            if isinstance(v, str) and v.strip()
        ][:_MAX_FACTS_PER_TURN]
        if kept:
            facts[number] = kept
    return facts


def extract_facts(
    turns: Sequence[tuple[str, str]],
    when: datetime | str,
    llm: Callable[[str], str],
    people: Sequence[str] | None = None,
) -> dict[int, list[str]]:
    """Ask ``llm`` for the facts stated in one conversation.

    Returns ``{message number: [facts]}`` for the messages that yielded
    any: the user's by default, everyone's when ``people`` names the
    participants of a conversation between people (see
    :func:`build_extraction_prompt`). Errors from ``llm`` propagate; a
    reply that cannot be read yields an empty result.
    """
    speaks = (lambda role: True) if people else (lambda role: role == "user")
    if not any(speaks(role) and text.strip() for role, text in turns):
        return {}
    return parse_facts(llm(build_extraction_prompt(turns, when, people)), len(turns))


__all__ = ["build_extraction_prompt", "extract_facts", "parse_facts"]
