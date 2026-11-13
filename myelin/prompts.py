"""Prompts for answering a question from recalled memories.

Retrieval decides what an answering model sees; the prompt decides what
it does with it. :func:`build_answer_prompt` wraps recalled context (see
:func:`myelin.format_context`) and a question in instructions for the
situations where models most often go wrong with memory:

* a fact that changed over time — the latest statement is the current one;
* totals, counts and orderings — list every relevant item before answering;
* dates — a conversation's date is when it took place, and "last
  Saturday" inside it has to be worked out from that date;
* requests for advice — use what the history shows about the person
  instead of declining because no ready-made answer is stored;
* missing information — say so, but only when the specific thing asked
  for is absent.

Two styles are provided. ``"reasoned"`` asks for brief working before the
answer and is the stronger choice for small or mid-sized models.
``"direct"`` asks for the answer only, which suits models that reason
internally and callers that need short output.
"""

from __future__ import annotations

from datetime import datetime

_RULES = """\
- The history is grouped by conversation, each headed by the date it took place. \
Statements such as "yesterday" or "last week" inside a conversation are relative to \
that conversation's date; work out the actual dates when the question depends on them.
- When the history gives different values for the same thing at different times, the \
most recent conversation holds the current value. Earlier values are history: mention \
them only if the question asks about the past or about what changed.
- For a total, a count, a comparison or an ordering, first list every relevant item \
from every conversation, with its date or amount, then work out the answer from that \
list. Do not stop at the first match.
- If the question asks for a suggestion, recommendation or advice, give one. Base it \
on what the history shows about the user's preferences, habits, constraints and past \
experiences, and say which of those you are relying on. Never refuse such a request \
for lack of information.
- If the question asks for a specific fact that the history does not contain, say \
that it was not mentioned, and say what related information was mentioned instead. \
Do not guess."""

_REASONED = """\
You are answering a question for a user, using excerpts from their past conversations \
with an assistant.

Rules:
{rules}

History:

{history}

Current date: {now}
Question: {question}

First write short notes: the facts from the history that bear on the question, with \
their dates, and any calculation. Then give your final answer on a new line starting \
with "Answer:"."""

_DIRECT = """\
You are answering a question for a user, using excerpts from their past conversations \
with an assistant.

Rules:
{rules}

History:

{history}

Current date: {now}
Question: {question}
Answer:"""

ANSWER_STYLES = {"reasoned": _REASONED, "direct": _DIRECT}


def build_answer_prompt(
    history: str,
    question: str,
    *,
    now: datetime | str | None = None,
    style: str = "reasoned",
) -> str:
    """Return a prompt asking a model to answer ``question`` from ``history``.

    ``history`` is recalled context, normally the output of
    :func:`myelin.format_context`. ``now`` is the moment the question is
    asked (a datetime, or text to show as-is); it defaults to the current
    time. ``style`` is ``"reasoned"`` or ``"direct"``.
    """
    if style not in ANSWER_STYLES:
        raise ValueError(
            f"unknown style {style!r}; expected one of {sorted(ANSWER_STYLES)}"
        )
    if now is None:
        now = datetime.now().astimezone()
    if isinstance(now, datetime):
        now = now.strftime("%Y-%m-%d (%a) %H:%M")
    return ANSWER_STYLES[style].format(
        rules=_RULES,
        history=history.strip() or "(no relevant history found)",
        now=now,
        question=question.strip(),
    )


__all__ = ["ANSWER_STYLES", "build_answer_prompt"]
