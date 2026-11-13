"""Turn search results into context for an answering model.

Two steps sit between ``search()`` and a prompt: deciding how many
results to pass on, and laying them out so the model can read them.

* :func:`fit_to_budget` keeps the best-ranked results that fit a token
  budget. A count such as ``top_k=10`` means very different amounts of
  text depending on how long the memories are; a budget does not.
* :func:`format_context` renders results oldest first, each group
  headed by its date, so the model can reason about order and recency.

Token counts are estimated at four characters per token, which is close
enough for budgeting English text without a tokenizer dependency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from myelin.types import SearchResult

_CHARS_PER_TOKEN = 4
# Metadata keys that mark memories as belonging to one conversation.
_SESSION_KEYS = ("session_id", "_session_id")


def estimate_tokens(text: str) -> int:
    """Approximate token count of ``text`` (never less than 1)."""
    return len(text) // _CHARS_PER_TOKEN + 1


def fit_to_budget(
    results: Iterable[SearchResult], token_budget: int
) -> list[SearchResult]:
    """Keep results, in rank order, while they fit in ``token_budget``.

    The top result is always kept, even if it alone exceeds the budget,
    so a non-empty search never yields empty context. Selection stops at
    the first result that does not fit; later, shorter results are not
    pulled forward past a better-ranked one.
    """
    kept: list[SearchResult] = []
    used = 0
    for result in results:
        cost = estimate_tokens(result.memory.content)
        if kept and used + cost > token_budget:
            break
        kept.append(result)
        used += cost
    return kept


def _session_of(result: SearchResult) -> str | None:
    for key in _SESSION_KEYS:
        value = result.memory.metadata.get(key)
        if value is not None:
            return str(value)
    return None


def _turn_of(result: SearchResult) -> int | None:
    turn = result.memory.metadata.get("turn")
    return turn if isinstance(turn, int) else None


def _stamp(when: datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M")


def format_context(results: Sequence[SearchResult]) -> str:
    """Render results as dated text, oldest first.

    Memories that share a timestamp and a session (``session_id`` in
    their metadata) are grouped under one date heading, in conversation
    order when a ``turn`` number is present and in rank order otherwise.
    Returns an empty string for no results.
    """
    groups: dict[tuple[datetime, str | None], list[tuple[int, SearchResult]]] = {}
    for rank, result in enumerate(results):
        key = (result.memory.t_valid, _session_of(result))
        groups.setdefault(key, []).append((rank, result))

    blocks: list[str] = []
    for (when, session), members in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        members.sort(
            key=lambda pair: (
                _turn_of(pair[1]) is None,
                _turn_of(pair[1]) or 0,
                pair[0],
            )
        )
        lines = "\n".join(result.memory.content for _, result in members)
        blocks.append(f"### {_stamp(when)}\n{lines}")
    return "\n\n".join(blocks)


__all__ = ["estimate_tokens", "fit_to_budget", "format_context"]
