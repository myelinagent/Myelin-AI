"""Turn search results into context for an answering model.

Two steps sit between ``search()`` and a prompt: deciding how many
results to pass on, and laying them out so the model can read them.

* :func:`fit_to_budget` keeps the best-ranked results that fit a token
  budget. A count such as ``top_k=10`` means very different amounts of
  text depending on how long the memories are; a budget does not.
* :func:`format_context` renders results oldest first, each group
  headed by its date, so the model can reason about order and recency.

* :func:`excerpt` cuts a long memory down to the passages that mention
  the query, so one long memory does not crowd out several short ones.

Token counts are estimated at four characters per token, which is close
enough for budgeting English text without a tokenizer dependency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime

from myelin.types import SearchResult

_CHARS_PER_TOKEN = 4
# Metadata keys that mark memories as belonging to one conversation.
_SESSION_KEYS = ("session_id", "_session_id")


# Passage boundaries: line breaks, and sentence ends followed by a space.
_PASSAGE_BREAK = re.compile(r"\n+|(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_COMMON_WORDS = frozenset(
    "a an the is are was were be been being of in on at to for with and or not "
    "what who where when why how do does did this that these those i you he she "
    "it we they me him her us them my your his its our their s have has had".split()
)
_ELLIPSIS = " … "


def estimate_tokens(text: str) -> int:
    """Approximate token count of ``text`` (never less than 1)."""
    return len(text) // _CHARS_PER_TOKEN + 1


def _stems(text: str) -> set[str]:
    """Crude word stems: lowercase words cut to six letters, so that
    "graduate", "graduated" and "graduation" compare equal."""
    return {w[:6] for w in _WORD.findall(text.lower()) if w not in _COMMON_WORDS}


def excerpt(text: str, query: str, max_tokens: int) -> str:
    """Shorten ``text`` to about ``max_tokens``, keeping what ``query`` is about.

    Text already within the limit is returned unchanged. Otherwise the
    text is split into passages (lines and sentences); passages sharing
    the most words with the query are kept, in their original order, and
    each gap is marked with an ellipsis. When no passage shares a word
    with the query the opening of the text is kept.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    limit = max_tokens * _CHARS_PER_TOKEN
    passages = [p for p in _PASSAGE_BREAK.split(text) if p.strip()]
    wanted = _stems(query)
    scored = [(len(wanted & _stems(p)), i) for i, p in enumerate(passages)]
    if not passages or max(score for score, _ in scored) == 0:
        return text[:limit].rstrip() + "…"

    chosen: set[int] = set()
    used = 0
    for score, i in sorted(scored, key=lambda pair: (-pair[0], pair[1])):
        if score == 0:
            break
        size = len(passages[i])
        if chosen and used + size > limit:
            continue
        chosen.add(i)
        used += size
    # Spend what is left on the passages around the matches, for context.
    for i in sorted(chosen):
        for neighbour in (i + 1, i - 1):
            if 0 <= neighbour < len(passages) and neighbour not in chosen:
                size = len(passages[neighbour])
                if used + size <= limit:
                    chosen.add(neighbour)
                    used += size

    parts: list[str] = []
    previous = -1
    for i in sorted(chosen):
        if parts and i != previous + 1:
            parts.append(_ELLIPSIS)
        elif parts:
            parts.append(" ")
        parts.append(passages[i][:limit])
        previous = i
    body = "".join(parts)
    lead = "… " if min(chosen) > 0 else ""
    tail = " …" if max(chosen) < len(passages) - 1 else ""
    return f"{lead}{body}{tail}"


def shown_text(result: SearchResult) -> str:
    """What a result contributes to context: its excerpt if it has one,
    otherwise the whole memory."""
    return result.excerpt if result.excerpt is not None else result.memory.content


def with_excerpts(
    results: Iterable[SearchResult],
    query: str,
    max_tokens: int,
    *,
    keep_whole: int = 0,
) -> list[SearchResult]:
    """Attach a query-focused excerpt to each result longer than
    ``max_tokens``, except the first ``keep_whole`` (the best ranked),
    which are passed through complete. Memories themselves are left
    untouched."""
    out: list[SearchResult] = []
    for rank, result in enumerate(results):
        content = result.memory.content
        if rank >= keep_whole and estimate_tokens(content) > max_tokens:
            result = replace(result, excerpt=excerpt(content, query, max_tokens))
        out.append(result)
    return out


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
        cost = estimate_tokens(shown_text(result))
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
        lines = "\n".join(shown_text(result) for _, result in members)
        blocks.append(f"### {_stamp(when)}\n{lines}")
    return "\n\n".join(blocks)


__all__ = [
    "estimate_tokens",
    "excerpt",
    "fit_to_budget",
    "format_context",
    "shown_text",
    "with_excerpts",
]
