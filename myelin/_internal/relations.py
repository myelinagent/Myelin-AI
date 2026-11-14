"""How memories relate to each other, for grading them together.

A memory's health should not be judged alone. A message that was never
cited itself but sits next to ones that keep proving useful (the same
conversation, or saying nearly the same thing) is more likely to be
needed than an unrelated message of the same age. This module turns
"related to useful memories" into a number that the health equation
adds to a memory's own record of use.

Two kinds of relation are used, both cheap to derive from what is
already stored:

* **Conversation**: memories that share ``metadata["session_id"]`` and
  are close in order (``metadata["turn"]`` when present, otherwise
  time and insertion order). Weight falls off with distance.
* **Similarity**: when embedding vectors are stored, the nearest
  neighbours of each useful memory, weighted by cosine similarity.

Support flows *from* memories with a record of use *to* their
neighbours. A store with no cites has no support anywhere, so health
there is exactly what it was without this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import log1p
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from myelin.types import Memory

# How far along a conversation support reaches, in messages.
SESSION_WINDOW: int = 3
# Nearest neighbours by embedding that receive support, and how alike
# they must be.
SIMILAR_NEIGHBOURS: int = 8
MIN_SIMILARITY: float = 0.5


def _use(memory: Memory) -> float:
    """A memory's own record of use, as the health equation counts it."""
    return log1p(max(0, memory.cite_correct_count))


def _conversation_order(memories: Sequence[Memory]) -> dict[str, list[Memory]]:
    sessions: dict[str, list[tuple[Any, int, Memory]]] = defaultdict(list)
    for position, memory in enumerate(memories):
        session = memory.metadata.get("session_id")
        if not isinstance(session, str):
            continue
        turn = memory.metadata.get("turn")
        key = (0, turn) if isinstance(turn, int) else (1, memory.t_valid)
        sessions[session].append((key, position, memory))
    return {
        session: [m for _, _, m in sorted(rows, key=lambda row: (row[0], row[1]))]
        for session, rows in sessions.items()
    }


def _most_similar(
    source: str, ids: Sequence[str], vectors: Mapping[str, Any]
) -> list[tuple[str, float]]:
    """The memories most like ``source`` by cosine similarity (vectors
    are unit length), at least ``MIN_SIMILARITY`` alike."""
    try:
        import numpy as np
    except ImportError:
        target = vectors[source]
        scored = [
            (sum(a * b for a, b in zip(vectors[i], target, strict=True)), i)
            for i in ids
            if i != source
        ]
    else:
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        scores = matrix @ np.asarray(vectors[source], dtype=np.float32)
        # Only the best few matter; do not sort a large store in full.
        keep = min(len(ids), SIMILAR_NEIGHBOURS + 1)
        best = np.argpartition(-scores, keep - 1)[:keep].tolist()
        scored = [(float(scores[row]), ids[row]) for row in best if ids[row] != source]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [
        (memory_id, score)
        for score, memory_id in scored[:SIMILAR_NEIGHBOURS]
        if score >= MIN_SIMILARITY
    ]


def neighbour_support(
    memories: Sequence[Memory],
    vectors: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Support each memory receives from related memories that were used.

    ``vectors`` maps memory id to a unit-length embedding (any sequence
    of floats, or a numpy row); omit it to use conversation order only.
    Returns raw support, 0.0 for memories with no useful neighbour.
    """
    support: dict[str, float] = defaultdict(float)
    used = [m for m in memories if m.cite_correct_count > 0 and m.health_state != "tombstoned"]
    if not used:
        return {}

    for ordered in _conversation_order(memories).values():
        for index, source in enumerate(ordered):
            strength = _use(source)
            if strength <= 0:
                continue
            for offset in range(1, SESSION_WINDOW + 1):
                for neighbour in (index - offset, index + offset):
                    if 0 <= neighbour < len(ordered):
                        support[ordered[neighbour].id] += strength / (1 + offset)

    if vectors:
        ids = [m.id for m in memories if m.id in vectors]
        if len(ids) > 1:
            for source in used:
                if source.id not in vectors:
                    continue
                strength = _use(source)
                for neighbour, similarity in _most_similar(source.id, ids, vectors):
                    support[neighbour] += strength * similarity
    return dict(support)


__all__ = ["MIN_SIMILARITY", "SESSION_WINDOW", "SIMILAR_NEIGHBOURS", "neighbour_support"]
