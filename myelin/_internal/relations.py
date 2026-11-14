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
            try:
                import numpy as np
            except ImportError:  # similarity edges need numpy
                np = None  # type: ignore[assignment]
            if np is not None:
                matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
                row_of = {memory_id: row for row, memory_id in enumerate(ids)}
                for source in used:
                    row = row_of.get(source.id)
                    if row is None:
                        continue
                    scores = matrix @ matrix[row]
                    scores[row] = -1.0
                    count = min(SIMILAR_NEIGHBOURS, len(ids) - 1)
                    nearest = np.argpartition(-scores, count - 1)[:count]
                    strength = _use(source)
                    for neighbour in nearest.tolist():
                        if scores[neighbour] >= MIN_SIMILARITY:
                            support[ids[neighbour]] += strength * float(scores[neighbour])
    return dict(support)


__all__ = ["MIN_SIMILARITY", "SESSION_WINDOW", "SIMILAR_NEIGHBOURS", "neighbour_support"]
