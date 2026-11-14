"""Let outcome feedback from ``cite()`` influence ranking.

``cite(memory_id, answer_correct=...)`` records whether a memory that
was shown to the answerer led to a right or a wrong answer. Until now
that record only fed the health score that drives forgetting. Here it
also nudges ranking: among memories that are already relevant to the
query, ones with a record of helping move up and ones with a record of
misleading move down.

The design follows what the literature found to work (MemRL, 2026):

* **Relevance gates, feedback re-orders.** Only a pool of the most
  relevant candidates is re-ranked. Feedback never pulls in a memory the
  query did not match; ranking by usefulness alone was unstable.
* **Standardised blend.** Relevance scores are converted to z-scores
  within the pool and the feedback term is added on that scale, so the
  blend means the same thing for BM25 scores and fused scores.
* **Evidence before effect.** A single cite proves little. The feedback
  term is scaled by how many cites a memory has, reaching full strength
  at ``FULL_CONFIDENCE_CITES``; an uncited memory is left exactly where
  relevance put it.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pstdev

from myelin.types import SearchResult

# How many relevance standard deviations a fully trusted, always-helpful
# memory gains (and an always-misleading one loses).
UTILITY_WEIGHT: float = 1.0
# Cites needed before a memory's record counts at full strength.
FULL_CONFIDENCE_CITES: int = 5
# The pool re-ranked is this many times the number of results wanted.
POOL_FACTOR: int = 3
POOL_MIN_EXTRA: int = 10


def pool_size(top_k: int) -> int:
    """How many candidates to fetch so that ``top_k`` can be re-ranked."""
    return max(top_k * POOL_FACTOR, top_k + POOL_MIN_EXTRA)


def utility(correct: int, incorrect: int, unused: int = 0) -> float:
    """Feedback record as a number in (-1, 1): positive for memories that
    mostly led to right answers, negative for wrong, 0 with no record.

    The success rate is smoothed (one imaginary right and one imaginary
    wrong answer) and scaled by how much evidence there is.
    """
    correct = max(0, correct)
    # Shown for a question and not used counts against a memory's rank
    # the same way a wrong answer does: it took a place something
    # useful could have had.
    incorrect = max(0, incorrect) + max(0, unused)
    total = correct + incorrect
    if total == 0:
        return 0.0
    rate = (correct + 1) / (total + 2)
    confidence = min(1.0, total / FULL_CONFIDENCE_CITES)
    return (2.0 * rate - 1.0) * confidence


def rerank_by_cites(
    results: Sequence[SearchResult], top_k: int, *, weight: float = UTILITY_WEIGHT
) -> list[SearchResult]:
    """Re-order relevance-ranked ``results`` using each memory's cite
    record, and return the best ``top_k``.

    Results keep their relevance ``score``. With no cited memory in the
    pool the input order is returned untouched.
    """
    pool = list(results)
    utilities = [
        utility(
            r.memory.cite_correct_count,
            r.memory.cite_incorrect_count,
            r.memory.shown_unused_count,
        )
        for r in pool
    ]
    if not any(utilities):
        return pool[:top_k]
    scores = [r.score for r in pool]
    mean = fmean(scores)
    spread = pstdev(scores) or 1.0
    blended = [
        ((score - mean) / spread + weight * u, -rank)
        for rank, (score, u) in enumerate(zip(scores, utilities, strict=True))
    ]
    order = sorted(range(len(pool)), key=lambda i: blended[i], reverse=True)
    return [pool[i] for i in order[:top_k]]


__all__ = [
    "FULL_CONFIDENCE_CITES",
    "UTILITY_WEIGHT",
    "pool_size",
    "rerank_by_cites",
    "utility",
]
