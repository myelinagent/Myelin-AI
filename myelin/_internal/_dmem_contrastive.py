"""Helpers for the contrastive_v1 utility backend in DMemGate.

Extracted from dmem_gate.py to keep that module under the 500-LOC CI
gate. The public surface is intentionally small:

- ``score_via_contrastive(gate, content, neighbors)`` — does the
  lazy-load, soft-fallback on encoder failure, and inference call.
- ``summarize_neighbors(neighbors)`` — short text fingerprint passed
  into the encoder's ``neighbor_context`` slot.

Both functions take the gate as their first argument so they can
mutate the gate's ``_contrastive_encoder`` / ``_contrastive_fallback_active``
attributes WITHOUT being instance methods on the dataclass.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from myelin._internal.dmem_gate import DMemGate

UtilityClass = Literal["transient", "short_term", "persistent"]
logger = logging.getLogger(__name__)


def score_via_contrastive(
    gate: "DMemGate",
    content: str,
    neighbors: Sequence[Sequence[float]],
) -> tuple[UtilityClass, float, str, bool]:
    """Return ``(class, score, reason, cached)`` from the contrastive
    encoder.

    Implements the lazy-load + soft-fallback semantics described in
    SPARC EC-4: if the encoder cannot be loaded (weights missing, SHA
    mismatch, torch import failed), flip ``gate`` into permanent
    llm_judge-fallback for the rest of the process lifetime and log a
    single warning.
    """
    if gate._contrastive_fallback_active:
        return gate._judge_utility(content)

    if gate._contrastive_encoder is None:
        try:
            from myelin._internal.contrastive_encoder import ContrastiveEncoder

            gate._contrastive_encoder = ContrastiveEncoder.get_instance()
        except Exception as e:
            logger.warning(
                "DMemGate contrastive_v1 encoder load failed (%s: %s); "
                "falling back to llm_judge for the rest of this run",
                type(e).__name__,
                e,
            )
            gate._contrastive_fallback_active = True
            if gate.llm_client is None:
                try:
                    gate.llm_client = gate._lazy_openai_client()
                except Exception as e2:  # pragma: no cover
                    logger.error(
                        "DMemGate fallback to llm_judge ALSO failed to build "
                        "OpenAI client (%s); returning soft-fallback decision",
                        e2,
                    )
                    fb_cls, fb_score, fb_reason = gate.llm_error_fallback
                    return fb_cls, fb_score, fb_reason, False
            return gate._judge_utility(content)

    ctx = summarize_neighbors(neighbors, max_chars=480)
    try:
        util, cls = gate._contrastive_encoder.score(
            content=content, neighbor_context=ctx
        )
    except Exception as e:
        logger.warning(
            "DMemGate contrastive_v1 score failed at inference (%s); "
            "soft-falling back to llm_judge for this turn",
            type(e).__name__,
        )
        return gate._judge_utility(content)
    reason = f"contrastive_v1:cls={cls}:util={util:.3f}"
    return cls, util, reason, False


def summarize_neighbors(
    neighbors: Sequence[Sequence[float]],
    *,
    max_chars: int = 480,
) -> str:
    """Compress the embedding-list neighbor window into a short text
    fingerprint the contrastive encoder ingests as context.

    The encoder doesn't need the raw float vectors (training data
    didn't have them either); it needs *something* that distinguishes
    "this user has talked about X" from "fresh topic." We summarize
    the window via its size + the first few coordinates of the mean
    vector, which is enough for the encoder to condition on neighbor
    density without leaking unbounded text.
    """
    if not neighbors:
        return "[no_neighbors]"
    n = len(neighbors)
    dim = len(neighbors[0]) if neighbors[0] else 0
    if dim == 0:
        return f"[neighbors:{n}]"
    coords = min(8, dim)
    mean = [sum(v[i] for v in neighbors) / n for i in range(coords)]
    parts = [f"[neighbors:{n}d{dim}:mean="]
    parts.extend(f"{m:+.3f}" for m in mean)
    parts.append("]")
    s = "".join(parts)
    return s[:max_chars]


__all__ = [
    "score_via_contrastive",
    "summarize_neighbors",
]
