"""Pure helpers for the DMemGate two-axis salience scorer.

Extracted from dmem_gate.py to keep that module under the 500-LOC CI
gate. These are all stateless functions / constants — no I/O, no
network calls — so they're safe to import from the `dmem_gate` module
and any external test code that wants to verify the parser / surprise
math in isolation.

Surface:

* ``_LLM_JUDGE_SYSTEM`` — system-prompt string for the gpt-4o-mini
  utility judge.
* ``parse_judge_response(raw)`` — tolerant JSON parser, falls back to
  ``transient`` / 0.0 on garbage.
* ``cosine(a, b)`` — cosine similarity between two equal-length float
  sequences.
* ``zscore_surprise(query, neighbors)`` — D-MEM §3.2 surprise signal:
  returns (raw_distance, z_score) clipped to [-3, 3].
* ``z_to_unit(z)`` — sigmoid([-3, 3]) → [0, 1] for multiplicative
  combination with utility_score.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Sequence
from typing import Literal

logger = logging.getLogger(__name__)

UtilityClass = Literal["transient", "short_term", "persistent"]


def resolve_float_env(
    env_name: str,
    default: float,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    """Read ``env_name`` as a float bounded to ``[lo, hi]``.

    Unset / unparseable / out-of-range → return default and emit a
    logger.warning. Used by the Phase G3 ablation sweep so a single
    Docker image can serve every (rpe, utility, window) cell of the
    design space without recompilation.
    """
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        logger.warning(
            "DMemGate env %s=%r unparseable; falling back to default %.3f",
            env_name, raw, default,
        )
        return default
    if not (lo <= v <= hi):
        logger.warning(
            "DMemGate env %s=%.3f outside [%.3f, %.3f]; falling back to %.3f",
            env_name, v, lo, hi, default,
        )
        return default
    return v


def resolve_int_env(
    env_name: str,
    default: int,
    *,
    lo: int = 1,
    hi: int = 4096,
) -> int:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        logger.warning(
            "DMemGate env %s=%r unparseable; falling back to default %d",
            env_name, raw, default,
        )
        return default
    if not (lo <= v <= hi):
        logger.warning(
            "DMemGate env %s=%d outside [%d, %d]; falling back to %d",
            env_name, v, lo, hi, default,
        )
        return default
    return v


_LLM_JUDGE_SYSTEM = (
    "You score whether a conversation turn is worth remembering "
    "long-term about the user.\n\n"
    "Return JSON with three keys: "
    '{"class": "transient|short_term|persistent", '
    '"score": <0.0-1.0>, "reason": "<<=20 words>"}\n\n'
    "Rules:\n"
    "- transient = chitchat, greetings, acknowledgments, time references "
    "that don't matter beyond this turn.\n"
    "- short_term = facts that matter for this session/topic but not "
    "durable across sessions (e.g. 'I'm tired tonight').\n"
    "- persistent = durable facts about the user (preferences, "
    "biographical, lasting beliefs, decisions, plans, named entities)."
)


def parse_judge_response(raw: str) -> tuple[UtilityClass, float, str]:
    """Tolerant JSON parser. Falls back to ``transient`` / 0.0 on garbage."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object embedded in extra text.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                d = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return "transient", 0.0, "parse-error"
        else:
            return "transient", 0.0, "parse-error"
    cls = str(d.get("class", "transient")).strip().lower()
    if cls not in ("transient", "short_term", "persistent"):
        cls = "transient"
    try:
        score = float(d.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    reason = str(d.get("reason", ""))[:120]
    return cls, score, reason  # type: ignore[return-value]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def zscore_surprise(
    query: Sequence[float],
    neighbors: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """Returns (raw_distance, z_score).

    raw_distance = 1 - mean_cosine_similarity_to_neighbors
    z_score = (raw_distance - mean(distances)) / std(distances), clipped.

    With < 3 neighbors we fall back to raw distance scaled to [0, 1] —
    the cold-start regime has no distribution to Z-normalize against.
    """
    if not neighbors:
        return 0.5, 0.0  # cold-start neutral
    sims = [cosine(query, n) for n in neighbors]
    dists = [1.0 - s for s in sims]
    raw = sum(dists) / len(dists)  # mean distance to neighbors
    if len(dists) < 3:
        # Not enough samples for a meaningful Z; map raw to [-1, 1].
        return raw, max(-1.0, min(1.0, (raw - 0.5) * 2.0))
    mu = sum(dists) / len(dists)
    var = sum((d - mu) ** 2 for d in dists) / len(dists)
    sigma = math.sqrt(var) if var > 0 else 1e-6
    z = (raw - mu) / sigma
    z_clipped = max(-3.0, min(3.0, z))
    return raw, z_clipped


def z_to_unit(z: float) -> float:
    """Convert Z-score in [-3, 3] to [0, 1] via sigmoid centered at 0."""
    return 1.0 / (1.0 + math.exp(-z))


__all__ = [
    "UtilityClass",
    "_LLM_JUDGE_SYSTEM",
    "parse_judge_response",
    "cosine",
    "zscore_surprise",
    "z_to_unit",
    "resolve_float_env",
    "resolve_int_env",
]
