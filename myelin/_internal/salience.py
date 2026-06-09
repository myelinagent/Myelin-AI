"""Carved 4-signal salience pipeline — stdlib-only port of `src/cortex/salience/`.

This is the v1 *heuristic* gate, ported verbatim from cortex into the
standalone myelin SDK so myelin can be benched apples-to-apples without
importing from the cortex tree. Three modes:

* ``off``        — no gating, every turn is stored (default).
* ``heuristic``  — this module. 4-signal linear-sum gate (novelty +
                   affect + reward + reference) with sigmoid scaling.
                   See ADR-005 + the 2026-05-20 addendum for the
                   diagnostic history; the verdict is that the
                   heuristic signals are sub-threshold by design on
                   natural conversational input, so this mode is
                   shipped only as the bench reference point for the
                   D-MEM gate.
* ``dmem``       — see :mod:`myelin._internal.dmem_gate`.

Phase B of `docs/plans/salience-gate-revisit-2026-05-20.md` lowered
``tau`` to 0.40 with ``require_reward=False`` so the heuristic mode
actually keeps ~13.5% of turns on PersonaMem-like input. That's the
config the bench uses to give the heuristic arm a fair shot.

No deps beyond stdlib. The cortex source kept `cortex.safety.pi_detector`
on the pipeline; in myelin we leave PI detection to the caller (each
adapter can wire its own safety scan upstream of `.add()`).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace


# --------------------------------------------------------------------
# Signal scorers (stdlib-only versions of cortex.salience.signals.*)
# --------------------------------------------------------------------

_POSITIVE = frozenset(
    "love loved loving great amazing wonderful excellent fantastic awesome "
    "perfect brilliant delighted thrilled happy grateful thankful".split()
)
_NEGATIVE = frozenset(
    "hate hated hating terrible awful horrible bad worst furious angry "
    "disappointed frustrated annoyed sad miserable broken".split()
)
_INTENSIFIERS = frozenset(
    "absolutely very really super extremely totally so".split()
)
_AFFECT_TOKEN_RE = re.compile(r"[A-Za-z]+")


def affect_score(text: str) -> float:
    """Keyword-lexicon affect proxy (port of ``signals/affect.py``)."""
    tokens = [t.lower() for t in _AFFECT_TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POSITIVE)
    neg = sum(1 for t in tokens if t in _NEGATIVE)
    ints = sum(1 for t in tokens if t in _INTENSIFIERS)
    base = min(1.0, (pos + neg) / 2.0)
    boost = 1.0 + 0.5 * min(2, ints)
    return max(0.0, min(1.0, base * boost))


_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_REFERENCE_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with and or "
    "not what who where when why how do does did this that these those".split()
)
_REFERENCE_MIN_CONTENT_LEN = 4


def reference_score(text: str) -> float:
    """Content-word density proxy (port of ``signals/reference.py``)."""
    tokens = [t.lower() for t in _REFERENCE_TOKEN_RE.findall(text)]
    if not tokens:
        return 0.0
    content = sum(
        1
        for t in tokens
        if t not in _REFERENCE_STOPWORDS and len(t) >= _REFERENCE_MIN_CONTENT_LEN
    )
    return max(0.0, min(1.0, content / max(1, len(tokens))))


def reward_score(outcome: Mapping[str, object] | None) -> float:
    """Outcome-driven reward (port of ``signals/reward.py``).

    PersonaMem turns carry no ``outcome`` field — ``reward_score(None)``
    cleanly returns 0.0, which is what the corpus needs.
    """
    if outcome is None:
        return 0.0
    if outcome.get("task_succeeded") is False:
        return 0.0
    s = 0.0
    if outcome.get("task_succeeded") is True:
        s += 0.5
    if outcome.get("user_thumbs_up") is True:
        s += 0.5
    if outcome.get("user_thumbs_down") is True:
        s -= 0.5
    if outcome.get("error") is True:
        s -= 0.3
    return max(0.0, min(1.0, s))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_NOVELTY_COLD_START_NEUTRAL = 0.5


def novelty_score(
    query_embedding: Sequence[float] | None,
    neighbor_embeddings: Sequence[Sequence[float]] = (),
) -> float:
    """1 - max_cosine novelty (port of ``signals/novelty.py``).

    Cold-start neutral when either the query embedding is missing or
    there are no neighbors yet — that's what the cortex implementation
    does and what the AAR-addendum diagnostic was scored against.
    """
    if query_embedding is None:
        return _NOVELTY_COLD_START_NEUTRAL
    if not neighbor_embeddings:
        return _NOVELTY_COLD_START_NEUTRAL
    max_sim = max(_cosine(query_embedding, n) for n in neighbor_embeddings)
    return max(0.0, min(1.0, 1.0 - max_sim))


# --------------------------------------------------------------------
# Heuristic gate (carved from gate.py + pipeline.py)
# --------------------------------------------------------------------

_WEIGHTS = {"novelty": 0.3, "affect": 0.2, "reward": 0.3, "reference": 0.2}
_SIGMOID_STEEPNESS = 10.0
_SIGMOID_CENTER = 0.5


@dataclass(frozen=True)
class SignalBundle:
    novelty: float
    affect: float
    reward: float
    reference: float


@dataclass(frozen=True)
class GateDecision:
    score: float
    tag: bool
    reason: str = ""
    # The four signals that produced this decision. Attached by
    # ``score_turn_heuristic`` so callers (MemoryClient telemetry, the
    # as-wired liveness probe) can observe the individual novelty/reward
    # axes without re-deriving them. ``None`` when a gate is evaluated
    # directly from a hand-built bundle.
    bundle: "SignalBundle | None" = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class HeuristicSalienceGate:
    """4-signal linear-sum gate with sigmoid scaling.

    Default tau is **0.40** (not 0.55 like cortex) to give the bench's
    heuristic arm a sigmoid-headroom that exists on PersonaMem-like
    natural conversation. Default ``require_reward=False`` so the
    kill-switch doesn't dominate (Phase B falsified the
    kill-switch-alone hypothesis; this config still keeps ~13.5% of
    turns).

    Re-bench mantra: this is the carved heuristic *for comparison
    purposes only.* The forward-going path is the D-MEM gate.
    """

    tau: float = 0.40
    require_reward: bool = False

    def evaluate(self, bundle: SignalBundle) -> GateDecision:
        linear = (
            bundle.novelty * _WEIGHTS["novelty"]
            + bundle.affect * _WEIGHTS["affect"]
            + bundle.reward * _WEIGHTS["reward"]
            + bundle.reference * _WEIGHTS["reference"]
        )
        score = _sigmoid((linear - _SIGMOID_CENTER) * _SIGMOID_STEEPNESS)
        reward_ok = bundle.reward > 0 if self.require_reward else True
        tag = (score >= self.tau) and reward_ok
        reason = (
            f"score={score:.3f} tau={self.tau:.2f} "
            f"reward_ok={reward_ok} linear={linear:.3f}"
        )
        return GateDecision(score=score, tag=tag, reason=reason)


def score_turn_heuristic(
    *,
    text: str,
    outcome: Mapping[str, object] | None = None,
    query_embedding: Sequence[float] | None = None,
    neighbor_embeddings: Sequence[Sequence[float]] = (),
    gate: HeuristicSalienceGate | None = None,
) -> GateDecision:
    """One-call entry point used by ``MemoryClient.add()``."""
    g = gate or HeuristicSalienceGate()
    bundle = SignalBundle(
        novelty=novelty_score(query_embedding, neighbor_embeddings),
        affect=affect_score(text),
        reward=reward_score(outcome),
        reference=reference_score(text),
    )
    decision = g.evaluate(bundle)
    return replace(decision, bundle=bundle)


__all__ = [
    "GateDecision",
    "HeuristicSalienceGate",
    "SignalBundle",
    "affect_score",
    "novelty_score",
    "reference_score",
    "reward_score",
    "score_turn_heuristic",
]
