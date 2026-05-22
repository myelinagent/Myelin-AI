"""D-MEM-style two-axis salience gate (the "trained-weights" arm).

The gate combines two signals to decide whether a turn is durable:

* **semantic surprise** — Z-score-normalized cosine distance vs the
  per-user neighbor window. Z-norm defeats embedding anisotropy.
* **utility** — either an LLM-judge call (``utility_backend="llm_judge"``)
  or the trained Memory-R1 contrastive encoder
  (``utility_backend="contrastive_v1"``, see ADR-019).

Final tag requires (utility_class != "transient") AND (utility_score ≥
utility_threshold) AND (rpe ≥ rpe_threshold) where rpe = surprise_unit
× utility_score.

Env overrides (consulted only when constructor kwarg is at sentinel):

* ``MYELIN_DMEM_RPE_THRESHOLD``       (default 0.30, range [0, 1])
* ``MYELIN_DMEM_UTILITY_THRESHOLD``   (default 0.30, range [0, 1])
* ``MYELIN_DMEM_NEIGHBOR_WINDOW``     (default 64,   range [4, 512])
* ``MYELIN_DMEM_UTILITY_BACKEND``     (default "llm_judge")
* ``MYELIN_DMEM_LLM_TIMEOUT``         (default 30s)
* ``MYELIN_DMEM_LLM_MAX_RETRIES``     (default 2, bounded [0, 5])
* ``MYELIN_DMEM_AUTO_FALLBACK_THRESHOLD`` (default 0.9, EC-3)

Helpers live in ``_dmem_helpers`` (parser, surprise) and
``_dmem_contrastive`` (encoder backend). See ``docs/sparc/
memory-r1-contrastive-spec.md`` and ``docs/plans/
salience-gate-revisit-2026-05-20.md`` for design history.

On LLM-judge exception (ReadTimeout / RateLimitError / RuntimeError /
etc.) the gate emits the configurable ``llm_error_fallback`` triple
(default short_term / 0.5) so a wedged judge does not silently
blacken-out the rest of the bench (Phase G2 post-stall lesson).

No PII or sensitive content is logged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

UtilityClass = Literal["transient", "short_term", "persistent"]
UtilityBackend = Literal["llm_judge", "contrastive_v1"]

# Sentinel for utility_backend so explicit kwargs win over env.
_BACKEND_SENTINEL: str = "__from_env__"

_VALID_BACKENDS: tuple[str, ...] = ("llm_judge", "contrastive_v1")


# --------------------------------------------------------------------
# Decision record
# --------------------------------------------------------------------

@dataclass(frozen=True)
class DMemDecision:
    """The two-axis gate's per-turn verdict.

    ``utility_class``, ``utility_score`` come from the LLM judge.
    ``surprise_zscore`` is the Z-normalized cosine novelty against the
    running neighbor window. ``rpe`` is the multiplicative product;
    ``tag`` is the final keep/drop. ``reason`` is a short audit string.

    ``llm_calls_made`` is 0 when the cache served the verdict, 1 when
    the LLM was called. Bench harness sums this to compute per-turn
    cost and cache hit rate.
    """

    utility_class: UtilityClass
    utility_score: float
    surprise_zscore: float
    rpe: float
    tag: bool
    reason: str
    llm_calls_made: int
    cached: bool


# --------------------------------------------------------------------
# Public protocols (so the caller can inject embedder / LLM)
# --------------------------------------------------------------------

class Embedder:
    """Minimal contract — anything with ``.encode(text) -> list[float]``.

    sentence-transformers' SentenceTransformer matches this shape:
    ``SentenceTransformer(...).encode(text)`` returns a numpy array
    that's convertible via ``.tolist()``. We wrap it lazily so the
    sentence-transformers import only happens when the D-MEM mode is
    actually selected.
    """

    def encode(self, text: str) -> Sequence[float]:  # pragma: no cover - protocol
        raise NotImplementedError


def _default_embedder() -> Embedder:
    """Lazy-load sentence-transformers/all-MiniLM-L6-v2."""
    from sentence_transformers import SentenceTransformer

    class _STAdapter:
        def __init__(self) -> None:
            self._m = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

        def encode(self, text: str) -> Sequence[float]:
            arr = self._m.encode(text, show_progress_bar=False)
            return arr.tolist()

    return _STAdapter()


# --------------------------------------------------------------------
# Re-exports from _dmem_helpers (extracted to satisfy the 500-LOC cap)
# --------------------------------------------------------------------

from myelin._internal._dmem_helpers import (  # noqa: E402
    _LLM_JUDGE_SYSTEM,
    parse_judge_response as _parse_judge_response,
    cosine as _cosine,
    zscore_surprise as _zscore_surprise,
    z_to_unit as _z_to_unit,
)


# --------------------------------------------------------------------
# DMemGate
# --------------------------------------------------------------------

_DEFAULT_RPE_THRESHOLD = 0.30
_DEFAULT_UTILITY_THRESHOLD = 0.30
_DEFAULT_NEIGHBOR_WINDOW_SIZE = 64

# Sentinels distinguish "caller passed an explicit value" from "use the
# env-var-or-baked-in default." None is a poor sentinel here because we
# want the dataclass field to type-check as float / int.
_RPE_SENTINEL = -1.0
_UTILITY_SENTINEL = -1.0
_WINDOW_SENTINEL = -1


# Env resolvers live in _dmem_helpers (extracted for the 500-LOC cap).
from myelin._internal._dmem_helpers import (  # noqa: E402
    resolve_float_env as _resolve_float_env,
    resolve_int_env as _resolve_int_env,
)


@dataclass
class DMemGate:
    """Two-axis gate: surprise × utility → multiplicative RPE → keep/drop.

    Default thresholds tuned for PersonaMem-shaped natural conversation
    per the research §5 prescriptive recommendations. The bench will
    sweep these to confirm.

    The three sweep-axis hyperparameters
    (``rpe_threshold``, ``utility_threshold``, ``neighbor_window_size``)
    are env-var-overridable when left at their sentinel defaults — see
    module docstring "Hyperparameter env-var pickup" for the env names
    and bounds. Explicit constructor kwargs ALWAYS win over env, so
    existing unit tests (which pass e.g. ``rpe_threshold=0.20``) keep
    working regardless of what's in os.environ.
    """

    # OpenAI-compatible client (lazy-built from OPENAI_API_KEY when None
    # and backend=llm_judge); embedder defaults to ST/all-MiniLM-L6-v2.
    llm_client: Any = None
    embedder: Embedder | None = None
    llm_model: str = "gpt-4o-mini"

    # Sweep-axis thresholds — explicit kwargs always win over env (see
    # module docstring for the env var names + bounds).
    rpe_threshold: float = _RPE_SENTINEL
    utility_threshold: float = _UTILITY_SENTINEL
    neighbor_window_size: int = _WINDOW_SENTINEL

    utility_cache_size: int = 10_000  # LRU cap for the judge cache

    # Which utility scorer: "llm_judge" (Phase G2/G3) or "contrastive_v1"
    # (Memory-R1 trained weights, ADR-019). EC-4: encoder-load failure
    # silently soft-falls back to llm_judge for the rest of the run.
    utility_backend: str = _BACKEND_SENTINEL

    # EC-3: fraction of recent-100-turn classes == "transient" above
    # which a one-shot warning is emitted.
    distribution_shift_threshold: float = -1.0

    # --- runtime state (populated lazily / per-user) ----------------

    _neighbor_windows: dict[str, deque] = field(default_factory=dict)
    _utility_cache: dict[str, tuple[UtilityClass, float, str]] = field(
        default_factory=dict
    )
    _cache_keys_lru: deque = field(default_factory=deque)
    _llm_calls_made: int = 0
    _llm_cache_hits: int = 0

    # Distribution-shift monitor state (EC-3).
    _recent_classes: deque = field(default_factory=lambda: deque(maxlen=100))
    _shift_warning_emitted: bool = False

    # Contrastive backend cached lookup (resolved on first evaluate()).
    _contrastive_encoder: Any = None
    _contrastive_fallback_active: bool = False

    # Soft-fallback triple on any LLM-judge exception. Defaults to
    # short_term/0.5 so a wedged judge doesn't blackout the bench.
    llm_error_fallback: tuple[UtilityClass, float, str] = (
        "short_term", 0.5, "judge-error-fallback"
    )

    def __post_init__(self) -> None:
        # Resolve sentinel-valued hyperparameters from env vars so the
        # Phase G3 ablation sweep can reuse one Docker image. Explicit
        # caller kwargs (anything that isn't the negative sentinel) win
        # unchanged.
        if self.rpe_threshold == _RPE_SENTINEL:
            self.rpe_threshold = _resolve_float_env(
                "MYELIN_DMEM_RPE_THRESHOLD",
                _DEFAULT_RPE_THRESHOLD,
                lo=0.0,
                hi=1.0,
            )
        if self.utility_threshold == _UTILITY_SENTINEL:
            self.utility_threshold = _resolve_float_env(
                "MYELIN_DMEM_UTILITY_THRESHOLD",
                _DEFAULT_UTILITY_THRESHOLD,
                lo=0.0,
                hi=1.0,
            )
        if self.neighbor_window_size == _WINDOW_SENTINEL:
            self.neighbor_window_size = _resolve_int_env(
                "MYELIN_DMEM_NEIGHBOR_WINDOW",
                _DEFAULT_NEIGHBOR_WINDOW_SIZE,
                lo=4,
                hi=512,
            )

        # Resolve utility_backend from env if not explicitly set.
        if self.utility_backend == _BACKEND_SENTINEL:
            env_backend = os.environ.get(
                "MYELIN_DMEM_UTILITY_BACKEND", "llm_judge"
            ).strip().lower()
            self.utility_backend = env_backend
        if self.utility_backend not in _VALID_BACKENDS:
            raise ValueError(
                f"DMemGate: unknown utility_backend={self.utility_backend!r}; "
                f"valid choices are {list(_VALID_BACKENDS)}"
            )

        # Distribution-shift threshold (EC-3).
        if self.distribution_shift_threshold == -1.0:
            self.distribution_shift_threshold = _resolve_float_env(
                "MYELIN_DMEM_AUTO_FALLBACK_THRESHOLD",
                0.9,
                lo=0.0,
                hi=1.0,
            )

        logger.info(
            "DMemGate hyperparameters: rpe=%.3f utility=%.3f window=%d "
            "backend=%s",
            self.rpe_threshold,
            self.utility_threshold,
            self.neighbor_window_size,
            self.utility_backend,
        )
        if self.embedder is None:
            self.embedder = _default_embedder()
        # LLM client is only required for the llm_judge backend OR as a
        # contrastive_v1 fallback if encoder load fails. Build it lazily
        # in either case so contrastive_v1 doesn't hard-require
        # OPENAI_API_KEY just to construct the gate.
        if self.llm_client is None and self.utility_backend == "llm_judge":
            self.llm_client = self._lazy_openai_client()

    @staticmethod
    def _lazy_openai_client() -> Any:
        """Build an OpenAI client with HARD-BOUNDED per-request timeout +
        max_retries. The 2026-05-20 local Phase G Run 3 stall (8h to
        process 77 rows of ctx 7/37, ingest time exploded 58s → 4562s
        across 4 contexts) was a textbook unbounded retry-on-rate-limit
        loop. The OpenAI Python SDK defaults to ``max_retries=2`` per
        ``OpenAI.__init__`` but the lazy client also needs an explicit
        per-request ``timeout`` so a wedged TCP socket can't hang the
        bench forever.
        """
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DMemGate needs OPENAI_API_KEY in env for the utility judge"
            )
        try:
            timeout_s = float(os.environ.get("MYELIN_DMEM_LLM_TIMEOUT", "30"))
        except ValueError:
            timeout_s = 30.0
        try:
            max_retries = int(os.environ.get("MYELIN_DMEM_LLM_MAX_RETRIES", "2"))
        except ValueError:
            max_retries = 2
        max_retries = max(0, min(max_retries, 5))
        timeout_s = max(1.0, min(timeout_s, 300.0))
        logger.info(
            "DMemGate lazy OpenAI client: timeout=%.1fs max_retries=%d",
            timeout_s,
            max_retries,
        )
        return OpenAI(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    # ----- public API ------------------------------------------------

    def evaluate(self, *, content: str, user_id: str = "default") -> DMemDecision:
        """Score one turn and return a DMemDecision.

        Side-effects: appends the content's embedding to the user's
        neighbor window AFTER the surprise computation (so the turn is
        compared against its predecessors, not itself).
        """
        # 1) Embed.
        emb = list(self.embedder.encode(content)) if self.embedder else []

        # 2) Surprise vs neighbor window.
        window = self._neighbor_windows.setdefault(
            user_id, deque(maxlen=self.neighbor_window_size)
        )
        neighbors = list(window)
        raw_dist, z = _zscore_surprise(emb, neighbors)
        surprise_unit = _z_to_unit(z)

        # 3) Utility scorer — branch on backend.
        if self.utility_backend == "contrastive_v1":
            utility_class, utility_score, judge_reason, cached = (
                self._contrastive_utility(content, neighbors)
            )
        else:
            utility_class, utility_score, judge_reason, cached = (
                self._judge_utility(content)
            )

        # Distribution-shift monitor (EC-3).
        self._recent_classes.append(utility_class)
        if (
            not self._shift_warning_emitted
            and len(self._recent_classes) == self._recent_classes.maxlen
        ):
            transient_rate = (
                self._recent_classes.count("transient") / len(self._recent_classes)
            )
            if transient_rate > self.distribution_shift_threshold:
                logger.warning(
                    "DMemGate distribution shift: transient_rate=%.2f "
                    "exceeds threshold=%.2f over last %d turns — "
                    "consider re-training utility encoder",
                    transient_rate,
                    self.distribution_shift_threshold,
                    len(self._recent_classes),
                )
                self._shift_warning_emitted = True

        # 4) Multiplicative RPE + categorical kill-switch.
        rpe = surprise_unit * utility_score
        tag = (
            utility_class != "transient"
            and utility_score >= self.utility_threshold
            and rpe >= self.rpe_threshold
        )
        reason = (
            f"cls={utility_class} util={utility_score:.2f} "
            f"surprise_unit={surprise_unit:.2f} z={z:.2f} "
            f"raw_dist={raw_dist:.2f} rpe={rpe:.2f} "
            f"thr={self.rpe_threshold:.2f} judge='{judge_reason}'"
        )

        # 5) Update neighbor window AFTER scoring this turn.
        if emb:
            window.append(emb)

        return DMemDecision(
            utility_class=utility_class,
            utility_score=utility_score,
            surprise_zscore=z,
            rpe=rpe,
            tag=tag,
            reason=reason,
            llm_calls_made=0 if cached else 1,
            cached=cached,
        )

    def reset_user(self, user_id: str) -> None:
        """Drop the neighbor window for a user — used between personas."""
        self._neighbor_windows.pop(user_id, None)

    def reset_all(self) -> None:
        """Drop ALL neighbor windows (cache stays — cache is content-keyed)."""
        self._neighbor_windows.clear()

    @property
    def stats(self) -> dict[str, Any]:
        total = self._llm_calls_made + self._llm_cache_hits
        return {
            "llm_calls_made": self._llm_calls_made,
            "llm_cache_hits": self._llm_cache_hits,
            "total_evaluations": total,
            "cache_hit_rate": (
                self._llm_cache_hits / total if total > 0 else 0.0
            ),
            "cache_size": len(self._utility_cache),
        }

    # ----- internals -------------------------------------------------

    def _contrastive_utility(
        self,
        content: str,
        neighbors: Sequence[Sequence[float]],
    ) -> tuple[UtilityClass, float, str, bool]:
        """Thin delegator — full logic lives in ``_dmem_contrastive``
        to keep this module under the 500-LOC cap."""
        from myelin._internal._dmem_contrastive import score_via_contrastive

        return score_via_contrastive(self, content, neighbors)

    def _judge_utility(
        self, content: str
    ) -> tuple[UtilityClass, float, str, bool]:
        """Return (class, score, reason, cached)."""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in self._utility_cache:
            self._llm_cache_hits += 1
            cls, score, reason = self._utility_cache[h]
            return cls, score, reason, True
        # cache miss → LLM call. Timeout + max_retries are bound by
        # the OpenAI client itself (constructed in _lazy_openai_client).
        # On any post-retry failure we fall back to the configurable
        # `llm_error_fallback` triple so a wedged judge does NOT silently
        # filter every subsequent bench turn to transient/0.0.
        try:
            r = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
            raw = r.choices[0].message.content or ""
        except Exception as e:  # network/auth/rate-limit/timeout/etc
            fb_cls, fb_score, fb_reason = self.llm_error_fallback
            logger.warning(
                "DMemGate LLM judge failed (%s); falling back to "
                "%s/%.2f — gate continues with soft-tag, bench progresses",
                type(e).__name__,
                fb_cls,
                fb_score,
            )
            return (
                fb_cls,
                fb_score,
                f"{fb_reason}:{type(e).__name__}",
                False,
            )
        cls, score, reason = _parse_judge_response(raw)
        # LRU insert
        if len(self._utility_cache) >= self.utility_cache_size:
            old = self._cache_keys_lru.popleft()
            self._utility_cache.pop(old, None)
        self._utility_cache[h] = (cls, score, reason)
        self._cache_keys_lru.append(h)
        self._llm_calls_made += 1
        return cls, score, reason, False


__all__ = [
    "DMemDecision",
    "DMemGate",
    "Embedder",
    "UtilityClass",
]
