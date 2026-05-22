"""Single source of truth for v1.2 tuning constants.

Pinned in `docs/sparc/cite-and-working-memory/phase_3_architecture.md` §3.
Every constant is env-overridable via the same strict-"1"-friendly path
the D-MEM gate uses (numeric env vars are parsed; non-parseable values
fall back to the baked default with a logger.warning).

The EC-A5 invariant — contradiction signal must dominate cite signal in
the apoptosis health equation — is enforced by a module-level assertion
at import time: `W_USE_LIFT * log1p(100) < |W_CONTRADICTION|`. A future
tuning sweep that breaks this invariant breaks the import.

No secrets here. No API keys. No URLs. Only algorithm parameters.
"""
from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Cite-aware health (Innovation A)
# ---------------------------------------------------------------------

# Default magnitudes pinned by Phase 3 §3.1.
_DEFAULT_W_USE_LIFT: float = 0.5
_DEFAULT_W_USE_PENALTY: float = -0.3
_DEFAULT_W_RECENCY_LIFT: float = 0.4
_DEFAULT_RECENCY_HALFLIFE_DAYS: float = 14.0

# ---------------------------------------------------------------------
# Working-memory layer (Innovation B)
# ---------------------------------------------------------------------

_DEFAULT_SESSION_TTL_SECONDS: int = 86_400        # 24h
_DEFAULT_MAX_CONCURRENT_SESSIONS: int = 1_000
_DEFAULT_OBSERVED_RING_MAX: int = 1_024
_DEFAULT_WORKING_SET_MAX: int = 256
_DEFAULT_SWEEPER_INTERVAL_SECONDS: int = 60

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def env_is_strict_one(name: str) -> bool:
    """Return True only for the literal string ``"1"`` — never for
    ``"true"``, ``"True"``, ``"yes"``, ``"1.0"``, etc.

    Per `feedback_strict_1_env_vars` in user memory: loose-truthy parses
    have repeatedly corrupted locked baselines by silently re-engaging
    ship-gate flags. v1.2 follows the same convention as the cortex
    pipeline.
    """
    return os.environ.get(name) == "1"


def _resolve_float_env(env_name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        logger.warning(
            "constants: env %s=%r unparseable; falling back to %.3f",
            env_name, raw, default,
        )
        return default
    if not (lo <= v <= hi):
        logger.warning(
            "constants: env %s=%.3f outside [%.3f, %.3f]; falling back to %.3f",
            env_name, v, lo, hi, default,
        )
        return default
    return v


def _resolve_int_env(env_name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        logger.warning(
            "constants: env %s=%r unparseable; falling back to %d",
            env_name, raw, default,
        )
        return default
    if not (lo <= v <= hi):
        logger.warning(
            "constants: env %s=%d outside [%d, %d]; falling back to %d",
            env_name, v, lo, hi, default,
        )
        return default
    return v


# ---------------------------------------------------------------------
# Resolved public constants
# ---------------------------------------------------------------------

# Cite-aware health. Bounds are wide-but-sane; the EC-A5 invariant
# (asserted below) is what actually keeps W_USE_LIFT safe.
W_USE_LIFT: float = _resolve_float_env(
    "MYELIN_W_USE_LIFT", _DEFAULT_W_USE_LIFT, lo=0.0, hi=2.0,
)
W_USE_PENALTY: float = _resolve_float_env(
    "MYELIN_W_USE_PENALTY", _DEFAULT_W_USE_PENALTY, lo=-2.0, hi=0.0,
)
W_RECENCY_LIFT: float = _resolve_float_env(
    "MYELIN_W_RECENCY_LIFT", _DEFAULT_W_RECENCY_LIFT, lo=0.0, hi=2.0,
)
RECENCY_HALFLIFE_DAYS: float = _resolve_float_env(
    "MYELIN_RECENCY_HALFLIFE_DAYS", _DEFAULT_RECENCY_HALFLIFE_DAYS,
    lo=0.5, hi=365.0,
)

# Working-memory layer.
SESSION_TTL_SECONDS: int = _resolve_int_env(
    "MYELIN_SESSION_TTL_SECONDS", _DEFAULT_SESSION_TTL_SECONDS,
    lo=60, hi=7 * 86_400,
)
MAX_CONCURRENT_SESSIONS: int = _resolve_int_env(
    "MYELIN_MAX_CONCURRENT_SESSIONS", _DEFAULT_MAX_CONCURRENT_SESSIONS,
    lo=1, hi=100_000,
)
OBSERVED_RING_MAX: int = _resolve_int_env(
    "MYELIN_OBSERVED_RING_MAX", _DEFAULT_OBSERVED_RING_MAX,
    lo=8, hi=65_536,
)
WORKING_SET_MAX: int = _resolve_int_env(
    "MYELIN_WORKING_SET_MAX", _DEFAULT_WORKING_SET_MAX,
    lo=4, hi=8_192,
)
SWEEPER_INTERVAL_SECONDS: int = _resolve_int_env(
    "MYELIN_SWEEPER_INTERVAL_SECONDS", _DEFAULT_SWEEPER_INTERVAL_SECONDS,
    lo=1, hi=3_600,
)

# ---------------------------------------------------------------------
# EC-A5 invariant — enforced at import
# ---------------------------------------------------------------------

# W_CONTRADICTION lives in _internal/health.py; we re-state its magnitude
# here for the invariant check. If health.py's W_CONTRADICTION changes,
# this literal must update too — it's a deliberate redundancy so a tuning
# sweep that bumps W_USE_LIFT past the safe range breaks the import
# instead of silently corrupting apoptosis behavior.
_W_CONTRADICTION_MAGNITUDE: float = 3.5

_USE_LIFT_AT_100_CITES: float = W_USE_LIFT * math.log1p(100)

assert _USE_LIFT_AT_100_CITES < _W_CONTRADICTION_MAGNITUDE, (
    "EC-A5 invariant violated: W_USE_LIFT * log1p(100) = "
    f"{_USE_LIFT_AT_100_CITES:.3f} >= |W_CONTRADICTION| = "
    f"{_W_CONTRADICTION_MAGNITUDE:.3f}. A single contradiction would "
    "no longer outweigh 100 correct cites — poisoned facts could "
    "survive apoptosis. Lower W_USE_LIFT (env MYELIN_W_USE_LIFT) or "
    "raise W_CONTRADICTION in health.py."
)

# Also assert W_USE_PENALTY magnitude < W_USE_LIFT (incorrect cites are
# weaker signal than correct cites; documented in Phase 3 §3.1).
assert abs(W_USE_PENALTY) < W_USE_LIFT, (
    f"|W_USE_PENALTY|={abs(W_USE_PENALTY):.3f} should be less than "
    f"W_USE_LIFT={W_USE_LIFT:.3f} per Phase 3 §3.1 rationale."
)

logger.debug(
    "myelin v1.2 constants: W_USE_LIFT=%.2f W_USE_PENALTY=%.2f "
    "W_RECENCY_LIFT=%.2f halflife=%.1fd ttl=%ds max_sessions=%d "
    "ring=%d working_set=%d sweep=%ds",
    W_USE_LIFT, W_USE_PENALTY, W_RECENCY_LIFT, RECENCY_HALFLIFE_DAYS,
    SESSION_TTL_SECONDS, MAX_CONCURRENT_SESSIONS, OBSERVED_RING_MAX,
    WORKING_SET_MAX, SWEEPER_INTERVAL_SECONDS,
)


__all__ = [
    "MAX_CONCURRENT_SESSIONS",
    "OBSERVED_RING_MAX",
    "RECENCY_HALFLIFE_DAYS",
    "SESSION_TTL_SECONDS",
    "SWEEPER_INTERVAL_SECONDS",
    "WORKING_SET_MAX",
    "W_RECENCY_LIFT",
    "W_USE_LIFT",
    "W_USE_PENALTY",
    "env_is_strict_one",
]
