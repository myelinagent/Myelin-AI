"""Apoptosis health scoring — carved from cortex/scheduler/jobs/_apoptosis_health.py.

Single source of truth for the per-memory health score that drives
the quarantine -> tombstone state machine. Weights match the v2
cheap-falsifier values that validated TPR=1.0 / FPR=0.0 at quarantine
on synthetic poison (commit a662fb4 in the parent repo). DO NOT tune
without re-running the F-BI-5 cohort smoke.

v1.2 extension (Innovation A): the equation gains three additive terms
reading the cite signal on Memory:
  + W_USE_LIFT     * log1p(cite_correct_count)
  + W_USE_PENALTY  * log1p(cite_incorrect_count)
  + W_RECENCY_LIFT * recency_factor(last_cited_at, now)

Defaults of those Memory fields (0, 0, None) make all three terms
contribute 0 — a v1 substrate with no cite signal produces the v1
health value unchanged.

The EC-A5 contradiction-dominance invariant is enforced at import time
in `myelin._internal.constants` — `W_USE_LIFT * log1p(100)` must be
strictly less than `|W_CONTRADICTION|` so a single contradiction always
outweighs 100 correct cites.

Thresholds:
  health < 0.4  -> quarantine candidate
  health < 0.2  -> caspase candidate (after 7-day window)

This module is import-safe: stdlib only, no numpy / torch / cortex.
"""
from __future__ import annotations

from datetime import UTC, datetime
from math import exp, log1p
from typing import TYPE_CHECKING

from myelin._internal.constants import (
    RECENCY_HALFLIFE_DAYS,
    W_RECENCY_LIFT,
    W_USE_LIFT,
    W_USE_PENALTY,
)

if TYPE_CHECKING:  # pragma: no cover
    from myelin.types import Memory

# v2-validated weights — see parent repo F-BI-5 cohort smoke for empirical
# basis (TPR=1.0, FPR=0.0 on 1000-fact-30-poison synthetic fixture).
W_AGE: float = -0.04
W_AGE_OVERFLOW: float = -5.0
W_CONTRADICTION: float = -3.5
W_LOW_CONF: float = -2.5
W_LOW_IMPORTANCE: float = -1.0
BIAS: float = 2.0

# Graded together with a memory's own use: support from related memories
# that were used (see myelin._internal.relations), and how often it was
# shown for a question without being used.
# Weights from the memory-over-time benchmark (bench/results/lifetime.md).
# Related use protects memories that sit beside useful ones. Being shown
# and not used does not count against health: a mild penalty (-0.3)
# pruned more and ranked slightly better, but hid more of what later
# questions needed, so it is off. It still lowers rank (cite_ranking).
W_RELATED: float = 2.0
W_UNUSED: float = 0.0
# Use, related use and recency together may not outweigh one
# contradiction (the EC-A5 rule in constants.py covers use alone).
MAX_USE_LIFT: float = 3.4

QUARANTINE_THRESHOLD: float = 0.4
CASPASE_THRESHOLD: float = 0.2

# v1 caspase window — quarantined memories must spend this long in
# quarantine before becoming caspase-eligible. Overridable per-call via
# `MemoryClient.prune(caspase_window_days=...)` for tests.
CASPASE_WINDOW_DAYS: int = 7


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    z = exp(x)
    return z / (1.0 + z)


def recency_factor(
    last_cited_at: datetime | None,
    now: datetime,
) -> float:
    """Return a recency lift in [0, 1].

    Exponential decay with half-life ``RECENCY_HALFLIFE_DAYS``. A fact
    cited just now returns 1.0; a fact never cited returns 0.0; a fact
    cited one half-life ago returns 0.5.

    Clock-skew safety: ``last_cited_at`` strictly in the future returns
    1.0 (treat as just-cited). Naive datetimes are coerced to UTC.
    """
    if last_cited_at is None:
        return 0.0
    if last_cited_at.tzinfo is None:
        last_cited_at = last_cited_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    seconds_since = (now - last_cited_at).total_seconds()
    if seconds_since <= 0:
        return 1.0

    halflife_seconds = RECENCY_HALFLIFE_DAYS * 86400.0
    return 0.5 ** (seconds_since / halflife_seconds)


def memory_health(
    memory: "Memory",
    *,
    now: datetime | None = None,
    contradiction_count: int = 0,
    related_support: float = 0.0,
) -> float:
    """Return health in [0, 1]. Higher = healthier.

    ``related_support`` is what
    :func:`myelin._internal.relations.neighbour_support` gives for this
    memory: how much its neighbours (same conversation, similar meaning)
    have been used.

    Health is driven by:
    - age in days (older = less healthy; sharply penalized past 30 days)
    - contradiction_count: active memories on (subject, predicate) with
      a different normalized object that are not older than this one
    - confidence (lower = less healthy)
    - importance (lower = less healthy, weakly)
    - v1.2 cite-correct count (log-compressed positive lift)
    - v1.2 cite-incorrect count (log-compressed penalty)
    - v1.2 recency factor (exponential decay since last cite)

    Caller computes `contradiction_count`; this function does NOT crawl
    the substrate.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    t = memory.t_valid
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    age_days = max(0, (now - t).days)
    age_overflow = max(0.0, (age_days - 30) / 30.0)

    conf = memory.confidence
    imp = memory.importance

    raw = (
        BIAS
        + W_AGE * age_days * 0.05
        + W_AGE_OVERFLOW * age_overflow * 0.04
        + W_CONTRADICTION * contradiction_count
        + W_LOW_CONF * (1.0 - conf)
        + W_LOW_IMPORTANCE * (1.0 - imp) * 0.3
        # ---- use, graded together: own use, related use, recency ----
        + min(
            MAX_USE_LIFT,
            W_USE_LIFT * log1p(max(0, memory.cite_correct_count))
            + W_RELATED * log1p(max(0.0, related_support))
            + W_RECENCY_LIFT * recency_factor(memory.last_cited_at, now),
        )
        + W_USE_PENALTY * log1p(max(0, memory.cite_incorrect_count))
        + W_UNUSED * log1p(max(0, memory.shown_unused_count))
    )
    return _sigmoid(raw)


def is_quarantine_candidate(
    memory: "Memory",
    *,
    now: datetime | None = None,
    contradiction_count: int = 0,
) -> bool:
    return memory_health(
        memory, now=now, contradiction_count=contradiction_count
    ) < QUARANTINE_THRESHOLD


def compute_contradiction_counts(memories: list["Memory"]) -> dict[str, int]:
    """For each Memory by id, count the active memories that supersede
    it: same (subject, predicate), a different normalized object, and
    not older than it.

    A newer fact supersedes an older one, so updating a single-valued
    fact penalizes only the stale value and the current value keeps a
    count of zero. Facts with the same ``t_valid`` cannot be ordered and
    contradict each other.

    For the SDK we don't have an explicit (subject, predicate, object)
    schema — facts are free-form content strings. We approximate the
    cortex contradiction model by grouping on `metadata["subject"]` and
    `metadata["predicate"]` when both are present; absent these keys, a
    memory contributes zero contradiction count.

    This mirrors the cortex `compute_contradiction_counts` semantics
    but operates over the SDK's looser substrate. Callers using event-
    anchored facts with explicit `subject` / `predicate` metadata get
    the same TPR/FPR; callers using free-form content get a graceful
    degradation (no false-positive contradictions).
    """
    by_subj_pred: dict[tuple[str, str], list[Memory]] = {}
    for m in memories:
        if m.t_invalid is not None or m.health_state == "tombstoned":
            continue
        subj = m.metadata.get("subject")
        pred = m.metadata.get("predicate")
        if not (isinstance(subj, str) and isinstance(pred, str)):
            continue
        by_subj_pred.setdefault((subj, pred), []).append(m)

    counts: dict[str, int] = {m.id: 0 for m in memories}
    for (_subject, predicate), bucket in by_subj_pred.items():
        if _is_multi_value(predicate):
            continue
        if len(bucket) < 2:
            continue
        norms = [_normalize(m.metadata.get("object", m.content)) for m in bucket]
        stamps = [_as_utc(m.t_valid) for m in bucket]
        for i, m in enumerate(bucket):
            counts[m.id] = sum(
                1
                for j, n in enumerate(norms)
                if i != j and n != norms[i] and stamps[j] >= stamps[i]
            )
    return counts


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def find_superseded(memories: list["Memory"]) -> dict[str, "Memory"]:
    """Map each outdated memory's id to the memory that replaced it.

    A memory is outdated when another current memory states a different
    value for the same (subject, predicate) at a strictly later time.
    The replacement returned is the earliest such later memory, so its
    ``t_valid`` is the moment the old value stopped being true. Memories
    already marked invalid, tombstoned memories, multi-value predicates
    and memories without subject/predicate metadata are left out, as in
    ``compute_contradiction_counts``.
    """
    buckets: dict[tuple[str, str], list[Memory]] = {}
    for m in memories:
        if m.t_invalid is not None or m.health_state == "tombstoned":
            continue
        subj = m.metadata.get("subject")
        pred = m.metadata.get("predicate")
        if not (isinstance(subj, str) and isinstance(pred, str)):
            continue
        if _is_multi_value(pred):
            continue
        buckets.setdefault((subj, pred), []).append(m)

    superseded: dict[str, Memory] = {}
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        ordered = sorted(bucket, key=lambda m: (_as_utc(m.t_valid), m.id))
        values = [_normalize(m.metadata.get("object", m.content)) for m in ordered]
        for i, old in enumerate(ordered):
            for j in range(i + 1, len(ordered)):
                newer = ordered[j]
                if values[j] != values[i] and _as_utc(newer.t_valid) > _as_utc(old.t_valid):
                    superseded[old.id] = newer
                    break
    return superseded


# Multi-value predicates — predicates where a single subject can hold
# many concurrent active values. Mirrors `MULTI_VALUE_PREDICATES` in
# `cortex.sleep.memory_schema` but pared to the personal-agent base
# vocabulary; callers can extend at runtime via `register_multi_value`.
_MULTI_VALUE: set[str] = {
    "prefers",
    "dislikes",
    "uses_tool",
    "works_on_project",
    "owns_project",
    "wants_outcome",
    "has_constraint",
    "has_role",
    "uses_workflow",
    "created_artifact",
    "needs_followup",
    "decided",
    "rejected_option",
    "has_deadline",
    "has_style_rule",
    "has_operational_rule",
    "has_family_member",
    "has_recurring_schedule",
    "has_household_routine",
    "cares_for_minor",
    "delegates_to_agent",
    "requires_approval_for",
    "forbids_agent_from",
    "learns_via",
    "understands_concept",
    "struggles_with",
    "prefers_explanation_style",
}


def _is_multi_value(predicate: str) -> bool:
    return predicate in _MULTI_VALUE


def register_multi_value(predicate: str) -> None:
    """Extend the multi-value predicate set at runtime."""
    _MULTI_VALUE.add(predicate)


_LEADING_ARTICLES = ("a ", "an ", "the ")
_STRIP = " .,!?:;\"'"


def _normalize(value: object) -> str:
    s = str(value).strip(_STRIP).lower()
    for prefix in _LEADING_ARTICLES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return " ".join(s.split())
