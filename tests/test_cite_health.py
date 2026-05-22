"""TDD anchors T-cite_health-1..8 + T-cite_health-cov-1..7 branch coverage.

Maps to spec: FR-A2, FR-A3, AC-A2, EC-A5.
Coverage anchors target health.py lines 87/89/121/123/127/155/179/190-194/241/249-254
to lift module coverage 77 -> 95%.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from myelin._internal.constants import (
    RECENCY_HALFLIFE_DAYS,
    W_RECENCY_LIFT,
    W_USE_LIFT,
    W_USE_PENALTY,
)
from myelin._internal.health import (
    QUARANTINE_THRESHOLD,
    _normalize,
    compute_contradiction_counts,
    is_quarantine_candidate,
    memory_health,
    recency_factor,
    register_multi_value,
)
from myelin.types import Memory


def _mem(**overrides) -> Memory:
    """Build a Memory with reasonable defaults."""
    base = {
        "id": "test-id",
        "content": "test content",
        "user_id": "u",
        "t_valid": datetime.now(UTC) - timedelta(days=1),
        "confidence": 0.5,
        "importance": 0.5,
    }
    base.update(overrides)
    return Memory(**base)


# T-cite_health-1
def test_default_cite_fields_no_op():
    """Memory with zero cite counters and None last_cited_at should
    score identically to the v1 equation."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    m_default = _mem(t_valid=now - timedelta(days=5))
    h_default = memory_health(m_default, now=now)
    # Should equal v1 equation: BIAS + W_AGE*5*0.05 + W_LOW_CONF*0.5 + ...
    # We don't recompute v1; just assert it's a deterministic value.
    assert 0.0 <= h_default <= 1.0
    # Recompute and confirm stability
    assert memory_health(m_default, now=now) == h_default


# T-cite_health-2 / FR-A3
def test_correct_cites_lift_health():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    base = _mem(t_valid=now - timedelta(days=5))
    cited = _mem(
        t_valid=now - timedelta(days=5),
        cite_correct_count=10,
        last_cited_at=now,
    )
    h_base = memory_health(base, now=now)
    h_cited = memory_health(cited, now=now)
    assert h_cited > h_base, "cite signal must lift health"


# T-cite_health-3 / FR-A3
def test_incorrect_cites_penalize_health():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    base = _mem(t_valid=now - timedelta(days=5))
    miscited = _mem(
        t_valid=now - timedelta(days=5),
        cite_incorrect_count=10,
    )
    h_base = memory_health(base, now=now)
    h_miscited = memory_health(miscited, now=now)
    assert h_miscited < h_base, "incorrect cites must penalize health"


# T-cite_health-4 — recency decay curve
def test_recency_decay_curve():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    # Just-cited
    assert recency_factor(now, now) == 1.0
    # Never-cited
    assert recency_factor(None, now) == 0.0
    # One half-life ago
    half = recency_factor(now - timedelta(days=RECENCY_HALFLIFE_DAYS), now)
    assert 0.49 < half < 0.51
    # Two half-lives ago
    quarter = recency_factor(now - timedelta(days=2 * RECENCY_HALFLIFE_DAYS), now)
    assert 0.24 < quarter < 0.26


# T-cite_health-5 / EC-A5 — contradiction dominance invariant
def test_contradiction_dominates_cite_lift():
    """A poisoned memory with 100 correct cites must STILL quarantine
    when contradiction_count >= 1. This is the load-bearing invariant
    for v1.2 — the constants module asserts it at import, and runtime
    must honor it too."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    poisoned_well_cited = _mem(
        t_valid=now - timedelta(days=10),
        confidence=0.3,
        importance=0.3,
        cite_correct_count=100,
        last_cited_at=now,
    )
    h = memory_health(poisoned_well_cited, now=now, contradiction_count=2)
    assert h < QUARANTINE_THRESHOLD, (
        f"EC-A5 violated: poison+100-cites+2-contradictions health={h:.4f} "
        f"not below QUARANTINE_THRESHOLD={QUARANTINE_THRESHOLD}"
    )


# T-cite_health-7 — clock-skew safety for future timestamps
def test_future_last_cited_at_full_recency():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    future = now + timedelta(hours=1)
    assert recency_factor(future, now) == 1.0


# T-cite_health-8 / FR-A2 — backward-compat invariant
def test_uncited_memory_v1_equation():
    """A memory with default cite fields (all zero / None) must produce
    the v1 health value. We verify by checking that adding zero-cite
    signal terms doesn't change the result."""
    now = datetime(2026, 5, 21, tzinfo=UTC)
    m = _mem(t_valid=now - timedelta(days=3), confidence=0.7, importance=0.6)
    h_actual = memory_health(m, now=now)
    # Recompute manually with v1 terms only:
    # BIAS + W_AGE*3*0.05 + W_LOW_CONF*(1-0.7) + W_LOW_IMP*(1-0.6)*0.3
    import math
    from myelin._internal.health import (
        BIAS, W_AGE, W_AGE_OVERFLOW, W_CONTRADICTION,
        W_LOW_CONF, W_LOW_IMPORTANCE,
    )
    age_days = 3
    age_overflow = 0.0
    raw_v1 = (
        BIAS
        + W_AGE * age_days * 0.05
        + W_AGE_OVERFLOW * age_overflow * 0.04
        + W_CONTRADICTION * 0
        + W_LOW_CONF * (1.0 - 0.7)
        + W_LOW_IMPORTANCE * (1.0 - 0.6) * 0.3
    )
    # cite terms all contribute 0 since defaults
    # log1p(0) = 0, recency_factor(None, ...) = 0
    raw_v12 = raw_v1 + W_USE_LIFT * 0 + W_USE_PENALTY * 0 + W_RECENCY_LIFT * 0
    h_expected = 1.0 / (1.0 + math.exp(-raw_v12))
    assert abs(h_actual - h_expected) < 1e-9


# T-cite_health-6 / AC-A2 — well-cited old memory survives apoptosis
def test_well_cited_old_memory_survives():
    from myelin import MemoryClient
    c = MemoryClient(cite_learning=True, user_id="u")
    try:
        m = c.add("often-cited fact", confidence=0.9, importance=0.8)
        # Simulate 20 correct cites
        for _ in range(20):
            c.cite(m.id, answer_correct=True)
        # Backdate t_valid to 30 days ago
        cur = c._backend._conn.cursor()  # noqa: SLF001
        backdated = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        cur.execute("UPDATE memories SET t_valid = ? WHERE id = ?", (backdated, m.id))
        c._backend._conn.commit()  # noqa: SLF001

        report = c.prune()
        # 30-day-old fact with 20 correct cites + last_cited very recent
        # should survive quarantine.
        assert report.quarantined_count == 0, (
            f"well-cited old memory should survive; got "
            f"{report.quarantined_count} quarantined"
        )
    finally:
        c.close()


# ============================================================================
# Branch-coverage anchors (T-cite_health-cov-1..7)
# Target health.py 77 -> 95% by exercising naive-datetime coercion, the
# quarantine helper, the contradiction-counter, the multi-value registry,
# and the article-stripping normalizer.
# ============================================================================


# T-cite_health-cov-1 — naive last_cited_at coerced to UTC (line 87)
def test_recency_factor_naive_last_cited_at_coerced():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    naive_past = datetime(2026, 5, 21) - timedelta(days=RECENCY_HALFLIFE_DAYS)
    assert naive_past.tzinfo is None
    half = recency_factor(naive_past, now)
    assert 0.49 < half < 0.51


# T-cite_health-cov-2 — naive `now` coerced to UTC (line 89)
def test_recency_factor_naive_now_coerced():
    aware_past = datetime(2026, 5, 21, tzinfo=UTC) - timedelta(
        days=RECENCY_HALFLIFE_DAYS
    )
    naive_now = datetime(2026, 5, 21)
    assert naive_now.tzinfo is None
    half = recency_factor(aware_past, naive_now)
    assert 0.49 < half < 0.51


# T-cite_health-cov-3 — memory_health defaults now=datetime.now(UTC) (line 121)
#                    and naive `now` parameter is coerced (line 123)
#                    and naive `t_valid` is coerced (line 127)
def test_memory_health_default_now_and_naive_coercion():
    # Default `now` (line 121)
    m1 = Memory(
        id="cov-3a",
        content="x",
        user_id="u",
        t_valid=datetime.now(UTC) - timedelta(days=1),
        confidence=0.7,
        importance=0.6,
    )
    h1 = memory_health(m1)
    assert 0.0 <= h1 <= 1.0

    # Naive `now` parameter (line 123)
    aware_t = datetime(2026, 5, 21, tzinfo=UTC) - timedelta(days=3)
    m2 = Memory(
        id="cov-3b",
        content="x",
        user_id="u",
        t_valid=aware_t,
        confidence=0.7,
        importance=0.6,
    )
    h2 = memory_health(m2, now=datetime(2026, 5, 21))
    assert 0.0 <= h2 <= 1.0

    # Naive `t_valid` (line 127). Pydantic Memory has t_valid: datetime so
    # passing a naive datetime is accepted; the function coerces internally.
    naive_t = datetime(2026, 5, 18)
    m3 = Memory(
        id="cov-3c",
        content="x",
        user_id="u",
        t_valid=naive_t,
        confidence=0.7,
        importance=0.6,
    )
    h3 = memory_health(m3, now=datetime(2026, 5, 21, tzinfo=UTC))
    assert 0.0 <= h3 <= 1.0


# T-cite_health-cov-4 — is_quarantine_candidate True/False paths (line 155)
def test_is_quarantine_candidate_both_branches():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    healthy = _mem(
        t_valid=now - timedelta(days=1),
        confidence=0.95,
        importance=0.95,
    )
    assert is_quarantine_candidate(healthy, now=now) is False

    poisoned = _mem(
        t_valid=now - timedelta(days=60),
        confidence=0.1,
        importance=0.1,
    )
    assert (
        is_quarantine_candidate(poisoned, now=now, contradiction_count=3)
        is True
    )


# T-cite_health-cov-5 — compute_contradiction_counts excludes tombstoned
# (line 179) and counts real contradictions on shared (subject, predicate)
# with different normalized objects (lines 190, 192-194 incl. _normalize).
def test_compute_contradiction_counts_full_paths():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    # Two memories on (alice, lives_in) with different objects -> contradict.
    a = _mem(
        id="a",
        t_valid=now,
        metadata={"subject": "alice", "predicate": "lives_in", "object": "Paris"},
    )
    b = _mem(
        id="b",
        t_valid=now,
        metadata={"subject": "alice", "predicate": "lives_in", "object": "Berlin"},
    )
    # Tombstoned memory must be ignored entirely (line 179).
    c = _mem(
        id="c",
        t_valid=now,
        metadata={
            "subject": "alice",
            "predicate": "lives_in",
            "object": "Tokyo",
        },
        health_state="tombstoned",
    )
    # Singleton bucket on (bob, lives_in) — len < 2 path (line 190).
    d = _mem(
        id="d",
        t_valid=now,
        metadata={"subject": "bob", "predicate": "lives_in", "object": "Rome"},
    )
    # Multi-value predicate bucket — skipped at line 188-189.
    e = _mem(
        id="e",
        t_valid=now,
        metadata={"subject": "alice", "predicate": "prefers", "object": "coffee"},
    )
    f = _mem(
        id="f",
        t_valid=now,
        metadata={"subject": "alice", "predicate": "prefers", "object": "tea"},
    )
    # Memory without subject/predicate metadata — skipped at line 182.
    g = _mem(id="g", t_valid=now, metadata={})

    counts = compute_contradiction_counts([a, b, c, d, e, f, g])
    assert counts["a"] == 1
    assert counts["b"] == 1
    assert counts["c"] == 0  # tombstoned, excluded
    assert counts["d"] == 0  # singleton
    assert counts["e"] == 0  # multi-value predicate
    assert counts["f"] == 0  # multi-value predicate
    assert counts["g"] == 0  # no subject/predicate


# T-cite_health-cov-6 — register_multi_value extends the registry (line 241)
def test_register_multi_value_extends_registry():
    now = datetime(2026, 5, 21, tzinfo=UTC)
    pred = "loves_topic_v12cov6"  # unique sentinel — won't collide with defaults
    a = _mem(
        id="a",
        t_valid=now,
        metadata={"subject": "alice", "predicate": pred, "object": "math"},
    )
    b = _mem(
        id="b",
        t_valid=now,
        metadata={"subject": "alice", "predicate": pred, "object": "music"},
    )
    # Before registration the predicate counts as single-value -> contradiction.
    pre = compute_contradiction_counts([a, b])
    assert pre["a"] == 1 and pre["b"] == 1

    register_multi_value(pred)

    post = compute_contradiction_counts([a, b])
    assert post["a"] == 0 and post["b"] == 0


# T-cite_health-cov-7 — _normalize strips articles + punctuation (lines 249-254)
def test_normalize_strips_articles_and_punctuation():
    assert _normalize("The Dog.") == "dog"
    assert _normalize("  an  Apple ") == "apple"
    assert _normalize("a  cat") == "cat"
    # No article — passthrough lower + whitespace-collapse.
    assert _normalize("  Hello  World ") == "hello world"
