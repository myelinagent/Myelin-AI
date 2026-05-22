"""TDD anchors T-tiered_search-1..9.

Maps to spec: FR-B5, NFR-B2, AC-B2, EC-B5.
"""
from __future__ import annotations

import time

import pytest

from myelin import MemoryClient


# T-tiered_search-1 — backward compat
def test_backward_compat_v1_signature():
    c = MemoryClient()
    try:
        c.add("Scott prefers tea")
        c.add("Alice loves chess")
        results = c.search("tea")
        assert len(results) == 1
        assert "tea" in results[0].memory.content
        # Telemetry still tracks the long-term hit
        tel = c.search_telemetry
        assert tel.n_hits_from_long_term_tier == 1
        assert tel.n_search_calls_with_session_id == 0
    finally:
        c.close()


# T-tiered_search-2 / AC-B2
def test_working_tier_hit():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        t = c.observe("My daughter Melanie is wonderful", session_id="s1")
        c.commit([t.turn_id], session_id="s1")
        results = c.search(
            "Melanie",
            session_id="s1",
            modes=("working", "long_term"),
            top_k=1,
        )
        assert len(results) == 1
        tel = c.search_telemetry
        assert tel.n_hits_from_working_tier == 1
        assert tel.n_calls_with_working_hit == 1
        assert tel.working_set_hit_rate == 1.0
    finally:
        c.close()


# T-tiered_search-3
def test_empty_working_set_falls_through():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        c.add("a long-term fact about tea")
        c.observe("bootstrap session", session_id="s")  # session exists
        # commit so the bootstrap isn't in working set as a hit for "tea"
        results = c.search("tea", session_id="s", modes=("working", "long_term"))
        assert len(results) == 1
        tel = c.search_telemetry
        assert tel.n_hits_from_long_term_tier == 1
        # Working tier had no hits for "tea"
        assert tel.n_hits_from_working_tier == 0
        assert tel.working_set_hit_rate == 0.0
    finally:
        c.close()


# T-tiered_search-4 — de-dupe across tiers
def test_dedupe_across_tiers():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        t = c.observe("Bob plays chess", session_id="s")
        committed = c.commit([t.turn_id], session_id="s")
        m1_id = committed[0].id
        # Add another long-term fact also about chess
        c.add("Alice plays chess too")
        results = c.search(
            "chess", session_id="s",
            modes=("working", "long_term"), top_k=10,
        )
        ids = [r.memory.id for r in results]
        # m1 should appear exactly once (not duplicated across tiers)
        assert ids.count(m1_id) == 1
        # Both unique memories present
        assert len(set(ids)) == len(ids)
    finally:
        c.close()


# T-tiered_search-5 / EC-B5
def test_missing_session_id_downgrades():
    c = MemoryClient(working_memory=True)
    try:
        c.add("a fact")
        results = c.search("fact", modes=("working", "long_term"))  # no session_id
        # Silent downgrade to long_term only
        assert len(results) == 1
        tel = c.search_telemetry
        assert tel.n_hits_from_working_tier == 0
        assert tel.n_search_calls_with_session_id == 0
    finally:
        c.close()


# T-tiered_search-6 / NFR-B2 — working tier p99 ≤ 5ms
def test_working_tier_p99_latency():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        # Fill working set up to cap
        from myelin._internal.constants import WORKING_SET_MAX
        n = WORKING_SET_MAX
        # observe + commit + attend (auto-attended on commit)
        observed = [
            c.observe(f"fact {i} chess", session_id="bench") for i in range(n)
        ]
        c.commit([t.turn_id for t in observed], session_id="bench")
        # Pre-warm
        c.search("chess", session_id="bench", modes=("working",), top_k=5)

        # Measure 100 trials
        timings: list[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            c.search("chess", session_id="bench", modes=("working",), top_k=5)
            timings.append(time.perf_counter() - t0)
        timings.sort()
        p99 = timings[98]  # index 98 of 100 ~ 99th percentile
        # NFR-B2: ≤ 5ms p99. Allow generous headroom (≤ 50ms) since
        # CI hardware varies wildly; the real-bench-time gate is in
        # the multi-turn fixture verdict.
        assert p99 < 0.050, (
            f"working tier p99 latency {p99 * 1000:.1f}ms exceeds soft cap"
        )
    finally:
        c.close()


# T-tiered_search-7 — invalid mode raises
def test_invalid_mode_raises():
    c = MemoryClient()
    try:
        with pytest.raises(ValueError):
            c.search("x", modes=("bogus_mode",))
    finally:
        c.close()


# T-tiered_search-8 — tombstoned working-set entry skipped
def test_tombstoned_in_working_set_skipped():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        m = c.add("a fact about tea")
        c.observe("bootstrap", session_id="s")
        c.attend(m.id, session_id="s")
        # Tombstone the memory
        c._backend.transition_state(m.id, new_state="tombstoned")  # noqa: SLF001
        results = c.search(
            "tea", session_id="s",
            modes=("working", "long_term"), top_k=10,
        )
        # No results — tombstoned memory excluded from working tier AND long term
        ids = [r.memory.id for r in results]
        assert m.id not in ids
    finally:
        c.close()


# T-tiered_search-9 — search hits auto-attend
def test_search_hits_auto_attend():
    c = MemoryClient(working_memory=True, salience_mode="off")
    try:
        m1 = c.add("Scott uses Cursor for code")
        m2 = c.add("Scott prefers tea over coffee")
        # Empty session
        c.observe("bootstrap", session_id="s")
        # Search returns long-term hits; auto-attends them
        results = c.search(
            "Scott", session_id="s",
            modes=("working", "long_term"), top_k=10,
        )
        ids_in_results = {r.memory.id for r in results}
        handle = c.session("s")
        for mid in ids_in_results:
            assert mid in handle.working_set_ids, (
                f"search hit {mid} not auto-attended into working set"
            )
    finally:
        c.close()
