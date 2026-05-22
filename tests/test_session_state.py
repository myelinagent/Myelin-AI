"""TDD anchors T-session_state-1..9.

Maps to spec: FR-B1, FR-B6, NFR-B4, EC-B3.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from myelin._internal.session import (
    SessionRegistry,
    SessionState,
    _new_turn_id,
)


@pytest.fixture
def registry():
    r = SessionRegistry(
        ttl_seconds=10,
        max_concurrent=5,
        observed_ring_max=3,
        sweeper_interval_seconds=10,
    )
    yield r
    r._shutdown()


# T-session_state-1
def test_lazy_session_creation(registry):
    s = registry.get_or_create("sess-A", "user-A")
    assert s.session_id == "sess-A"
    assert s.user_id == "user-A"
    assert s.n_committed == 0
    assert len(s.observed_buffer) == 0


# T-session_state-1 (idempotency)
def test_get_or_create_idempotent(registry):
    a = registry.get_or_create("s", "u")
    b = registry.get_or_create("s", "u")
    assert a is b, "same session_id must return the same SessionState"


# T-session_state-2 — ring buffer FIFO
def test_observed_buffer_fifo(registry):
    state = registry.get_or_create("s", "u")
    for i in range(5):
        state.append_observed(turn_id=f"t{i}", content=f"turn {i}", metadata=None)
    contents = [t.content for t in state.observed_buffer]
    assert contents == ["turn 2", "turn 3", "turn 4"]  # cap=3 -> last 3


# T-session_state-3 — working set LRU
def test_working_set_lru_cap(registry, monkeypatch):
    """Cap is enforced by SessionState.attend(). The constant is
    imported into session.py at module-load time, so we patch it on
    the session module — patching constants.py does nothing once
    session.py is loaded."""
    state = registry.get_or_create("s", "u")
    from myelin._internal import session as session_mod
    monkeypatch.setattr(session_mod, "WORKING_SET_MAX", 5)
    for i in range(10):
        state.attend(f"m{i}")
    assert len(state.working_set) == 5
    assert "m0" not in state.working_set
    assert "m9" in state.working_set


# T-session_state-4 — TTL sweeper expires idle sessions
def test_expired_session_emits_event():
    events: list[dict] = []
    r = SessionRegistry(
        ttl_seconds=1,
        max_concurrent=5,
        observed_ring_max=10,
        sweeper_interval_seconds=1,
    )
    r._event_sink = lambda e: events.append(e)
    try:
        state = r.get_or_create("expire-me", "user-C")
        state.last_active_at = datetime.now(UTC) - timedelta(seconds=5)
        time.sleep(2.0)
        assert r.lookup("expire-me") is None
        assert any(
            e["event"] == "session.expired"
            and e["session_id"] == "expire-me"
            for e in events
        )
    finally:
        r._shutdown()


# T-session_state-5 — LRU eviction at MAX_CONCURRENT_SESSIONS
def test_lru_eviction_at_cap():
    r = SessionRegistry(
        ttl_seconds=100,
        max_concurrent=2,
        observed_ring_max=10,
        sweeper_interval_seconds=100,
    )
    try:
        r.get_or_create("s1", "u")
        time.sleep(0.01)
        r.get_or_create("s2", "u")
        time.sleep(0.01)
        r.get_or_create("s3", "u")
        assert r.lookup("s1") is None
        assert r.lookup("s2") is not None
        assert r.lookup("s3") is not None
    finally:
        r._shutdown()


# T-session_state-6 / EC-B3
def test_concurrent_observe_safe(registry):
    state = registry.get_or_create("s", "u")

    def hit(i: int) -> None:
        state.append_observed(turn_id=_new_turn_id(), content=f"t{i}", metadata=None)

    threads = [threading.Thread(target=hit, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # observed_ring_max=3 for this registry — only last 3 remain
    assert len(state.observed_buffer) == 3


# T-session_state-7 / commit drain
def test_drain_selective(registry):
    state = registry.get_or_create("s", "u")
    t_a = state.append_observed(turn_id="A", content="a", metadata=None)
    t_b = state.append_observed(turn_id="B", content="b", metadata=None)
    t_c = state.append_observed(turn_id="C", content="c", metadata=None)
    drained = state.drain_observed(["B"])
    assert [d.turn_id for d in drained] == ["B"]
    assert [t.turn_id for t in state.observed_buffer] == ["A", "C"]
    assert state.n_committed == 1


# T-session_state-8 / EC-B1, EC-B6
def test_drain_missing_raises(registry):
    state = registry.get_or_create("s", "u")
    state.append_observed(turn_id="A", content="a", metadata=None)
    with pytest.raises(LookupError):
        state.drain_observed(["nonexistent-id"])
    # Original turn still in buffer
    assert [t.turn_id for t in state.observed_buffer] == ["A"]


# T-session_state-9 — re-attend promotes to MRU
def test_reattend_promotes_mru(registry):
    state = registry.get_or_create("s", "u")
    state.attend("m1")
    state.attend("m2")
    state.attend("m3")
    state.attend("m1")  # promote
    keys = list(state.working_set.keys())
    assert keys == ["m2", "m3", "m1"]


# Singleton + reset_for_testing
def test_singleton_and_reset():
    a = SessionRegistry.get_instance()
    b = SessionRegistry.get_instance()
    assert a is b
    SessionRegistry.reset_for_testing()
    c = SessionRegistry.get_instance()
    assert c is not a
    SessionRegistry.reset_for_testing()
