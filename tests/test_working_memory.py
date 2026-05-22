"""TDD anchors T-working_methods-1..10.

Maps to spec: FR-B1, FR-B2, FR-B3, FR-B4, EC-B1, EC-B2, EC-B6.
"""
from __future__ import annotations

import pytest

from myelin import MemoryClient


@pytest.fixture
def client():
    c = MemoryClient(working_memory=True, salience_mode="off")
    yield c
    c.close()


# T-working_methods-1
def test_observe_is_ram_only(client):
    t = client.observe("hello", session_id="s")
    assert t.turn_id
    # No row in memories table yet
    cur = client._backend._conn.cursor()  # noqa: SLF001
    n = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert n == 0


# T-working_methods-2
def test_commit_with_no_gate(client):
    t = client.observe("a fact", session_id="s")
    committed = client.commit([t.turn_id], session_id="s")
    assert len(committed) == 1
    assert committed[0].content == "a fact"
    # Row in memories
    cur = client._backend._conn.cursor()  # noqa: SLF001
    n = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert n == 1


# T-working_methods-3 — salience gate at commit drops correctly
def test_commit_respects_salience_drop():
    """A stub gate that always tags False should cause commit() to drop
    the turn (no Memory persisted)."""
    class StubDecision:
        def __init__(self):
            self.utility_class = "transient"
            self.utility_score = 0.0
            self.surprise_zscore = 0.0
            self.rpe = 0.0
            self.tag = False
            self.cached = False
            self.reason = "stubbed-drop"

    class StubGate:
        def __init__(self):
            self.stats = {"llm_calls_made": 0}

        def evaluate(self, *, content, user_id):
            return StubDecision()

    c = MemoryClient(working_memory=True, salience_mode="dmem", dmem_gate=StubGate())
    try:
        t = c.observe("stuff", session_id="s")
        committed = c.commit([t.turn_id], session_id="s")
        assert committed == []  # gate dropped
        cur = c._backend._conn.cursor()  # noqa: SLF001
        n = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 0
    finally:
        c.close()


# T-working_methods-4 / NFR-B3 — observe does NOT invoke judge
def test_observe_does_not_invoke_judge():
    class TrackingGate:
        def __init__(self):
            self.stats = {"llm_calls_made": 0}
            self.n_calls = 0

        def evaluate(self, *, content, user_id):
            self.n_calls += 1
            class D:
                utility_class = "persistent"
                utility_score = 0.9
                surprise_zscore = 0.0
                rpe = 0.9
                tag = True
                cached = False
                reason = "tracked"
            return D()

    gate = TrackingGate()
    c = MemoryClient(working_memory=True, salience_mode="dmem", dmem_gate=gate)
    try:
        observed = [c.observe(f"turn {i}", session_id="s") for i in range(50)]
        # Gate has NOT been invoked yet — observe is gate-free.
        assert gate.n_calls == 0, "observe() must not invoke the gate"

        # Commit only 10 of the 50
        c.commit([t.turn_id for t in observed[:10]], session_id="s")
        # Gate fired 10 times (one per committed turn)
        assert gate.n_calls == 10
    finally:
        c.close()


# T-working_methods-5 / FR-B3
def test_attend_adds_to_working_set(client):
    m = client.add("an existing fact")
    client.attend(m.id, session_id="s")
    handle = client.session("s")
    assert m.id in handle.working_set_ids


# T-working_methods-6 / EC-B2
def test_attend_on_tombstoned_noop(client):
    m = client.add("to be killed")
    client._backend.transition_state(m.id, new_state="tombstoned")  # noqa: SLF001
    client.observe("bootstrap session", session_id="s")  # create session
    client.attend(m.id, session_id="s")
    handle = client.session("s")
    assert m.id not in handle.working_set_ids


# T-working_methods-7 / EC-B6
def test_commit_expired_session_raises(client):
    with pytest.raises(LookupError):
        client.commit(["any-turn-id"], session_id="never-created")


# T-working_methods-8 — observe fallback when working memory disabled
def test_observe_fallback_when_disabled(monkeypatch):
    monkeypatch.delenv("MYELIN_WORKING_MEMORY", raising=False)
    c = MemoryClient(working_memory=False)
    try:
        t = c.observe("fallback content", session_id="ignored")
        # Memory IS persisted (fallback path)
        m = c.get(t.turn_id)
        assert m is not None
        assert m.content == "fallback content"
    finally:
        c.close()


# T-working_methods-9 / EC-B1
def test_commit_cross_session_raises(client):
    t1 = client.observe("a", session_id="s1")
    client.observe("b", session_id="s2")
    # turn_id from s1 passed to commit for s2 -> LookupError
    with pytest.raises(LookupError):
        client.commit([t1.turn_id], session_id="s2")


# T-working_methods-10 / FR-B4
def test_session_handle_shape(client):
    t1 = client.observe("a", session_id="s")
    client.observe("b", session_id="s")
    client.commit([t1.turn_id], session_id="s")
    handle = client.session("s")
    assert handle.session_id == "s"
    assert handle.n_observed == 1  # one remaining
    assert handle.n_committed == 1
    assert len(handle.working_set_ids) == 1  # auto-attended on commit


def test_observe_empty_session_id_raises(client):
    with pytest.raises(ValueError):
        client.observe("x", session_id="")


def test_session_unknown_raises(client):
    with pytest.raises(LookupError):
        client.session("never-created")
