"""TDD anchors T-cite_event-1..9.

Maps to spec: FR-A1, FR-A4, FR-A5, FR-A6, NFR-A4, EC-A1, EC-A2, EC-A4.
"""
from __future__ import annotations

import logging

import pytest

from myelin import MemoryClient


@pytest.fixture
def client():
    c = MemoryClient(cite_learning=True)
    yield c
    c.close()


# T-cite_event-1
def test_cite_correct_increments_counter(client):
    m = client.add("Scott prefers tea.")
    client.cite(m.id, answer_correct=True)
    m_after = client.get(m.id)
    assert m_after.cite_correct_count == 1
    assert m_after.cite_incorrect_count == 0
    # Verify cite_events row written
    cur = client._backend._conn.cursor()  # noqa: SLF001
    row = cur.execute(
        "SELECT COUNT(*) FROM cite_events WHERE memory_id = ?", (m.id,)
    ).fetchone()
    assert row[0] == 1


# T-cite_event-2 / EC-A1
def test_cite_on_tombstoned_is_noop(client, caplog):
    m = client.add("To be tombstoned")
    client._backend.transition_state(m.id, new_state="tombstoned")
    with caplog.at_level(logging.WARNING, logger="myelin.client"):
        client.cite(m.id, answer_correct=True)
    cur = client._backend._conn.cursor()  # noqa: SLF001
    row = cur.execute(
        "SELECT COUNT(*) FROM cite_events WHERE memory_id = ?", (m.id,)
    ).fetchone()
    assert row[0] == 0  # no event written
    assert any("unknown or tombstoned" in r.message for r in caplog.records)


# T-cite_event-3 / EC-A4
def test_cite_with_none_correct_skipped(client):
    m = client.add("fact")
    client.cite(m.id, answer_correct=None)  # type: ignore[arg-type]
    m_after = client.get(m.id)
    assert m_after.cite_correct_count == 0
    assert m_after.cite_incorrect_count == 0


# T-cite_event-4 / NFR-A4
def test_cite_disabled_when_env_off(monkeypatch):
    monkeypatch.delenv("MYELIN_CITE_LEARNING", raising=False)
    c = MemoryClient()  # no kwarg, no env
    try:
        m = c.add("fact")
        c.cite(m.id, answer_correct=True)
        m_after = c.get(m.id)
        assert m_after.cite_correct_count == 0
    finally:
        c.close()


# T-cite_event-5 / EC-A3
def test_100_thread_cite_race(client):
    import threading
    m = client.add("concurrent target")
    n_threads = 100

    def hit() -> None:
        client.cite(m.id, answer_correct=True)

    threads = [threading.Thread(target=hit) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m_after = client.get(m.id)
    assert m_after.cite_correct_count == n_threads, (
        f"expected {n_threads} correct cites, got "
        f"{m_after.cite_correct_count} (atomic counter race)"
    )
    cur = client._backend._conn.cursor()  # noqa: SLF001
    n_events = cur.execute(
        "SELECT COUNT(*) FROM cite_events WHERE memory_id = ?", (m.id,)
    ).fetchone()[0]
    assert n_events == n_threads


# T-cite_event-6 / FR-A6
def test_abstain_event_recorded(client):
    client.abstain_event("a question with no answer in substrate")
    cur = client._backend._conn.cursor()  # noqa: SLF001
    n = cur.execute("SELECT COUNT(*) FROM abstain_events").fetchone()[0]
    assert n == 1


# T-cite_event-7 — query PII hashing under strict-"1" env
def test_query_pii_hashed_when_env_set(monkeypatch):
    monkeypatch.setenv("MYELIN_CITE_HASH_QUERIES", "1")
    c = MemoryClient(cite_learning=True)
    try:
        m = c.add("fact")
        c.cite(m.id, answer_correct=True, query="user's secret PII")
        cur = c._backend._conn.cursor()  # noqa: SLF001
        row = cur.execute(
            "SELECT query_hash_or_text FROM cite_events WHERE memory_id = ?",
            (m.id,),
        ).fetchone()
        assert row[0] is not None
        assert row[0] != "user's secret PII"
        assert len(row[0]) == 16  # 16-char hex hash
        assert all(c in "0123456789abcdef" for c in row[0])
    finally:
        c.close()


# T-cite_event-8 / NFR-A4 — loose-truthy env values are refused
@pytest.mark.parametrize("env_val", ["true", "True", "yes", "1.0", "TRUE"])
def test_loose_truthy_env_does_not_enable(monkeypatch, env_val):
    monkeypatch.setenv("MYELIN_CITE_LEARNING", env_val)
    c = MemoryClient()  # no kwarg
    try:
        m = c.add("fact")
        c.cite(m.id, answer_correct=True)
        m_after = c.get(m.id)
        assert m_after.cite_correct_count == 0, (
            f"loose-truthy env {env_val!r} must NOT enable cite learning"
        )
    finally:
        c.close()


# T-cite_event-9 — session bridge (touches Step 11)
def test_cite_with_session_id_attends():
    c = MemoryClient(cite_learning=True, working_memory=True)
    try:
        m = c.add("Alice loves chess.")
        # Pre-create the session by observing one turn
        c.observe("bootstrap", session_id="sess-1")
        c.cite(m.id, answer_correct=True, session_id="sess-1")
        handle = c.session("sess-1")
        assert m.id in handle.working_set_ids, (
            "cite with session_id must auto-attend memory to working set"
        )
    finally:
        c.close()


# Telemetry shape
def test_cite_telemetry_shape(client):
    m1 = client.add("a")
    m2 = client.add("b")
    client.cite(m1.id, answer_correct=True)
    client.cite(m2.id, answer_correct=False)
    client.abstain_event("missing query")

    tel = client.cite_telemetry
    assert tel["cite_learning_enabled"] is True
    assert tel["n_cite_calls"] == 2
    assert tel["n_cite_correct"] == 1
    assert tel["n_cite_incorrect"] == 1
    assert tel["n_abstain_events"] == 1
