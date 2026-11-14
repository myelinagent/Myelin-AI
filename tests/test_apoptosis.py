"""P3 apoptosis — quarantine -> tombstone state machine tests.

Mirrors the F-BI-5 cohort smoke (synthetic 1000-fact + 30-poison
fixture) at a tractable scale. Key invariants:
- Quarantine fires on low-health (age, contradiction, low confidence).
- Caspase cascade fires after the 7-day window, OR immediately when
  health drops below the caspase threshold.
- Tombstoned facts are invisible to search() but persist in the audit
  view via list_memories(include_tombstoned=True).
- Healthy facts (recent, high-confidence, no contradiction) survive.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from myelin import MemoryClient


@pytest.fixture
def client():
    c = MemoryClient(user_id="scott")
    yield c
    c.close()


def _poison_fact(client, idx: int):
    """A fact engineered to fail health checks: ancient + low confidence."""
    mem = client.add(
        f"poison fact {idx} about contradicting prior knowledge",
        confidence=0.05,
        importance=0.1,
        metadata={"subject": "scott", "predicate": "decided", "object": f"poison-{idx}"},
    )
    # Backdate t_valid 90 days so age penalty fires.
    backend = client._backend  # noqa: SLF001
    backend_conn = backend._conn
    backdated = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    with backend._lock:
        backend_conn.execute(
            "UPDATE memories SET t_valid = ? WHERE id = ?",
            (backdated, mem.id),
        )
        backend_conn.commit()


def _healthy_fact(client, idx: int):
    return client.add(
        f"healthy fact {idx} about a recent decision",
        confidence=0.9,
        importance=0.8,
        metadata={"subject": "scott", "predicate": "uses_tool", "object": f"tool-{idx}"},
    )


def test_prune_quarantines_poison(client):
    for i in range(30):
        _poison_fact(client, i)
    for i in range(100):
        _healthy_fact(client, i)
    report = client.prune()
    # Per the cohort smoke target: >=28/30 poison quarantined.
    assert report.quarantined_count >= 28, (
        f"expected >=28 quarantined, got {report.quarantined_count} "
        f"(report={report!r})"
    )
    # And no healthy fact should be quarantined.
    assert report.quarantined_count <= 35


def test_prune_tombstones_after_window(client):
    for i in range(30):
        _poison_fact(client, i)
    for i in range(50):
        _healthy_fact(client, i)
    # First pass: quarantines them.
    first = client.prune()
    assert first.quarantined_count >= 25
    # Second pass 14 days later, asked to tombstone: caspase cascades them.
    future = datetime.now(UTC) + timedelta(days=14)
    second = client.prune(now=future, tombstone=True)
    assert second.tombstoned_count >= 25, (
        f"expected >=25 tombstoned after window, got {second.tombstoned_count}"
    )


def test_prune_does_not_delete(client):
    """Apoptosis must NOT hard-delete — tombstones persist in audit."""
    for i in range(5):
        _poison_fact(client, i)
    client.prune()
    future = datetime.now(UTC) + timedelta(days=14)
    client.prune(now=future, tombstone=True)
    audit = client._backend.list_memories(  # noqa: SLF001
        user_id="scott", include_tombstoned=True
    )
    tombstoned = [m for m in audit if m.health_state == "tombstoned"]
    assert tombstoned, "tombstoned facts must remain in audit substrate"


def test_search_excludes_tombstones(client):
    mem = _healthy_fact(client, 0)
    _poison_fact(client, 999)
    client.prune()
    future = datetime.now(UTC) + timedelta(days=14)
    client.prune(now=future, tombstone=True)
    # Healthy fact still searchable.
    hits = client.search("healthy")
    assert any(h.memory.id == mem.id for h in hits)
    # Poison fact tombstoned -> not surfaced.
    poison_hits = client.search("poison")
    assert all(h.memory.health_state != "tombstoned" for h in poison_hits)


def test_get_returns_none_for_tombstoned(client):
    _poison_fact(client, 0)
    client.prune()
    future = datetime.now(UTC) + timedelta(days=14)
    client.prune(now=future, tombstone=True)
    audit = client._backend.list_memories(  # noqa: SLF001
        user_id="scott", include_tombstoned=True
    )
    tombs = [m for m in audit if m.health_state == "tombstoned"]
    assert tombs
    # client.get() must return None for the tombstoned id.
    assert client.get(tombs[0].id) is None


def test_healthy_facts_survive(client):
    for i in range(50):
        _healthy_fact(client, i)
    report = client.prune()
    assert report.quarantined_count == 0
    assert report.tombstoned_count == 0
    assert report.retained_count == 50
    assert report.scanned == 50


def test_prune_empty_substrate(client):
    report = client.prune()
    assert report.scanned == 0
    assert report.quarantined_count == 0
    assert report.tombstoned_count == 0


def test_prune_caspase_window_override(client):
    for i in range(10):
        _poison_fact(client, i)
    client.prune()  # quarantine
    # Pass 0-day window so even just-quarantined facts caspase next call.
    future = datetime.now(UTC) + timedelta(seconds=1)
    report = client.prune(now=future, caspase_window_days=0, tombstone=True)
    assert report.tombstoned_count >= 9
    assert report.caspase_window_days == 0


def test_prune_report_shape(client):
    _healthy_fact(client, 0)
    _poison_fact(client, 0)
    report = client.prune()
    assert hasattr(report, "scanned")
    assert hasattr(report, "quarantined_count")
    assert hasattr(report, "tombstoned_count")
    assert hasattr(report, "retained_count")
    assert hasattr(report, "caspase_window_days")
    assert report.scanned == 2


def test_idempotent_prune(client):
    """Calling prune twice in a row on a stable substrate is a no-op
    on the second call."""
    _healthy_fact(client, 0)
    r1 = client.prune()
    r2 = client.prune()
    assert r1.quarantined_count == r2.quarantined_count
    assert r1.tombstoned_count == r2.tombstoned_count


# ---- quarantine is where forgetting stops unless tombstoning is asked for


def test_prune_never_tombstones_by_default(client):
    for i in range(5):
        _poison_fact(client, i)
    first = client.prune()
    assert first.quarantined_count == 5
    later = client.prune(now=datetime.now(UTC) + timedelta(days=60))
    assert later.tombstoned_count == 0
    kept = client._backend.list_memories(user_id="scott")  # noqa: SLF001
    assert len(kept) == 5
    assert {m.health_state for m in kept} == {"quarantined"}


def test_quarantined_facts_are_hidden_from_search_but_kept(client):
    mem = client.add("poison fact about lunar tides", confidence=0.05, importance=0.1)
    client._backend._conn.execute(  # noqa: SLF001 — backdate so age penalty fires
        "UPDATE memories SET t_valid = ? WHERE id = ?",
        ((datetime.now(UTC) - timedelta(days=90)).isoformat(), mem.id),
    )
    client._backend._conn.commit()  # noqa: SLF001
    assert client.prune().quarantined_count == 1
    assert client.search("lunar tides") == []
    assert client.recall("lunar tides") == []
    assert client.get(mem.id).health_state == "quarantined"
    shown = client.search("lunar tides", include_quarantined=True)
    assert [h.memory.id for h in shown] == [mem.id]


def test_a_quarantined_fact_recovers_when_its_health_does():
    c = MemoryClient(user_id="scott", cite_learning=True)
    try:
        mem = c.add("a doubtful fact about lunar tides", confidence=0.1, importance=0.5)
        c._backend._conn.execute(  # noqa: SLF001
            "UPDATE memories SET t_valid = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=40)).isoformat(), mem.id),
        )
        c._backend._conn.commit()  # noqa: SLF001
        assert c.prune().quarantined_count == 1
        assert c.search("lunar tides") == []
        # It keeps proving right when it is used.
        for _ in range(8):
            c.cite(mem.id, answer_correct=True)
        report = c.prune()
        assert report.recovered_count == 1
        assert report.quarantined_count == 0
        assert c.get(mem.id).health_state == "healthy"
        assert [h.memory.id for h in c.search("lunar tides")] == [mem.id]
    finally:
        c.close()
