"""Event-anchored fact tests — P1 H044 v2 semantics.

Append-only writes with event provenance; `as_of=t` queries return the
right historical state across superseding writes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from myelin import Event, MemoryClient


@pytest.fixture
def client():
    c = MemoryClient(user_id="scott")
    yield c
    c.close()


def test_add_with_event_object(client):
    ev = Event(
        event_id="meeting-001",
        type="meeting",
        anchor_datetime=datetime(2026, 1, 15, 14, 0, tzinfo=UTC),
        location="Zoom",
        participant_ids=("scott", "alice"),
        evidence_turn_ids=("turn-12", "turn-13"),
    )
    mem = client.add("Scott decided to switch to weekly syncs.", event=ev)
    assert mem.event is not None
    assert mem.event.event_id == "meeting-001"
    assert mem.event.type == "meeting"
    assert mem.event.location == "Zoom"
    assert mem.event.participant_ids == ("scott", "alice")


def test_add_with_event_dict(client):
    mem = client.add(
        "Scott decided to switch to weekly syncs.",
        event={
            "event_id": "meeting-001",
            "type": "meeting",
            "anchor_datetime": "2026-01-15T14:00:00+00:00",
            "location": "Zoom",
            "participant_ids": ["scott", "alice"],
        },
    )
    assert mem.event is not None
    assert mem.event.event_id == "meeting-001"
    assert mem.event.anchor_datetime == datetime(2026, 1, 15, 14, 0, tzinfo=UTC)


def test_event_persists_after_search(client):
    client.add(
        "Scott prefers tea.",
        event={"event_id": "ev-1", "type": "preference-stated"},
    )
    hits = client.search("tea")
    assert hits[0].memory.event is not None
    assert hits[0].memory.event.event_id == "ev-1"


def test_event_persists_through_get(client):
    mem = client.add(
        "Scott prefers tea.",
        event={"event_id": "ev-1", "type": "preference"},
    )
    fetched = client.get(mem.id)
    assert fetched.event is not None
    assert fetched.event.event_id == "ev-1"
    assert fetched.event.type == "preference"


def test_append_only_writes_dont_clobber(client):
    """P1 H044 v2 — multiple writes on the same (subject, predicate)
    coexist as separate memory rows. No destructive supersession."""
    client.add(
        "Scott prefers tea.",
        metadata={"subject": "Scott", "predicate": "prefers"},
        event={"event_id": "ev-1"},
    )
    client.add(
        "Scott prefers coffee.",
        metadata={"subject": "Scott", "predicate": "prefers"},
        event={"event_id": "ev-2"},
    )
    hits = client.search("Scott prefers")
    contents = {h.memory.content for h in hits}
    # Both facts coexist — multi-value predicate, no auto-invalidation.
    assert "Scott prefers tea." in contents
    assert "Scott prefers coffee." in contents


def test_as_of_returns_historical_state(client):
    """A fact written at t1 must NOT be visible to a search with
    as_of=t0 < t1."""
    jan_mem = client.add("January fact.")
    feb_mem = client.add("February fact.")
    # The backend assigns t_valid=now at write time; we backdate both
    # rows via a test-only shim so the as_of axis is testable without
    # waiting real wall-clock time.
    backend = client._backend  # noqa: SLF001 — test access
    with backend._lock:
        backend._conn.execute(
            "UPDATE memories SET t_valid = ? WHERE id = ?",
            ("2026-01-10T00:00:00+00:00", jan_mem.id),
        )
        backend._conn.execute(
            "UPDATE memories SET t_valid = ? WHERE id = ?",
            ("2026-03-01T00:00:00+00:00", feb_mem.id),
        )
        backend._conn.commit()
    # as_of=Jan 15 should only see the January fact (March one not yet valid).
    hits_t0 = client.search("fact", as_of=datetime(2026, 1, 15, tzinfo=UTC))
    contents = {h.memory.content for h in hits_t0}
    assert "January fact." in contents
    assert "February fact." not in contents
    # No as_of -> both visible.
    hits_now = client.search("fact")
    assert len(hits_now) == 2


def test_as_of_with_naive_datetime_works(client):
    """as_of accepts naive datetimes (treated as UTC) — common
    caller mistake we should handle gracefully."""
    client.add("a fact about something")
    naive = datetime(2030, 1, 1)  # no tzinfo
    hits = client.search("fact", as_of=naive)
    assert len(hits) == 1


def test_event_id_only_minimal_dict(client):
    """Event dicts with only event_id should work — other fields default."""
    mem = client.add("...", event={"event_id": "just-an-id"})
    assert mem.event is not None
    assert mem.event.event_id == "just-an-id"
    assert mem.event.type == ""
    assert mem.event.location == ""


def test_multiple_facts_share_event(client):
    """Multiple facts can anchor to the same event_id (idempotent upsert)."""
    ev = {"event_id": "ev-shared", "type": "meeting"}
    m1 = client.add("Alice mentioned X.", event=ev)
    m2 = client.add("Bob mentioned Y.", event=ev)
    assert m1.event.event_id == m2.event.event_id == "ev-shared"
    # Both retrievable.
    hits = client.search("mentioned")
    assert len(hits) == 2
