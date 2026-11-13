"""Batch memory fetch and batch event loading in the SQLite backend."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from myelin import Event, MemoryClient
from myelin.backends.sqlite import SqliteBackend


@pytest.fixture
def backend():
    b = SqliteBackend()
    yield b
    b.close()


def _add(backend: SqliteBackend, content: str, *, event: Event | None = None):
    return backend.add_memory(content=content, user_id="u", event=event)


def _event(event_id: str) -> Event:
    return Event(
        event_id=event_id,
        type="conversation",
        anchor_datetime=datetime(2026, 1, 1, tzinfo=UTC),
        location="kitchen",
        participant_ids=("alice", "bob"),
    )


def _count_event_queries(backend: SqliteBackend) -> list[str]:
    seen: list[str] = []
    backend._conn.set_trace_callback(
        lambda sql: seen.append(sql) if "FROM events" in sql else None
    )
    return seen


def test_get_batch_empty(backend):
    assert backend.get_batch([]) == []


def test_get_batch_preserves_request_order(backend):
    ids = [_add(backend, f"fact {i}").id for i in range(5)]
    wanted = [ids[3], ids[0], ids[4]]
    assert [m.id for m in backend.get_batch(wanted)] == wanted


def test_get_batch_omits_unknown_and_tombstoned(backend):
    live = _add(backend, "live fact")
    dead = _add(backend, "dead fact")
    backend.transition_state(dead.id, new_state="tombstoned")
    got = backend.get_batch(["nope", dead.id, live.id])
    assert [m.id for m in got] == [live.id]


def test_get_batch_dedupes_repeated_ids(backend):
    m = _add(backend, "only once")
    assert [x.id for x in backend.get_batch([m.id, m.id])] == [m.id]


def test_get_batch_matches_get(backend):
    plain = _add(backend, "no event")
    anchored = _add(backend, "with event", event=_event("ev-1"))
    for m in (plain, anchored):
        assert backend.get_batch([m.id]) == [backend.get(m.id)]
    assert backend.get(anchored.id).event == _event("ev-1")
    assert backend.get("nope") is None


def test_get_batch_beyond_sql_parameter_limit(backend):
    ids = [_add(backend, f"fact {i}").id for i in range(2100)]
    assert [m.id for m in backend.get_batch(ids)] == ids


def test_list_memories_loads_events_in_one_query(backend):
    for i in range(20):
        _add(backend, f"anchored {i}", event=_event(f"ev-{i}"))
    _add(backend, "unanchored")
    queries = _count_event_queries(backend)
    memories = backend.list_memories(user_id="u")
    assert len(queries) == 1
    assert len(memories) == 21
    by_content = {m.content: m for m in memories}
    assert by_content["unanchored"].event is None
    for i in range(20):
        assert by_content[f"anchored {i}"].event == _event(f"ev-{i}")


def test_list_memories_without_events_skips_event_query(backend):
    _add(backend, "plain")
    queries = _count_event_queries(backend)
    assert len(backend.list_memories(user_id="u")) == 1
    assert queries == []


def test_shared_event_attached_to_every_memory(backend):
    a = _add(backend, "first", event=_event("shared"))
    b = _add(backend, "second", event=_event("shared"))
    got = backend.get_batch([a.id, b.id])
    assert [m.event.event_id for m in got] == ["shared", "shared"]


def test_tombstoned_listing_keeps_events(backend):
    m = _add(backend, "going away", event=_event("ev-t"))
    backend.transition_state(m.id, new_state="tombstoned")
    listed = backend.list_memories(user_id="u", include_tombstoned=True)
    assert [x.event.event_id for x in listed] == ["ev-t"]


def test_search_loads_events_in_one_query(backend):
    for i in range(10):
        _add(backend, f"tea fact {i}", event=_event(f"ev-{i}"))
    queries = _count_event_queries(backend)
    results = backend.search(query="tea", user_id="u", top_k=10)
    assert len(queries) == 1
    assert all(r.memory.event is not None for r in results)


def _working_client() -> tuple[MemoryClient, str]:
    c = MemoryClient(working_memory=True, salience_mode="off")
    t = c.observe("My daughter Melanie is wonderful", session_id="s1")
    committed = c.commit([t.turn_id], session_id="s1")
    return c, committed[0].id


def test_working_set_search_uses_get_batch():
    c, memory_id = _working_client()
    try:
        calls: list[list[str]] = []
        real = c._backend.get_batch

        def spy(ids):
            calls.append(list(ids))
            return real(ids)

        c._backend.get_batch = spy
        results = c.search("Melanie", session_id="s1", modes=("working",))
        assert [r.memory.id for r in results] == [memory_id]
        assert calls == [[memory_id]]
    finally:
        c.close()


def test_working_set_search_falls_back_without_get_batch():
    """A backend that only implements get() still serves the working tier."""

    class GetOnly:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "get_batch":
                raise AttributeError(name)
            return getattr(self._inner, name)

    c, memory_id = _working_client()
    try:
        c._backend = GetOnly(c._backend)
        results = c.search("Melanie", session_id="s1", modes=("working",))
        assert [r.memory.id for r in results] == [memory_id]
    finally:
        c.close()
