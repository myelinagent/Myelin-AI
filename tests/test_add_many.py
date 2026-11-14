"""Batch writes: MemoryClient.add_many and SqliteBackend.add_memories."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from myelin import Event, MemoryClient
from myelin.backends.sqlite import SqliteBackend


@pytest.fixture
def client():
    c = MemoryClient(user_id="u")
    yield c
    c.close()


def test_strings_and_mappings_are_stored_in_order(client):
    when = datetime(2023, 5, 1, tzinfo=UTC)
    stored = client.add_many(
        [
            "Scott prefers tea",
            {"content": "Scott moved to Berlin", "t_valid": when,
             "metadata": {"session_id": "s1"}, "confidence": 0.9},
        ]
    )
    assert [m.content for m in stored] == ["Scott prefers tea", "Scott moved to Berlin"]
    assert stored[1].t_valid == when
    assert stored[1].metadata == {"session_id": "s1"}
    assert stored[1].confidence == 0.9
    assert client.get(stored[0].id).content == "Scott prefers tea"
    assert [r.memory.id for r in client.search("Berlin")] == [stored[1].id]


def test_user_scope_default_and_per_item(client):
    stored = client.add_many(
        ["for the default user", {"content": "for alice", "user_id": "alice"}]
    )
    assert [m.user_id for m in stored] == ["u", "alice"]
    other = client.add_many(["for bob"], user_id="bob")
    assert other[0].user_id == "bob"


def test_events_are_accepted_as_objects_or_dicts(client):
    stored = client.add_many(
        [
            {"content": "anchored one", "event": Event(event_id="e1", type="chat")},
            {"content": "anchored two", "event": {"event_id": "e2"}},
        ]
    )
    assert [client.get(m.id).event.event_id for m in stored] == ["e1", "e2"]


def test_empty_batch_is_fine(client):
    assert client.add_many([]) == []


@pytest.mark.parametrize(
    ("item", "error"),
    [
        ({"metadata": {}}, "needs a 'content' string"),
        ({"content": 7}, "needs a 'content' string"),
        ({"content": "x", "colour": "red"}, "unexpected keys"),
        ({"content": "x", "event": 5}, "event must be"),
    ],
)
def test_a_bad_item_stores_nothing(client, item, error):
    with pytest.raises(TypeError, match=error):
        client.add_many(["a good one", item])
    assert client._backend.list_memories(user_id="u") == []
    assert client.search("good") == []


def test_backend_batch_is_all_or_nothing():
    backend = SqliteBackend()
    with pytest.raises(ValueError, match="embedding vector"):
        backend.add_memories(
            [
                {"content": "fine", "user_id": "u", "vector": [1.0, 0.0]},
                {"content": "bad vector", "user_id": "u", "vector": [0.0, 0.0]},
            ]
        )
    assert backend.list_memories(user_id="u") == []
    assert backend.search(query="fine", user_id="u") == []
    # The backend is usable afterwards and single writes commit again.
    kept = backend.add_memory(content="later", user_id="u")
    assert backend.get(kept.id) is not None
    backend.close()


def test_batch_embeds_in_one_call_when_the_embedder_can():
    class Batching:
        def __init__(self):
            self.batches: list[list[str]] = []
            self.singles: list[str] = []

        def encode(self, text):
            self.singles.append(text)
            return [1.0, float(len(text))]

        def encode_batch(self, texts):
            self.batches.append(list(texts))
            return [[1.0, float(len(t))] for t in texts]

    embedder = Batching()
    c = MemoryClient(user_id="u", embedder=embedder)
    try:
        stored = c.add_many(["one fact", "another fact", "a third fact"])
        assert embedder.batches == [["one fact", "another fact", "a third fact"]]
        assert embedder.singles == []
        assert c._backend._conn.execute(
            "SELECT COUNT(*) FROM memory_vectors"
        ).fetchone()[0] == 3
        assert len(stored) == 3
    finally:
        c.close()


def test_batch_falls_back_to_single_embedding():
    calls: list[str] = []

    def encode(text):
        calls.append(text)
        return [1.0, 2.0]

    c = MemoryClient(user_id="u", embedder=encode)
    try:
        c.add_many(["one fact", "another fact"])
        assert calls == ["one fact", "another fact"]
    finally:
        c.close()


def test_a_failing_batch_embedder_still_stores_the_memories():
    class Broken:
        def encode(self, text):
            raise RuntimeError("down")

        def encode_batch(self, texts):
            raise RuntimeError("down")

    c = MemoryClient(user_id="u", embedder=Broken())
    try:
        stored = c.add_many(["Scott prefers tea"])
        assert [r.memory.id for r in c.search("tea")] == [stored[0].id]
    finally:
        c.close()


def test_with_a_salience_gate_items_go_through_it_one_by_one():
    c = MemoryClient(user_id="u", salience_mode="heuristic", salience_embedder=lambda t: [1.0])
    try:
        results = c.add_many(["ok", "I absolutely love this amazing wonderful excellent tea"])
        assert len(results) == 2
        assert c.salience_telemetry["n_add_calls"] == 2
    finally:
        c.close()


def test_backends_without_batch_support_are_written_one_at_a_time(client):
    class NoBatch:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "add_memories":
                raise AttributeError(name)
            return getattr(self._inner, name)

    client._backend = NoBatch(client._backend)
    stored = client.add_many(["Scott prefers tea", "Scott moved to Berlin"])
    assert len(stored) == 2
    assert len(client.search("Scott")) == 2


def test_add_many_counts_towards_telemetry(client):
    client.add_many(["a", "b", "c"])
    assert client.salience_telemetry["n_add_calls"] == 3
