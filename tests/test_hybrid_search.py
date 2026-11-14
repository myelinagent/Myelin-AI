"""Hybrid (lexical + embedding) search."""

from __future__ import annotations

import builtins
import sys
from datetime import UTC, datetime

import pytest

from myelin import MemoryClient
from myelin.backends.sqlite import SqliteBackend

# A toy embedder: each word in a small vocabulary is one dimension, and
# words meaning the same thing share a dimension. "dog" and "greyhound"
# are close in this space although they share no letters.
_CONCEPTS = {
    "dog": 0, "greyhound": 0, "puppy": 0, "pet": 0,
    "tea": 1, "oolong": 1, "drink": 1, "beverage": 1,
    "paris": 2, "berlin": 2, "city": 2, "lives": 2,
    "weather": 3, "rain": 3,
}


class ToyEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)
        vector = [0.0] * 4
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            if word in _CONCEPTS:
                vector[_CONCEPTS[word]] += 1.0
        return vector if any(vector) else [0.01, 0.01, 0.01, 0.01]


@pytest.fixture
def embedder():
    return ToyEmbedder()


@pytest.fixture
def client(embedder):
    c = MemoryClient(user_id="u", embedder=embedder)
    yield c
    c.close()


def _ids(results):
    return [r.memory.id for r in results]


def test_vector_channel_finds_what_words_alone_miss(client):
    target = client.add("Scott adopted a greyhound")
    client.add("It may rain; the weather is poor")
    lexical_only = MemoryClient(backend=client._backend, user_id="u")
    assert lexical_only.search("What pet does he have?") == []
    assert _ids(client.search("What pet does he have?"))[0] == target.id


def test_lexical_matches_still_count(client):
    exact = client.add("The serial number is ZX81")
    client.add("Scott drinks oolong tea")
    assert _ids(client.search("ZX81"))[0] == exact.id


def test_results_are_ranked_and_respect_top_k(client):
    for i in range(10):
        client.add(f"Note {i}: oolong tea")
    hits = client.search("tea", top_k=4)
    assert len(hits) == 4
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_is_scoped_to_the_user(client):
    client.add("alice has a greyhound", user_id="alice")
    mine = client.add("bob has a puppy", user_id="bob")
    assert _ids(client.search("dog", user_id="bob")) == [mine.id]


def test_as_of_applies_to_fused_results(client):
    old = client.add("Scott lives in Paris", t_valid=datetime(2023, 1, 1, tzinfo=UTC))
    client.add("Scott lives in Berlin", t_valid=datetime(2024, 1, 1, tzinfo=UTC))
    hits = client.search("city", as_of=datetime(2023, 6, 1))  # noqa: DTZ001 — naive on purpose
    assert _ids(hits) == [old.id]


def test_update_delete_and_tombstone_keep_vectors_in_step(client, embedder):
    mem = client.add("Scott drinks oolong tea")
    other = client.add("The weather is poor")
    client.update(mem.id, content="Scott adopted a greyhound")
    assert _ids(client.search("pet"))[0] == mem.id
    client.delete(mem.id)
    client._backend.transition_state(other.id, new_state="tombstoned")
    assert client.search("pet") == []
    assert client._backend._conn.execute(
        "SELECT COUNT(*) FROM memory_vectors"
    ).fetchone()[0] == 0


def test_backfill_embeds_memories_written_without_an_embedder(embedder):
    plain = MemoryClient(user_id="u")
    target = plain.add("Scott adopted a greyhound")
    plain.add("Alice has a puppy", user_id="alice")
    upgraded = MemoryClient(backend=plain._backend, user_id="u", embedder=embedder)
    try:
        assert upgraded.search("pet") == []
        assert upgraded.backfill_embeddings(user_id="u") == 1
        assert _ids(upgraded.search("pet")) == [target.id]
        assert upgraded.backfill_embeddings() == 1  # alice's
        assert upgraded.backfill_embeddings() == 0
        assert plain.backfill_embeddings() == 0  # no embedder: nothing to do
    finally:
        plain.close()


def test_committed_turns_are_embedded(embedder):
    c = MemoryClient(user_id="u", embedder=embedder, working_memory=True)
    try:
        turn = c.observe("Scott adopted a greyhound", session_id="s")
        (mem,) = c.commit([turn.turn_id], session_id="s")
        assert _ids(c.search("pet")) == [mem.id]
    finally:
        c.close()


def test_a_failing_embedder_degrades_to_lexical_search(caplog):
    class Broken:
        def encode(self, text):
            raise RuntimeError("model not loaded")

    c = MemoryClient(user_id="u", embedder=Broken())
    try:
        mem = c.add("Scott drinks oolong tea")
        assert _ids(c.search("oolong")) == [mem.id]
    finally:
        c.close()


def test_a_plain_callable_works_as_an_embedder():
    c = MemoryClient(user_id="u", embedder=ToyEmbedder().encode)
    try:
        mem = c.add("Scott adopted a greyhound")
        assert _ids(c.search("pet")) == [mem.id]
    finally:
        c.close()


def test_vectors_from_another_model_size_are_ignored(client):
    mem = client.add("Scott adopted a greyhound")
    other = MemoryClient(
        backend=client._backend, user_id="u", embedder=lambda text: [1.0, 2.0]
    )
    assert _ids(other.search("greyhound")) == [mem.id]  # lexical still works
    assert other.search("pet") == []


@pytest.mark.parametrize("bad", [[], [0.0, 0.0], [float("nan"), 1.0]])
def test_unusable_vectors_are_rejected(bad):
    backend = SqliteBackend()
    with pytest.raises(ValueError, match="embedding vector"):
        backend.add_memory(content="x", user_id="u", vector=bad)
    assert backend.list_memories(user_id="u") == []
    backend.close()


def test_hybrid_search_works_without_numpy(monkeypatch, embedder):
    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_numpy)
    c = MemoryClient(user_id="u", embedder=embedder)
    try:
        target = c.add("Scott adopted a greyhound")
        c.add("The weather is poor")
        assert _ids(c.search("pet"))[0] == target.id
    finally:
        c.close()


def test_queries_are_embedded_once_and_only_with_an_embedder(client, embedder):
    client.add("Scott adopted a greyhound")
    embedder.calls.clear()
    client.search("pet")
    assert embedder.calls == ["pet"]


def test_static_embedder_reports_a_missing_dependency(monkeypatch):
    from myelin.embedders import StaticEmbedder

    monkeypatch.setitem(sys.modules, "model2vec", None)
    with pytest.raises(ImportError, match=r"myelin-ai\[static\]"):
        StaticEmbedder().encode("hello")


def test_vectors_written_through_another_connection_are_searched(tmp_path):
    """The in-memory vector cache must notice another process's writes."""
    from myelin.backends.sqlite import SqliteBackend

    path = str(tmp_path / "shared.db")
    reader, writer = SqliteBackend(path), SqliteBackend(path)
    try:
        first = writer.add_memories(
            [{"content": "alpha", "user_id": "u", "vector": [1.0, 0.0]}]
        )[0]
        assert reader._vector_ranking("u", [1.0, 0.0], 5) == [first.id]
        second = writer.add_memories(
            [{"content": "beta", "user_id": "u", "vector": [0.0, 1.0]}]
        )[0]
        assert reader._vector_ranking("u", [0.0, 1.0], 5) == [second.id, first.id]
    finally:
        reader.close()
        writer.close()


def test_vector_ranking_top_k_matches_a_full_sort():
    import random

    from myelin.backends.sqlite import SqliteBackend

    rng = random.Random(4)
    backend = SqliteBackend()
    try:
        backend.add_memories(
            [
                {"content": f"m{i}", "user_id": "u",
                 "vector": [rng.gauss(0, 1) for _ in range(8)]}
                for i in range(300)
            ]
            + [{"content": "tie", "user_id": "u", "vector": [1.0] + [0.0] * 7}] * 3
        )
        query = [1.0] + [0.0] * 7
        full = backend._vector_ranking("u", query, 1000)
        assert len(full) == 303
        for k in (1, 2, 10, 302):
            assert backend._vector_ranking("u", query, k) == full[:k]
    finally:
        backend.close()
