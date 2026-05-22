"""Happy-path + edge cases for the MemoryClient CRUD surface."""
from __future__ import annotations

import pytest

from myelin import MemoryClient, Memory, SearchResult


@pytest.fixture
def client():
    c = MemoryClient()
    yield c
    c.close()


def test_add_returns_memory_with_id(client):
    mem = client.add("Scott prefers tea.")
    assert isinstance(mem, Memory)
    assert mem.id
    assert mem.content == "Scott prefers tea."
    assert mem.health_state == "healthy"
    assert mem.user_id == "default"


def test_search_returns_relevant_hit(client):
    client.add("Scott prefers tea.")
    client.add("Alice loves chess.")
    results = client.search("tea")
    assert results, "search should return at least one hit"
    assert isinstance(results[0], SearchResult)
    assert "tea" in results[0].memory.content.lower()


def test_search_empty_query_returns_empty(client):
    client.add("Scott prefers tea.")
    assert client.search("") == []
    # whitespace / pure-stopwords should also return empty
    assert client.search("the   ") == []


def test_search_no_hits_returns_empty(client):
    client.add("Scott prefers tea.")
    assert client.search("zebra") == []


def test_search_respects_top_k(client):
    for i in range(20):
        client.add(f"fact number {i} about tea.")
    results = client.search("tea", top_k=5)
    assert len(results) == 5


def test_search_scoring_orders_by_relevance(client):
    client.add("tea")
    client.add("tea tea tea about tea infusion")
    client.add("coffee primarily")
    results = client.search("tea")
    # The richer-overlap fact should outrank the bare one.
    assert results[0].score >= results[-1].score


def test_get_returns_memory(client):
    mem = client.add("Scott prefers tea.")
    fetched = client.get(mem.id)
    assert fetched is not None
    assert fetched.id == mem.id
    assert fetched.content == "Scott prefers tea."


def test_get_unknown_returns_none(client):
    assert client.get("does-not-exist") is None


def test_update_changes_content(client):
    mem = client.add("Scott prefers tea.")
    updated = client.update(mem.id, content="Scott prefers coffee.")
    assert updated.content == "Scott prefers coffee."
    refetched = client.get(mem.id)
    assert refetched is not None
    assert refetched.content == "Scott prefers coffee."


def test_update_changes_metadata(client):
    mem = client.add("Scott prefers tea.", metadata={"tag": "preference"})
    updated = client.update(mem.id, metadata={"tag": "preference", "score": 0.9})
    assert updated.metadata["score"] == 0.9


def test_update_unknown_raises(client):
    with pytest.raises(KeyError):
        client.update("does-not-exist", content="...")


def test_delete_removes_memory(client):
    mem = client.add("Scott prefers tea.")
    assert client.delete(mem.id) is True
    assert client.get(mem.id) is None
    # second delete is a noop
    assert client.delete(mem.id) is False


def test_delete_unknown_returns_false(client):
    assert client.delete("does-not-exist") is False


def test_user_id_scoping():
    c = MemoryClient(user_id="alice")
    c.add("Alice fact.")
    c.add("Bob fact.", user_id="bob")
    alice_hits = c.search("fact")
    bob_hits = c.search("fact", user_id="bob")
    assert {r.memory.content for r in alice_hits} == {"Alice fact."}
    assert {r.memory.content for r in bob_hits} == {"Bob fact."}
    c.close()


def test_persistence_across_instances(tmp_path):
    db = tmp_path / "test.db"
    c1 = MemoryClient(db_path=str(db))
    c1.add("persistent fact about tea.")
    c1.close()
    c2 = MemoryClient(db_path=str(db))
    hits = c2.search("tea")
    assert len(hits) == 1
    assert "tea" in hits[0].memory.content
    c2.close()


def test_metadata_roundtrip(client):
    mem = client.add(
        "Scott prefers tea.",
        metadata={"subject": "Scott", "predicate": "prefers", "object": "tea"},
    )
    fetched = client.get(mem.id)
    assert fetched.metadata["subject"] == "Scott"
    assert fetched.metadata["predicate"] == "prefers"


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        MemoryClient(backend="postgres")


def test_event_dict_requires_event_id():
    c = MemoryClient()
    with pytest.raises(ValueError):
        c.add("...", event={"type": "meeting"})
    c.close()


def test_event_wrong_type_raises():
    c = MemoryClient()
    with pytest.raises(TypeError):
        c.add("...", event=42)
    c.close()
