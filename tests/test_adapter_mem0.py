"""Mem0-compatible adapter tests.

The adapter at :mod:`myelin.adapters.mem0` exposes Mem0's documented
``MemoryClient`` surface backed by the Myelin substrate. These tests
exercise the surface directly — they do **not** require ``mem0ai`` to
be installed. A small optional cross-check against the real ``mem0``
package is gated behind ``pytest.importorskip``.
"""
from __future__ import annotations

import pytest

from myelin.adapters.mem0 import MemoryClient


@pytest.fixture
def client():
    c = MemoryClient()
    yield c
    c.close()


# ---- import surface ------------------------------------------------


def test_drop_in_import_path():
    """``from myelin.adapters.mem0 import MemoryClient`` mirrors
    ``from mem0 import MemoryClient``."""
    from myelin.adapters.mem0 import MemoryClient as Adapter

    assert Adapter.__name__ == "MemoryClient"
    # Constructor accepts a config dict and arbitrary kwargs without
    # raising — Mem0 callers often pass long-form config blobs.
    inst = Adapter(config={"db_path": ":memory:", "vector_store": "ignored"})
    inst.close()


# ---- add: messages-list + string forms -----------------------------


def test_add_messages_list_form(client):
    out = client.add(
        [
            {"role": "user", "content": "I love tea."},
            {"role": "assistant", "content": "Noted."},
        ],
        user_id="alice",
    )
    assert isinstance(out, dict)
    assert out["id"]
    assert "user: I love tea." in out["memory"]
    assert "assistant: Noted." in out["memory"]
    assert out["user_id"] == "alice"
    assert "created_at" in out
    assert "updated_at" in out


def test_add_string_form(client):
    out = client.add("hello world", user_id="bob")
    assert out["memory"] == "hello world"
    assert out["user_id"] == "bob"


def test_add_invalid_messages_raises(client):
    with pytest.raises(TypeError):
        client.add(42, user_id="alice")  # type: ignore[arg-type]


def test_add_writes_agent_id_run_id_into_metadata(client):
    out = client.add(
        "scoped fact",
        user_id="alice",
        agent_id="search-bot",
        run_id="run-42",
        metadata={"tag": "preference"},
    )
    assert out["metadata"]["agent_id"] == "search-bot"
    assert out["metadata"]["run_id"] == "run-42"
    assert out["metadata"]["tag"] == "preference"


# ---- search: Mem0 dict shape --------------------------------------


def test_search_returns_mem0_dicts(client):
    client.add("Alice prefers tea.", user_id="alice")
    client.add("Alice loves chess.", user_id="alice")
    hits = client.search("tea", user_id="alice")
    assert hits, "expected at least one hit"
    first = hits[0]
    assert isinstance(first, dict)
    # Mem0 keys present:
    for key in ("id", "memory", "user_id", "metadata", "score"):
        assert key in first, f"missing Mem0 key {key!r}"
    # Not a SearchResult or Memory dataclass — pure dict.
    assert not hasattr(first, "memory") or isinstance(first["memory"], str)


def test_search_respects_limit(client):
    for i in range(15):
        client.add(f"tea fact {i}", user_id="alice")
    hits = client.search("tea", user_id="alice", limit=5)
    assert len(hits) == 5


def test_search_scope_filters_by_agent_id(client):
    client.add("alpha fact", user_id="alice", agent_id="agent-a")
    client.add("beta fact", user_id="alice", agent_id="agent-b")
    hits = client.search("fact", user_id="alice", agent_id="agent-a")
    contents = {h["memory"] for h in hits}
    assert contents == {"alpha fact"}


def test_search_scope_filters_by_run_id(client):
    client.add("r1 fact", user_id="alice", run_id="run-1")
    client.add("r2 fact", user_id="alice", run_id="run-2")
    hits = client.search("fact", user_id="alice", run_id="run-1")
    assert {h["memory"] for h in hits} == {"r1 fact"}


# ---- get + get_all -------------------------------------------------


def test_get_returns_mem0_dict(client):
    added = client.add("a memory", user_id="alice")
    fetched = client.get(added["id"])
    assert fetched is not None
    assert fetched["id"] == added["id"]
    assert fetched["memory"] == "a memory"
    assert fetched["user_id"] == "alice"


def test_get_unknown_returns_none(client):
    assert client.get("nonexistent-id") is None


def test_get_all_respects_user_id_filter(client):
    client.add("alice-1", user_id="alice")
    client.add("alice-2", user_id="alice")
    client.add("bob-1", user_id="bob")
    alice_rows = client.get_all(user_id="alice")
    bob_rows = client.get_all(user_id="bob")
    assert {r["memory"] for r in alice_rows} == {"alice-1", "alice-2"}
    assert {r["memory"] for r in bob_rows} == {"bob-1"}


def test_get_all_respects_limit(client):
    for i in range(12):
        client.add(f"fact {i}", user_id="alice")
    rows = client.get_all(user_id="alice", limit=5)
    assert len(rows) == 5


# ---- update --------------------------------------------------------


def test_update_with_string_data(client):
    added = client.add("old content", user_id="alice")
    updated = client.update(added["id"], "new content")
    assert updated["memory"] == "new content"


def test_update_with_dict_data(client):
    added = client.add(
        "first", user_id="alice", metadata={"tag": "v1"}
    )
    updated = client.update(
        added["id"],
        {"memory": "second", "metadata": {"tag": "v2"}},
    )
    assert updated["memory"] == "second"
    assert updated["metadata"]["tag"] == "v2"


def test_update_unknown_raises(client):
    with pytest.raises(KeyError):
        client.update("does-not-exist", "x")


# ---- delete / delete_all ------------------------------------------


def test_delete_returns_ack_dict(client):
    added = client.add("doomed", user_id="alice")
    ack = client.delete(added["id"])
    assert ack["deleted"] is True
    assert ack["id"] == added["id"]
    assert client.get(added["id"]) is None


def test_delete_unknown_returns_falsy_ack(client):
    ack = client.delete("nope")
    assert ack["deleted"] is False


def test_delete_all_scoped_by_user_id(client):
    client.add("alice-1", user_id="alice")
    client.add("alice-2", user_id="alice")
    client.add("bob-1", user_id="bob")
    ack = client.delete_all(user_id="alice")
    assert ack["deleted_count"] == 2
    assert client.get_all(user_id="alice") == []
    # bob's memories survive
    bob_rows = client.get_all(user_id="bob")
    assert len(bob_rows) == 1
    assert bob_rows[0]["memory"] == "bob-1"


def test_delete_all_scoped_by_agent_id(client):
    client.add("a", user_id="alice", agent_id="agent-x")
    client.add("b", user_id="alice", agent_id="agent-y")
    ack = client.delete_all(user_id="alice", agent_id="agent-x")
    assert ack["deleted_count"] == 1
    remaining = client.get_all(user_id="alice")
    assert len(remaining) == 1
    assert remaining[0]["memory"] == "b"


# ---- history -------------------------------------------------------


def test_history_returns_revisions_in_order(client):
    # Three event-anchored revisions of the same (subject, predicate).
    e1 = {"event_id": "ev-1", "type": "meeting", "anchor_datetime": "2024-01-01T00:00:00"}
    e2 = {"event_id": "ev-2", "type": "meeting", "anchor_datetime": "2024-06-01T00:00:00"}
    e3 = {"event_id": "ev-3", "type": "meeting", "anchor_datetime": "2024-12-01T00:00:00"}
    # We need to use the underlying myelin client to attach events,
    # since Mem0's add() doesn't expose `event=`. The adapter still
    # reads them back through history().
    inner = client._client
    m1 = inner.add(
        "Scott prefers tea.",
        user_id="alice",
        event=e1,
        metadata={"subject": "Scott", "predicate": "prefers"},
    )
    inner.add(
        "Scott prefers coffee.",
        user_id="alice",
        event=e2,
        metadata={"subject": "Scott", "predicate": "prefers"},
    )
    inner.add(
        "Scott prefers matcha.",
        user_id="alice",
        event=e3,
        metadata={"subject": "Scott", "predicate": "prefers"},
    )
    hist = client.history(m1.id)
    assert isinstance(hist, list)
    assert len(hist) == 3
    contents = [h["memory"] for h in hist]
    assert contents == [
        "Scott prefers tea.",
        "Scott prefers coffee.",
        "Scott prefers matcha.",
    ]
    # Each entry carries the event-anchor info.
    event_ids = [h["event"]["event_id"] for h in hist]
    assert event_ids == ["ev-1", "ev-2", "ev-3"]


def test_history_without_event_anchor_returns_single_row(client):
    added = client.add("standalone fact", user_id="alice")
    hist = client.history(added["id"])
    assert len(hist) == 1
    assert hist[0]["id"] == added["id"]


def test_history_unknown_returns_empty(client):
    assert client.history("nonexistent-id") == []


# ---- reset ---------------------------------------------------------


def test_reset_empties_substrate(client):
    client.add("a", user_id="alice")
    client.add("b", user_id="bob")
    ack = client.reset()
    assert ack["reset"] is True
    assert client.get_all(user_id="alice") == []
    assert client.get_all(user_id="bob") == []


# ---- Myelin extensions --------------------------------------------


def test_consolidate_returns_dict(client):
    client.add("Scott prefers tea.", user_id="alice")
    report = client.consolidate(user_id="alice")
    assert isinstance(report, dict)
    for key in (
        "hit_rate",
        "gaps_surfaced",
        "new_facts_extracted",
        "dream_queries_generated",
        "llm_calls",
        "facts_sampled",
    ):
        assert key in report, f"missing key {key!r} in consolidation dict"
    # facts_sampled increases as we add facts, even without an LLM
    # client wired in.
    assert report["facts_sampled"] >= 1


def test_prune_returns_dict(client):
    client.add("Scott prefers tea.", user_id="alice")
    report = client.prune(user_id="alice")
    assert isinstance(report, dict)
    for key in (
        "quarantined_count",
        "tombstoned_count",
        "retained_count",
        "scanned",
        "caspase_window_days",
    ):
        assert key in report, f"missing key {key!r} in apoptosis dict"
    assert report["scanned"] >= 1


# ---- Optional interop with real mem0ai (skipped if not installed) -


def test_optional_interop_with_real_mem0():
    """Sanity check that ``mem0.MemoryClient`` exists with a parallel
    surface — gated behind ``importorskip`` so the test does not
    require ``mem0ai`` to pass."""
    mem0 = pytest.importorskip("mem0")
    assert hasattr(mem0, "MemoryClient"), (
        "expected mem0.MemoryClient on the published package"
    )
    # The adapter intentionally mirrors method names that exist on the
    # upstream class; spot-check a few:
    upstream = mem0.MemoryClient
    for name in ("add", "search", "get", "delete", "reset"):
        assert hasattr(
            upstream, name
        ), f"upstream mem0.MemoryClient missing {name}"
