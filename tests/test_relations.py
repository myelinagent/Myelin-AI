"""Grading memories together: use, related use, unused showings, age."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from myelin import MemoryClient
from myelin._internal import health
from myelin._internal.relations import SESSION_WINDOW, neighbour_support
from myelin.types import Memory

NOW = datetime(2025, 6, 1, tzinfo=UTC)


def _memory(name, *, session=None, turn=None, used=0, unused=0, age_days=0):
    metadata = {}
    if session is not None:
        metadata = {"session_id": session, "turn": turn}
    return Memory(
        id=name, content=name, user_id="u", t_valid=NOW - timedelta(days=age_days),
        metadata=metadata, cite_correct_count=used, shown_unused_count=unused,
    )


def test_no_use_anywhere_means_no_support():
    memories = [_memory(f"m{i}", session="s", turn=i) for i in range(5)]
    assert neighbour_support(memories) == {}


def test_support_reaches_conversation_neighbours_and_fades_with_distance():
    memories = [_memory(f"m{i}", session="s", turn=i, used=3 if i == 5 else 0) for i in range(12)]
    other = _memory("elsewhere", session="t", turn=0)
    support = neighbour_support([*memories, other])
    assert support["m4"] == support["m6"] > support["m3"] > support["m2"] > 0
    assert "m5" not in support  # a memory does not support itself
    assert f"m{5 + SESSION_WINDOW + 1}" not in support
    assert "elsewhere" not in support


def test_support_reaches_similar_memories():
    memories = [_memory("used", used=2), _memory("alike"), _memory("unlike")]
    vectors = {"used": [1.0, 0.0], "alike": [0.9, 0.436], "unlike": [0.0, 1.0]}
    support = neighbour_support(memories, vectors)
    assert support.get("alike", 0) > 0
    assert "unlike" not in support
    assert neighbour_support(memories) == {}  # no vectors, no sessions: no edges


def test_health_grades_use_related_use_and_unused_showings_together():
    old = dict(age_days=200)
    alone = health.memory_health(_memory("a", **old), now=NOW)
    related = health.memory_health(_memory("a", **old), now=NOW, related_support=2.0)
    used = health.memory_health(_memory("a", used=3, **old), now=NOW)
    unused = health.memory_health(_memory("a", unused=6, **old), now=NOW)
    assert alone < health.QUARANTINE_THRESHOLD < related
    assert used > alone > unused
    fresh = health.memory_health(_memory("a"), now=NOW)
    assert fresh > alone  # age still counts


def test_one_contradiction_outweighs_any_amount_of_use_and_related_use():
    popular = _memory("a", used=10_000)
    lifted = health.memory_health(popular, now=NOW, related_support=10_000.0)
    contradicted = health.memory_health(
        popular, now=NOW, related_support=10_000.0, contradiction_count=1
    )
    plain = health.memory_health(_memory("a"), now=NOW)
    assert contradicted < plain < lifted


def test_prune_keeps_the_unused_neighbour_of_a_used_memory():
    clock = [NOW - timedelta(days=200)]
    client = MemoryClient(user_id="u", cite_learning=True, clock=lambda: clock[0])
    try:
        talk = client.add_conversation(
            [("user", "My sister Ana lives in Porto."), ("assistant", "Noted, Porto.")]
        )
        stray = client.add("The weather was mild that week.")
        assert talk[0].metadata["session_id"] == talk[1].metadata["session_id"]
        assert [m.metadata["turn"] for m in talk] == [0, 1]
        clock[0] = NOW
        for _ in range(3):
            client.cite(talk[0].id, answer_correct=True)
        report = client.prune()
        assert client.get(talk[0].id).health_state == "healthy"
        assert client.get(talk[1].id).health_state == "healthy"  # never cited, but related
        assert client.get(stray.id).health_state == "quarantined"
        assert report.quarantined_count == 1
    finally:
        client.close()
