"""Outcome feedback from cite() influencing search ranking."""

from __future__ import annotations

import pytest

from myelin import MemoryClient
from myelin._internal.cite_ranking import (
    FULL_CONFIDENCE_CITES,
    pool_size,
    rerank_by_cites,
    utility,
)
from myelin.types import Memory, SearchResult


def _result(name: str, score: float, correct: int = 0, incorrect: int = 0) -> SearchResult:
    from datetime import UTC, datetime

    memory = Memory(
        id=name, content=name, user_id="u", t_valid=datetime(2024, 1, 1, tzinfo=UTC),
        cite_correct_count=correct, cite_incorrect_count=incorrect,
    )
    return SearchResult(memory=memory, score=score)


def _names(results):
    return [r.memory.id for r in results]


# ---- the utility of a cite record --------------------------------------


def test_utility_is_zero_without_cites():
    assert utility(0, 0) == 0.0


def test_utility_sign_follows_the_record():
    assert utility(5, 0) > 0
    assert utility(0, 5) < 0
    assert utility(3, 3) == 0.0


def test_utility_grows_with_evidence_and_stays_bounded():
    assert 0 < utility(1, 0) < utility(3, 0) < utility(FULL_CONFIDENCE_CITES, 0)
    assert -1 < utility(0, 1000) < 0 < utility(1000, 0) < 1
    assert utility(1, 0) < 0.15  # one cite proves little


def test_utility_ignores_negative_counts():
    assert utility(-3, 0) == 0.0


# ---- re-ranking ---------------------------------------------------------


def test_uncited_pool_is_returned_in_relevance_order():
    pool = [_result("a", 3.0), _result("b", 2.0), _result("c", 1.0)]
    assert _names(rerank_by_cites(pool, 2)) == ["a", "b"]


def test_a_helpful_record_lifts_a_close_contender():
    pool = [_result("plain", 2.1), _result("helpful", 2.0, correct=8), _result("far", 0.5)]
    assert _names(rerank_by_cites(pool, 3)) == ["helpful", "plain", "far"]


def test_a_misleading_record_sinks_a_memory():
    pool = [_result("misleading", 2.1, incorrect=8), _result("plain", 2.0), _result("far", 0.5)]
    assert _names(rerank_by_cites(pool, 3))[0] == "plain"


def test_feedback_cannot_beat_a_large_relevance_gap():
    pool = [
        _result("exact", 20.0),
        *[_result(f"mid{i}", 2.0) for i in range(8)],
        _result("weak_but_loved", 1.0, correct=50),
    ]
    assert _names(rerank_by_cites(pool, 3))[0] == "exact"


def test_scores_are_left_as_relevance_scores():
    pool = [_result("a", 2.1), _result("b", 2.0, correct=8)]
    assert {r.memory.id: r.score for r in rerank_by_cites(pool, 2)} == {"a": 2.1, "b": 2.0}


def test_identical_scores_do_not_divide_by_zero():
    pool = [_result("a", 1.0), _result("b", 1.0, correct=5)]
    assert _names(rerank_by_cites(pool, 2)) == ["b", "a"]


def test_empty_pool_and_pool_size():
    assert rerank_by_cites([], 5) == []
    assert pool_size(10) == 30
    assert pool_size(1) == 11


# ---- through the client -------------------------------------------------


@pytest.fixture
def client():
    c = MemoryClient(user_id="u", cite_learning=True)
    yield c
    c.close()


def _two_close_memories(c):
    first = c.add("Scott takes the red bus to work every day")
    second = c.add("Scott takes the red bus to work most days")
    assert _names(c.search("red bus work", top_k=2))[0] in {first.id, second.id}
    top = c.search("red bus work", top_k=2)
    return top[0].memory.id, top[1].memory.id


def test_cites_reorder_search_results(client):
    leader, runner_up = _two_close_memories(client)
    for _ in range(6):
        client.cite(runner_up, answer_correct=True)
        client.cite(leader, answer_correct=False)
    assert _names(client.search("red bus work", top_k=2)) == [runner_up, leader]


def test_cites_never_pull_in_an_unrelated_memory(client):
    related = client.add("Scott takes the red bus to work")
    unrelated = client.add("The weather in Lisbon is lovely")
    for _ in range(20):
        client.cite(unrelated.id, answer_correct=True)
    assert _names(client.search("red bus work")) == [related.id]


def test_top_k_is_respected_after_reranking(client):
    for i in range(30):
        client.add(f"Scott takes bus number {i} to work")
    assert len(client.search("bus work", top_k=5)) == 5


def test_ranking_is_unchanged_when_cite_ranking_is_off():
    c = MemoryClient(user_id="u", cite_learning=True, cite_ranking=False)
    try:
        leader, runner_up = _two_close_memories(c)
        for _ in range(6):
            c.cite(runner_up, answer_correct=True)
            c.cite(leader, answer_correct=False)
        assert _names(c.search("red bus work", top_k=2)) == [leader, runner_up]
    finally:
        c.close()


def test_ranking_is_unchanged_when_cite_learning_is_off():
    c = MemoryClient(user_id="u", cite_learning=False)
    try:
        leader, runner_up = _two_close_memories(c)
        # Counts written directly, as if another client had cited.
        c._backend._conn.execute(
            "UPDATE memories SET cite_correct_count = 9 WHERE id = ?", (runner_up,)
        )
        c._backend._conn.commit()
        assert _names(c.search("red bus work", top_k=2)) == [leader, runner_up]
    finally:
        c.close()


def test_recall_uses_the_same_ranking(client):
    leader, runner_up = _two_close_memories(client)
    for _ in range(6):
        client.cite(runner_up, answer_correct=True)
        client.cite(leader, answer_correct=False)
    assert _names(client.recall("red bus work"))[:2] == [runner_up, leader]
