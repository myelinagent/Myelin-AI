"""BM25 full-text search, budget-sized recall and context formatting."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from myelin import (
    MemoryClient,
    estimate_tokens,
    excerpt,
    fit_to_budget,
    format_context,
)
from myelin.backends.sqlite import SqliteBackend, _fts_query
from myelin.types import Memory, SearchResult


@pytest.fixture
def client():
    c = MemoryClient(user_id="u")
    yield c
    c.close()


def _contents(results):
    return [r.memory.content for r in results]


# ---- ranking ----------------------------------------------------------


def test_rare_terms_outrank_common_ones(client):
    for i in range(20):
        client.add(f"Scott mentioned the weather on day {i}")
    target = client.add("Scott adopted a greyhound")
    hits = client.search("Scott greyhound", top_k=3)
    assert hits[0].memory.id == target.id
    assert hits[0].score > hits[1].score > 0


def test_search_matches_word_forms(client):
    client.add("I graduated with a degree in Business Administration")
    client.add("The weather was lovely in Lisbon")
    assert _contents(client.search("What degree did I graduate with?")) == [
        "I graduated with a degree in Business Administration"
    ]


def test_query_punctuation_and_operators_are_not_syntax(client):
    client.add("The NEAR protocol AND the OR gate")
    for query in ['what about "quotes"?', "AND OR NOT NEAR(", "col:value*", "a-b^c", "(((", "'"]:
        client.search(query)  # must not raise
    assert len(client.search("NEAR protocol")) == 1


@pytest.mark.parametrize("query", ["", "   ", "the of and", "?!"])
def test_query_without_searchable_terms_returns_nothing(client, query):
    client.add("Scott prefers tea")
    assert client.search(query) == []


def test_fts_query_quotes_and_dedupes_terms():
    assert _fts_query("Tea, tea and the TEA-pot!") == '"tea" OR "pot"'


def test_search_is_scoped_to_the_user(client):
    client.add("alice likes oolong", user_id="alice")
    client.add("bob likes oolong", user_id="bob")
    assert _contents(client.search("oolong", user_id="alice")) == ["alice likes oolong"]


def test_top_k_zero_returns_nothing(client):
    client.add("Scott prefers tea")
    assert client.search("tea", top_k=0) == []


# ---- the index follows the memories table ------------------------------


def test_update_reindexes_content(client):
    mem = client.add("Scott prefers tea")
    client.update(mem.id, content="Scott prefers espresso")
    assert client.search("tea") == []
    assert [r.memory.id for r in client.search("espresso")] == [mem.id]


def test_metadata_only_update_keeps_content_searchable(client):
    mem = client.add("Scott prefers tea")
    client.update(mem.id, metadata={"tag": "drink"})
    assert [r.memory.id for r in client.search("tea")] == [mem.id]


def test_deleted_and_tombstoned_memories_leave_the_index(client):
    gone = client.add("Scott prefers tea")
    pruned = client.add("Scott prefers tea in the morning")
    kept = client.add("Scott prefers tea at night")
    client.delete(gone.id)
    client._backend.transition_state(pruned.id, new_state="tombstoned")
    assert [r.memory.id for r in client.search("tea")] == [kept.id]
    rows = client._backend._conn.execute(
        "SELECT (SELECT COUNT(*) FROM memories_fts), (SELECT COUNT(*) FROM memories_fts_rows)"
    ).fetchone()
    assert rows == (1, 1)


def test_quarantined_memories_are_hidden_unless_asked_for(client):
    mem = client.add("Scott prefers tea")
    kept = client.add("Scott prefers tea in the morning")
    client._backend.transition_state(
        mem.id, new_state="quarantined", quarantine_entered_at=datetime.now(UTC)
    )
    assert [r.memory.id for r in client.search("tea")] == [kept.id]
    both = client.search("tea", include_quarantined=True)
    assert {r.memory.id for r in both} == {mem.id, kept.id}
    # Hidden rows do not use up the result limit.
    assert [r.memory.id for r in client.search("tea", top_k=1)] == [kept.id]


def test_as_of_filters_after_ranking(client):
    old = client.add("Scott lives in Paris", t_valid=datetime(2023, 1, 1, tzinfo=UTC))
    for i in range(30):
        client.add(f"Scott lives in Berlin now, note {i}", t_valid=datetime(2024, 1, 1, tzinfo=UTC))
    hits = client.search("Scott lives", top_k=5, as_of=datetime(2023, 6, 1, tzinfo=UTC))
    assert [r.memory.id for r in hits] == [old.id]


def test_store_written_before_the_index_existed_is_indexed_on_open(tmp_path):
    db = str(tmp_path / "old.db")
    first = SqliteBackend(db_path=db)
    first.add_memory(content="Scott prefers tea", user_id="u")
    # Simulate a store created by a version without the index.
    first._conn.executescript("DROP TABLE memories_fts; DROP TABLE memories_fts_rows;")
    first._conn.commit()
    first.close()

    reopened = SqliteBackend(db_path=db)
    assert [r.memory.content for r in reopened.search(query="tea", user_id="u")] == [
        "Scott prefers tea"
    ]
    reopened.add_memory(content="Scott also likes oolong tea", user_id="u")
    assert len(reopened.search(query="tea", user_id="u")) == 2
    reopened.close()


def test_search_falls_back_without_fts5(monkeypatch):
    """A SQLite built without FTS5 still searches, by token overlap."""

    def no_fts(self):
        return False

    monkeypatch.setattr(SqliteBackend, "_init_fts", no_fts)
    c = MemoryClient(user_id="u")
    try:
        assert c._backend._fts is False
        mem = c.add("Scott prefers oolong tea")
        c.add("The weather in Lisbon")
        assert [r.memory.id for r in c.search("oolong")] == [mem.id]
        c.update(mem.id, content="Scott prefers espresso")
        assert c.delete(mem.id) is True
    finally:
        c.close()


def test_fts5_is_available_in_this_python():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.close()


# ---- budget-sized recall ----------------------------------------------


def _result(content: str, *, when: datetime | None = None, **metadata) -> SearchResult:
    memory = Memory(
        id=content[:8],
        content=content,
        user_id="u",
        t_valid=when or datetime(2023, 5, 1, 10, 0, tzinfo=UTC),
        metadata=metadata,
    )
    return SearchResult(memory=memory, score=1.0)


def test_fit_to_budget_keeps_rank_order_until_full():
    results = [_result("a" * 400), _result("b" * 400), _result("c" * 400)]
    assert estimate_tokens("a" * 400) == 101
    assert len(fit_to_budget(results, 250)) == 2
    assert len(fit_to_budget(results, 10_000)) == 3


def test_fit_to_budget_always_keeps_the_top_result():
    results = [_result("a" * 4000), _result("b")]
    assert _contents(fit_to_budget(results, 10)) == ["a" * 4000]
    assert fit_to_budget([], 10) == []


def test_fit_to_budget_does_not_skip_ahead_to_smaller_results():
    results = [_result("a" * 400), _result("b" * 4000), _result("c")]
    assert _contents(fit_to_budget(results, 150)) == ["a" * 400]


def test_recall_returns_more_for_short_memories_than_top_k_would(client):
    for i in range(40):
        client.add(f"Scott tea note number {i}")
    assert len(client.search("tea")) == 10
    assert len(client.recall("tea", token_budget=6000)) == 40
    assert all(r.excerpt is None for r in client.recall("tea"))
    assert len(client.recall("tea", token_budget=6000, max_results=15)) == 15
    small = client.recall("tea", token_budget=30)
    assert 1 <= len(small) < 10


def test_recall_rejects_a_non_positive_budget(client):
    with pytest.raises(ValueError, match="token_budget must be positive"):
        client.recall("tea", token_budget=0)


# ---- context formatting ------------------------------------------------


def test_format_context_is_chronological_and_grouped_by_session():
    day1 = datetime(2023, 5, 1, 10, 0, tzinfo=UTC)
    day2 = day1 + timedelta(days=1)
    results = [
        _result("assistant: Lisbon is lovely.", when=day2, session_id="s2", turn=1),
        _result("user: I adopted a greyhound.", when=day1, session_id="s1", turn=0),
        _result("user: Priya moved to Lisbon.", when=day2, session_id="s2", turn=0),
    ]
    assert format_context(results) == (
        "### 2023-05-01 10:00\n"
        "user: I adopted a greyhound.\n\n"
        "### 2023-05-02 10:00\n"
        "user: Priya moved to Lisbon.\n"
        "assistant: Lisbon is lovely."
    )


def test_format_context_keeps_rank_order_without_turn_numbers():
    results = [_result("second fact ranked first"), _result("first fact ranked second")]
    assert format_context(results).splitlines()[1:] == [
        "second fact ranked first",
        "first fact ranked second",
    ]


def test_format_context_separates_sessions_at_the_same_time():
    results = [_result("from a", session_id="a"), _result("from b", session_id="b")]
    assert format_context(results).count("### 2023-05-01 10:00") == 2
    assert format_context([]) == ""


def test_recall_and_format_context_work_end_to_end(client):
    client.add("Scott moved to Berlin", t_valid=datetime(2024, 3, 1, tzinfo=UTC))
    client.add("Scott lived in Paris", t_valid=datetime(2023, 3, 1, tzinfo=UTC))
    text = format_context(client.recall("Where has Scott lived or moved?"))
    assert text.index("Paris") < text.index("Berlin")
    assert text.startswith("### 2023-03-01 00:00")


# ---- excerpts of long memories -----------------------------------------

_FILLER = " ".join(f"Sentence {i} talks about the weather." for i in range(120))


def test_excerpt_leaves_short_text_alone():
    assert excerpt("Scott prefers tea.", "tea", 300) == "Scott prefers tea."


def test_excerpt_keeps_the_passage_about_the_query():
    text = f"{_FILLER} The smoker you bought needs seasoning first. {_FILLER}"
    short = excerpt(text, "What kitchen appliance did I buy? A smoker?", 60)
    assert "smoker you bought needs seasoning" in short
    assert short.startswith("… ") and short.endswith(" …")
    assert estimate_tokens(short) <= 70


def test_excerpt_matches_word_forms_and_marks_gaps():
    text = (
        f"I graduated in 2015. {_FILLER} "
        f"My graduation was held outdoors. {_FILLER}"
    )
    short = excerpt(text, "When did I graduate?", 40)
    assert "graduated in 2015" in short
    assert "graduation was held outdoors" in short
    assert " … " in short


def test_excerpt_falls_back_to_the_opening_when_nothing_matches():
    short = excerpt(_FILLER, "greyhound", 20)
    assert short == _FILLER[:80].rstrip() + "…"


def test_excerpt_handles_one_enormous_passage():
    blob = "tea " * 5000
    assert estimate_tokens(excerpt(blob, "tea", 100)) <= 110


def test_recall_excerpts_long_memories_but_keeps_them_whole(client):
    long_reply = f"{_FILLER} Oolong is a partly oxidised tea. {_FILLER}"
    stored = client.add(long_reply)
    client.add("Scott prefers oolong tea")
    hits = client.recall(
        "oolong tea", token_budget=6000, max_result_tokens=50, whole_results=0
    )
    by_id = {h.memory.id: h for h in hits}
    assert by_id[stored.id].memory.content == long_reply
    assert "Oolong is a partly oxidised tea" in by_id[stored.id].excerpt
    assert estimate_tokens(by_id[stored.id].excerpt) < 80
    assert "Sentence 60 " not in format_context(hits)
    assert "Oolong is a partly oxidised tea" in format_context(hits)

    whole = client.recall("oolong tea", token_budget=6000, max_result_tokens=None)
    assert all(h.excerpt is None for h in whole)


def test_excerpts_let_more_memories_fit_the_budget(client):
    for i in range(8):
        client.add(f"{_FILLER} Note {i}: Scott drinks oolong tea.")
    whole = client.recall("oolong tea", token_budget=1500, max_result_tokens=None)
    short = client.recall(
        "oolong tea", token_budget=1500, max_result_tokens=100, whole_results=0
    )
    assert len(whole) == 1
    assert len(short) == 8


def test_recall_keeps_the_best_ranked_results_whole(client):
    for i in range(6):
        client.add(f"{_FILLER} Note {i}: Scott drinks oolong tea.")
    hits = client.recall("oolong tea", token_budget=20_000, max_result_tokens=100)
    assert [h.excerpt is None for h in hits] == [True, True, True, False, False, False]
    none_whole = client.recall(
        "oolong tea", token_budget=20_000, max_result_tokens=100, whole_results=0
    )
    assert all(h.excerpt is not None for h in none_whole)


def test_recall_rejects_a_non_positive_excerpt_size(client):
    with pytest.raises(ValueError, match="max_result_tokens"):
        client.recall("tea", max_result_tokens=0)


def test_search_still_finds_matches_ranked_below_many_hidden_ones(client):
    """The index shortlist must not hide results when most of the best
    matches are filtered out."""
    hidden = [client.add("green tea green tea green tea") for _ in range(120)]
    for mem in hidden:
        client._backend.transition_state(
            mem.id, new_state="quarantined", quarantine_entered_at=datetime.now(UTC)
        )
    kept = client.add("I sometimes drink tea with a long tail of other unrelated words here")
    hits = client.search("green tea", top_k=5)
    assert [h.memory.id for h in hits] == [kept.id]
    assert len(client.search("green tea", top_k=5, include_quarantined=True)) == 5
