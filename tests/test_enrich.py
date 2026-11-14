"""Extra index text on memories, and write-time fact extraction."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from myelin import MemoryClient, format_context
from myelin.backends.sqlite import SqliteBackend
from myelin.enrich import build_extraction_prompt, extract_facts, parse_facts


@pytest.fixture
def client():
    c = MemoryClient(user_id="u")
    yield c
    c.close()


def _ids(results):
    return [r.memory.id for r in results]


# ---- index text ---------------------------------------------------------


def test_a_memory_is_found_by_its_index_text(client):
    mem = client.add(
        "user: I finally picked her up on Saturday!",
        index_text="The user adopted a greyhound named Comet on 2023-05-13.",
    )
    client.add("user: The weather has been lovely.")
    assert client.search("greyhound adoption date") != []
    assert _ids(client.search("greyhound"))[0] == mem.id


def test_index_text_is_never_shown_as_content(client):
    client.add("user: I picked her up on Saturday!", index_text="The user adopted a greyhound.")
    (hit,) = client.recall("greyhound")
    assert hit.memory.content == "user: I picked her up on Saturday!"
    assert hit.memory.index_text == "The user adopted a greyhound."
    assert "greyhound" not in format_context([hit])


def test_set_index_text_adds_and_replaces(client):
    mem = client.add("user: I picked her up on Saturday!")
    assert client.search("greyhound") == []
    assert client.set_index_text(mem.id, "The user adopted a greyhound.") is True
    assert _ids(client.search("greyhound")) == [mem.id]
    client.set_index_text(mem.id, "The user bought a kayak.")
    assert client.search("greyhound") == []
    assert _ids(client.search("kayak")) == [mem.id]
    assert client.get(mem.id).index_text == "The user bought a kayak."
    assert client.set_index_text("nope", "x") is False


def test_updating_content_keeps_the_index_text(client):
    mem = client.add("user: I picked her up!", index_text="The user adopted a greyhound.")
    client.update(mem.id, content="user: I picked her up on Saturday morning!")
    assert _ids(client.search("greyhound")) == [mem.id]
    assert _ids(client.search("morning")) == [mem.id]


def test_add_many_takes_index_text(client):
    stored = client.add_many(
        [
            {"content": "user: picked her up!", "index_text": "The user adopted a greyhound."},
            "user: lovely weather",
        ]
    )
    assert _ids(client.search("greyhound")) == [stored[0].id]
    assert stored[1].index_text == ""


def test_index_text_is_embedded_with_the_content():
    seen: list[str] = []

    def encode(text):
        seen.append(text)
        return [1.0, float(len(text))]

    c = MemoryClient(user_id="u", embedder=encode)
    try:
        mem = c.add("user: picked her up!", index_text="The user adopted a greyhound.")
        assert seen[-1] == "user: picked her up!\nThe user adopted a greyhound."
        c.set_index_text(mem.id, "The user bought a kayak.")
        assert seen[-1] == "user: picked her up!\nThe user bought a kayak."
    finally:
        c.close()


def test_index_text_survives_reopening_and_reindexing(tmp_path):
    db = str(tmp_path / "m.db")
    first = SqliteBackend(db_path=db)
    first.add_memory(content="picked her up", user_id="u", index_text="adopted a greyhound")
    first._conn.executescript("DROP TABLE memories_fts; DROP TABLE memories_fts_rows;")
    first._conn.commit()
    first.close()
    reopened = SqliteBackend(db_path=db)
    assert len(reopened.search(query="greyhound", user_id="u")) == 1
    reopened.close()


def test_a_store_without_the_column_is_migrated(tmp_path):
    db = str(tmp_path / "old.db")
    old = SqliteBackend(db_path=db)
    old.add_memory(content="Scott prefers tea", user_id="u")
    old._conn.execute("ALTER TABLE memories DROP COLUMN index_text")
    old._conn.commit()
    old.close()
    reopened = SqliteBackend(db_path=db)
    (mem,) = reopened.list_memories(user_id="u")
    assert mem.index_text == ""
    assert reopened.set_index_text(mem.id, "oolong") is True
    assert len(reopened.search(query="oolong", user_id="u")) == 1
    reopened.close()


def test_index_text_counts_in_the_fallback_scan(monkeypatch):
    monkeypatch.setattr(SqliteBackend, "_init_fts", lambda self: False)
    c = MemoryClient(user_id="u")
    try:
        mem = c.add("picked her up", index_text="adopted a greyhound")
        assert _ids(c.search("greyhound")) == [mem.id]
    finally:
        c.close()


# ---- extraction ---------------------------------------------------------

_TURNS = [
    ("user", "I adopted a greyhound last Saturday."),
    ("assistant", "Congratulations! " + "Greyhounds are gentle. " * 40),
    ("user", "Thanks!"),
]


def test_prompt_numbers_messages_dates_the_conversation_and_trims_replies():
    prompt = build_extraction_prompt(_TURNS, datetime(2023, 5, 20, tzinfo=UTC))
    assert "It took place on Saturday 2023-05-20." in prompt
    assert "[0] USER: I adopted a greyhound last Saturday." in prompt
    assert "[2] USER: Thanks!" in prompt
    assistant_line = next(line for line in prompt.splitlines() if line.startswith("[1]"))
    assert len(assistant_line) < 340 and assistant_line.endswith("…")
    assert build_extraction_prompt(_TURNS, "2023/05/20 (Sat) 02:21").count("2023/05/20") == 1


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"0": ["The user adopted a greyhound on 2023-05-13."]}',
         {0: ["The user adopted a greyhound on 2023-05-13."]}),
        ('Sure!\n```json\n{"0": ["A."], "2": ["B.", "C."]}\n```', {0: ["A."], 2: ["B.", "C."]}),
        ('{"0": "A single string."}', {0: ["A single string."]}),
        ("{}", {}),
        ("no json here", {}),
        ('{"0": ["A."', {}),
        ("[1, 2]", {}),
        ('{"x": ["A."], "-1": ["B."], "9": ["C."], "1": [""], "2": [7]}', {}),
    ],
)
def test_parse_facts_is_tolerant(reply, expected):
    assert parse_facts(reply, n_turns=3) == expected


def test_parse_facts_caps_count_and_length():
    reply = '{"0": [' + ", ".join(f'"{"x" * 500} {i}"' for i in range(20)) + "]}"
    facts = parse_facts(reply)[0]
    assert len(facts) == 6
    assert all(len(f) == 300 for f in facts)


def test_extract_facts_calls_the_model_once():
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return '{"0": ["The user adopted a greyhound on 2023-05-13."]}'

    facts = extract_facts(_TURNS, datetime(2023, 5, 20, tzinfo=UTC), llm)
    assert facts == {0: ["The user adopted a greyhound on 2023-05-13."]}
    assert len(prompts) == 1


def test_extract_facts_skips_the_model_when_there_is_nothing_from_the_user():
    def llm(prompt: str) -> str:
        raise AssertionError("should not be called")

    assert extract_facts([("assistant", "Hello!")], "2023-05-20", llm) == {}
    assert extract_facts([], "2023-05-20", llm) == {}


def test_extracted_facts_make_a_turn_findable_end_to_end(client):
    facts = extract_facts(
        _TURNS, "2023-05-20",
        lambda prompt: '{"0": ["The user adopted a greyhound on 2023-05-13."]}',
    )
    stored = client.add_many(
        {"content": f"{role}: {text}", "index_text": " ".join(facts.get(i, []))}
        for i, (role, text) in enumerate(_TURNS)
    )
    assert _ids(client.search("2023-05-13"))[0] == stored[0].id


def test_people_prompt_names_everyone_and_keeps_messages_whole():
    long_reply = "x" * 500
    turns = [("Ana", "I adopted a beagle yesterday."), ("Ben", long_reply)]
    prompt = build_extraction_prompt(turns, datetime(2023, 5, 20, tzinfo=UTC), people=["Ana", "Ben"])
    assert "between Ana and Ben" in prompt
    assert "Saturday 2023-05-20" in prompt
    assert "[0] Ana: I adopted a beagle yesterday." in prompt
    assert long_reply in prompt
    assert "The user" not in prompt
    three = build_extraction_prompt(turns, "2023-05-20", people=["Ana", "Ben", "Cy"])
    assert "between Ana, Ben and Cy" in three


def test_extract_facts_between_people_reads_every_speaker():
    seen = []

    def llm(prompt):
        seen.append(prompt)
        return '{"0": ["Ana adopted a beagle on 2023-05-19."], "1": ["Ben likes dogs."]}'

    turns = [("Ana", "I adopted a beagle yesterday."), ("Ben", "I love dogs.")]
    facts = extract_facts(turns, "2023-05-20", llm, people=["Ana", "Ben"])
    assert facts == {0: ["Ana adopted a beagle on 2023-05-19."], 1: ["Ben likes dogs."]}
    assert len(seen) == 1
    assert extract_facts([("Ana", "  ")], "2023-05-20", llm, people=["Ana"]) == {}


# ---- add_conversation ----------------------------------------------------


def test_add_conversation_stores_dated_messages_and_indexes_facts(client):
    when = datetime(2023, 5, 20, tzinfo=UTC)
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return '{"0": ["Ana adopted a beagle puppy on 2023-05-19."]}'

    stored = client.add_conversation(
        [("Ana", "Pickle came home yesterday!"), ("Ben", "  "), ("Ben", "So happy for you.")],
        user_id="u", when=when, llm=llm, people=["Ana", "Ben"],
    )
    assert [m.content for m in stored] == [
        "Ana: Pickle came home yesterday!", "Ben: So happy for you."
    ]
    assert all(m.t_valid == when for m in stored)
    assert len(prompts) == 1 and "between Ana and Ben" in prompts[0]
    hits = client.search("beagle puppy", user_id="u")
    assert hits and hits[0].memory.content == "Ana: Pickle came home yesterday!"


def test_add_conversation_without_a_model_or_when_it_fails(client):
    plain = client.add_conversation([("user", "I prefer tea.")], user_id="u")
    assert [m.content for m in plain] == ["user: I prefer tea."]

    def broken(prompt):
        raise RuntimeError("model unavailable")

    kept = client.add_conversation([("user", "I moved to Lisbon.")], user_id="u", llm=broken)
    assert [m.content for m in kept] == ["user: I moved to Lisbon."]
    assert client.search("Lisbon", user_id="u")
    assert client.add_conversation([], user_id="u", llm=broken) == []
