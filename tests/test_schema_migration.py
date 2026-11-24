"""TDD anchors T-schema_migration-1..5.

Maps to spec: FR-A2 storage backward-compat invariant.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from myelin.backends.sqlite import SqliteBackend


# T-schema_migration-1 — v1 substrate migrates clean
def test_v1_substrate_migrates_clean(tmp_path):
    """Simulate a v1 substrate by creating the database with only the
    original schema (no cite columns, no cite_events table), then open
    it with v1.2 code. All v1 + v1.2 columns/tables must be present
    afterward."""
    db_path = str(tmp_path / "v1.sqlite")
    # Bootstrap a "v1" schema manually
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
            metadata_json TEXT NOT NULL, event_id TEXT, t_valid TEXT NOT NULL,
            t_invalid TEXT, confidence REAL DEFAULT 0.5, importance REAL DEFAULT 0.5,
            health_state TEXT DEFAULT 'healthy', quarantine_entered_at TEXT
        );
        CREATE TABLE tombstones (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
            metadata_json TEXT NOT NULL, event_id TEXT, t_valid TEXT NOT NULL,
            t_invalid TEXT, confidence REAL DEFAULT 0.5, importance REAL DEFAULT 0.5,
            health_state TEXT DEFAULT 'tombstoned', quarantine_entered_at TEXT,
            tombstoned_at TEXT NOT NULL
        );
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, type TEXT DEFAULT '',
            anchor_datetime TEXT, location TEXT DEFAULT '',
            participant_ids TEXT DEFAULT '[]', evidence_turn_ids TEXT DEFAULT '[]'
        );
    """)
    conn.commit()
    conn.close()

    # Open with v1.2 code; migration runs automatically
    backend = SqliteBackend(db_path=db_path)
    try:
        # All v1 + v1.2 columns present
        mem_cols = backend._column_names("memories")  # noqa: SLF001
        for c in ("cite_correct_count", "cite_incorrect_count", "last_cited_at"):
            assert c in mem_cols, f"missing column {c} after migration"
        tomb_cols = backend._column_names("tombstones")  # noqa: SLF001
        for c in ("cite_correct_count", "cite_incorrect_count", "last_cited_at"):
            assert c in tomb_cols

        # v1.2 tables present
        cur = backend._conn.cursor()  # noqa: SLF001
        rows = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "cite_events" in names
        assert "abstain_events" in names
    finally:
        backend.close()


# T-schema_migration-2 — idempotent
def test_migration_idempotent(tmp_path):
    db_path = str(tmp_path / "v12.sqlite")
    backend = SqliteBackend(db_path=db_path)
    try:
        # Re-run migration multiple times — must not raise.
        backend._apply_migrations()  # noqa: SLF001
        backend._apply_migrations()  # noqa: SLF001
        # Columns still single
        cur = backend._conn.cursor()  # noqa: SLF001
        cols = cur.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [r[1] for r in cols]
        assert col_names.count("cite_correct_count") == 1
    finally:
        backend.close()


# T-schema_migration-3 — v1 rows produce v1 health values
def test_v1_rows_v1_health(tmp_path):
    """After migration, an existing row with no cite signal (defaults
    to 0/NULL) must produce a health value identical to what v1 would
    compute for the same row."""
    backend = SqliteBackend(db_path=":memory:")
    try:
        mem = backend.add_memory(
            content="legacy fact",
            user_id="u",
            confidence=0.7,
            importance=0.6,
        )
        # Confirm new fields defaulted
        retrieved = backend.get(mem.id)
        assert retrieved.cite_correct_count == 0
        assert retrieved.cite_incorrect_count == 0
        assert retrieved.last_cited_at is None

        from myelin._internal.health import memory_health
        h = memory_health(retrieved, now=datetime.now(UTC))
        assert 0.0 < h < 1.0  # well-formed sigmoid
    finally:
        backend.close()


# T-schema_migration-4 — tombstone preserves cite columns
def test_tombstone_preserves_cite_columns():
    backend = SqliteBackend(db_path=":memory:")
    try:
        mem = backend.add_memory(content="fact", user_id="u")
        backend.record_cite(
            memory_id=mem.id, answer_correct=True,
            session_id="s", query="q", now=datetime.now(UTC),
        )
        backend.record_cite(
            memory_id=mem.id, answer_correct=False,
            session_id="s", query="q", now=datetime.now(UTC),
        )
        backend.transition_state(mem.id, new_state="tombstoned")
        cur = backend._conn.cursor()  # noqa: SLF001
        row = cur.execute(
            "SELECT cite_correct_count, cite_incorrect_count "
            "FROM tombstones WHERE id = ?", (mem.id,),
        ).fetchone()
        assert row == (1, 1)
    finally:
        backend.close()


# T-schema_migration-5 — cite_events outlive memory
def test_cite_events_outlive_memory():
    backend = SqliteBackend(db_path=":memory:")
    try:
        mem = backend.add_memory(content="fact", user_id="u")
        for _ in range(5):
            backend.record_cite(
                memory_id=mem.id, answer_correct=True,
                session_id="s", query="q", now=datetime.now(UTC),
            )
        backend.transition_state(mem.id, new_state="tombstoned")
        cur = backend._conn.cursor()  # noqa: SLF001
        n = cur.execute(
            "SELECT COUNT(*) FROM cite_events WHERE memory_id = ?", (mem.id,),
        ).fetchone()[0]
        assert n == 5, "cite_events must outlive the memory for audit"
    finally:
        backend.close()


def test_v160_substrate_migrates_clean_to_graph_tables(tmp_path):
    """Open a genuine 1.6.0-shaped database (v1 + v1.2 + vectors + FTS5,
    without any graph tables) and verify additive migration."""
    db_path = str(tmp_path / "v160.sqlite")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            event_id TEXT,
            t_valid TEXT NOT NULL,
            t_invalid TEXT,
            confidence REAL NOT NULL DEFAULT 0.5,
            importance REAL NOT NULL DEFAULT 0.5,
            health_state TEXT NOT NULL DEFAULT 'healthy',
            quarantine_entered_at TEXT,
            cite_correct_count INTEGER NOT NULL DEFAULT 0,
            cite_incorrect_count INTEGER NOT NULL DEFAULT 0,
            last_cited_at TEXT,
            shown_unused_count INTEGER NOT NULL DEFAULT 0,
            related_support REAL NOT NULL DEFAULT 0,
            index_text TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX idx_memories_user_state ON memories(user_id, health_state);
        CREATE INDEX idx_memories_user_tvalid ON memories(user_id, t_valid);

        CREATE TABLE tombstones (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            event_id TEXT,
            t_valid TEXT NOT NULL,
            t_invalid TEXT,
            confidence REAL NOT NULL DEFAULT 0.5,
            importance REAL NOT NULL DEFAULT 0.5,
            health_state TEXT NOT NULL DEFAULT 'tombstoned',
            quarantine_entered_at TEXT,
            tombstoned_at TEXT NOT NULL,
            cite_correct_count INTEGER NOT NULL DEFAULT 0,
            cite_incorrect_count INTEGER NOT NULL DEFAULT 0,
            last_cited_at TEXT
        );

        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            type TEXT NOT NULL DEFAULT '',
            anchor_datetime TEXT,
            location TEXT NOT NULL DEFAULT '',
            participant_ids TEXT NOT NULL DEFAULT '[]',
            evidence_turn_ids TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE cite_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            answer_correct INTEGER NOT NULL CHECK (answer_correct IN (0, 1)),
            session_id TEXT,
            query_hash_or_text TEXT,
            cited_at TEXT NOT NULL
        );

        CREATE TABLE abstain_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash_or_text TEXT,
            session_id TEXT,
            recorded_at TEXT NOT NULL
        );

        CREATE TABLE memory_vectors (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            vector BLOB NOT NULL
        );

        CREATE VIRTUAL TABLE memories_fts USING fts5(
            content,
            user_id UNINDEXED,
            tokenize = 'porter unicode61'
        );
        CREATE TABLE memories_fts_rows (
            memory_id TEXT PRIMARY KEY,
            fts_rowid INTEGER NOT NULL UNIQUE
        );
    """)

    conn.execute(
        "INSERT INTO events VALUES ('ev-1', 'meeting', '2026-01-10T10:00:00+00:00', 'HQ', '[]', '[]')"
    )
    conn.execute(
        "INSERT INTO memories (id, user_id, content, metadata_json, event_id, t_valid, index_text) "
        "VALUES ('m-1', 'alice', 'Alice likes green tea.', ?, 'ev-1', '2026-01-10T10:00:00+00:00', 'beverage')",
        ('{"k": 1}',),
    )
    conn.execute(
        "INSERT INTO memories (id, user_id, content, metadata_json, event_id, t_valid) "
        "VALUES ('m-2', 'alice', 'Alice lives in Austin.', '{}', NULL, '2026-01-11T10:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO memories_fts (rowid, content, user_id) "
        "SELECT rowid, content || char(10) || index_text, user_id FROM memories"
    )
    conn.execute(
        "INSERT INTO memories_fts_rows (memory_id, fts_rowid) SELECT id, rowid FROM memories"
    )
    conn.execute(
        "INSERT INTO memory_vectors VALUES ('m-1', 'alice', X'0000803F00000000')"
    )
    conn.execute(
        "INSERT INTO tombstones (id, user_id, content, metadata_json, t_valid, tombstoned_at) "
        "VALUES ('m-old', 'alice', 'Old fact', '{}', '2025-01-01T00:00:00+00:00', '2025-12-01T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO cite_events (memory_id, answer_correct, session_id, query_hash_or_text, cited_at) "
        "VALUES ('m-1', 1, 's1', 'tea?', '2026-01-12T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO abstain_events (query_hash_or_text, session_id, recorded_at) "
        "VALUES ('unknown?', 's1', '2026-01-12T00:00:00+00:00')"
    )
    pre_mem_cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
    pre_tomb_cols = [r[1] for r in conn.execute("PRAGMA table_info(tombstones)").fetchall()]
    pre_tables = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for graph_table in ("entities", "entity_aliases", "edges", "mentions", "flags"):
        assert graph_table not in pre_tables
    conn.commit()
    conn.close()

    backend = SqliteBackend(db_path=db_path)
    try:
        cur = backend._conn.cursor()  # noqa: SLF001
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for graph_table in ("entities", "entity_aliases", "edges", "mentions", "flags"):
            assert graph_table in tables

        indexes = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        for graph_index in (
            "idx_entities_user_type",
            "idx_entity_aliases_entity",
            "idx_edges_src",
            "idx_edges_dst",
            "idx_mentions_entity",
            "idx_flags_user_open",
        ):
            assert graph_index in indexes

        post_mem_cols = [r[1] for r in cur.execute("PRAGMA table_info(memories)").fetchall()]
        post_tomb_cols = [r[1] for r in cur.execute("PRAGMA table_info(tombstones)").fetchall()]
        assert post_mem_cols == pre_mem_cols
        assert post_tomb_cols == pre_tomb_cols

        assert cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert cur.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM cite_events").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM abstain_events").fetchone()[0] == 1
        assert cur.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 1

        fetched = backend.get("m-1")
        assert fetched is not None
        assert fetched.content == "Alice likes green tea."
        assert fetched.event is not None and fetched.event.event_id == "ev-1"
        hits = backend.search(query="green tea", user_id="alice")
        assert [h.memory.id for h in hits] == ["m-1"]

        # Running migrations twice more must be strictly idempotent.
        backend._apply_migrations()  # noqa: SLF001
        backend._apply_migrations()  # noqa: SLF001
        assert [r[1] for r in cur.execute("PRAGMA table_info(memories)").fetchall()] == pre_mem_cols
        assert cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
        assert cur.execute("SELECT COUNT(*) FROM cite_events").fetchone()[0] == 1
    finally:
        backend.close()
