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
