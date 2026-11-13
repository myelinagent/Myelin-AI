"""SQLite-backed Backend implementation.

Single-file persistence with bi-temporal append-only fact storage,
event-anchored provenance, and apoptosis state tracking. Pure stdlib —
no compile step, no native deps. Layout:

  memories       : the live + quarantined substrate; tombstoned rows
                   move to `tombstones` so the active retrieval view
                   stays small.
  tombstones     : audit-trail rows. `search()` never reads from here.
  events         : event-provenance nodes referenced by memories.event_id.
  cite_events    : v1.2 — append-only cite event audit (FR-A1, FR-A6).
  abstain_events : v1.2 — append-only abstain event audit (FR-A6).

Indexes target the hot retrieval path: user_id + state for `search()`
and `list_memories()`, and the (user_id, t_valid) ordering used by
`as_of=t` reconstruction.

Retrieval is BM25 over a SQLite FTS5 index (``memories_fts``) kept in
step with the ``memories`` table, with English stemming. FTS5 ships with
the SQLite bundled in CPython, so the backend stays dependency-free; on a
SQLite built without it, search falls back to a token-overlap scan of the
user's memories.

v1.2 migration discipline: ``_apply_migrations`` is idempotent on every
connect. ``PRAGMA table_info`` introspection adds missing columns; new
tables use ``CREATE TABLE IF NOT EXISTS``. A v1 substrate opens cleanly
under v1.2 code with no data loss; equation collapses to v1 form when
new fields default to 0/NULL.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from typing import Any

from myelin.types import Event, Memory, SearchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    event_id        TEXT,
    t_valid         TEXT NOT NULL,
    t_invalid       TEXT,
    confidence      REAL NOT NULL DEFAULT 0.5,
    importance      REAL NOT NULL DEFAULT 0.5,
    health_state    TEXT NOT NULL DEFAULT 'healthy',
    quarantine_entered_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_user_state
    ON memories(user_id, health_state);
CREATE INDEX IF NOT EXISTS idx_memories_user_tvalid
    ON memories(user_id, t_valid);

CREATE TABLE IF NOT EXISTS tombstones (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata_json   TEXT NOT NULL,
    event_id        TEXT,
    t_valid         TEXT NOT NULL,
    t_invalid       TEXT,
    confidence      REAL NOT NULL DEFAULT 0.5,
    importance      REAL NOT NULL DEFAULT 0.5,
    health_state    TEXT NOT NULL DEFAULT 'tombstoned',
    quarantine_entered_at TEXT,
    tombstoned_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    type            TEXT NOT NULL DEFAULT '',
    anchor_datetime TEXT,
    location        TEXT NOT NULL DEFAULT '',
    participant_ids TEXT NOT NULL DEFAULT '[]',
    evidence_turn_ids TEXT NOT NULL DEFAULT '[]'
);
"""

# v1.2 new tables. CREATE IF NOT EXISTS is idempotent on its own; this
# block runs after the original _SCHEMA. ALTER TABLE additions on the
# pre-existing memories and tombstones tables are handled separately
# via PRAGMA introspection in ``_apply_migrations`` because SQLite has
# no native "ALTER TABLE IF NOT EXISTS COLUMN".
_SCHEMA_V1_2_NEW_TABLES = """
CREATE TABLE IF NOT EXISTS cite_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id           TEXT    NOT NULL,
    answer_correct      INTEGER NOT NULL CHECK (answer_correct IN (0, 1)),
    session_id          TEXT,
    query_hash_or_text  TEXT,
    cited_at            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cite_events_memory_id
    ON cite_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_cite_events_cited_at
    ON cite_events(cited_at);

CREATE TABLE IF NOT EXISTS abstain_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash_or_text  TEXT,
    session_id          TEXT,
    recorded_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_abstain_events_recorded_at
    ON abstain_events(recorded_at);
"""

# v1.2 cite-signal columns added to memories and tombstones via
# idempotent ALTER. Each tuple is (column_name, sql_fragment).
_V1_2_CITE_COLUMNS = (
    ("cite_correct_count",   "cite_correct_count INTEGER NOT NULL DEFAULT 0"),
    ("cite_incorrect_count", "cite_incorrect_count INTEGER NOT NULL DEFAULT 0"),
    ("last_cited_at",        "last_cited_at TEXT"),
)

# Full-text index over live memories. Only ``content`` is searchable;
# ``user_id`` rides along as a filter. ``memories_fts_rows`` maps each
# memory to its row in the index so one memory can be updated or removed
# without scanning the index.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    user_id UNINDEXED,
    tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS memories_fts_rows (
    memory_id   TEXT PRIMARY KEY,
    fts_rowid   INTEGER NOT NULL UNIQUE
);
"""

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_QUERY_TERM_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with and or not "
    "what who where when why how do does did this that these those i you he she "
    "it we they me him her us them my your his its our their s".split()
)


def _tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS
    }


def _fts_query(text: str) -> str:
    """Turn free text into an FTS5 query matching any of its terms.

    Terms are quoted so punctuation and FTS5 operators in the text are
    never parsed as syntax. Returns "" when no searchable term is left.
    """
    terms = dict.fromkeys(
        t.lower()
        for t in _QUERY_TERM_RE.findall(text)
        if t.lower() not in _STOPWORDS
    )
    return " OR ".join(f'"{t}"' for t in terms)


# SQLite caps bound parameters per statement (999 on builds before
# 3.32), so IN (...) lookups are issued in chunks below that limit.
_MAX_SQL_PARAMS = 900


def _chunks(items: list[str]) -> Iterator[list[str]]:
    for start in range(0, len(items), _MAX_SQL_PARAMS):
        yield items[start : start + _MAX_SQL_PARAMS]


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _iso_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


class SqliteBackend:
    """SQLite-backed substrate. See module docstring for layout.

    Thread-safe via a single lock around the connection. SQLite's
    serializable isolation handles the concurrent-reader case fine; the
    lock matters for the in-process counters and the multi-statement
    apoptosis transitions.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self._apply_migrations()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- migration ----------------------------------------------

    def _apply_migrations(self) -> None:
        """Idempotent schema migration. Runs on every connect.

        Steps:
        1. Apply original v1 schema (CREATE TABLE IF NOT EXISTS).
        2. PRAGMA-introspect memories + tombstones and add v1.2 cite
           columns where missing.
        3. Apply v1.2 new-table schema (CREATE TABLE IF NOT EXISTS).

        Idempotent: a v1.2 substrate opening under v1.2 code skips every
        ALTER and skips every CREATE because all columns + tables
        already exist.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.executescript(_SCHEMA)

            # v1.2 cite columns on memories
            existing_mem_cols = self._column_names("memories")
            for col_name, col_sql in _V1_2_CITE_COLUMNS:
                if col_name not in existing_mem_cols:
                    cursor.execute(
                        f"ALTER TABLE memories ADD COLUMN {col_sql}"
                    )

            # v1.2 cite columns on tombstones (mirror, for audit trail)
            existing_tomb_cols = self._column_names("tombstones")
            for col_name, col_sql in _V1_2_CITE_COLUMNS:
                if col_name not in existing_tomb_cols:
                    cursor.execute(
                        f"ALTER TABLE tombstones ADD COLUMN {col_sql}"
                    )

            cursor.executescript(_SCHEMA_V1_2_NEW_TABLES)
            self._conn.commit()
            self._fts = self._init_fts()

    def _init_fts(self) -> bool:
        """Create the full-text index, indexing any memories written
        before it existed. Returns False when this SQLite has no FTS5."""
        cursor = self._conn.cursor()
        try:
            cursor.executescript(_FTS_SCHEMA)
        except sqlite3.OperationalError:
            return False
        indexed = cursor.execute(
            "SELECT COUNT(*) FROM memories_fts_rows"
        ).fetchone()[0]
        stored = cursor.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if indexed != stored:
            cursor.execute("DELETE FROM memories_fts")
            cursor.execute("DELETE FROM memories_fts_rows")
            cursor.execute(
                "INSERT INTO memories_fts (rowid, content, user_id) "
                "SELECT rowid, content, user_id FROM memories"
            )
            cursor.execute(
                "INSERT INTO memories_fts_rows (memory_id, fts_rowid) "
                "SELECT id, rowid FROM memories"
            )
        self._conn.commit()
        return True

    def _column_names(self, table: str) -> set[str]:
        """Return the set of column names for ``table``. Uses
        ``PRAGMA table_info`` which is read-only and safe to call on
        the live connection."""
        cursor = self._conn.cursor()
        rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
        # PRAGMA table_info rows: (cid, name, type, notnull, dflt_value, pk)
        return {row[1] for row in rows}

    # ---- public Backend protocol ---------------------------------

    def add_memory(
        self,
        *,
        content: str,
        user_id: str,
        metadata: dict | None = None,
        event: Event | None = None,
        t_valid: datetime | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
    ) -> Memory:
        if t_valid is None:
            t_valid = datetime.now(UTC)
        elif t_valid.tzinfo is None:
            t_valid = t_valid.replace(tzinfo=UTC)

        mem_id = str(uuid.uuid4())
        meta_dict = dict(metadata or {})

        with self._lock:
            if event is not None:
                self._upsert_event(event)
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO memories
                   (id, user_id, content, metadata_json, event_id,
                    t_valid, t_invalid, confidence, importance,
                    health_state, quarantine_entered_at)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 'healthy', NULL)""",
                (
                    mem_id,
                    user_id,
                    content,
                    json.dumps(meta_dict),
                    event.event_id if event else None,
                    _dt_to_iso(t_valid),
                    confidence,
                    importance,
                ),
            )
            if self._fts:
                cur.execute(
                    "INSERT INTO memories_fts (content, user_id) VALUES (?, ?)",
                    (content, user_id),
                )
                cur.execute(
                    "INSERT INTO memories_fts_rows (memory_id, fts_rowid) "
                    "VALUES (?, ?)",
                    (mem_id, cur.lastrowid),
                )
            self._conn.commit()

        return Memory(
            id=mem_id,
            content=content,
            user_id=user_id,
            t_valid=t_valid,
            metadata=meta_dict,
            event=event,
            confidence=confidence,
            importance=importance,
            health_state="healthy",
        )

    def search(
        self,
        *,
        query: str,
        user_id: str,
        top_k: int = 10,
        as_of: datetime | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        if self._fts:
            return self._search_fts(query, user_id, top_k, as_of)
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        rows = self._scan_for_user(user_id, as_of=as_of)
        scored: list[tuple[float, Memory]] = []
        for mem in rows:
            score = self._score(mem, q_tokens)
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda kv: (-kv[0], kv[1].id))
        return [SearchResult(memory=m, score=s) for s, m in scored[:top_k]]

    def _search_fts(
        self, query: str, user_id: str, top_k: int, as_of: datetime | None
    ) -> list[SearchResult]:
        """BM25-ranked search over the full-text index."""
        match = _fts_query(query)
        if not match:
            return []
        if as_of is not None and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        sql = (
            "SELECT r.memory_id, bm25(memories_fts) AS rank "
            "FROM memories_fts JOIN memories_fts_rows r "
            "ON r.fts_rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND user_id = ? "
            "ORDER BY rank, r.memory_id"
        )
        params: list[Any] = [match, user_id]
        if as_of is None:
            sql += " LIMIT ?"
            params.append(top_k)
        with self._lock:
            ranked = self._conn.execute(sql, params).fetchall()
            memories = {m.id: m for m in self.get_batch([r[0] for r in ranked])}
        results: list[SearchResult] = []
        for memory_id, rank in ranked:
            memory = memories.get(memory_id)
            if memory is None:
                continue
            if as_of is not None and (
                memory.t_valid > as_of
                or (memory.t_invalid is not None and memory.t_invalid <= as_of)
            ):
                continue
            # bm25() is lower-is-better and negative; expose it as a
            # higher-is-better score.
            results.append(SearchResult(memory=memory, score=-rank))
            if len(results) >= top_k:
                break
        return results

    def get(self, memory_id: str) -> Memory | None:
        found = self.get_batch([memory_id])
        return found[0] if found else None

    def get_batch(self, memory_ids: Sequence[str]) -> list[Memory]:
        """Fetch several memories by id without one query per id.

        Returns live (non-tombstoned) memories in the order their ids
        were first requested. Unknown and tombstoned ids are omitted,
        and a repeated id yields one memory.
        """
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        with self._lock:
            cur = self._conn.cursor()
            raw: list[dict[str, Any]] = []
            for chunk in _chunks(ids):
                rows = cur.execute(
                    "SELECT * FROM memories WHERE id IN "
                    f"({_placeholders(len(chunk))}) "
                    "AND health_state != 'tombstoned'",
                    chunk,
                ).fetchall()
                cols = [d[0] for d in cur.description]
                raw.extend(dict(zip(cols, r, strict=True)) for r in rows)
            by_id = {m.id: m for m in self._rows_to_memories(raw)}
        return [by_id[i] for i in ids if i in by_id]

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        with self._lock:
            existing = self.get(memory_id)
            if existing is None:
                raise KeyError(f"Memory {memory_id!r} not found or tombstoned")
            new_content = content if content is not None else existing.content
            new_meta = (
                metadata if metadata is not None else existing.metadata
            )
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE memories SET content = ?, metadata_json = ? WHERE id = ?",
                (new_content, json.dumps(new_meta), memory_id),
            )
            if self._fts and new_content != existing.content:
                cur.execute(
                    "UPDATE memories_fts SET content = ? WHERE rowid = "
                    "(SELECT fts_rowid FROM memories_fts_rows WHERE memory_id = ?)",
                    (new_content, memory_id),
                )
            self._conn.commit()
            existing.content = new_content
            existing.metadata = new_meta
            return existing

    def delete(self, memory_id: str) -> bool:
        # Hard delete — for caller-initiated removal. Apoptosis uses
        # `transition_state(... new_state='tombstoned')` which preserves
        # the audit trail; an explicit delete removes that trail too, so
        # the content and its cite history are gone from every table.
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cur.rowcount
            self._unindex(cur, memory_id)
            cur.execute("DELETE FROM tombstones WHERE id = ?", (memory_id,))
            deleted += cur.rowcount
            cur.execute(
                "DELETE FROM cite_events WHERE memory_id = ?", (memory_id,)
            )
            self._conn.commit()
            return deleted > 0

    def list_memories(
        self,
        *,
        user_id: str,
        include_tombstoned: bool = False,
        only_healthy: bool = False,
    ) -> list[Memory]:
        with self._lock:
            cur = self._conn.cursor()
            if only_healthy:
                rows = cur.execute(
                    "SELECT * FROM memories WHERE user_id = ? AND health_state = 'healthy' "
                    "ORDER BY t_valid",
                    (user_id,),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT * FROM memories WHERE user_id = ? ORDER BY t_valid",
                    (user_id,),
                ).fetchall()
            cols = [d[0] for d in cur.description]
            live = self._rows_to_memories(
                [dict(zip(cols, r, strict=True)) for r in rows]
            )
            if not include_tombstoned:
                return live
            # Tombstones table has an extra `tombstoned_at` column at the
            # end. SELECT * brings it through; _row_to_memory ignores
            # unknown keys so the shape lines up.
            tomb_rows = cur.execute(
                "SELECT * FROM tombstones WHERE user_id = ? ORDER BY tombstoned_at",
                (user_id,),
            ).fetchall()
            tomb_cols = [d[0] for d in cur.description]
            tombstoned = self._rows_to_memories(
                [dict(zip(tomb_cols, r, strict=True)) for r in tomb_rows]
            )
            return live + tombstoned

    def transition_state(
        self,
        memory_id: str,
        *,
        new_state: str,
        quarantine_entered_at: datetime | None = None,
    ) -> bool:
        if new_state not in {"healthy", "quarantined", "tombstoned"}:
            raise ValueError(f"invalid health_state {new_state!r}")
        with self._lock:
            cur = self._conn.cursor()
            if new_state == "tombstoned":
                # Move row from memories -> tombstones atomically.
                # Read with explicit column list (resilient to future ALTERs).
                existing = cur.execute(
                    "SELECT id, user_id, content, metadata_json, event_id, "
                    "t_valid, t_invalid, confidence, importance, health_state, "
                    "quarantine_entered_at, cite_correct_count, "
                    "cite_incorrect_count, last_cited_at "
                    "FROM memories WHERE id = ?",
                    (memory_id,),
                ).fetchone()
                if existing is None:
                    return False
                now_iso = _dt_to_iso(datetime.now(UTC))
                (mid, user_id, content, metadata_json, event_id,
                 t_valid, t_invalid, confidence, importance, _health_state,
                 quarantine_entered, cite_correct, cite_incorrect,
                 last_cited) = existing
                cur.execute(
                    """INSERT INTO tombstones
                       (id, user_id, content, metadata_json, event_id, t_valid,
                        t_invalid, confidence, importance, health_state,
                        quarantine_entered_at, tombstoned_at,
                        cite_correct_count, cite_incorrect_count, last_cited_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'tombstoned',
                               ?, ?, ?, ?, ?)""",
                    (mid, user_id, content, metadata_json, event_id,
                     t_valid, t_invalid, confidence, importance,
                     quarantine_entered, now_iso,
                     cite_correct, cite_incorrect, last_cited),
                )
                cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                self._unindex(cur, memory_id)
                self._conn.commit()
                return True

            cur.execute(
                "UPDATE memories SET health_state = ?, quarantine_entered_at = ? "
                "WHERE id = ?",
                (
                    new_state,
                    _dt_to_iso(quarantine_entered_at),
                    memory_id,
                ),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ---- v1.2 cite-event surface --------------------------------

    def record_cite(
        self,
        *,
        memory_id: str,
        answer_correct: bool,
        session_id: str | None,
        query: str | None,
        now: datetime,
    ) -> None:
        """Increment per-memory cite counter + insert cite_events row.

        Atomic per-memory: SQLite serializes ``UPDATE`` per-row.
        No-op when ``memory_id`` is unknown or tombstoned. The event
        row is NOT written on a race-miss (the memory was tombstoned
        between the client-side ``get`` and this call) — keeps audit
        consistent with apoptosis state.
        """
        counter_col = (
            "cite_correct_count" if answer_correct else "cite_incorrect_count"
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"UPDATE memories SET {counter_col} = {counter_col} + 1, "
                "last_cited_at = ? "
                "WHERE id = ? AND health_state != 'tombstoned'",
                (_dt_to_iso(now), memory_id),
            )
            if cur.rowcount == 0:
                # Memory tombstoned / unknown — skip the event row too.
                return
            cur.execute(
                "INSERT INTO cite_events "
                "(memory_id, answer_correct, session_id, "
                " query_hash_or_text, cited_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    memory_id,
                    1 if answer_correct else 0,
                    session_id,
                    query,
                    _dt_to_iso(now),
                ),
            )
            self._conn.commit()

    def record_abstain(
        self,
        *,
        query: str | None,
        session_id: str | None,
        now: datetime,
    ) -> None:
        """Record a zero-result retrieval event. FR-A6."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO abstain_events "
                "(query_hash_or_text, session_id, recorded_at) "
                "VALUES (?, ?, ?)",
                (query, session_id, _dt_to_iso(now)),
            )
            self._conn.commit()

    # ---- adapter-support helpers --------------------------------

    def _clear_all(self) -> None:
        """Wipe all memories, tombstones, events, cite_events,
        abstain_events. Used by the Mem0 adapter's ``reset()`` — not
        part of the public Backend Protocol.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM memories")
            if self._fts:
                cur.execute("DELETE FROM memories_fts")
                cur.execute("DELETE FROM memories_fts_rows")
            cur.execute("DELETE FROM tombstones")
            cur.execute("DELETE FROM events")
            cur.execute("DELETE FROM cite_events")
            cur.execute("DELETE FROM abstain_events")
            self._conn.commit()

    # ---- internals ----------------------------------------------

    def _unindex(self, cur: sqlite3.Cursor, memory_id: str) -> None:
        """Drop a memory from the full-text index (it left ``memories``)."""
        if self._fts:
            cur.execute(
                "DELETE FROM memories_fts WHERE rowid = "
                "(SELECT fts_rowid FROM memories_fts_rows WHERE memory_id = ?)",
                (memory_id,),
            )
            cur.execute(
                "DELETE FROM memories_fts_rows WHERE memory_id = ?", (memory_id,)
            )

    def _upsert_event(self, event: Event) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO events
               (event_id, type, anchor_datetime, location,
                participant_ids, evidence_turn_ids)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id) DO UPDATE SET
                   type=excluded.type,
                   anchor_datetime=excluded.anchor_datetime,
                   location=excluded.location,
                   participant_ids=excluded.participant_ids,
                   evidence_turn_ids=excluded.evidence_turn_ids""",
            (
                event.event_id,
                event.type,
                _dt_to_iso(event.anchor_datetime),
                event.location,
                json.dumps(list(event.participant_ids)),
                json.dumps(list(event.evidence_turn_ids)),
            ),
        )

    def _scan_for_user(
        self, user_id: str, *, as_of: datetime | None
    ) -> list[Memory]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT * FROM memories WHERE user_id = ? AND health_state != 'tombstoned'",
                (user_id,),
            ).fetchall()
            cols = [d[0] for d in cur.description]
            out: list[Memory] = []
            t_cmp = (
                as_of.replace(tzinfo=UTC) if as_of and as_of.tzinfo is None else as_of
            )
            for mem in self._rows_to_memories(
                [dict(zip(cols, r, strict=True)) for r in rows]
            ):
                if t_cmp is not None:
                    if mem.t_valid > t_cmp:
                        continue
                    if mem.t_invalid is not None and mem.t_invalid <= t_cmp:
                        continue
                out.append(mem)
            return out

    def _score(self, mem: Memory, q_tokens: set[str]) -> float:
        c_tokens = _tokens(mem.content)
        overlap = len(c_tokens & q_tokens)
        if overlap == 0:
            return 0.0
        # Lightweight lexical score: overlap weighted by content density.
        # Mirrors the cortex T2Store's lexical-graph channel before BM25
        # fusion was added — good enough for the SDK's default backend.
        return overlap + 0.5 * (overlap / max(1, len(c_tokens)))

    def _load_events(self, event_ids: Iterable[str]) -> dict[str, Event]:
        """Load events by id, keyed by event_id. Unknown ids are absent."""
        ids = list(dict.fromkeys(event_ids))
        events: dict[str, Event] = {}
        cur = self._conn.cursor()
        for chunk in _chunks(ids):
            ev_rows = cur.execute(
                "SELECT * FROM events WHERE event_id IN "
                f"({_placeholders(len(chunk))})",
                chunk,
            ).fetchall()
            ev_cols = [d[0] for d in cur.description]
            for ev_row in ev_rows:
                ev = dict(zip(ev_cols, ev_row, strict=True))
                events[ev["event_id"]] = Event(
                    event_id=ev["event_id"],
                    type=ev.get("type", "") or "",
                    anchor_datetime=_iso_to_dt(ev.get("anchor_datetime")),
                    location=ev.get("location", "") or "",
                    participant_ids=tuple(
                        json.loads(ev.get("participant_ids") or "[]")
                    ),
                    evidence_turn_ids=tuple(
                        json.loads(ev.get("evidence_turn_ids") or "[]")
                    ),
                )
        return events

    def _rows_to_memories(self, rows: list[dict[str, Any]]) -> list[Memory]:
        """Convert rows to memories, loading all their events up front
        instead of issuing one event query per row."""
        events = self._load_events(
            r["event_id"] for r in rows if r.get("event_id")
        )
        return [self._row_to_memory(r, events) for r in rows]

    def _row_to_memory(
        self, row: dict[str, Any], events: dict[str, Event] | None = None
    ) -> Memory:
        ev_id = row.get("event_id")
        if not ev_id:
            event = None
        elif events is not None:
            event = events.get(ev_id)
        else:
            event = self._load_events([ev_id]).get(ev_id)
        return Memory(
            id=row["id"],
            content=row["content"],
            user_id=row["user_id"],
            t_valid=_iso_to_dt(row["t_valid"]),  # type: ignore[arg-type]
            t_invalid=_iso_to_dt(row.get("t_invalid")),
            metadata=json.loads(row["metadata_json"]),
            event=event,
            confidence=float(row.get("confidence", 0.5)),
            importance=float(row.get("importance", 0.5)),
            health_state=row.get("health_state", "healthy"),
            quarantine_entered_at=_iso_to_dt(
                row.get("quarantine_entered_at")
            ),
            # v1.2 cite fields — defaults preserve v1 behavior when
            # reading a substrate without these columns. row.get() with
            # default keeps the read tolerant.
            cite_correct_count=int(row.get("cite_correct_count") or 0),
            cite_incorrect_count=int(row.get("cite_incorrect_count") or 0),
            last_cited_at=_iso_to_dt(row.get("last_cited_at")),
        )
