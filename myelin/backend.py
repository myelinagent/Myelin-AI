"""Backend storage protocol.

The Backend is the persistence + retrieval layer. v1 ships one
implementation (SqliteBackend in `myelin.backends.sqlite`); future
versions may add postgres / RVF / vector-native backends. The client
talks only to this Protocol — it does not import from
`myelin.backends.*` directly.

The protocol is intentionally small and stateful (the backend owns the
substrate; the client is a thin facade). All methods are sync; async
backends should wrap their async I/O at the boundary.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from myelin.types import Event, Memory, SearchResult


class Backend(Protocol):
    """Storage contract for the Myelin SDK.

    Implementations must enforce:
    - `add` returns the stored Memory with `id` populated.
    - `search` returns SearchResult list ordered by score desc.
    - `as_of` filtering reconstructs the substrate as it existed at `t`:
       facts with `t_valid <= t` AND (`t_invalid is None OR t_invalid > t`).
    - Tombstoned memories are NEVER returned by `search` or `get`.
    """

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
        ...

    def search(
        self,
        *,
        query: str,
        user_id: str,
        top_k: int = 10,
        as_of: datetime | None = None,
    ) -> list[SearchResult]:
        ...

    def get(self, memory_id: str) -> Memory | None:
        ...

    # Optional: ``get_batch(memory_ids) -> list[Memory]`` returns the live
    # memories for several ids in request order, omitting unknown and
    # tombstoned ids. It is deliberately not part of this Protocol so
    # existing backends stay conformant; MemoryClient uses it when
    # present and falls back to per-id ``get()`` otherwise.

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        ...

    def delete(self, memory_id: str) -> bool:
        ...

    # Bulk operations used by consolidate / prune. Kept on the protocol
    # so swapping backends does not require reimplementing the
    # mechanisms — they call back through this surface.

    def list_memories(
        self,
        *,
        user_id: str,
        include_tombstoned: bool = False,
        only_healthy: bool = False,
    ) -> list[Memory]:
        ...

    def transition_state(
        self,
        memory_id: str,
        *,
        new_state: str,
        quarantine_entered_at: datetime | None = None,
    ) -> bool:
        """Move a memory between {healthy, quarantined, tombstoned}.

        Returns True on success, False if memory_id is unknown. The
        backend is responsible for moving tombstoned memories out of
        the active retrieval view while keeping them queryable for
        audit (see `list_memories(include_tombstoned=True)`)."""
        ...

    # ---- v1.2 cite-event surface (Innovation A) -------------------
    # Backends that do not implement cite/abstain raise
    # NotImplementedError pointing callers at the migration guide.
    # The SqliteBackend in this package implements all three.

    def record_cite(
        self,
        *,
        memory_id: str,
        answer_correct: bool,
        session_id: str | None,
        query: str | None,
        now: datetime,
    ) -> None:
        """Record one ``cite`` event and atomically increment the
        per-memory cite counter.

        No-op on tombstoned or unknown memory_id (no raise — see
        ``client.cite`` for the no-raise contract that callers rely on
        when iterating retrieval results).
        """
        ...

    def record_abstain(
        self,
        *,
        query: str | None,
        session_id: str | None,
        now: datetime,
    ) -> None:
        """Record one ``abstain`` event — a search that returned zero
        results (deliberate abstention). FR-A6.
        """
        ...
