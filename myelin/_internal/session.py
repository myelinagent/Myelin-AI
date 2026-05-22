"""Session state and registry for the working-memory layer.

Owns:
  - per-session ring buffer of observed-but-uncommitted turns
  - per-session working set of long-term memory ids (LRU OrderedDict)
  - TTL-based session expiry via a daemon sweeper thread
  - LRU eviction at the MAX_CONCURRENT_SESSIONS cap

Does NOT own:
  - the long-term storage substrate (that's the Backend)
  - the salience gate (that runs at commit() time, in the client)
  - the cite-event log (separate substrate, see Backend.record_cite)

The SessionRegistry is a process-singleton. Multiple MemoryClient
instances in the same process share one registry — the correct
semantic for multi-user single-process deployments. Multi-process
deployments (e.g. gunicorn workers) each get their own registry;
session ids do NOT migrate across processes in v1.2. Redis-backed
cross-process registry is v1.3.

Concurrency:
  - SessionRegistry._registry_lock protects the sessions dict.
  - Each SessionState has its own RLock for its mutable contents.
  - Lock order is always registry -> session -> backend (latter is
    held by SqliteBackend, not by this module).

No PII flows through this module. observed turns and working-set
entries are RAM-only and dropped on process exit. Anything that ought
to be durable goes through commit() -> Backend.add_memory().
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar

from myelin._internal.constants import (
    MAX_CONCURRENT_SESSIONS,
    OBSERVED_RING_MAX,
    SESSION_TTL_SECONDS,
    SWEEPER_INTERVAL_SECONDS,
    WORKING_SET_MAX,
)
from myelin.types import ObservedTurn

logger = logging.getLogger(__name__)


def _new_turn_id() -> str:
    """Roughly-sortable turn id. ``{millis_since_epoch}-{uuid4_short}``.

    Not a full ULID — that would be a new dep. The millis prefix is
    enough to order observed turns lexicographically within a session,
    which is what callers iterate by; the uuid suffix is enough to
    guarantee uniqueness across collocated processes.
    """
    millis = int(time.time() * 1000)
    suffix = uuid.uuid4().hex[:12]
    return f"{millis:013d}-{suffix}"


@dataclass
class SessionState:
    """A single session's mutable state. One instance per active
    ``session_id`` per process. Protected by ``lock`` (RLock).

    ``observed_buffer`` is a deque of ``ObservedTurn`` with maxlen
    ``OBSERVED_RING_MAX``. FIFO eviction on overflow.

    ``working_set`` is an OrderedDict mapping memory_id -> attended_at.
    LRU eviction at ``WORKING_SET_MAX``. The order is LRU-first,
    MRU-last (matches OrderedDict.popitem(last=False) semantics).
    """

    session_id: str
    user_id: str
    started_at: datetime
    last_active_at: datetime
    observed_buffer: deque[ObservedTurn] = field(default_factory=deque)
    working_set: "OrderedDict[str, datetime]" = field(default_factory=OrderedDict)
    n_committed: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def append_observed(
        self,
        *,
        turn_id: str,
        content: str,
        metadata: dict | None,
    ) -> ObservedTurn:
        """Append an ObservedTurn under the session lock. Returns the
        appended record. FIFO-evicts the oldest entry when the deque
        is at maxlen — caller can detect by counting subsequent calls."""
        observed = ObservedTurn(
            turn_id=turn_id,
            content=content,
            observed_at=datetime.now(UTC),
            monotonic=time.monotonic(),
            metadata=dict(metadata or {}),
        )
        with self.lock:
            self.observed_buffer.append(observed)
            self.last_active_at = observed.observed_at
        return observed

    def attend(self, memory_id: str) -> None:
        """Add ``memory_id`` to the working set or promote it to MRU
        position. LRU-evicts the oldest entry when the set exceeds
        ``WORKING_SET_MAX``."""
        now = datetime.now(UTC)
        with self.lock:
            if memory_id in self.working_set:
                # Reattend -> MRU promotion (move to last)
                del self.working_set[memory_id]
            self.working_set[memory_id] = now
            while len(self.working_set) > WORKING_SET_MAX:
                evicted_id, _ = self.working_set.popitem(last=False)
                logger.debug(
                    "working set LRU eviction: session=%s evicted=%s",
                    self.session_id, evicted_id,
                )
            self.last_active_at = now

    def drain_observed(self, turn_ids: list[str]) -> list[ObservedTurn]:
        """Remove the named ObservedTurns from the buffer and return
        them. Preserves order of remaining turns in the buffer.

        Raises LookupError if any requested ``turn_id`` is missing —
        either from a different session (EC-B1), or already drained,
        or evicted by FIFO overflow (EC-B6).
        """
        with self.lock:
            wanted = set(turn_ids)
            drained: list[ObservedTurn] = []
            remaining: deque[ObservedTurn] = deque(
                maxlen=self.observed_buffer.maxlen
            )
            for turn in self.observed_buffer:
                if turn.turn_id in wanted:
                    drained.append(turn)
                    wanted.discard(turn.turn_id)
                else:
                    remaining.append(turn)
            if wanted:
                raise LookupError(
                    f"drain_observed: turn_ids not found in buffer "
                    f"(session={self.session_id}): {sorted(wanted)}"
                )
            self.observed_buffer = remaining
            self.n_committed += len(drained)
            self.last_active_at = datetime.now(UTC)
            return drained


class SessionRegistry:
    """Process-singleton registry of active sessions.

    Construct via ``SessionRegistry.get_instance()`` — direct
    instantiation works for tests but the singleton path is what
    MemoryClient uses.

    Owns the TTL sweeper daemon thread. Started lazily on first
    ``get_or_create``. Tests call ``reset_for_testing()`` in fixtures
    to ensure cross-test isolation.
    """

    _instance: ClassVar["SessionRegistry | None"] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SessionRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Tear down the singleton — used by pytest fixtures so test
        isolation holds. Waits up to 1s for the sweeper to exit."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance._shutdown()
            cls._instance = None

    def __init__(
        self,
        *,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_SESSIONS,
        observed_ring_max: int = OBSERVED_RING_MAX,
        sweeper_interval_seconds: int = SWEEPER_INTERVAL_SECONDS,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_concurrent = max_concurrent
        self._observed_ring_max = observed_ring_max
        self._sweeper_interval_seconds = sweeper_interval_seconds

        self._sessions: dict[str, SessionState] = {}
        self._registry_lock = threading.RLock()
        self._sweeper_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Event-bus hook — set by MemoryClient.__init__ when an
        # observability sink is wired. Default no-op so the registry
        # doesn't grow dependencies.
        self._event_sink: "callable | None" = None

    def get_or_create(self, session_id: str, user_id: str) -> SessionState:
        """Idempotent — returns the existing session if present,
        creating one otherwise. Touches ``last_active_at`` on every
        call. Triggers LRU eviction if at cap.

        Lazy-starts the sweeper thread on the first call.
        """
        with self._registry_lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                existing.last_active_at = datetime.now(UTC)
                self._ensure_sweeper_running()
                return existing

            if len(self._sessions) >= self._max_concurrent:
                self._evict_lru_session(except_id=session_id)

            now = datetime.now(UTC)
            state = SessionState(
                session_id=session_id,
                user_id=user_id,
                started_at=now,
                last_active_at=now,
                observed_buffer=deque(maxlen=self._observed_ring_max),
                working_set=OrderedDict(),
            )
            self._sessions[session_id] = state
            self._ensure_sweeper_running()
            logger.debug("session created: %s (user=%s)", session_id, user_id)
            return state

    def lookup(self, session_id: str) -> SessionState | None:
        """Return the SessionState for ``session_id`` or None if it
        does not exist (was never created OR was TTL-expired)."""
        with self._registry_lock:
            return self._sessions.get(session_id)

    def expire(self, session_id: str) -> None:
        """Explicitly drop a session (test affordance + sweeper helper).
        Emits the session.expired event if a sink is wired."""
        self._expire_session(session_id)

    # ---- sweeper -------------------------------------------------

    def _ensure_sweeper_running(self) -> None:
        if self._sweeper_thread is not None and self._sweeper_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._sweeper_thread = threading.Thread(
            target=self._sweeper_loop,
            name="myelin-session-sweeper",
            daemon=True,
        )
        self._sweeper_thread.start()

    def _sweeper_loop(self) -> None:
        while not self._shutdown_event.is_set():
            # Wait with timeout so shutdown is responsive (no busy poll).
            if self._shutdown_event.wait(self._sweeper_interval_seconds):
                return
            self._sweep_expired()

    def _sweep_expired(self) -> None:
        now = datetime.now(UTC)
        expired_ids: list[str] = []
        with self._registry_lock:
            for sid, state in self._sessions.items():
                idle = (now - state.last_active_at).total_seconds()
                if idle >= self._ttl_seconds:
                    expired_ids.append(sid)
        for sid in expired_ids:
            self._expire_session(sid)

    def _expire_session(self, session_id: str) -> None:
        with self._registry_lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return  # raced — another caller already expired it

        with state.lock:
            n_observed = len(state.observed_buffer)
            n_working = len(state.working_set)

        logger.info(
            "session expired: id=%s user=%s observed=%d working=%d",
            session_id, state.user_id, n_observed, n_working,
        )

        # Emit observability event if a sink is wired. The event shape
        # matches Phase 3 §9.3.
        sink = self._event_sink
        if sink is not None:
            try:
                sink({
                    "event": "session.expired",
                    "ts": datetime.now(UTC).isoformat(),
                    "session_id": session_id,
                    "user_id": state.user_id,
                    "n_observed_dropped": n_observed,
                    "n_working_set": n_working,
                    "started_at": state.started_at.isoformat(),
                    "last_active_at": state.last_active_at.isoformat(),
                })
            except Exception:  # noqa: BLE001 — sink errors must not propagate
                logger.exception("event sink raised for session.expired")

    def _evict_lru_session(self, *, except_id: str) -> None:
        with self._registry_lock:
            candidates = [
                s for s in self._sessions.values()
                if s.session_id != except_id
            ]
            if not candidates:
                raise RuntimeError(
                    f"MAX_CONCURRENT_SESSIONS ({self._max_concurrent}) "
                    "reached and no other session is evictable. Raise "
                    "MYELIN_MAX_CONCURRENT_SESSIONS or shorten "
                    "MYELIN_SESSION_TTL_SECONDS."
                )
            lru = min(candidates, key=lambda s: s.last_active_at)
            logger.warning(
                "LRU session eviction: evicting %s (last_active=%s)",
                lru.session_id, lru.last_active_at.isoformat(),
            )
        self._expire_session(lru.session_id)

    def _shutdown(self) -> None:
        self._shutdown_event.set()
        thread = self._sweeper_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self._registry_lock:
            self._sessions.clear()


__all__ = [
    "SessionRegistry",
    "SessionState",
    "_new_turn_id",
]
