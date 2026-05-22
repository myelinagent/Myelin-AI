"""Mem0-compatible adapter for the Myelin substrate.

This module exposes :class:`MemoryClient` — a drop-in replacement for
``mem0.MemoryClient`` (from the ``mem0ai`` package, see
https://docs.mem0.ai/python-sdk). Callers who today use::

    from mem0 import MemoryClient

can switch a single import line to::

    from myelin.adapters.mem0 import MemoryClient

and pick up the event-anchored, dream-synthesis, and apoptosis-based
substrate built in Phase A of the Myelin SDK. Method signatures, kwargs
and the dict shapes returned by ``add``, ``search``, ``get`` etc. match
Mem0's documented contract — no further refactoring required at the
call site.

Two design properties matter:

1. **No runtime dependency on ``mem0ai``.** The adapter implements the
   surface directly; ``mem0`` does not need to be installed. The shape
   contract is reproduced here against Mem0's published API.
2. **Myelin extensions are additive.** Two extra methods —
   :meth:`MemoryClient.consolidate` (dream-synthesis self-test) and
   :meth:`MemoryClient.prune` (apoptosis quarantine → tombstone) — sit
   alongside the Mem0 surface. Drop-in callers can ignore them; callers
   who want the substrate's full mechanics call them explicitly.

The adapter's return dicts use Mem0's documented keys:
``id``, ``memory``, ``user_id``, ``metadata``, ``created_at``,
``updated_at``, ``score`` (search-only), and ``event`` (history-only,
myelin-specific). Datetimes are serialised as ISO-8601 strings.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from myelin.client import MemoryClient as _MyelinClient
from myelin.types import (
    ApoptosisReport,
    ConsolidationReport,
    Memory,
    SearchResult,
)

__all__ = ["MemoryClient"]


def _coerce_messages(messages: list[dict] | str) -> str:
    """Collapse Mem0's ``messages``/``str`` add-input into a single string.

    Mem0's ``add`` accepts either a bare string OR a list-of-message
    dicts (``[{"role": "user", "content": "..."}, ...]``). The Myelin
    substrate stores natural-language facts, so we concatenate the
    messages into a single content blob — same convention used by
    Mem0's own LLM-extraction path.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                raise TypeError(
                    f"each message must be a dict; got {type(m).__name__}"
                )
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", "")).strip()
            if role and content:
                parts.append(f"{role}: {content}")
            elif content:
                parts.append(content)
        return "\n".join(parts)
    raise TypeError(
        "messages must be a str or list[dict]; got "
        f"{type(messages).__name__}"
    )


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _memory_to_mem0_dict(mem: Memory) -> dict[str, Any]:
    """Map an internal :class:`Memory` to Mem0's dict shape.

    Keys mirror ``mem0.MemoryClient.get(...)``'s documented response:
    ``id``, ``memory``, ``user_id``, ``metadata``, ``created_at``,
    ``updated_at``, ``hash``. ``updated_at`` defaults to ``t_valid``
    because the Myelin substrate is event-anchored append-only — an
    "update" appends a new revision rather than mutating the row.
    """
    return {
        "id": mem.id,
        "memory": mem.content,
        "user_id": mem.user_id,
        "metadata": dict(mem.metadata or {}),
        "created_at": _iso(mem.t_valid),
        "updated_at": _iso(mem.t_valid),
    }


def _result_to_mem0_dict(result: SearchResult) -> dict[str, Any]:
    out = _memory_to_mem0_dict(result.memory)
    out["score"] = result.score
    return out


def _consolidation_to_dict(report: ConsolidationReport) -> dict[str, Any]:
    return {
        "hit_rate": report.hit_rate,
        "gaps_surfaced": report.gaps_surfaced,
        "new_facts_extracted": report.new_facts_extracted,
        "dream_queries_generated": report.dream_queries_generated,
        "llm_calls": report.llm_calls,
        "facts_sampled": report.facts_sampled,
    }


def _apoptosis_to_dict(report: ApoptosisReport) -> dict[str, Any]:
    return {
        "quarantined_count": report.quarantined_count,
        "tombstoned_count": report.tombstoned_count,
        "retained_count": report.retained_count,
        "scanned": report.scanned,
        "caspase_window_days": report.caspase_window_days,
    }


def _scope_filter(
    mem: Memory,
    *,
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
) -> bool:
    """Mem0-style scope filtering across (user_id, agent_id, run_id).

    ``user_id`` is the primary key in the Myelin substrate; ``agent_id``
    and ``run_id`` are stored in metadata so we can filter on them
    without changing the schema. A ``None`` filter is a wildcard.
    """
    if user_id is not None and mem.user_id != user_id:
        return False
    if agent_id is not None and mem.metadata.get("agent_id") != agent_id:
        return False
    if run_id is not None and mem.metadata.get("run_id") != run_id:
        return False
    return True


class MemoryClient:
    """Mem0-compatible drop-in client backed by Myelin's substrate.

    Construct once per process and share across requests::

        from myelin.adapters.mem0 import MemoryClient

        client = MemoryClient()
        client.add(
            [{"role": "user", "content": "I love tea."}],
            user_id="alice",
        )
        hits = client.search("what do I drink?", user_id="alice")
        # hits is a list[dict]; hits[0]["memory"] == "user: I love tea."

    The constructor accepts Mem0's ``config`` dict for forward
    compatibility, but the only keys it actively consumes are
    ``db_path`` (Myelin sqlite path) and ``user_id`` (default scope).
    Unknown keys are accepted and ignored — Mem0 callers can pass their
    full LLM/vector-store config without raising.

    See :meth:`consolidate` and :meth:`prune` for the Myelin extensions
    that aren't part of the upstream Mem0 surface.
    """

    def __init__(
        self,
        *,
        config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        cfg: dict[str, Any] = dict(config or {})
        # Allow direct kwargs to override config dict — mirrors mem0's
        # tolerant constructor where users sometimes pass through.
        cfg.update(kwargs)
        db_path = cfg.get("db_path", ":memory:")
        user_id = cfg.get("user_id")
        llm_client = cfg.get("llm_client")
        backend = cfg.get("backend", "sqlite")
        self._client = _MyelinClient(
            backend=backend,
            db_path=db_path,
            user_id=user_id,
            llm_client=llm_client,
        )

    # ---- Mem0-surface methods -----------------------------------

    def add(
        self,
        messages: list[dict] | str,
        *,
        user_id: str,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a memory. Accepts either a string or Mem0's messages-list.

        Returns a Mem0-shaped dict for the stored memory.
        """
        content = _coerce_messages(messages)
        meta: dict[str, Any] = dict(metadata or {})
        if agent_id is not None:
            meta["agent_id"] = agent_id
        if run_id is not None:
            meta["run_id"] = run_id
        mem = self._client.add(
            content,
            user_id=user_id,
            metadata=meta,
        )
        return _memory_to_mem0_dict(mem)

    def search(
        self,
        query: str,
        *,
        user_id: str,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search the substrate. Returns a Mem0-shaped list of result dicts.

        ``limit`` is Mem0's name for ``top_k``. ``agent_id`` and ``run_id``
        further filter the result set after retrieval.
        """
        # Over-fetch so post-filter on agent_id/run_id still returns
        # ``limit`` rows in the steady-state case.
        over = max(limit * 4, limit) if (agent_id or run_id) else limit
        raw = self._client.search(query, user_id=user_id, top_k=over)
        filtered = [
            r for r in raw
            if _scope_filter(
                r.memory,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
        ]
        return [_result_to_mem0_dict(r) for r in filtered[:limit]]

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """Fetch a memory by id. Returns None for unknown / tombstoned ids."""
        mem = self._client.get(memory_id)
        if mem is None:
            return None
        return _memory_to_mem0_dict(mem)

    def get_all(
        self,
        *,
        user_id: str,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List memories for ``user_id``, optionally narrowed by agent/run.

        Tombstoned memories are excluded. Returns up to ``limit`` rows
        ordered by ``t_valid`` ascending (same order as the underlying
        backend's ``list_memories``).
        """
        backend = self._client._backend
        rows = backend.list_memories(user_id=user_id)
        filtered = [
            m for m in rows
            if _scope_filter(
                m,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
            )
        ]
        return [_memory_to_mem0_dict(m) for m in filtered[:limit]]

    def update(self, memory_id: str, data: str | dict) -> dict[str, Any]:
        """Mutate an existing memory in place.

        ``data`` may be a bare string (interpreted as new content) or a
        dict with ``memory`` / ``content`` and/or ``metadata`` keys.
        Mirrors Mem0's tolerant ``update`` signature.
        """
        if isinstance(data, str):
            new_content: str | None = data
            new_meta: dict | None = None
        elif isinstance(data, dict):
            new_content = data.get("memory") or data.get("content")
            new_meta = data.get("metadata")
        else:
            raise TypeError(
                "data must be a str or dict; got " f"{type(data).__name__}"
            )
        mem = self._client.update(
            memory_id,
            content=new_content,
            metadata=new_meta,
        )
        return _memory_to_mem0_dict(mem)

    def delete(self, memory_id: str) -> dict[str, Any]:
        """Delete a memory by id. Returns a Mem0-shaped ack dict."""
        deleted = self._client.delete(memory_id)
        return {
            "id": memory_id,
            "deleted": deleted,
            "message": (
                "Memory deleted successfully."
                if deleted
                else "Memory not found."
            ),
        }

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete every memory matching the given scope filters.

        At least one of ``user_id``, ``agent_id``, or ``run_id`` should
        be provided to scope the operation. Returns a Mem0-shaped ack
        dict with the count of deletions.
        """
        backend = self._client._backend
        candidates: list[Memory] = []
        if user_id is not None:
            candidates = backend.list_memories(user_id=user_id)
        else:
            # No user_id given — Mem0 allows agent_id-only / run_id-only
            # deletion. Walk all users by inspecting the metadata; the
            # SqliteBackend can list a single user at a time, so we
            # discover users by scanning the memories table once.
            for uid in _all_user_ids(backend):
                candidates.extend(backend.list_memories(user_id=uid))
        deleted = 0
        for mem in candidates:
            if not _scope_filter(
                mem, user_id=user_id, agent_id=agent_id, run_id=run_id
            ):
                continue
            if backend.delete(mem.id):
                deleted += 1
        return {
            "deleted_count": deleted,
            "message": f"Deleted {deleted} memories.",
        }

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """Return the change-history of a memory.

        Wired to the Myelin substrate via
        :meth:`myelin.MemoryClient.list_revisions` — returns every
        event-anchored revision of the same ``(subject, predicate)``
        pair the source memory was written with, ordered oldest-first.

        Each entry is a Mem0-shaped dict with an extra ``event`` key
        carrying the event-anchor's ``event_id`` (or ``None`` if the
        revision wasn't event-anchored).
        """
        revisions = self._client.list_revisions(memory_id)
        out: list[dict[str, Any]] = []
        for mem in revisions:
            entry = _memory_to_mem0_dict(mem)
            entry["event"] = (
                {
                    "event_id": mem.event.event_id,
                    "type": mem.event.type,
                    "anchor_datetime": _iso(mem.event.anchor_datetime),
                }
                if mem.event is not None
                else None
            )
            out.append(entry)
        return out

    def reset(self) -> dict[str, Any]:
        """Empty the substrate. Returns a Mem0-shaped ack dict.

        Removes every memory, tombstone, and event row across all users.
        Intended for test fixtures and one-off scrub operations.
        """
        backend = self._client._backend
        clear = getattr(backend, "_clear_all", None)
        if callable(clear):
            clear()
            return {
                "reset": True,
                "message": "Substrate reset successfully.",
            }
        # Fallback for backends that don't expose _clear_all: walk every
        # user and hard-delete. Keeps the adapter portable across the
        # Backend Protocol's bare surface.
        for uid in _all_user_ids(backend):
            for mem in backend.list_memories(
                user_id=uid, include_tombstoned=True
            ):
                backend.delete(mem.id)
        return {
            "reset": True,
            "message": "Substrate reset (per-user fallback path).",
        }

    # ---- Myelin extensions (additive; not part of Mem0's surface) -

    def consolidate(self, *, user_id: str) -> dict[str, Any]:
        """Trigger the F-ML-2 dream-synthesis self-test (Myelin extension).

        Returns the consolidation report as a dict. Drop-in Mem0 callers
        don't have to invoke this; it's available for callers who want
        the underlying substrate's full mechanics.
        """
        report = self._client.consolidate(user_id=user_id)
        return _consolidation_to_dict(report)

    def prune(self, *, user_id: str) -> dict[str, Any]:
        """Trigger the P3 apoptosis sweep (Myelin extension).

        Returns the apoptosis report as a dict. Drop-in Mem0 callers
        don't have to invoke this; it's available for callers who want
        audit-preserving "forget on purpose" semantics.
        """
        report = self._client.prune(user_id=user_id)
        return _apoptosis_to_dict(report)

    # ---- lifecycle ----------------------------------------------

    def close(self) -> None:
        """Close the underlying backend. Optional but recommended in tests."""
        self._client.close()


def _all_user_ids(backend: Any) -> list[str]:
    """Best-effort enumeration of user_ids in a backend.

    Used by ``delete_all`` / ``reset`` when no ``user_id`` filter is
    supplied. Pokes at the SqliteBackend's connection directly because
    the public Backend Protocol does not (yet) include a list-users
    method — keeping the protocol minimal is a Phase-A constraint.
    """
    conn = getattr(backend, "_conn", None)
    lock = getattr(backend, "_lock", None)
    if conn is None or lock is None:
        return []
    with lock:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT DISTINCT user_id FROM memories"
        ).fetchall()
    return [r[0] for r in rows]
