"""MemoryClient — the public 7-method API surface.

The whole SDK collapses to this one class. It owns:
- A Backend (defaults to SqliteBackend) for persistence + retrieval.
- An optional `llm_client` for `consolidate()` dream-synthesis.
- A default `user_id` scope so single-user agent harnesses can omit it
  on every call.

What the client deliberately does NOT own:
- An answerer. The whole point of the SDK is that the caller brings
  their own LLM for downstream reasoning. `consolidate()` uses the
  caller's `llm_client` for the self-test step.
- Embedding infrastructure. The default lexical retrieval works
  reasonably well; embedding-based retrieval is an `[embeddings]`
  extra. The shape of the eventual hook is `MemoryClient(embedder=...)`
  — not implemented in v1.
- Tools / agentic loop. This is a memory layer, not an agent harness.
  Wire myelin into Mem0 / Letta / OpenClaw / your own harness.

Method surface (the 7):
- `add` / `search` / `get` / `update` / `delete`  — CRUD-ish
- `consolidate`  — F-ML-2 dream-synthesis
- `prune`        — P3 apoptosis (quarantine → tombstone)
"""
from __future__ import annotations

import hashlib
import logging
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from myelin._internal.constants import env_is_strict_one
from myelin._internal.dream import run_dream_cycle
from myelin._internal.health import (
    CASPASE_THRESHOLD,
    CASPASE_WINDOW_DAYS,
    QUARANTINE_THRESHOLD,
    compute_contradiction_counts,
    memory_health,
)
from myelin._internal.salience import (
    HeuristicSalienceGate,
    score_turn_heuristic,
)
from myelin._internal.session import SessionRegistry, _new_turn_id
from myelin.backend import Backend
from myelin.backends.sqlite import SqliteBackend
from myelin.types import (
    ApoptosisReport,
    ConsolidationReport,
    Event,
    Memory,
    ObservedTurn,
    SearchResult,
    SearchTelemetry,
    SessionHandle,
)

logger = logging.getLogger(__name__)

# Module-level set tracking which warn-once messages have fired this
# process. Avoids log spam for callers iterating retrieval results
# against a disabled feature.
_warned_once: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Emit ``message`` at WARNING level the first time ``key`` is
    seen this process; silent thereafter. Used for feature-disabled
    fallback paths so callers iterating cite/observe calls don't get
    log spam."""
    if key in _warned_once:
        return
    _warned_once.add(key)
    logger.warning(message)


_DEFAULT_USER = "default"

# Salience-mode sentinel used by the dropped-turn path so the caller
# can distinguish "ingest failed" from "gate dropped the turn." Memory
# objects with id="" never come from the backend.
_DROPPED_BY_SALIENCE_ID = ""


class MemoryClient:
    """Drop-in memory layer for AI agent harnesses.

    Construct once per process, share across requests. The client is
    safe for use from multiple threads (the underlying SqliteBackend
    serializes access via an internal lock).

    The default scope is `user_id="default"` so single-user setups
    don't have to pass `user_id=` on every call. Multi-user setups
    should construct one MemoryClient per user, or pass `user_id=` on
    every call.

    Example:
        >>> from myelin import MemoryClient
        >>> client = MemoryClient()
        >>> mem = client.add("Scott prefers tea.")
        >>> hits = client.search("What does Scott like?")
        >>> hits[0].memory.content
        'Scott prefers tea.'
    """

    def __init__(
        self,
        *,
        backend: str | Backend = "sqlite",
        db_path: str = ":memory:",
        user_id: str | None = None,
        llm_client: Any | None = None,
        salience_mode: str = "off",
        salience_tau: float = 0.40,
        salience_require_reward: bool = False,
        salience_embedder: Any | None = None,
        salience_neighbor_window: int = 32,
        dmem_gate: Any | None = None,
        cite_learning: bool | None = None,
        working_memory: bool | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Construct a MemoryClient.

        ``salience_mode`` (added 2026-05-20 for Phase C):

        * ``"off"``       — no gating; every ``add()`` stores the turn.
                            This is the v0.1.0 default behavior, unchanged.
        * ``"heuristic"`` — wire the carved 4-signal gate at ingest.
                            Tag-rate ~13.5% on PersonaMem-like input under
                            (tau=0.40, require_reward=False); included only
                            as a bench reference point.
        * ``"dmem"``      — D-MEM-style two-axis gate (semantic surprise ×
                            LLM-judged utility). Phase C of
                            `docs/plans/salience-gate-revisit-2026-05-20.md`.
                            Requires ``dmem_gate=DMemGate(...)`` instance
                            (or the client lazy-builds one from the env's
                            OPENAI_API_KEY).

        Turns that the gate drops still return a synthetic ``Memory`` with
        ``id=""`` so the caller can detect the drop without raising.

        ``clock`` supplies "now" for the client: the default ``t_valid``
        of ``add()``, the default ``now`` of ``prune()`` and the
        timestamp of ``cite()`` events. Pass one to replay a recorded
        history at its own dates (benchmarks, backfills, deterministic
        tests). Defaults to the current UTC time.
        """
        self._clock = clock
        if isinstance(backend, str):
            if backend == "sqlite":
                self._backend: Backend = SqliteBackend(db_path=db_path)
            else:
                raise ValueError(
                    f"unknown backend {backend!r}; supported: 'sqlite' "
                    "(pass a Backend instance for custom impls)"
                )
        else:
            self._backend = backend
        self._user_id = user_id or _DEFAULT_USER
        self._llm_client = llm_client

        # --- salience wiring -------------------------------------
        if salience_mode not in {"off", "heuristic", "dmem"}:
            raise ValueError(
                f"unknown salience_mode {salience_mode!r}; "
                "supported: 'off', 'heuristic', 'dmem'"
            )
        self._salience_mode = salience_mode
        self._heuristic_gate = (
            HeuristicSalienceGate(
                tau=salience_tau, require_reward=salience_require_reward
            )
            if salience_mode == "heuristic"
            else None
        )
        # B2 (ADR-026): the heuristic gate's novelty axis is wired to a real
        # embedder + per-user rolling neighbour window, so novelty varies as
        # the conversation moves instead of pinning at the 0.5 cold-start
        # constant. An explicit ``salience_embedder`` (anything with
        # ``.encode(text) -> Sequence[float]``) wins; otherwise the default
        # sentence-transformers embedder is lazy-built on first heuristic add
        # (so the dependency is only paid when heuristic mode is selected,
        # mirroring the D-MEM path). On embedder-load failure novelty falls
        # back to cold-start neutral rather than crashing ``add()``.
        self._salience_embedder_arg = salience_embedder
        self._salience_embedder_resolved: Any | None = None
        self._salience_embedder_failed = False
        self._salience_neighbor_window = max(1, int(salience_neighbor_window))
        self._heuristic_windows: dict[str, deque] = {}
        self._dmem_gate = None
        if salience_mode == "dmem":
            if dmem_gate is None:
                from myelin._internal.dmem_gate import DMemGate

                self._dmem_gate = DMemGate()
            else:
                self._dmem_gate = dmem_gate
        # Per-instance gate telemetry — bench harness reads these.
        self._n_add_calls = 0
        self._n_add_dropped = 0

        # --- v1.2 feature flags --------------------------------------
        # When the constructor kwarg is None, fall back to strict-"1"
        # env. Explicit True/False from the caller always wins. Per
        # `feedback_strict_1_env_vars` user memo: loose-truthy parses
        # have repeatedly corrupted locked baselines.
        if cite_learning is None:
            self._cite_learning_enabled = env_is_strict_one("MYELIN_CITE_LEARNING")
        else:
            self._cite_learning_enabled = bool(cite_learning)
        if working_memory is None:
            self._working_memory_enabled = env_is_strict_one("MYELIN_WORKING_MEMORY")
        else:
            self._working_memory_enabled = bool(working_memory)

        # v1.2 cite telemetry counters — bench harness reads these.
        self._n_cite_calls = 0
        self._n_cite_correct = 0
        self._n_cite_incorrect = 0
        self._n_abstain_events = 0

        # v1.2 working-memory plumbing. SessionRegistry is a process-
        # singleton; multiple MemoryClient instances in one process
        # share the same registry, which matches the multi-user
        # single-process semantics.
        self._session_registry = SessionRegistry.get_instance()

        # v1.2 search-telemetry counters — used by AC-B4 and the
        # vacuous-PASS pre-flight to compute working_set_hit_rate.
        # _n_hits_* counters are per-hit (every novel hit at that tier
        # increments). _n_calls_with_working_hit is per-call — at most
        # one increment per search() call regardless of how many
        # working-tier hits the call produced. The rate is the latter
        # divided by total session-scoped calls so the metric stays in
        # [0, 1].
        # Where the next consolidate() starts sampling, per user.
        self._consolidate_cursor: dict[str, int] = {}

        self._n_search_calls = 0
        self._n_search_with_session = 0
        self._n_hits_working = 0
        self._n_hits_session = 0
        self._n_hits_long_term = 0
        self._n_calls_with_working_hit = 0

    def _now(self) -> datetime:
        """Current time from the injected clock, as a tz-aware datetime."""
        now = self._clock() if self._clock is not None else datetime.now(UTC)
        return now if now.tzinfo is not None else now.replace(tzinfo=UTC)

    # ---- salience: heuristic novelty wiring (B2, ADR-026) -------

    def _heuristic_embedder(self) -> Any | None:
        """Resolve the heuristic-novelty embedder, lazy-building the default
        sentence-transformers adapter on first use. Returns ``None`` (→ cold-
        start neutral novelty) if no embedder is available or a build fails."""
        if self._salience_embedder_arg is not None:
            return self._salience_embedder_arg
        if self._salience_embedder_resolved is not None:
            return self._salience_embedder_resolved
        if self._salience_embedder_failed:
            return None
        try:
            from myelin._internal.dmem_gate import _default_embedder

            self._salience_embedder_resolved = _default_embedder()
            return self._salience_embedder_resolved
        except Exception as exc:  # noqa: BLE001 — degrade, never crash add()
            self._salience_embedder_failed = True
            logger.warning(
                "heuristic salience embedder unavailable (%s: %s); novelty "
                "falls back to cold-start neutral",
                type(exc).__name__,
                exc,
            )
            return None

    def _heuristic_novelty_inputs(
        self, content: str, scope: str
    ) -> tuple[list[float] | None, list[list[float]]]:
        """Embed ``content`` and return ``(query_embedding, neighbours)`` from
        the user's rolling window, then append the embedding to the window so
        the turn is scored against its predecessors, not itself (mirrors the
        D-MEM neighbour-window discipline)."""
        embedder = self._heuristic_embedder()
        emb: list[float] | None = None
        if embedder is not None:
            try:
                emb = list(embedder.encode(content))
            except Exception as exc:  # noqa: BLE001 — degrade to cold-start
                logger.warning(
                    "heuristic novelty embed failed (%s); cold-start neutral",
                    type(exc).__name__,
                )
                emb = None
        window = self._heuristic_windows.setdefault(
            scope, deque(maxlen=self._salience_neighbor_window)
        )
        neighbors = list(window)
        if emb:
            window.append(emb)
        return emb, neighbors

    # ---- CRUD (5/7) ---------------------------------------------

    def add(
        self,
        content: str,
        *,
        user_id: str | None = None,
        metadata: dict | None = None,
        event: dict | Event | None = None,
        confidence: float = 0.5,
        importance: float = 0.5,
        outcome: Mapping[str, object] | None = None,
        t_valid: datetime | None = None,
    ) -> Memory:
        """Add a fact to the substrate.

        `event` can be either an `Event` instance or a dict with the
        same keys: `event_id`, `type`, `anchor_datetime`, `location`,
        `participant_ids`, `evidence_turn_ids`. Only `event_id` is
        required when passing a dict.

        Event-anchored facts (P1 H044 v2): pass `event=...` to anchor
        this fact to a provenance node. Subsequent `search(..., as_of=t)`
        calls reconstruct historical state correctly across superseding
        writes. Without `event=`, the fact is timestamped at write time
        only — `as_of=t` still works but you lose the provenance chain.

        `outcome` (B2, ADR-026): an optional outcome mapping
        (`task_succeeded` / `user_thumbs_up` / `user_thumbs_down` /
        `error`) threaded into the heuristic salience gate's reward axis.
        Only consulted in `salience_mode="heuristic"`; ignored otherwise.
        When omitted the reward signal is 0.0 (the PersonaMem default).

        `t_valid` is when the fact became true. It defaults to the
        client's clock (the current time); pass it when ingesting
        history so `search(as_of=...)` and age-based health see the
        fact at its real date. Naive values are read as UTC.
        """
        ev: Event | None
        if event is None:
            ev = None
        elif isinstance(event, Event):
            ev = event
        elif isinstance(event, dict):
            ev = _event_from_dict(event)
        else:
            raise TypeError(
                f"event must be Event, dict, or None; got {type(event).__name__}"
            )

        # --- salience gate (Phase C, 2026-05-20) -------------------
        # Gate runs BEFORE backend persistence so dropped turns never
        # consume a row. Drop returns an empty Memory sentinel rather
        # than raising — callers iterating over turn streams shouldn't
        # have to wrap every add() in a try/except for an expected
        # outcome of the design.
        scope = user_id or self._user_id
        self._n_add_calls += 1
        gate_meta: dict[str, Any] = {}
        if self._salience_mode == "heuristic":
            q_emb, neighbors = self._heuristic_novelty_inputs(content, scope)
            decision = score_turn_heuristic(
                text=content,
                outcome=outcome,
                query_embedding=q_emb,
                neighbor_embeddings=neighbors,
                gate=self._heuristic_gate,
            )
            gate_meta = {
                "salience_mode": "heuristic",
                "salience_score": decision.score,
                "salience_tag": decision.tag,
                "salience_reason": decision.reason,
                "salience_novelty": decision.bundle.novelty if decision.bundle else None,
                "salience_reward": decision.bundle.reward if decision.bundle else None,
            }
            if not decision.tag:
                self._n_add_dropped += 1
                return _make_dropped_memory(scope, content, gate_meta)
        elif self._salience_mode == "dmem":
            assert self._dmem_gate is not None
            decision = self._dmem_gate.evaluate(content=content, user_id=scope)
            gate_meta = {
                "salience_mode": "dmem",
                "salience_utility_class": decision.utility_class,
                "salience_utility_score": decision.utility_score,
                "salience_surprise_z": decision.surprise_zscore,
                "salience_rpe": decision.rpe,
                "salience_tag": decision.tag,
                "salience_cached": decision.cached,
                "salience_reason": decision.reason,
            }
            if not decision.tag:
                self._n_add_dropped += 1
                return _make_dropped_memory(scope, content, gate_meta)

        # Merge gate telemetry into metadata so downstream reads can
        # surface the decision when needed (optional; default off).
        merged_metadata = dict(metadata or {})
        if gate_meta:
            merged_metadata["_salience"] = gate_meta

        return self._backend.add_memory(
            content=content,
            user_id=scope,
            metadata=merged_metadata,
            event=ev,
            t_valid=t_valid if t_valid is not None else self._now(),
            confidence=confidence,
            importance=importance,
        )

    @property
    def salience_telemetry(self) -> dict[str, Any]:
        """Per-instance counters for the salience gate.

        Bench harnesses call this after a context's worth of ingest
        to summarize tag-rate, dropped-count, and (for D-MEM) cache
        hit rate + LLM-judge cost.
        """
        out: dict[str, Any] = {
            "salience_mode": self._salience_mode,
            "n_add_calls": self._n_add_calls,
            "n_add_dropped": self._n_add_dropped,
            "n_add_kept": self._n_add_calls - self._n_add_dropped,
            "tag_rate": (
                (self._n_add_calls - self._n_add_dropped) / self._n_add_calls
                if self._n_add_calls > 0
                else 0.0
            ),
        }
        if self._dmem_gate is not None:
            out["dmem"] = self._dmem_gate.stats
        return out

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        top_k: int = 10,
        as_of: datetime | None = None,
        session_id: str | None = None,
        modes: tuple[str, ...] = ("long_term",),
    ) -> list[SearchResult]:
        """Semantic-ish lexical search over the substrate.

        ``top_k`` caps the result list. ``as_of`` (bi-temporal) returns
        only facts valid at that timestamp — i.e. ``t_valid <= as_of``
        AND (``t_invalid is None OR t_invalid > as_of``). Tombstoned
        facts are never returned.

        v1.2 ``modes`` extension (FR-B5): search in tiered order. Valid
        modes are "working", "session", "long_term". Default
        ``("long_term",)`` preserves v1 behavior exactly. When
        ``session_id`` is None, "working" and "session" modes are
        silently dropped with a warn-once log line (EC-B5).

        Hits from a higher tier auto-attend (FR-B3) into the session's
        working set when ``session_id`` is provided, so subsequent
        in-session retrieval hits the hot path.
        """
        # ---- 1. Validate modes ----
        valid_modes = {"working", "session", "long_term"}
        for m in modes:
            if m not in valid_modes:
                raise ValueError(
                    f"unknown search mode {m!r}; valid: {sorted(valid_modes)}"
                )

        # ---- 2. Telemetry ----
        self._n_search_calls += 1
        if session_id is not None:
            self._n_search_with_session += 1

        # ---- 3. EC-B5 silent downgrade ----
        if session_id is None:
            unsupported = [m for m in modes if m in ("working", "session")]
            if unsupported:
                _warn_once(
                    "search_no_session_id",
                    "search(modes=" + repr(unsupported) + ") requires "
                    "session_id; falling back to long_term only "
                    "(this warning fires once).",
                )
            effective_modes = tuple(m for m in modes if m == "long_term") or ("long_term",)
        else:
            effective_modes = modes

        scope = user_id or self._user_id
        # Every tier compares against tz-aware timestamps, so a naive
        # ``as_of`` is read as UTC here rather than per tier.
        if as_of is not None and as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        # ---- 4. Iterate tiers in order ----
        aggregated: list[SearchResult] = []
        seen_ids: set[str] = set()

        call_had_working_hit = False
        for mode in effective_modes:
            if len(aggregated) >= top_k:
                break

            remaining = top_k - len(aggregated)
            if mode == "working":
                tier_hits = self._search_working_set(
                    query=query, session_id=session_id,
                    top_k=remaining, as_of=as_of, user_id=scope,
                )
                hit_counter = "_n_hits_working"
            elif mode == "session":
                tier_hits = self._search_session_scope(
                    query=query, session_id=session_id,
                    top_k=remaining, as_of=as_of, user_id=scope,
                )
                hit_counter = "_n_hits_session"
            else:  # long_term
                tier_hits = self._backend.search(
                    query=query, user_id=scope,
                    top_k=remaining, as_of=as_of,
                )
                hit_counter = "_n_hits_long_term"

            new_hits_this_tier = 0
            for hit in tier_hits:
                if hit.memory.id in seen_ids:
                    continue
                seen_ids.add(hit.memory.id)
                aggregated.append(hit)
                new_hits_this_tier += 1
                if len(aggregated) >= top_k:
                    break

            if new_hits_this_tier > 0:
                # Per-hit counter (monotone over client lifetime).
                setattr(
                    self, hit_counter,
                    getattr(self, hit_counter) + new_hits_this_tier,
                )
                if mode == "working":
                    call_had_working_hit = True

        # Per-call counter for AC-B4 rate computation. At most one
        # increment per search() call regardless of working-tier hit
        # count, so the rate stays in [0, 1].
        if call_had_working_hit and session_id is not None:
            self._n_calls_with_working_hit += 1

        # ---- 5. Auto-attend hits into the session's working set ----
        if session_id is not None and self._working_memory_enabled:
            session_state = self._session_registry.get_or_create(
                session_id=session_id, user_id=scope
            )
            for hit in aggregated:
                session_state.attend(memory_id=hit.memory.id)

        return aggregated

    def _search_working_set(
        self,
        *,
        query: str,
        session_id: str | None,
        top_k: int,
        as_of: datetime | None,
        user_id: str,
    ) -> list[SearchResult]:
        """Search the session's working set (RAM-resident memory ids).

        Resolves each working-set memory id via Backend.get() and
        scores with the same lexical algorithm as the long_term tier.
        Returns ordered SearchResults (highest score first).
        """
        if session_id is None:
            return []
        state = self._session_registry.lookup(session_id)
        if state is None:
            return []

        with state.lock:
            memory_ids = list(state.working_set.keys())
        if not memory_ids:
            return []

        # Use Backend.search's tokenizer for consistency. We score
        # outside the backend's hot path because we already know the
        # candidate ids and want to skip the full-table scan.
        from myelin.backends.sqlite import _tokens as _backend_tokens

        q_tokens = _backend_tokens(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, Memory]] = []
        for memory in self._get_many(memory_ids):
            if memory.user_id != user_id:
                continue
            if as_of is not None:
                if memory.t_valid > as_of:
                    continue
                if memory.t_invalid is not None and memory.t_invalid <= as_of:
                    continue
            # Reuse the backend's _score for parity with long_term ranking.
            score = self._backend._score(memory, q_tokens)  # noqa: SLF001
            if score > 0:
                scored.append((score, memory))

        scored.sort(key=lambda kv: (-kv[0], kv[1].id))
        return [SearchResult(memory=m, score=s) for s, m in scored[:top_k]]

    def _get_many(self, memory_ids: list[str]) -> list[Memory]:
        """Resolve ids to live memories, skipping unknown/tombstoned ones.

        Uses the backend's ``get_batch`` when it has one; backends that
        only implement ``get`` are looked up id by id.
        """
        get_batch = getattr(self._backend, "get_batch", None)
        if callable(get_batch):
            return list(get_batch(memory_ids))
        found = (self._backend.get(mid) for mid in memory_ids)
        return [m for m in found if m is not None]

    def _search_session_scope(
        self,
        *,
        query: str,
        session_id: str | None,
        top_k: int,
        as_of: datetime | None,
        user_id: str,
    ) -> list[SearchResult]:
        """Search memories tagged with this session in their metadata.

        commit() records ``_session_id`` in metadata; this filter
        surfaces memories committed during this session but not yet
        attended into the working set.
        """
        if session_id is None:
            return []
        # Over-fetch then filter — simpler than adding an index column
        # in v1.2. Profiler may push us to a dedicated column in v1.3.
        raw = self._backend.search(
            query=query, user_id=user_id,
            top_k=top_k * 5, as_of=as_of,
        )
        session_hits = [
            r for r in raw
            if r.memory.metadata.get("_session_id") == session_id
        ]
        return session_hits[:top_k]

    @property
    def search_telemetry(self) -> SearchTelemetry:
        """Per-client cumulative search telemetry. Bench harness reads
        this after a run to compute ``working_set_hit_rate`` (AC-B4)."""
        return SearchTelemetry(
            n_search_calls=self._n_search_calls,
            n_search_calls_with_session_id=self._n_search_with_session,
            n_hits_from_working_tier=self._n_hits_working,
            n_hits_from_session_tier=self._n_hits_session,
            n_hits_from_long_term_tier=self._n_hits_long_term,
            n_calls_with_working_hit=self._n_calls_with_working_hit,
        )

    # ---- v1.2 working-memory layer (Innovation B) --------------

    def observe(
        self,
        content: str,
        *,
        session_id: str,
        metadata: dict | None = None,
        user_id: str | None = None,
    ) -> ObservedTurn:
        """Record a turn into the session ring buffer WITHOUT
        persisting it to the long-term substrate.

        ``user_id`` scopes a session this call creates (default: the
        client's ``user_id``). A session keeps the user it was created
        for, and ``commit()`` stores its turns under that user; passing
        a different ``user_id`` for an existing session raises
        ``ValueError`` rather than mixing two users' turns.

        ``commit(turn_ids, session_id=...)`` promotes observed turns
        into the long-term substrate. Until commit(), the turn is RAM-
        only and lost on process restart (EC-B4).

        Fallback when working memory is disabled: behaves as ``add()``
        with a warn-once log. The returned ObservedTurn's ``turn_id``
        equals the resulting Memory id — callers can migrate to
        observe() without rewriting downstream code.
        """
        if not self._working_memory_enabled:
            _warn_once(
                "working_memory_disabled",
                "MemoryClient.observe() called but working memory is "
                "disabled (MYELIN_WORKING_MEMORY != '1' AND "
                "constructor kwarg not set). Falling back to add() so "
                "callers can migrate gradually. This warning fires once.",
            )
            mem = self.add(content, metadata=metadata, user_id=user_id)
            return ObservedTurn(
                turn_id=mem.id,
                content=content,
                observed_at=mem.t_valid,
                monotonic=datetime.now(UTC).timestamp(),
                metadata=dict(metadata or {}),
            )

        if not session_id or not isinstance(session_id, str):
            raise ValueError("observe() requires a non-empty session_id")

        state = self._session_registry.get_or_create(
            session_id=session_id, user_id=user_id or self._user_id
        )
        if user_id is not None and state.user_id != user_id:
            raise ValueError(
                f"session {session_id!r} belongs to user "
                f"{state.user_id!r}; cannot observe as {user_id!r}"
            )
        turn_id = _new_turn_id()
        return state.append_observed(
            turn_id=turn_id, content=content, metadata=metadata
        )

    def commit(
        self,
        turn_ids: list[str],
        *,
        session_id: str,
    ) -> list[Memory]:
        """Promote observed turns to the long-term substrate.

        Applies the salience gate at promotion time (NFR-B3 cost win —
        only commit() pays for the LLM judge, not every observe()).
        Each promoted memory is auto-attended into the session working
        set.

        Memories are stored under the session's user. If persisting a
        turn fails, that turn and the ones after it go back into the
        session buffer and the error propagates, so the call can be
        retried; turns already stored stay stored.

        Raises:
            RuntimeError: working memory disabled.
            LookupError: session expired (EC-B6) OR turn_ids do not
                belong to this session (EC-B1) OR some turn_ids were
                already drained / FIFO-evicted.
        """
        if not self._working_memory_enabled:
            raise RuntimeError(
                "commit() called but working memory disabled "
                "(MYELIN_WORKING_MEMORY != '1' AND constructor kwarg "
                "not set)"
            )
        if not turn_ids:
            return []

        state = self._session_registry.lookup(session_id)
        if state is None:
            raise LookupError(
                f"commit() on expired or unknown session: {session_id!r}"
            )

        drained = state.drain_observed(turn_ids)

        committed: list[Memory] = []
        for index, turn in enumerate(drained):
            try:
                mem = self._commit_turn(turn, state.user_id, session_id)
            except BaseException:
                state.restore_observed(drained[index:])
                raise
            if mem is not None:
                committed.append(mem)
                state.attend(memory_id=mem.id)

        return committed

    def _commit_turn(
        self, turn: ObservedTurn, user_id: str, session_id: str
    ) -> Memory | None:
        """Gate and persist one observed turn. Returns ``None`` when the
        salience gate drops it."""
        merged_metadata = dict(turn.metadata or {})
        merged_metadata["_session_id"] = session_id

        # Salience gate at commit time (NFR-B3).
        if self._salience_mode == "heuristic":
            turn_outcome = (turn.metadata or {}).get("outcome")
            q_emb, neighbors = self._heuristic_novelty_inputs(
                turn.content, user_id
            )
            decision = score_turn_heuristic(
                text=turn.content,
                outcome=turn_outcome,
                query_embedding=q_emb,
                neighbor_embeddings=neighbors,
                gate=self._heuristic_gate,
            )
            if not decision.tag:
                # Dropped by salience; do NOT persist. Counted in
                # the same telemetry as add()-side drops.
                self._n_add_dropped += 1
                return None
            merged_metadata["_salience"] = {
                "salience_mode": "heuristic",
                "salience_score": decision.score,
                "salience_tag": decision.tag,
                "salience_reason": decision.reason,
                "salience_novelty": decision.bundle.novelty if decision.bundle else None,
                "salience_reward": decision.bundle.reward if decision.bundle else None,
            }
        elif self._salience_mode == "dmem":
            assert self._dmem_gate is not None
            decision = self._dmem_gate.evaluate(
                content=turn.content, user_id=user_id
            )
            if not decision.tag:
                self._n_add_dropped += 1
                return None
            merged_metadata["_salience"] = {
                "salience_mode": "dmem",
                "salience_utility_class": decision.utility_class,
                "salience_utility_score": decision.utility_score,
                "salience_surprise_z": decision.surprise_zscore,
                "salience_rpe": decision.rpe,
                "salience_tag": decision.tag,
                "salience_cached": decision.cached,
                "salience_reason": decision.reason,
            }

        return self._backend.add_memory(
            content=turn.content,
            user_id=user_id,
            metadata=merged_metadata,
            event=None,
            confidence=0.5,
            importance=0.5,
        )

    def attend(
        self,
        memory_id: str,
        *,
        session_id: str,
    ) -> None:
        """Add ``memory_id`` to ``session_id``'s working set.

        No-op + WARNING when working memory is disabled OR the memory
        is tombstoned/unknown (EC-B2). Does NOT resurrect a tombstoned
        memory."""
        if not self._working_memory_enabled:
            _warn_once(
                "attend_disabled",
                "MemoryClient.attend() called but working memory is "
                "disabled. No-op.",
            )
            return
        memory = self._backend.get(memory_id)
        if memory is None:
            logger.warning(
                "attend no-op: memory_id %r unknown or tombstoned",
                memory_id,
            )
            return
        state = self._session_registry.get_or_create(
            session_id=session_id, user_id=self._user_id
        )
        state.attend(memory_id=memory_id)

    def session(self, session_id: str) -> SessionHandle:
        """Return a read-only snapshot of ``session_id``'s state.

        Raises:
            RuntimeError: working memory disabled.
            LookupError: session does not exist (never created OR
                TTL-expired).
        """
        if not self._working_memory_enabled:
            raise RuntimeError(
                "session() called but working memory disabled"
            )
        state = self._session_registry.lookup(session_id)
        if state is None:
            raise LookupError(
                f"session() on expired or unknown session: {session_id!r}"
            )
        with state.lock:
            return SessionHandle(
                session_id=state.session_id,
                started_at=state.started_at,
                last_active_at=state.last_active_at,
                n_observed=len(state.observed_buffer),
                n_committed=state.n_committed,
                working_set_ids=tuple(state.working_set.keys()),
            )

    def get(self, memory_id: str) -> Memory | None:
        """Fetch a memory by id. Returns None for unknown / tombstoned ids."""
        return self._backend.get(memory_id)

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        """In-place edit of an existing memory.

        Does NOT append a new fact — overwrites the existing row. For
        event-anchored append-only semantics, call `add()` with a new
        Event instead.

        Raises KeyError if `memory_id` is unknown or tombstoned.
        """
        return self._backend.update(
            memory_id, content=content, metadata=metadata
        )

    def delete(self, memory_id: str) -> bool:
        """Hard-delete a memory. Returns True if deleted, False if absent.

        For audit-preserving "forget on purpose" semantics, use
        `prune()` which goes through the quarantine → tombstone state
        machine. `delete()` removes the row entirely.
        """
        return self._backend.delete(memory_id)

    # ---- Mechanisms (2/7) ---------------------------------------

    def consolidate(
        self,
        *,
        user_id: str | None = None,
        max_samples: int = 32,
        retriever: Callable[[str], Iterable[str]] | None = None,
    ) -> ConsolidationReport:
        """Trigger F-ML-2 dream-synthesis self-test.

        Samples post-sleep healthy facts, generates dream queries
        whose ground-truth answer is the fact, runs the caller's
        `llm_client` against each query, and grades hits by substring
        match. Returns a `ConsolidationReport` with `hit_rate`,
        `gaps_surfaced`, and `new_facts_extracted` (v1: always 0; v1.1
        will add the extractor hook).

        When `llm_client` was None at client construction, the LLM-
        dependent step is a no-op — the report is well-formed with
        `hit_rate=0.0`, `llm_calls=0`, but `facts_sampled` and
        `dream_queries_generated` are still populated so the caller
        can verify the sampling step ran.

        `max_samples` caps the number of facts self-tested this call
        (each fact generates 3 dream queries by default). Successive
        calls walk through the user's facts oldest-first and wrap
        around, so repeated consolidation covers the whole store rather
        than re-testing the same oldest facts.

        `retriever` grounds the self-test in the store: a callable that
        takes a dream query and returns context snippets to show the
        answerer, e.g. ``lambda q: [r.memory.content for r in
        client.search(q, top_k=5)]``. Without it the answerer sees only
        the query.
        """
        scope = user_id or self._user_id
        memories = self._backend.list_memories(
            user_id=scope, only_healthy=True
        )
        if memories and max_samples > 0:
            start = self._consolidate_cursor.get(scope, 0) % len(memories)
            memories = memories[start:] + memories[:start]
            self._consolidate_cursor[scope] = start + min(
                max_samples, len(memories)
            )
        return run_dream_cycle(
            memories=memories,
            llm_client=self._llm_client,
            max_samples=max_samples,
            retriever=retriever,
        )

    def prune(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
        caspase_window_days: int = CASPASE_WINDOW_DAYS,
    ) -> ApoptosisReport:
        """Trigger P3 apoptosis: quarantine → tombstone state machine.

        Two-phase per call:
        1. **Quarantine promotion.** Each active fact's health is
           computed (`memory_health(memory, contradiction_count=...)`).
           Facts with `health < QUARANTINE_THRESHOLD` (0.4) are
           transitioned to `quarantined` and stamped with
           `quarantine_entered_at`.
        2. **Caspase cascade.** Facts that have been quarantined for
           `caspase_window_days` (default 7) OR whose current health is
           below `CASPASE_THRESHOLD` (0.2) are transitioned to
           `tombstoned`. Tombstoned facts are invisible to `search()`
           but persist in the audit substrate.

        `now` defaults to UTC now; pass an explicit datetime for
        deterministic tests (e.g. `now = datetime.now(UTC) +
        timedelta(days=14)` to fast-forward past the caspase window).

        Returns `ApoptosisReport(quarantined_count, tombstoned_count,
        retained_count, scanned, caspase_window_days)`.
        """
        target_user = user_id or self._user_id
        if now is None:
            now = self._now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        memories = self._backend.list_memories(user_id=target_user)
        contradiction_counts = compute_contradiction_counts(memories)

        report = ApoptosisReport(
            scanned=len(memories),
            caspase_window_days=caspase_window_days,
        )
        caspase_cutoff = now - timedelta(days=caspase_window_days)

        # --- Phase 2 first: tombstone aged quarantined facts ----
        # Run caspase BEFORE quarantine promotion so a fact that's been
        # quarantined for >= window days is tombstoned this cycle, not
        # next. Order matches the cortex F_BI_5 scheduler ordering.
        for mem in memories:
            if mem.health_state != "quarantined":
                continue
            entered = mem.quarantine_entered_at
            if entered is None:
                continue
            if entered.tzinfo is None:
                entered = entered.replace(tzinfo=UTC)
            health = memory_health(
                mem,
                now=now,
                contradiction_count=contradiction_counts.get(mem.id, 0),
            )
            should_caspase = (
                entered <= caspase_cutoff or health < CASPASE_THRESHOLD
            )
            if should_caspase:
                self._backend.transition_state(
                    mem.id, new_state="tombstoned"
                )
                report.tombstoned_count += 1

        # --- Phase 1: quarantine sick active facts --------------
        for mem in memories:
            if mem.health_state != "healthy":
                continue
            health = memory_health(
                mem,
                now=now,
                contradiction_count=contradiction_counts.get(mem.id, 0),
            )
            if health < QUARANTINE_THRESHOLD:
                self._backend.transition_state(
                    mem.id, new_state="quarantined", quarantine_entered_at=now
                )
                report.quarantined_count += 1

        report.retained_count = (
            report.scanned
            - report.quarantined_count
            - report.tombstoned_count
        )
        return report

    # ---- v1.2 cite/abstain (Innovation A) ----------------------

    def cite(
        self,
        memory_id: str,
        *,
        answer_correct: bool,
        session_id: str | None = None,
        query: str | None = None,
    ) -> None:
        """Record that ``memory_id`` was rendered into the answerer's
        prompt for a query, and report whether the answer was correct.

        This is the write side of Innovation A — the read side is the
        cite-aware health terms in :mod:`myelin._internal.health`. The
        bench harness calls this once per retrieved+rendered memory per
        question; the caller's scoring rubric supplies ``answer_correct``.

        Args:
            memory_id: opaque id from ``add()`` / ``search()`` / ``get()``.
            answer_correct: True if the answer was scored correct; False
                if scored incorrect. Pass nothing (or None) to skip the
                call — useful when the scorer is incomplete.
            session_id: optional session this cite belongs to. When the
                working-memory feature is enabled AND ``session_id`` is
                passed, the cite also triggers a side-channel ``attend``
                so the memory is added to the session's working set.
                (The side-channel attend is wired in Step 11; until then
                only the cite event is recorded.)
            query: optional natural-language query that produced this
                retrieval. When ``MYELIN_CITE_HASH_QUERIES=1`` is set,
                the value is SHA-256-hashed (first 16 hex chars)
                before storage; otherwise truncated to 256 chars.

        No-op when ``MYELIN_CITE_LEARNING != "1"`` (warn-once),
        when ``memory_id`` is unknown or tombstoned (warn), or when
        ``answer_correct`` resolves to None (debug log).
        """
        if not self._cite_learning_enabled:
            _warn_once(
                "cite_disabled",
                "MemoryClient.cite() called but cite learning is "
                "disabled (MYELIN_CITE_LEARNING != '1' AND "
                "constructor kwarg not set). All cite calls this "
                "process will be no-ops. This warning fires once.",
            )
            return

        if answer_correct is None:  # type: ignore[unreachable]
            logger.debug("cite skipped: answer_correct=None")
            return

        # Verify the memory exists and isn't tombstoned. Backend.get()
        # already returns None for tombstoned/unknown; we treat both
        # paths as "no-op + warn" because the cite path is fire-and-
        # forget (callers iterate retrieval results).
        memory = self._backend.get(memory_id)
        if memory is None:
            logger.warning(
                "cite no-op: memory_id %r unknown or tombstoned",
                memory_id,
            )
            return

        safe_query = self._safe_query(query)
        self._n_cite_calls += 1
        if answer_correct:
            self._n_cite_correct += 1
        else:
            self._n_cite_incorrect += 1

        self._backend.record_cite(
            memory_id=memory_id,
            answer_correct=bool(answer_correct),
            session_id=session_id,
            query=safe_query,
            now=self._now(),
        )

        # Cite-session bridge — wires the working-set attend side
        # effect. Only fires when BOTH cite-learning AND working-memory
        # are enabled AND a session_id is provided. The actual
        # SessionRegistry call happens in Step 11; here we leave the
        # extension point.
        if session_id is not None and self._working_memory_enabled:
            self._maybe_attend_from_cite(memory_id, session_id)

    def abstain_event(
        self,
        query: str | None,
        *,
        session_id: str | None = None,
    ) -> None:
        """Record that a ``search()`` returned zero results (deliberate
        abstention). Surfaces gap-cases for dream-synthesis to
        re-prioritize self-test toward the missing fact-class.

        FR-A6. No-op when ``MYELIN_CITE_LEARNING != "1"``.
        """
        if not self._cite_learning_enabled:
            return
        safe_query = self._safe_query(query)
        self._n_abstain_events += 1
        self._backend.record_abstain(
            query=safe_query,
            session_id=session_id,
            now=datetime.now(UTC),
        )

    def _safe_query(self, query: str | None) -> str | None:
        """Hash or truncate the query text per the
        ``MYELIN_CITE_HASH_QUERIES`` env flag. Returns ``None`` for
        None input."""
        if query is None:
            return None
        if env_is_strict_one("MYELIN_CITE_HASH_QUERIES"):
            return hashlib.sha256(query[:32].encode("utf-8")).hexdigest()[:16]
        return query[:256]

    def _maybe_attend_from_cite(
        self, memory_id: str, session_id: str
    ) -> None:
        """Cite-session bridge (FR-B3 link from Innovation A).

        Citing a memory in an active session auto-attends it into the
        working set so subsequent in-session retrieval can hit the hot
        path. Only fires when working memory is enabled AND the
        session exists; otherwise silent no-op so cite() stays useful
        without the working-memory layer engaged.
        """
        state = self._session_registry.lookup(session_id)
        if state is None:
            return
        state.attend(memory_id=memory_id)

    @property
    def cite_telemetry(self) -> dict[str, Any]:
        """Per-instance cite counters surfaced for the bench harness."""
        return {
            "cite_learning_enabled": self._cite_learning_enabled,
            "n_cite_calls": self._n_cite_calls,
            "n_cite_correct": self._n_cite_correct,
            "n_cite_incorrect": self._n_cite_incorrect,
            "n_abstain_events": self._n_abstain_events,
        }

    # ---- diagnostic helpers (not part of the 7) ----------------

    def close(self) -> None:
        """Close the underlying backend connection. Optional — the
        process exit cleanup is fine for most callers."""
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    # ---- adapter-support helpers (additive; not part of the 7) -

    def list_revisions(self, memory_id: str) -> list[Memory]:
        """Return all event-anchored revisions of the same logical fact
        as ``memory_id``, ordered by ``t_valid`` ascending.

        Used by adapters (e.g. ``myelin.adapters.mem0.MemoryClient.history``)
        to expose Mem0-style change history. A "revision" is any other
        memory in the same ``user_id`` scope whose metadata shares the
        same ``(subject, predicate)`` pair as the source memory.

        If the source memory has no ``subject``/``predicate`` in its
        metadata — i.e. it wasn't written as a fact triple — the returned
        list contains only the source memory itself. Tombstoned memories
        are not included (consistent with ``get()`` semantics).

        Returns ``[]`` if ``memory_id`` is unknown or tombstoned. This
        method is additive to the Phase-A 7-method surface and may
        evolve in v1.1 once the Event-aware revision graph is wired
        in directly.
        """
        anchor = self._backend.get(memory_id)
        if anchor is None:
            return []
        subject = anchor.metadata.get("subject")
        predicate = anchor.metadata.get("predicate")
        if not subject or not predicate:
            return [anchor]
        siblings = self._backend.list_memories(user_id=anchor.user_id)
        revisions = [
            mem
            for mem in siblings
            if mem.metadata.get("subject") == subject
            and mem.metadata.get("predicate") == predicate
        ]
        revisions.sort(key=lambda m: m.t_valid)
        return revisions


def _make_dropped_memory(user_id: str, content: str, gate_meta: dict) -> Memory:
    """Return a synthetic Memory representing a salience-dropped turn.

    ``id=""`` (falsy) lets callers detect the drop with a simple
    truthiness check. Health state is ``"tombstoned"`` so any code path
    that filters by health still discards it correctly. The content is
    preserved verbatim only in metadata under ``_dropped_content_hash``
    (sha256 prefix) — the full text is NOT stored anywhere.
    """
    import hashlib
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    md = dict(gate_meta or {})
    md["_dropped_content_hash"] = digest
    return Memory(
        id=_DROPPED_BY_SALIENCE_ID,
        user_id=user_id,
        content="",
        t_valid=datetime.now(UTC),
        t_invalid=None,
        metadata=md,
        event=None,
        confidence=0.0,
        importance=0.0,
        health_state="tombstoned",
        quarantine_entered_at=None,
    )


def _event_from_dict(d: dict) -> Event:
    if "event_id" not in d:
        raise ValueError("event dict must include 'event_id'")
    anchor = d.get("anchor_datetime")
    if isinstance(anchor, str):
        anchor = datetime.fromisoformat(anchor)
    return Event(
        event_id=d["event_id"],
        type=d.get("type", ""),
        anchor_datetime=anchor,
        location=d.get("location", ""),
        participant_ids=tuple(d.get("participant_ids", ())),
        evidence_turn_ids=tuple(d.get("evidence_turn_ids", ())),
    )
