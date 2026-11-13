"""Public dataclasses exposed by the Myelin SDK.

These types form the stable surface of the public API. Internal storage
representations may differ; the client converts to/from these at the
boundary so callers never see vendored cortex internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "ApoptosisReport",
    "ConsolidationReport",
    "Event",
    "Memory",
    "ObservedTurn",
    "SearchResult",
    "SearchTelemetry",
    "SessionHandle",
]


@dataclass
class Event:
    """Event-anchored provenance node.

    Mirrors APEX-MEM §3.1 — `Event(event_id, type, anchor_datetime,
    location, participant_ids, evidence_turn_ids)` — pared down to the
    fields we actually need on the public surface. Stored alongside the
    Memory in the substrate; surfaced on SearchResult.event when present.

    Anchoring a fact to an Event is what makes `as_of=t` queries return
    historically-coherent state. Without an event, the fact's only time
    stamp is `t_valid` and the validity collapses to write-time.
    """

    event_id: str
    type: str = ""
    anchor_datetime: datetime | None = None
    location: str = ""
    participant_ids: tuple[str, ...] = ()
    evidence_turn_ids: tuple[str, ...] = ()


@dataclass
class Memory:
    """A single stored fact.

    `id` is opaque to the caller — never construct one yourself; obtain
    it from `add()` or `search()`. `content` is the natural-language
    fact as written. `metadata` is arbitrary JSON-serializable dict.
    `event` is populated when the fact was anchored to an Event at
    write time (see `MemoryClient.add(event=...)`).

    `t_valid` is when the fact entered the substrate (default: write
    time). `t_invalid` is None unless the fact was superseded by a
    later non-event-anchored write — for event-anchored facts the
    convention is append-only (per P1 H044 v2) and `t_invalid` stays
    None forever; readers reconstruct historical state via `as_of=t`
    over the t_valid axis.

    `health_state` is the apoptosis state machine: one of
    {"healthy", "quarantined", "tombstoned"}. Tombstoned memories are
    invisible to `search()` but persist for audit; surfaced through
    `prune()` reports only.

    v1.2 cite signal fields (FR-A2):
    - `cite_correct_count`: how many times the answerer used this
      memory and got the right answer.
    - `cite_incorrect_count`: how many times the answerer used this
      memory and got the wrong answer.
    - `last_cited_at`: most recent cite event timestamp. None when
      never cited (distinct from "cited at epoch 0").

    All three default such that the v1 health equation collapses
    unchanged when these are not populated.
    """

    id: str
    content: str
    user_id: str
    t_valid: datetime
    t_invalid: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    event: Event | None = None
    confidence: float = 0.5
    importance: float = 0.5
    health_state: str = "healthy"
    quarantine_entered_at: datetime | None = None
    # ---- v1.2 cite signal (FR-A2) ----
    cite_correct_count: int = 0
    cite_incorrect_count: int = 0
    last_cited_at: datetime | None = None


@dataclass
class SearchResult:
    """A scored search hit.

    `memory` is the matched record. `score` is a non-negative float
    where higher = better — relative within a single `search()` call
    only, not comparable across calls.

    `excerpt` is set when the result was shortened for context (see
    `MemoryClient.recall(max_result_tokens=...)`): the passages of the
    memory that bear on the query. `memory.content` is always complete.
    """

    memory: Memory
    score: float
    excerpt: str | None = None


@dataclass
class ConsolidationReport:
    """Result of `MemoryClient.consolidate()`.

    `hit_rate` is the fraction of generated dream queries whose
    answerer-output substring-matched the seed fact — the same metric
    F-ML-2 gates on in `scripts/vertex/_harness.py`. `gaps_surfaced` is
    the count of dream queries that missed (1 - hit_rate fraction).
    `new_facts_extracted` is the count of facts extracted from the
    answerer's responses and added back to the substrate.

    `llm_calls` is the count of times the caller's `llm_client` was
    invoked. When `llm_client` is None at client construction,
    `llm_calls=0` and `hit_rate=0.0` — the LLM-dependent step is a
    no-op but the report is still well-formed.
    """

    hit_rate: float = 0.0
    gaps_surfaced: int = 0
    new_facts_extracted: int = 0
    dream_queries_generated: int = 0
    llm_calls: int = 0
    facts_sampled: int = 0


@dataclass
class ApoptosisReport:
    """Result of `MemoryClient.prune()`.

    `quarantined_count`: how many active facts were promoted to
    quarantined during this call.

    `tombstoned_count`: how many quarantined facts (from prior calls)
    aged past the caspase window during this call and were
    tombstoned. Tombstoned facts are removed from the active
    retrieval view but persist in the audit substrate.

    `retained_count`: active facts that survived this pass with no
    state change.

    The state machine is `active → quarantined → tombstoned`. Once a
    fact is tombstoned, `search()` will never return it; `get(id)`
    returns None as if it were deleted. The fact persists as an
    audit-trail row queryable via internal forensic APIs only.
    """

    quarantined_count: int = 0
    tombstoned_count: int = 0
    retained_count: int = 0
    scanned: int = 0
    caspase_window_days: int = 7


# ---------------------------------------------------------------------
# v1.2 working-memory primitive (Innovation B)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedTurn:
    """A turn that has been ``observe()``'d but not yet ``commit()``'d.

    Lives only in RAM. Lost on process restart — explicit ``commit()``
    is required to promote into the long-term substrate. The pair
    (``turn_id``, ``content``) is the minimum a caller needs to later
    request promotion.
    """

    turn_id: str
    content: str
    observed_at: datetime
    monotonic: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionHandle:
    """Read-only view of a session's state.

    Returned by ``MemoryClient.session(session_id)``. Snapshot at the
    moment of the call — subsequent ``observe()`` / ``commit()`` /
    ``attend()`` calls do NOT mutate the handle. Re-fetch via
    ``client.session(session_id)`` for fresh state.
    """

    session_id: str
    started_at: datetime
    last_active_at: datetime
    n_observed: int
    n_committed: int
    working_set_ids: tuple[str, ...]   # ordered LRU -> MRU


@dataclass(frozen=True)
class SearchTelemetry:
    """Per-client cumulative search telemetry.

    Populated by ``MemoryClient.search()`` calls. The bench harness
    reads this after a run to populate the verdict-row
    ``working_set_hit_rate`` field. Counters are monotone within a
    process lifetime.

    ``n_hits_from_*_tier`` counters are per-hit (every novel hit at
    that tier increments). ``n_calls_with_working_hit`` is per-call
    (at most one increment per search() call regardless of how many
    working-tier hits were returned) — this is what
    ``working_set_hit_rate`` divides so the metric stays in [0, 1].
    """

    n_search_calls: int
    n_search_calls_with_session_id: int
    n_hits_from_working_tier: int
    n_hits_from_session_tier: int
    n_hits_from_long_term_tier: int
    n_calls_with_working_hit: int = 0

    @property
    def working_set_hit_rate(self) -> float:
        """Fraction of session-scoped search calls that returned at
        least one hit from the working tier. Bounded [0, 1]. Returns
        0.0 when no session-scoped calls have run."""
        denom = self.n_search_calls_with_session_id
        if denom == 0:
            return 0.0
        return self.n_calls_with_working_hit / denom
