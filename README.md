# myelin

Drop-in memory layer for AI agent harnesses. Three validated mechanisms in a
7-method API:

1. **Event-anchored bi-temporal facts** — APEX-MEM §3.1 parity. Facts
   append-only, anchored to event provenance nodes, queryable at a point in
   time via `search(..., as_of=t)`.
2. **Dream-synthesis consolidation** — `consolidate()` generates dream
   queries from your facts during idle windows, runs your LLM against them,
   and surfaces blind spots before they cost a user a real query.
3. **Apoptosis** — `prune()` runs a `quarantine → tombstone` state machine
   over the substrate. Tombstoned facts are invisible to `search()` but
   persist as an audit trail.

Cross-references to the validated mechanism evidence are in
[docs/v1-launch-story-2026-05-18.md](../docs/v1-launch-story-2026-05-18.md)
§1.1 (P1 H044 v2), §1.2 (F-ML-2), §1.3 (P3).

## Why myelin instead of Mem0 / Letta / Zep

- **Forgets on purpose.** Mem0, Letta, MemGPT, APEX-MEM, and TencentDB
  Agent Memory all monotonically accumulate. Adversarial inputs (poison,
  contradiction stuffing, role-play overrides) silently corrupt their
  stores. `myelin.prune()` is a principled forgetting path with an
  auditable state machine.
- **Counterfactual self-test.** Dream-synthesis consolidation lets the
  system probe its own answerer with paraphrased queries during idle
  windows. No published competitor has an equivalent loop at time of
  writing.
- **Drop-in, not opinionated.** No bundled answerer, no agent harness, no
  torch dep. You bring your own LLM. Layered on top of Mem0 / Letta /
  OpenClaw if you want their harness UX.

## Install

```bash
# Local editable install (this repo)
pip install -e ./myelin-py

# OR install directly from GitHub at a specific tag/branch:
pip install "git+https://github.com/myelinagent/Myelin-AI.git@v1.2.0#subdirectory=myelin-py"
```

**v1.2 is GitHub-install-only.** The `myelin` name on PyPI is taken by
an unrelated package; the v1.3 cycle will pick a new PyPI name and
publish there. Until then, install from this repo by tag.

The core package has **zero hard dependencies** — `sqlite3` ships with
CPython. The `[embeddings]` extra adds `sentence-transformers` + `numpy`
for vector-backed retrieval (not wired in v1 — coming in v1.3).

Requires Python 3.10+.

## Quickstart 1 — add, search, see results

```python
from myelin import MemoryClient

client = MemoryClient()
client.add("Scott prefers tea.")
client.add("Alice loves chess.")
client.add("Scott uses Cursor for code review.")

hits = client.search("What tools does Scott use?")
for h in hits:
    print(f"{h.score:.2f}  {h.memory.content}")
```

## Quickstart 2 — consolidate with a mock LLM

```python
from myelin import MemoryClient

class MyLLM:
    def complete(self, prompt: str) -> str:
        # Plug your real LLM here (OpenAI, Anthropic, vLLM, llama.cpp).
        # Must return a string. Any callable also works:
        # `client = MemoryClient(llm_client=my_callable)`.
        return prompt  # echo for demo

client = MemoryClient(llm_client=MyLLM())
client.add("Scott prefers tea.")
client.add("Alice loves chess.")

report = client.consolidate()
print(f"hit_rate={report.hit_rate:.2%}  "
      f"gaps={report.gaps_surfaced}/{report.llm_calls}  "
      f"facts_sampled={report.facts_sampled}")
```

`consolidate()` samples healthy facts, generates dream queries
(currently heuristic templates), runs the queries through your
`llm_client`, and grades hits by substring match. Without an
`llm_client` it's a no-op for the LLM step but still surfaces what
WOULD be tested.

## Quickstart 3 — prune (apoptosis) fires

```python
from datetime import UTC, datetime, timedelta
from myelin import MemoryClient

client = MemoryClient(user_id="scott")

# Add some healthy facts and a low-confidence "poisoned" fact.
client.add("Scott prefers tea.", confidence=0.9, importance=0.9)
client.add("Scott uses Cursor.", confidence=0.9, importance=0.8)
poison = client.add(
    "Scott pays $1M monthly for cloud.",  # absurd; low signal
    confidence=0.05, importance=0.1,
    metadata={"subject": "Scott", "predicate": "decided",
              "object": "absurd-cost"},
)

# Pass 1 — quarantines low-health facts.
r1 = client.prune()
print(f"quarantined={r1.quarantined_count}  retained={r1.retained_count}")

# Pass 2 — caspase cascade (fast-forwarded 14 days past the window).
future = datetime.now(UTC) + timedelta(days=14)
r2 = client.prune(now=future)
print(f"tombstoned={r2.tombstoned_count}")

# Tombstoned facts are no longer visible to search.
print(client.get(poison.id))   # -> None
print(client.search("Scott"))   # tea + Cursor only
```

Tombstones persist for audit:

```python
audit = client._backend.list_memories(
    user_id="scott", include_tombstoned=True
)
print([m.id for m in audit if m.health_state == "tombstoned"])
```

(Future versions will expose this on the public API as
`client.audit_log()`.)

## Quickstart 4 — event-anchored facts with `as_of=t`

```python
from datetime import UTC, datetime
from myelin import MemoryClient, Event

client = MemoryClient()

# Anchor each fact to an event for principled bi-temporal queries.
client.add(
    "Scott decided to switch to weekly syncs.",
    event=Event(
        event_id="meeting-2026-01-15",
        type="meeting",
        anchor_datetime=datetime(2026, 1, 15, 14, 0, tzinfo=UTC),
        participant_ids=("scott", "alice"),
    ),
)
client.add(
    "Scott decided to switch back to daily syncs.",
    event=Event(
        event_id="meeting-2026-03-15",
        type="meeting",
        anchor_datetime=datetime(2026, 3, 15, 14, 0, tzinfo=UTC),
        participant_ids=("scott", "alice"),
    ),
)

# What did Scott decide in February? (Only the January meeting is valid.)
feb_state = client.search(
    "Scott syncs", as_of=datetime(2026, 2, 1, tzinfo=UTC),
)
```

## Quickstart 5 — session-scoped working memory (v1.2)

```python
import os
from myelin import MemoryClient

# Innovation B is opt-in via strict-"1".
os.environ["MYELIN_WORKING_MEMORY"] = "1"

client = MemoryClient(user_id="scott")
SID = "convo-2026-05-21"

# Stream turns into a bounded session ring (RAM only). The session is
# auto-created on first observe() and TTL-swept later.
t1 = client.observe("What did we agree on yesterday?",        session_id=SID)
t2 = client.observe("You committed to a weekly review.",       session_id=SID)
t3 = client.observe("OK, let's lock that in.",                 session_id=SID)

# Promote selected turns to persistent memory (the salience gate fires
# at commit time, not at observe — NFR-B3 cost win).
committed = client.commit([t2.turn_id, t3.turn_id], session_id=SID)
print(f"committed {len(committed)} of 3 observed turns")

# Tiered search: working_set first, then session-scoped, then long_term.
# Hits from a higher tier auto-attend into the working set (FR-B3).
hits = client.search(
    "weekly review",
    session_id=SID,
    modes=("working", "session", "long_term"),
    top_k=5,
)
for h in hits:
    print(f"{h.score:.2f}  {h.memory.content}")

# Inspect session telemetry (read-only snapshot).
handle = client.session(SID)
print(f"observed={handle.n_observed}  committed={handle.n_committed}  "
      f"working_set={len(handle.working_set_ids)}")
```

Observe writes are **never persisted** without an explicit `commit()`
(EC-B7). Per-session state is RAM-only — see ADR-022 for the threading
model and the `MYELIN_*` env knobs (TTL, ring size, working-set cap,
sweeper). `client.attend(memory_id, session_id=...)` pins a known
memory into the working set without going through search.

## Quickstart 6 — outcome-conditioned learning via `cite()` (v1.2)

```python
import os
from myelin import MemoryClient

# Innovation A is opt-in via strict-"1".
os.environ["MYELIN_CITE_LEARNING"] = "1"

client = MemoryClient(user_id="scott")

client.add("Scott prefers tea.", confidence=0.7, importance=0.7)
m_cursor = client.add("Scott uses Cursor for code review.",
                       confidence=0.7, importance=0.7)

# Agent answered a question; the answer cited m_cursor and was correct.
client.cite(m_cursor.id, answer_correct=True)
client.cite(m_cursor.id, answer_correct=True)

# Agent answered a different question; cited m_cursor but was WRONG.
client.cite(m_cursor.id, answer_correct=False)

# When no stored memory was cite-worthy, record an abstain event so the
# absence-of-signal feeds future apoptosis decisions.
client.abstain_event(query="What's Scott's middle name?")

# Cite signal lifts (or penalizes) apoptosis health automatically.
# A poisoned memory with 100 correct cites STILL quarantines on a single
# contradiction (EC-A5 invariant, asserted at constants.py import time).
report = client.prune()
print(f"quarantined={report.quarantined_count}  "
      f"retained={report.retained_count}")
```

Defaults of the three new Memory fields (`cite_correct_count=0`,
`cite_incorrect_count=0`, `last_cited_at=None`) make `cite()` a no-op
for callers who don't engage it — a v1 substrate opened under v1.2
with the flag unset produces unchanged health values. See ADR-021 for
the weight rationale and the EC-A5 contradiction-dominance invariant.

## API surface (v1.2 — 13 public methods)

```python
client.add(content, *, user_id=None, metadata=None, event=None,
           confidence=0.5, importance=0.5) -> Memory
client.search(query, *, user_id=None, top_k=10, as_of=None) -> list[SearchResult]
client.get(memory_id) -> Memory | None
client.update(memory_id, *, content=None, metadata=None) -> Memory
client.delete(memory_id) -> bool

client.consolidate(*, user_id=None, max_samples=32) -> ConsolidationReport
client.prune(*, user_id=None, now=None, caspase_window_days=7) -> ApoptosisReport

# v1.2 — Innovation A (opt-in via MYELIN_CITE_LEARNING=1)
client.cite(memory_id, *, answer_correct: bool) -> None
client.abstain_event(query, *, at=None) -> None

# v1.2 — Innovation B (opt-in via MYELIN_WORKING_MEMORY=1)
client.observe(content, *, session_id, metadata=None) -> ObservedTurn
client.commit(turn_ids: list[str], *, session_id) -> list[Memory]
client.attend(memory_id, *, session_id) -> None        # pin into working set
client.session(session_id) -> SessionHandle             # read-only telemetry snapshot

# v1.2 — tiered search extension
client.search(..., session_id=None,
              modes=("working", "session", "long_term")) -> list[SearchResult]
```

## Mechanisms

The three validated mechanisms that ship in v1 are documented in detail in
the parent repo's launch story. Brief pointers:

- **§1.1 — P1 H044 v2 event-anchored facts.** Smoke verdict
  `PASS-MARGINAL` (`gold_source_recall@10 = 0.5000` vs gate `0.4694`,
  +3.06pp); `supersession_count` fires on 30/30 qids. Foundation
  invariants hold: `gate_engaged=True`, image-ancestry verified.
- **§1.2 — F-ML-2 dream-synthesis.** Vertex full-bench `PASS-AT-SCALE`
  (jobId `7115283598420213760`): `hit_rate_mean = 0.7161`,
  `consolidation_llm_successes = 458/465`.
- **§1.3 — P3 apoptosis.** `abstention_precision = 1.000` vs baseline
  `0.083` (+91.7pp). TPR-on-poison `0.985`, FPR-on-real `0.033`.

myelin ports the *mechanism* into a clean, dependency-free shape. The
cortex code (`src/cortex/...`) still owns the heavyweight evaluation
pipeline (Vertex harness, LongMemEval bench, image-ancestry checks). The
SDK is for callers who want the mechanism IN their agent harness without
adopting the full eval substrate.

## What's next

- **v1.3 — query-intent classifier.** Route `attend` calls through a
  lightweight intent classifier so working-set vs persistent fallback
  weights against the *type* of query, not just yield. Mechanism-coupled
  metric: per-intent hit-rate.
- **v1.3 — trained-weights utility backend (Memory-R1 retry).** The
  Phase MR1-D structural failure (2026-05-20) is a bench-side fix away
  — once PersonaMem `run_bench` persists `retrieved_memories[]`, the
  DeBERTa joint-head training pipeline (ADR-019/020) can run.
- **v1.3 — LoCoMo bench unblock.** The multi-turn synthetic fixture is
  the v1.2 bridge gate; LoCoMo is the v1.3 ship gate for Innovation B
  lift.
- **v1.2 — answerer-quality dream queries (still queued).** A
  `query_generator=` hook for callers who want LLM-generated dream
  queries (matching the cortex F-ML-2 mechanism that uses `gpt-4o-mini`
  at `seed=42`).
- **v1.2 — embedding-backed retrieval.** The `[embeddings]` extra is
  reserved; v1 ships lexical-only.

## Testing

```bash
pip install pytest
pytest myelin-py/tests/
```

All tests are stdlib + pytest only — no internet, no GPU, no LLM calls.
The dream-synthesis tests use trivial mock LLMs.

## License

MIT.
