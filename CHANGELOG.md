# Changelog

All notable changes to the `myelin` Python SDK are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project uses semantic versioning.

## [1.3.1] — 2026-05-22

First release published from the dedicated `myelinagent/Myelin-AI`
repository. **No functional or API changes from 1.3.0** — same surface,
same benchmarks, same dependencies. This release exists to:

- Move the canonical source from the research monorepo
  (`Skobyn/cortex-m`, where the library lived under `myelin-py/`) to a
  dedicated clean distribution repo at `myelinagent/Myelin-AI`.
- Re-anchor PyPI Trusted Publishing on the new repo (the publisher
  binding was re-registered on PyPI for `myelinagent/Myelin-AI`).
- Update `[project.urls]` to point at the new homepage / issues /
  changelog URLs.

`pip install myelin-ai==1.3.1` is byte-equivalent to `1.3.0` in terms of
importable surface; the bump is metadata-only.

## [1.3.0] — 2026-05-22 (GitHub release only)

Promotes orchestration patterns validated on PersonaMem n=589 from the
`scripts/benchmarks/` research scaffold into the library. Two new
submodules; no breaking changes; nothing removed.

### Added

- **`myelin.serve`** — HTTP sidecar that wraps `MemoryClient` as JSON-over-HTTP
  with endpoints compatible with the OpenClaw memory-plugin convention
  (ADR-017). Endpoints: `/tools/memory_{search,get,store,update,forget,consolidate,prune}`,
  `/hooks/{before_agent_start,agent_end}`, `/admin/reset`, `/health`.
  - Run: `python -m myelin.serve --host 127.0.0.1 --port 8765`
  - Embed: `from myelin.serve import build_server`
  - Loopback only by default; no auth (wrap behind a reverse proxy if exposed
    beyond local host).

- **`myelin.agent.OrchestratedAgent`** — provider-agnostic memory-aware agent
  implementing the v1.6 classifier-routed multi-step pattern. Cheap classifier
  (gpt-4o-mini by default) labels each question as `factual` (→ multi-step
  with `memory_search` tool calls) or `synthesis` (→ top-10 prefetch + tool
  available). Supported answerer providers:
  - `openai/*` (e.g. `openai/gpt-4o`, `openai/gpt-4o-mini`)
  - `google/*` (e.g. `google/gemini-3.5-flash`) — Gemini client is configured
    with a 30s `HttpOptions.timeout` to guard against the SDK's internal
    retry-on-disconnect hang documented in user memory.
  - Token usage and cost are tracked on `OrchestratedAgent.last_usage`; pricing
    is overridable via `OrchestratedAgent.PRICE_TABLE`.

### Quickstart

```python
from myelin import MemoryClient
from myelin.agent import OrchestratedAgent

client = MemoryClient(user_id="u1")
client.add("The user prefers green tea over coffee.")
client.add("The user is allergic to peanuts.")

agent = OrchestratedAgent(client=client, model="openai/gpt-4o")
answer = agent.answer(
    question="What does the user prefer to drink?",
    options="(a) coffee\n(b) green tea\n(c) matcha\n(d) water",
)
print(answer)              # → "<final_answer>(b)"
print(agent.last_usage)    # → {"input_tokens": ..., "cost_usd": ..., ...}
```

### Benchmark provenance

The v1.6 classifier-routed pattern (PersonaMem n=589, gpt-4o) lifted overall
accuracy from 0.6486 (v1.2 direct-prompt) to 0.7097, the largest single-step
gain in the v1.3-v1.9 series. Three hypotheses were tested and **falsified**
on the same bench — they are intentionally **not** in this release:

- **HyDE query expansion** (v1.8 H2): 0.6949 vs v1.6 0.7097, regression.
- **LLM-distilled fact ingestion** (v1.7 H1): 0.7114, within noise.
- **K=3 self-consistency voting** (v1.9 H3): 0.7140, within noise; synthesis
  axis regressed by −4.8 pp under temperature sampling.

Full per-axis tables and verdict commits live in
`data/registry/test-results-registry.jsonl` for anyone who wants to
reproduce or extend.

### Compatibility

- No changes to existing v1.0 / v1.1 / v1.2 surfaces. v1.2 `cite()` and
  `session()` work unchanged with the new agent module.
- No new required dependencies. `openai>=1.0` (for OpenAI provider) and
  `google-genai` (for Gemini provider) are optional — only imported when
  the corresponding model spec is used.

## [1.2.0] — 2026-05-21 (GitHub release only)

Two additive innovations on top of the v1.0/v1.1 substrate. Both are
opt-in via strict-`"1"` env vars (per `feedback_strict_1_env_vars` — no
loose-truthy parsing). A v1 substrate with neither flag set produces the
v1 health value unchanged.

**Install path for v1.2**: the PyPI `myelin` name is owned by an
unrelated package, so v1.2 ships as a GitHub release only. Install via
`pip install "git+https://github.com/myelinagent/Myelin-AI.git@v1.2.0#subdirectory=myelin-py"`.
A different PyPI name will be picked for v1.3.

### Added

- **Innovation A — outcome-conditioned learning via `cite()`**
  - `MemoryClient.cite(memory_id, answer_correct=...)` records cite
    outcomes against stored memories. Correct cites lift health; incorrect
    cites penalize; recency-of-cite decays exponentially (half-life
    `MYELIN_RECENCY_HALFLIFE_DAYS`, default 14 days).
  - `MemoryClient.abstain_event(...)` records the negative-evidence path
    when no memory was cite-worthy (FR-A6).
  - Three new `Memory` fields populated transparently:
    `cite_correct_count`, `cite_incorrect_count`, `last_cited_at`. Default
    values (0, 0, None) leave the v1 health equation unchanged.
  - Apoptosis equation gains three additive terms:
    `+W_USE_LIFT · log1p(cite_correct_count)`,
    `+W_USE_PENALTY · log1p(cite_incorrect_count)`,
    `+W_RECENCY_LIFT · recency_factor(last_cited_at, now)`.
    See `docs/sparc/cite-and-working-memory/phase_3_architecture.md` §3
    for the pinned constants and Phase 2 cite-health doc for the algorithm.
  - **EC-A5 invariant — contradiction dominance.** `constants.py`
    asserts at import time that
    `W_USE_LIFT · log1p(100) < |W_CONTRADICTION|` so a single
    contradiction always outweighs 100 correct cites. A tuning sweep that
    breaks this invariant breaks the import — deliberate brittleness.

- **Innovation B — session-scoped working memory**
  - `MemoryClient.session(session_id=...)` returns a `SessionHandle`
    context manager.
  - `client.observe(turn)` writes to a ring-buffered `ObservedTurn`
    stream (bounded by `MYELIN_OBSERVED_RING_MAX`, default 1024).
  - `client.commit(turn_ids=...)` promotes observed turns into the
    persistent substrate as `Memory` rows. Observe-without-commit data is
    NEVER persisted (EC-B7).
  - `client.attend(query, ...)` performs a session-scoped working-set
    search; tiered search transparently falls back to the persistent
    substrate when working-set yield is below a threshold.
  - `SessionRegistry` singleton with per-session locks (no global lock),
    background TTL sweeper (`MYELIN_SWEEPER_INTERVAL_SECONDS`), and
    per-process bound (`MYELIN_MAX_CONCURRENT_SESSIONS`).
  - New types: `ObservedTurn`, `SessionHandle`, `SearchTelemetry`
    (re-exported from `myelin`). See architecture doc §4 for full data
    shapes.

- **Backend Protocol additions** (`myelin.backend.Backend`, see arch §6)
  - `record_cite(memory_id, correct: bool, cited_at: datetime) -> None`
  - `record_abstain(query: str, query_hash: str | None, at: datetime) -> None`
  - The bundled `myelin.backends.sqlite.SqliteBackend` implements both
    + a forward-only schema migration writing two new tables
    (`cite_events`, `abstain_events`) and three new columns on
    `memories`. Migration is idempotent and runs at backend open time;
    no manual step is required.

- **Bench-harness invariants** (`scripts/benchmarks/_invariants.py`)
  - `assert_bench_invariants(...)` asserts strict-`"1"` env engagement,
    non-zero cite-call counts when `cite_enabled`, working-set hit-rate
    sanity, and the EC-A5 TPR-on-poison floor. Verdict rows are written
    as `phase_aborted` (not `phase_complete`) when this assertion raises
    — closes the orchestrator-wrapper vacuous-PASS class (per
    `feedback_orchestrator_wrapper_vacuous_pass`).
  - `build_verdict_extension(...)` emits 11 new fields into each verdict
    row covering both v1.2 paths so a paid bench produces auditable
    mechanism-coupled telemetry alongside the answerer-coupled accuracy.

- **Multi-turn synthetic fixture bench**
  (`scripts/benchmarks/multi_turn_fixture/`)
  - Per `feedback_plan_acceptance_vs_bench_sampling`: PersonaMem is
    single-question-per-conversation and cannot show Innovation B lift.
    The synthetic 30-conv × 50-turn fixture is the bridge benchmark
    until a real multi-turn substrate (LoCoMo) unblocks.

### Changed

- `memory_health(...)` now includes the three cite-aware terms (Innovation
  A). Pre-existing callers see no behavior change when `cite_correct_count`,
  `cite_incorrect_count`, and `last_cited_at` are at their defaults.

### Environment variables (new, all opt-in)

| Variable | Type | Range / Format | Default | Purpose |
|----------|------|----------------|---------|---------|
| `MYELIN_CITE_LEARNING` | strict-`"1"` | literal `"1"` only | unset | Enable Innovation A (cite outcome-conditioned health). |
| `MYELIN_WORKING_MEMORY` | strict-`"1"` | literal `"1"` only | unset | Enable Innovation B (session-scoped working memory). |
| `MYELIN_CITE_HASH_QUERIES` | strict-`"1"` | literal `"1"` only | unset | SHA-256-hash query strings on `abstain_event` rows for privacy. |
| `MYELIN_W_USE_LIFT` | float | 0.0–2.0 | 0.5 | Lift weight per `log1p(correct_count)` in apoptosis health. |
| `MYELIN_W_USE_PENALTY` | float | −2.0–0.0 | −0.3 | Penalty weight per `log1p(incorrect_count)`. |
| `MYELIN_W_RECENCY_LIFT` | float | 0.0–2.0 | 0.4 | Recency-factor weight. |
| `MYELIN_RECENCY_HALFLIFE_DAYS` | float | 0.5–365.0 | 14.0 | Exponential-decay half-life since last cite. |
| `MYELIN_SESSION_TTL_SECONDS` | int | 60–604800 | 86400 | Session inactivity TTL before sweeper evicts. |
| `MYELIN_MAX_CONCURRENT_SESSIONS` | int | 1–100000 | 1000 | Per-process active-session cap. |
| `MYELIN_OBSERVED_RING_MAX` | int | 8–65536 | 1024 | Per-session observed-turn ring-buffer cap. |
| `MYELIN_WORKING_SET_MAX` | int | 4–8192 | 256 | Per-session committed working-set cap. |
| `MYELIN_SWEEPER_INTERVAL_SECONDS` | int | 1–3600 | 60 | TTL sweeper poll interval. |

Numeric vars fall back to the baked default with `logger.warning` on
unparseable input or out-of-range values — they do not raise.

### Migration notes (v1.0/v1.1 → v1.2)

- **Take a backup of the SQLite file before first open under 1.2.0.** The
  schema migration is forward-only — there is no automated 1.2 → 1.1
  downgrade path. The migration adds two tables (`cite_events`,
  `abstain_events`) and three columns on `memories`; it is idempotent
  and re-running on an already-migrated DB is a no-op.
- A v1 substrate opened under v1.2 with both flags unset behaves
  identically to v1: cite columns default to `(0, 0, NULL)`, the
  apoptosis equation produces unchanged values, and no working-memory
  state is allocated.
- To opt in incrementally, enable one flag at a time. Innovation B has
  no dependency on Innovation A and vice-versa.
- The strict-`"1"` env-var convention means `"true"`, `"True"`, `"yes"`,
  and `"1.0"` will NOT engage the flags. This is intentional — see
  `feedback_strict_1_env_vars` in user memory for the rationale.

### Documentation

- `docs/sparc/cite-and-working-memory/phase_3_architecture.md` — pinned
  constants, lock hierarchy, build order, data shapes.
- `docs/adr/021-cite-outcome-conditioned-learning.md` (queued)
- `docs/adr/022-working-memory-primitive.md` (queued)
- `docs/adr/023-bench-harness-vacuous-pass-preflight.md` (queued)
- `docs/adr/024-sqlite-schema-migration-discipline.md` (queued)
- `docs/runbooks/myelin-v1.2-operations.md` (queued)

### Test coverage

- 235 tests pass (228 v1 baseline + 7 new branch-coverage).
- `myelin._internal.health` lifted 77 % → 100 %.
- Total project coverage 88 %.
- Multi-turn fixture bench green: `working_set_hit_rate=1.000`,
  `phase=complete`.

### Known limitations / deferred

- **LoCoMo bench is still unblocked.** Until LoCoMo lands, the
  multi-turn synthetic fixture is the gate for Innovation B lift; the
  paid PersonaMem run captures Innovation A mechanism telemetry but
  cannot show Innovation B lift (single-question-per-conversation).
- `tests/test_weights_fetcher.py::test_pinned_sha256` remains skipped
  pre-training. This is a Memory-R1 contrastive-weights concern, not a
  v1.2 cite/working-memory concern.
