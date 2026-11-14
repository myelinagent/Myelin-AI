# Changelog

All notable changes to the `myelin-ai` Python SDK are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the
project uses semantic versioning.

## [Unreleased]

Stability pass: crash, stall and data-loss paths found in a whole-repo review.

### Changed

- **Hybrid search ranks its first 20 results by a balanced fusion.** The
  vector-heavy fusion pushed the right short messages out of the top
  results (LoCoMo, all evidence in the top 10: 39.6%, against 54.9% for
  lexical search alone). The first 20 places now use equal weights and a
  small rank constant, the rest are unchanged: 51.8% on LoCoMo, 77.5%
  unchanged on LongMemEval-S, and the same evidence within a token
  budget on both. See `bench/results/fusion.md`.
- **Excerpts use their whole budget.** A memory up to 1.5 times
  `max_result_tokens` is returned whole, and an excerpt grows outward
  from the passages that match the query until the limit is reached (a
  114-token reply used to be cut to 17 tokens). LongMemEval-S with
  `gpt-5-mini`: 91.3% over four runs (89.6 to 93.0), against 90.3%.
- **`recall()` shortens long memories to 100 tokens instead of 300**
  (`max_result_tokens`; the three best results are still whole). Nearly
  twice as many memories fit the same budget: on LongMemEval-S the share
  of questions with all evidence in context goes from 92.3% to 97.0%, and
  accuracy with `gpt-5-mini` from 89.0% to 90.3% (two runs each). Pass
  `max_result_tokens=300` for the previous behaviour.
- **Faster full-text search on large stores.** The best matches are now
  picked inside the index before memory rows are read: at 200,000
  memories, median `recall()` went from 119 ms to 45 ms and the 95th
  percentile from 600 ms to 190 ms. Results are unchanged apart from the
  order of exact ties. `python -m bench.scale` reproduces the measurement.
- **Forgetting stops at quarantine.** `prune()` no longer tombstones on its
  own. Each call now:
  1. marks a fact as superseded when a later fact gives a different value
     for the same `subject` and `predicate`: `t_invalid` is set and the
     replacing fact's id is stored in `metadata["_superseded_by"]`;
  2. restores a quarantined fact whose health has recovered;
  3. quarantines healthy facts whose health is below the threshold.

  Tombstoning needs `prune(tombstone=True)`, which applies the previous
  seven-day window. `ApoptosisReport` gains `superseded_count` and
  `recovered_count`. The Mem0 adapter, OpenClaw tool and sidecar accept
  `tombstone` too.
- **Search returns only current, healthy facts.** Quarantined facts are
  hidden (they used to be returned), and so are superseded facts unless
  `as_of` names a time when they held. `search()` and `recall()` take
  `include_quarantined=True` and `include_superseded=True` to see them;
  `get()` always returns them.

  **If you relied on the old behaviour:** call `prune(tombstone=True)` to
  keep automatic removal, and pass `include_quarantined=True` to keep
  quarantined facts in results.
- **Search ranks with BM25 over a SQLite full-text index, with word
  stemming.** It replaces the unweighted token-overlap scan: rare words now
  count for more than common ones, "graduate" matches "graduated", and a
  query no longer re-reads every stored memory. Existing stores are indexed
  the first time they are opened. `SearchResult.score` is now a BM25 score
  (higher is better, not comparable with earlier values). On a SQLite built
  without FTS5 the previous scan is used. On LongMemEval-S, the share of
  questions with all evidence in the top 10 rose from 80.9% to 86.0% and
  search time fell from 293 ms to 7 ms per question
  (`bench/results/longmemeval_s_stage1.md`).

### Added

- **Sidecar endpoints** `/tools/memory_recall` (budgeted matches plus the
  text for a prompt), `/tools/memory_store_many` and
  `/tools/memory_cite_used`.
- **`MemoryClient.cite_used(results, used)`** records which of the shown
  memories an answer relied on. Used ones are cited as useful; the rest
  are counted as shown but not used (`Memory.shown_unused_count`), which
  lowers their rank and leaves their health alone. With
  `format_context(numbered=True)`, `build_answer_prompt(ask_used=True)`
  and `parse_used()` the answering model supplies the list. On held-out
  LoCoMo questions all evidence is in the top 5 for 47.8% of questions
  against 42.4% without feedback.
- **`python -m bench.cite_feedback`**: a held-out test of cite-informed
  ranking, with no model. Feedback that names the useful memory is worth
  about 5 points of top-5 evidence with hybrid search; feedback that only
  says whether the answer was right is not.
- **`MemoryClient.add_conversation(turns, when=, llm=, people=)`** stores a
  conversation one memory per message and, given a model, indexes facts
  extracted from it in the same call.
- **`extract_facts(..., people=[...])`** extracts dated facts from a
  conversation between named people, about every speaker. Indexed with
  each utterance on LoCoMo it raises accuracy from 89.0% to 91.9%
  (`gpt-5-mini`) and from 78.7% to 82.4% (`gpt-4o-mini`).
- **LoCoMo runner** (`python -m bench.locomo`): 89.0% judged correct with
  `gpt-5-mini` answering and 78.7% with `gpt-4o-mini`, hybrid search,
  6,000-token context. See `bench/results/locomo.md`.
- **`recall(query, token_budget=6000)`** returns as many top results as fit
  a token budget, for building an answering model's context. With it, 90.9%
  of LongMemEval-S questions have all their evidence in context.
- **`recall(max_result_tokens=300)`** counts a longer memory against the
  budget as an excerpt of the passages that bear on the query
  (`SearchResult.excerpt`); the stored memory is untouched. `excerpt()` is
  exported. End to end on LongMemEval-S with `gpt-4o-mini` answering,
  accuracy went from 60.4% (1.4.0 search, top 10) to 64.8%
  (`bench/results/longmemeval_s_accuracy.md`).
- **Hybrid search: `MemoryClient(embedder=...)`.** With an embedder, each
  memory is embedded when written and each query when searched, and the
  vector ranking is fused with BM25 (reciprocal-rank fusion; see the
  fusion entry under Changed). Vectors live in SQLite and are compared by a full scan; numpy is used
  when installed and is not required. `myelin.embedders.StaticEmbedder`
  (extra `myelin-ai[static]`) is a CPU-only model2vec embedder.
  `backfill_embeddings()` embeds memories written earlier. If the embedder
  raises, writes and searches continue lexically. On LongMemEval-S the share
  of questions with every evidence turn in the answerer's context rose from
  85.3% to 92.3%, and accuracy with `gpt-4o` answering from 82.8% to 87.4%
  (`bench/results/longmemeval_s_stage2.md`).
- **Index text and write-time fact extraction.** `add(index_text=...)` (also
  an `add_many` item key, and `set_index_text()` for existing memories)
  stores extra text a memory is findable by. It is indexed and embedded with
  the content and never returned as content. `myelin.enrich.extract_facts`
  asks a model you supply for short, standalone, dated facts about the user
  in one conversation, to use as that index text. Opt-in: the default write
  path still makes no model calls. On LongMemEval-S, on top of hybrid
  search, indexing extracted facts did not measurably change accuracy
  (89.8% against 89.2%, within noise), and showing them to the answerer
  lowered it (88.0%); see `bench/results/longmemeval_s_stage2.md`.
- **`cite()` feedback now influences ranking.** With cite learning enabled,
  long-term search re-ranks a pool of the most relevant candidates: memories
  whose cites were mostly correct move up, mostly incorrect move down, and
  uncited memories stay where relevance put them. Feedback never pulls in a
  memory the query did not match, and a single cite has little effect (full
  strength at five). `MemoryClient(cite_ranking=False)` keeps cites for
  forgetting only. Not yet measured on a benchmark: doing so without leaking
  the answer key needs a held-out protocol.
- **`add_many(items)`** writes a batch of facts in one transaction (all or
  none) and embeds them in one call when the embedder has `encode_batch`
  (`StaticEmbedder` does). Loading about 494 turns with embeddings went from
  2.37 s to 0.88 s. `SqliteBackend.add_memories` is the backend half.
- **`build_answer_prompt(history, question, now=..., style=...)`** wraps
  recalled context in instructions for the cases where answering models go
  wrong with memory: changed facts, totals and orderings, dates relative to
  a conversation, requests for advice, and questions the history does not
  cover. Styles `"reasoned"` and `"direct"`. On LongMemEval-S it took
  `gpt-4o-mini` from 64.8% to 77.2% on the same retrieval; `gpt-4o` scores
  82.8%.
- **`recall(whole_results=3)`** leaves the best-ranked results unshortened,
  so an answer inside a long memory is not lost to an excerpt.
- **`format_context(results)`** renders results oldest first under date
  headings; `fit_to_budget` and `estimate_tokens` are exported with it.
- **Headless benchmark harness** (`bench/`, not part of the published
  package). `python -m bench.longmemeval run` takes a memory system through
  LongMemEval from one command: fresh store per question, retrieval recall,
  pinned-prompt answers, grading with the benchmark's judge prompts,
  per-question checkpoints and a run manifest. Ships a BM25 control and an
  oracle control, and `--token-budget` to size the answerer's context.
  Baseline numbers are in `bench/results/`.
- **`add(t_valid=...)`** stores a fact at the time it became true instead of
  the time it was written, so recorded history can be ingested at its real
  dates.
- **`MemoryClient(clock=...)`** supplies "now" for `add()`, `prune()` and
  `cite()`, for replaying history and for deterministic runs.
- **Sidecar `--db-path`** (and `build_server(db_path=...)`) keeps memories in
  a SQLite file instead of in memory; `/admin/reset` clears it.
- **Test workflow** runs the suite on pull requests and pushes to `main`.

### Fixed

- **Hybrid search now sees vectors written by another process.** The
  in-memory vector cache was only refreshed by this process's own writes,
  so memories added through another connection were missing from the
  embedding channel until then.
- **`requires-python` is now `>=3.11`.** The package declared support for
  Python 3.10 but imports `datetime.UTC`, which 3.10 does not have, so it
  failed on import there. The metadata now matches what runs.
- **`prune()` no longer removes the current value of an updated fact.** When
  a single-valued fact is written again with a different object, only the
  older memory is penalized as contradicted; the newer one supersedes it and
  stays healthy. Previously both were quarantined and tombstoned. Facts with
  an identical `t_valid` still contradict each other.
- **`commit()` no longer loses turns when a write fails part-way.** Turns
  that were not stored go back into the session buffer and the error
  propagates, so the call can be retried.
- **`commit()` stores memories under the session's user**, not the client's
  default user. `observe()` gains an optional `user_id=`; passing a different
  user for an existing session raises `ValueError`.
- **`search(as_of=<naive datetime>)` no longer raises `TypeError`** when the
  working tier is searched; naive values are read as UTC in every tier.
- **`consolidate()` accepts `retriever=`** as the README documents, and
  successive calls rotate through the store instead of re-testing the same
  oldest `max_samples` facts. A non-text answer from the answerer counts as a
  miss instead of raising.
- **HTTP sidecar no longer freezes on one stalled client.** Requests are
  served on threads with a 30 s socket timeout and an 8 MB body cap. Non-object
  JSON bodies return 400 and an unknown `memory_id` on update returns 404.
- **`OrchestratedAgent` survives malformed tool arguments** from the model
  (reported back as a tool error; `top_k` is clamped to 1–50) and its OpenAI
  client now has a 60 s timeout.

### Changed

- **`delete()` also removes tombstoned rows and the memory's cite events**, so
  an explicit delete leaves no copy of the content behind. It returns `True`
  when a tombstoned memory was removed.
- **Mem0 adapter `delete_all()` requires a filter** (`user_id`, `agent_id` or
  `run_id`) and raises `ValueError` without one; use `reset()` to empty the
  store. Tombstoned memories in scope are deleted too.

## [1.4.0] — 2026-06-09

Three bug fixes to load-bearing mechanisms (D-MEM surprise, dream grader,
heuristic salience wiring) plus an opt-in store-grounding upgrade to the dream
cycle. Six new optional kwargs across the public API; no kwargs removed; the
existing v1.3 surface still works with prior defaults preserved.

This is the first release where `pip install myelin-ai==1.4.0` ships the
substrate as it was intended in 1.3 — the previously-pinned scoring axes and
grader were silently no-oping in production.

### Fixed

- **D-MEM surprise score now actually moves with the query.**
  `myelin._internal._dmem_helpers.zscore_surprise(query, neighbors)` was
  Z-scoring `raw_distance` against the same `dists` array it was the mean of,
  so `raw == mu` and `z == 0` for every input. The query never moved the
  score, which silently flattened the entire surprise axis. The reference
  distribution is now derived via leave-one-out mean cosine-distance of each
  neighbour to the rest (`_neighbor_reference_dists`) — a distribution that's
  independent of the query. Callers can also thread their own rolling
  per-user history via the new `reference=` kwarg. Cold-start fallback widens
  from `[0, 1]` to `[-1, 1]` (signed deviation).

- **Dream grader now scores recall, not parroting.** Dream queries previously
  included the fact verbatim in the prompt (`"What do you remember about:
  {fact}"`), so the answerer could echo the question and score a hit. Fix:
  extract a held-out "answer key" from each fact (longest distinctive
  non-stopword) and mask it from the prompt (`"Fill in the blank from
  memory: ___"`); grade against the held-out key only.

- **Heuristic salience novelty axis is actually wired.** The heuristic-mode
  salience gate's `novelty` signal was pinned at the 0.5 cold-start neutral
  for every turn because no embedder was threaded. `MemoryClient` now
  lazy-builds a sentence-transformers embedder on first heuristic `add()`
  (deferred dep, only paid in heuristic mode), maintains a per-scope rolling
  deque of past embeddings as the neighbour window, and feeds
  `(query_embedding, neighbors)` into `score_turn_heuristic`. Embed failure
  degrades to cold-start neutral (never raises in `add()`). Configurable via
  the new `salience_embedder=` and `salience_neighbor_window=` constructor
  kwargs (defaults: lazy-built default embedder, window 32). `GateDecision`
  now exposes the originating `SignalBundle` so callers can read per-axis
  `novelty` / `reward` without re-deriving them.

### Added

- **`MemoryClient.add(..., outcome=...)` — reward signal for the salience
  gate.** Optional `Mapping[str, object]` with keys `task_succeeded`,
  `user_thumbs_up`, `user_thumbs_down`, `error`. Only consulted in
  `salience_mode="heuristic"`; ignored in `"off"` / `"dmem"`. Defaults to
  `None` — backwards compatible (reward stays 0.0 in that path, matching
  v1.3 behavior).

- **`run_dream_cycle(..., retriever=callable)` — store-grounded dream
  queries.** When supplied, each dream query is prepended with retrieved
  memory context (`"Remembered facts:\n- …\n\n{query}"`) so the answerer
  must recall from the store rather than guess from the elided prompt.
  Default `retriever=None` preserves the v1.3 ungrounded behavior exactly;
  a retriever miss (empty result or raise) falls back to the bare query
  (never a crash). The intended wiring is:

  ```python
  client.consolidate(
      retriever=lambda q: [r.memory.content for r in client.search(q, top_k=5)],
  )
  ```

  Reference re-derivation on the LongMemEval-oracle smoke: honest
  `hit_rate = 0.867` with store-grounding, clearing the recall bar that the
  ungrounded cloze couldn't reach without echoing.

- **`/metrics` endpoint on the HTTP sidecar.** `GET /metrics` returns
  `{"tool_invocations": int, "reset_count": int}` — the per-process counter
  is bumped on every successful `/tools/*` and `/hooks/*` call and on
  `/admin/reset`. Built for preflight integration checks: a bench can assert
  `tool_invocations > 0` before treating an "all-PASS" result as non-vacuous.

### Changed

- **Behavior changes that aren't drop-in equivalent.** The dream grader
  (`grade_hit`) and the D-MEM surprise score (`zscore_surprise`) return
  materially different values from 1.3 for the same inputs — by design,
  because the v1.3 behaviour was a no-op. Snapshot tests that pinned to the
  old "substring contains fact" grader output or the always-zero z-score
  will need to be updated. Threshold tuning calibrated against the 1.3
  surprise output should be re-run.

### New optional kwargs

| Surface | Kwarg | Default | Purpose |
|---|---|---|---|
| `MemoryClient.__init__` | `salience_embedder` | `None` (lazy default) | Override the heuristic-novelty embedder. |
| `MemoryClient.__init__` | `salience_neighbor_window` | `32` | Per-scope rolling neighbour window length. |
| `MemoryClient.add` | `outcome` | `None` | Outcome mapping threaded into the heuristic salience gate's reward axis. |
| `myelin._internal.dream.run_dream_cycle` | `retriever` | `None` | Optional `Callable[[query], Iterable[str]]` for store-grounded dream queries. |
| `myelin._internal._dmem_helpers.zscore_surprise` | `reference` | `None` (neighbour-derived) | Optional caller-supplied reference distribution. |

### Compatibility

- All new kwargs are optional with defaults that preserve prior behavior — no
  signatures break.
- The grader and surprise-score behavior changes are bug fixes; they DO change
  numerical output for the same inputs. Anyone with snapshot tests or
  calibrated thresholds against the 1.3 values will need to retune.
- `myelin.agent.OrchestratedAgent` (v1.3 addition) is unchanged.
- `cite()`, `session()`, working-memory tier, and the HTTP sidecar's existing
  endpoints are unchanged.

### Migration notes (1.3 → 1.4)

- **No mandatory action.** Existing call sites continue to work.
- **If you were calling `consolidate()` and observing low `hit_rate`** —
  it was being suppressed by the broken grader rewarding parrots. After
  upgrading, expect `hit_rate` to drop for substrates that previously had
  high-but-vacuous numbers. To recover real recall, wire `retriever=` to
  `client.search`.
- **If you were running `salience_mode="heuristic"` and seeing the gate
  not discriminate** — that was the unwired novelty axis. After upgrading,
  the heuristic gate will actually fire. Re-tune `salience_tau` if needed.

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
