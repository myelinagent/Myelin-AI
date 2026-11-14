# myelin-ai

`pip install myelin-ai` — a drop-in memory layer for AI agents. Event-anchored
bi-temporal facts, store-grounded dream-synthesis consolidation, and an
apoptosis-based forgetting path that quarantines low-health memories before
they can corrupt future answers. Zero hard dependencies in the core
(`sqlite3` only).

```bash
pip install myelin-ai             # core
pip install "myelin-ai[openai]"   # + OpenAI provider for OrchestratedAgent
pip install "myelin-ai[google]"   # + Gemini provider for OrchestratedAgent
```

Requires Python 3.10+. The import name remains `myelin`.

---

## What's new in 1.4.0

Three load-bearing mechanisms that were silently no-oping in 1.3 are now
actually wired. One new feature: store-grounded dream synthesis.

| Area | 1.3 behavior | 1.4 behavior |
|---|---|---|
| D-MEM surprise score (`zscore_surprise`) | `z == 0` for every input — broken normalization | Z-scored against a query-independent reference; signed and discriminating |
| Dream grader (`grade_hit`) | Substring-match against the prompt — answerer could echo and score a hit | Answer key held out of the prompt, graded against the elided key |
| Heuristic salience novelty axis | Pinned at 0.5 cold-start neutral (no embedder wired) | Lazy sentence-transformers embedder + per-scope rolling neighbour window |
| Dream cycle grounding | Ungrounded cloze: answerer guesses from query alone | Optional `retriever=` grounds queries against the memory store |
| Salience reward axis | Not pluggable | New `outcome=` kwarg on `add()` |
| Sidecar liveness | `/health` only | Adds `/metrics` for vacuous-PASS guards |

**This is the first PyPI release where the substrate behaves as designed.** If
you were on 1.3, upgrade — see the
[migration notes](https://github.com/myelinagent/Myelin-AI/blob/main/CHANGELOG.md).

---

## Why `myelin-ai` instead of Mem0 / Letta / Zep

- **Forgets on purpose.** Mem0, Letta, MemGPT, and similar agent-memory layers
  all monotonically accumulate. Adversarial inputs (poison, contradiction
  stuffing, role-play overrides) silently corrupt their stores.
  `client.prune()` is a principled forgetting path with an auditable
  quarantine → tombstone state machine.
- **Counterfactual self-test.** `client.consolidate(retriever=...)` lets the
  system probe its own answerer with held-out paraphrased queries during idle
  windows, grounded against the actual memory store.
- **Drop-in, not opinionated.** Zero hard deps in the core. No bundled
  answerer, no torch dep at install. You bring your own LLM. The
  `OrchestratedAgent` and HTTP sidecar are opt-in extras.

---

## Quickstart — the 1.4.0 happy path

The canonical wiring: add facts with outcome signal, run a store-grounded
dream cycle, and let apoptosis prune low-health residue.

```python
from myelin import MemoryClient

client = MemoryClient(
    user_id="scott",
    salience_mode="heuristic",       # 1.4: novelty + reward axes actually wired
)

# 1.4: outcome= threads success/failure into the heuristic gate's reward axis.
client.add(
    "Scott prefers oolong tea.",
    outcome={"task_succeeded": True, "user_thumbs_up": True},
)
client.add(
    "Scott uses Cursor for code review.",
    outcome={"task_succeeded": True},
)
client.add("Alice loves chess.")

# Hot retrieval.
for h in client.search("What does Scott drink?", top_k=3):
    print(f"{h.score:.2f}  {h.memory.content}")
```

### Store-grounded dream synthesis (the 1.4.0 marquee)

```python
# 1.4: retriever= turns the dream cycle from an ungrounded cloze into a
# real recall test. Each held-out query is prepended with retrieved memory
# context so the answerer must recall from the store, not guess.
class EchoLLM:
    def complete(self, prompt: str) -> str:
        # Replace with your real LLM (OpenAI, Anthropic, vLLM, llama.cpp...)
        return prompt

client = MemoryClient(user_id="scott", llm_client=EchoLLM())

report = client.consolidate(
    retriever=lambda q: [r.memory.content for r in client.search(q, top_k=5)],
)
print(f"hit_rate={report.hit_rate:.2%}  "
      f"facts_sampled={report.facts_sampled}  "
      f"llm_calls={report.llm_calls}")
```

Without `retriever=`, the cycle still runs but uses the v1.3-style ungrounded
cloze (kept for backward compat). The grader, however, is fixed in 1.4 either
way — `grade_hit` now scores against a held-out answer key, not a substring
of the prompt.

---

## Answering from memory (unreleased, on `main`)

Three steps turn stored memories into an answer: recall what fits a token
budget, lay it out in date order, and wrap it in a prompt that tells the
model how to use it.

```python
from datetime import UTC, datetime
from myelin import MemoryClient, build_answer_prompt, format_context

client = MemoryClient(user_id="scott")

# Load history at the time it happened (one transaction, one embedding call).
client.add_many([
    {"content": "Scott moved to Berlin.",
     "t_valid": datetime(2025, 3, 2, tzinfo=UTC),
     "metadata": {"session_id": "s1", "turn": 0}},
    {"content": "Scott is back in Paris for good.",
     "t_valid": datetime(2025, 9, 14, tzinfo=UTC),
     "metadata": {"session_id": "s2", "turn": 0}},
])

question = "Where does Scott live now?"
hits = client.recall(question, token_budget=6000)   # as many top hits as fit
prompt = build_answer_prompt(format_context(hits), question)
answer = your_llm(prompt)                           # bring your own model
```

Without an embedder, search is lexical: a question finds memories that
share words with it (here, "Scott"). Add an embedder, below, to match on
meaning as well.

- **`recall`** ranks with BM25 over a SQLite full-text index and returns
  the best results that fit `token_budget`. Long memories count as an
  excerpt of the passages that bear on the question (`result.excerpt`),
  except the three best-ranked, which stay whole. Stored memories are never
  altered.
- **`format_context`** renders results oldest first under date headings.
- **`build_answer_prompt`** adds rules for the cases models get wrong:
  facts that changed (the latest wins), totals and orderings, dates stated
  relative to a conversation, requests for advice, and questions the
  history does not cover. `style="reasoned"` (default) asks for brief notes
  and a final `Answer:` line; `style="direct"` asks for the answer only.

### Hybrid search

Add an embedding channel beside BM25 by passing an embedder. The core
package stays dependency-free; the bundled embedder is a CPU-only static
model behind an extra.

```bash
pip install "myelin-ai[static]"
```

```python
from myelin import MemoryClient
from myelin.embedders import StaticEmbedder

client = MemoryClient(db_path="memory.db", embedder=StaticEmbedder())
client.backfill_embeddings()    # embed memories written before this
```

Any object with `encode(text) -> sequence of floats` (and optionally
`encode_batch(texts)`) works in place of `StaticEmbedder`. If the embedder
raises, writes and searches carry on lexically.

### Facts at write time (opt-in)

`add_conversation` stores a conversation one memory per message. Given a
model, it also makes one call per conversation to extract short dated
facts and indexes them with the messages they came from. Messages are
still stored and returned whole.

```python
def llm(prompt: str) -> str:        # any model, any provider
    ...

client.add_conversation(
    [("Ana", "Pickle came home yesterday!"), ("Ben", "So happy for you.")],
    when=datetime(2023, 5, 20, tzinfo=UTC),
    llm=llm, people=["Ana", "Ben"],     # omit people for a user/assistant chat
)
client.search("when did Ana get her dog")   # finds the message via "Ana adopted a dog on 2023-05-19."
```

On LoCoMo this raises accuracy by about 3 points; on LongMemEval-S it
made no measurable difference (tables below).

### Measured on LongMemEval-S

All 500 questions, judged by `gpt-4o` with the benchmark's own prompts, one
run each (one standard error is about 1.7 points). Reproduce with
`python -m bench.longmemeval`; see [`bench/README.md`](bench/README.md).

| Setup | `gpt-4o-mini` answering | `gpt-4o` answering | `gpt-5-mini` answering |
|---|---:|---:|---:|
| 1.4.0 search, top 10, plain prompt | 60.4 | not run | not run |
| BM25 + budget + excerpts, plain prompt | 64.8 | not run | not run |
| + `build_answer_prompt` | 77.2 | 82.8 | not run |
| + hybrid search (`StaticEmbedder`) | 78.0 | 87.4 | 89.2 |
| Perfect retrieval (control) | 79.4 | 93.6 | 93.4 |

These are not comparable with scores published by other systems beyond
orientation: harnesses, prompts and answer models differ. Full tables,
controls and caveats are in [`bench/results/`](bench/results/).

### Measured on LoCoMo

1,540 questions over ten long conversations, hybrid search, the same
6,000-token budget, judged by `gpt-4o-mini` with the prompt published
LoCoMo scores use. One run each. Reproduce with `python -m bench.locomo`.

| | `gpt-4o-mini` answering | `gpt-5-mini` answering |
|---|---:|---:|
| Judged correct (overall) | 78.7 | 89.0 |
| single-hop / multi-hop | 88.6 / 77.7 | 93.2 / 84.8 |
| temporal / open-domain | 59.8 / 58.3 | 86.6 / 71.9 |
| **+ facts extracted at write time** (`myelin.enrich`), overall | 82.4 | **91.9** |
| + those facts also shown to the answerer, overall | 83.6 | 91.3 |

Fact extraction is opt-in and costs one model call per conversation. It
pays here and made no measurable difference on LongMemEval-S.

The same caution about comparing with published scores applies; see
[`bench/results/locomo.md`](bench/results/locomo.md).

---

## Memory-aware agent (`OrchestratedAgent`, v1.3)

A provider-agnostic multi-step agent. A cheap classifier (`gpt-4o-mini` by
default) routes each question as `factual` (multi-step with `memory_search`
tool calls) or `synthesis` (top-10 prefetch + tool available).

```python
from myelin import MemoryClient
from myelin.agent import OrchestratedAgent

client = MemoryClient(user_id="u1")
client.add("The user prefers green tea over coffee.")
client.add("The user is allergic to peanuts.")

# Works against OpenAI (openai/*) or Gemini (google/*) — pick what you have.
agent = OrchestratedAgent(client=client, model="openai/gpt-4o")
answer = agent.answer(
    question="What does the user prefer to drink?",
    options="(a) coffee\n(b) green tea\n(c) matcha\n(d) water",
    user_id="u1",
)
print(answer)              # → "<final_answer>(b)"
print(agent.last_usage)    # → {"input_tokens": ..., "cost_usd": ..., ...}
```

The Gemini client is wired with a 30s `HttpOptions.timeout` to defend against
the SDK's internal retry-on-disconnect hang. Pricing is overridable via
`OrchestratedAgent.PRICE_TABLE` for cost tracking.

---

## Salience gate — `mode="heuristic"` (1.4 wiring)

The heuristic gate combines four axes — novelty, reward, affect, reference —
into a single score and tags each turn `transient` / `short_term` /
`long_term`. In 1.4 the novelty axis actually moves (it was pinned at 0.5 in
1.3) and the reward axis is pluggable via `outcome=`.

```python
client = MemoryClient(
    user_id="scott",
    salience_mode="heuristic",
    salience_tau=0.40,                 # gate threshold (unchanged from 1.3)
    salience_neighbor_window=32,       # 1.4: rolling per-scope embedding window
    # salience_embedder=my_embedder,   # 1.4: override the lazy-built default
)

m = client.add(
    "Scott decided to switch to weekly syncs.",
    outcome={"user_thumbs_up": True},   # → +reward
)
```

Embedder resolution order: explicit `salience_embedder=` argument →
lazy-built sentence-transformers default → cold-start neutral on failure
(never raises). Pay the dep only when you use heuristic mode.

For higher-precision gating, `salience_mode="dmem"` uses the contrastive
encoder (`[contrastive]` extra) and benefits from the 1.4 surprise-score
fix automatically.

---

## Apoptosis — `client.prune()`

On `main` (unreleased), forgetting stops at quarantine. `prune()` retires
facts that a later fact has replaced, hides unhealthy facts from search, and
removes nothing unless you ask.

```python
from datetime import UTC, datetime, timedelta

client = MemoryClient(user_id="scott")
day = timedelta(days=1)
now = datetime.now(UTC)

# A fact that changes: same subject and predicate, a new object.
paris = client.add("Scott lives in Paris.", t_valid=now - 30 * day,
                   metadata={"subject": "Scott", "predicate": "lives_in",
                             "object": "Paris"})
berlin = client.add("Scott lives in Berlin.", t_valid=now - 2 * day,
                    metadata={"subject": "Scott", "predicate": "lives_in",
                              "object": "Berlin"})
# A fact with little to recommend it.
poison = client.add("Scott pays $1M monthly for cloud.",
                    confidence=0.05, importance=0.1)

r = client.prune()
print(r.superseded_count, r.quarantined_count, r.tombstoned_count)   # 1 1 0

# Current search returns the current value only.
assert [h.memory.id for h in client.search("Where does Scott live")] == [berlin.id]
# The old value is kept and found for the time it held.
then = client.search("Where does Scott live", as_of=now - 10 * day)
assert [h.memory.id for h in then] == [paris.id]

# The quarantined fact is hidden from search but still stored.
assert client.search("cloud") == []
assert client.get(poison.id).health_state == "quarantined"
assert client.search("cloud", include_quarantined=True)
```

What each pass does:

1. **Supersede.** A later fact with a different value for the same
   `subject` and `predicate` sets `t_invalid` on the earlier one and records
   its own id in the earlier one's `metadata["_superseded_by"]`.
2. **Recover.** A quarantined fact whose health is back above the threshold
   (for example after correct `cite()`s) returns to healthy.
3. **Quarantine.** A healthy fact whose health falls below the threshold is
   hidden from search and kept.

To remove facts that have sat in quarantine past the window, ask for it:

```python
client.prune(tombstone=True)        # quarantined ≥ 7 days → audit table
client.delete(memory_id)            # remove one fact and its history now
```

In 1.4.0, `prune()` tombstoned quarantined facts automatically after seven
days, and quarantined facts still appeared in search.

---

## Bi-temporal event anchoring — `as_of=`

```python
from datetime import UTC, datetime
from myelin import MemoryClient, Event

client = MemoryClient()
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

# "What was Scott's stance in February?" — only the January meeting is valid.
feb_state = client.search(
    "Scott syncs", as_of=datetime(2026, 2, 1, tzinfo=UTC),
)
```

---

## Session-scoped working memory (1.2)

Opt in with `MYELIN_WORKING_MEMORY=1` (strict-`"1"` parsing — `"true"` /
`"yes"` won't engage).

```python
import os
os.environ["MYELIN_WORKING_MEMORY"] = "1"

client = MemoryClient(user_id="scott")
SID = "convo-2026-06-09"

t1 = client.observe("What did we agree on yesterday?", session_id=SID)
t2 = client.observe("You committed to a weekly review.", session_id=SID)

# observe() is RAM-only. commit() runs the salience gate and persists.
committed = client.commit([t2.turn_id], session_id=SID)

# Tiered search across working → session → long-term.
hits = client.search(
    "weekly review",
    session_id=SID,
    modes=("working", "session", "long_term"),
    top_k=5,
)
```

Observe-without-commit data is **never persisted**. Per-session state is
RAM-only with a TTL sweeper.

---

## Outcome-conditioned health — `cite()` (1.2)

Opt in with `MYELIN_CITE_LEARNING=1`.

```python
import os
os.environ["MYELIN_CITE_LEARNING"] = "1"

client = MemoryClient(user_id="scott")
m_cursor = client.add("Scott uses Cursor for code review.",
                       confidence=0.7, importance=0.7)

# Cite the memory after using it. Correct → lifts apoptosis health.
client.cite(m_cursor.id, answer_correct=True)
client.cite(m_cursor.id, answer_correct=False)   # incorrect → penalty

# When NO stored memory was cite-worthy, record an abstain event.
client.abstain_event(query="What's Scott's middle name?")
```

Contradiction-dominance invariant: a poisoned memory with 100 correct cites
still quarantines on a single contradiction (asserted at `constants.py`
import time).

On `main` (unreleased), cites also nudge ranking: among memories relevant to
a query, those cited mostly in correct answers move up and those cited in
incorrect ones move down. Uncited memories keep their position, feedback
never pulls in a memory the query did not match, and one cite has little
effect (full strength at five). Pass `MemoryClient(cite_ranking=False)` to
use cites for forgetting only. This has not been measured on a benchmark.

---

## HTTP sidecar + `/metrics` (1.3 + 1.4)

Wraps the client as JSON-over-HTTP, OpenClaw-compatible. The 1.4 addition
is `GET /metrics` for vacuous-PASS preflight guards.

```bash
myelin-serve --host 127.0.0.1 --port 8765   # console script (1.3.1+)
# or: python -m myelin.serve --host 127.0.0.1 --port 8765
```

```bash
curl http://127.0.0.1:8765/health    # {"ok": true, "service": "myelin-sidecar"}
curl http://127.0.0.1:8765/metrics   # {"tool_invocations": 0, "reset_count": 0}

curl -X POST http://127.0.0.1:8765/tools/memory_store \
  -H 'content-type: application/json' \
  -d '{"content": "Scott uses Cursor.", "user_id": "scott"}'

curl http://127.0.0.1:8765/metrics   # {"tool_invocations": 1, "reset_count": 0}
```

Loopback only by default. No auth — wrap behind a reverse proxy if exposed.
Bench harnesses can assert `tool_invocations > 0` before treating an
"all-PASS" result as non-vacuous.

Endpoints: `/tools/memory_{search,get,store,update,forget,consolidate,prune}`,
`/hooks/{before_agent_start,agent_end}`, `/admin/reset`, `/health`,
`/metrics` (1.4).

---

## API surface

```python
# Core CRUD
client.add(content, *, user_id=None, metadata=None, event=None,
           confidence=0.5, importance=0.5, outcome=None,
           t_valid=None, index_text=None) -> Memory                       # outcome= 1.4
client.add_many(items, *, user_id=None) -> list[Memory]   # text or dicts of add() arguments
client.add_conversation(turns, *, user_id=None, when=None, llm=None, people=None) -> list[Memory]
client.set_index_text(memory_id, index_text) -> bool      # extra text to be found by
client.search(query, *, user_id=None, top_k=10, as_of=None,
              session_id=None, modes=("working","session","long_term")) -> list[SearchResult]
client.recall(query, *, token_budget=6000, max_results=100,
              max_result_tokens=300, whole_results=3, user_id=None,
              as_of=None, session_id=None,
              modes=("long_term",)) -> list[SearchResult]
format_context(results) -> str        # from myelin import format_context
build_answer_prompt(history, question, *, now=None,
                    style="reasoned") -> str
client.get(memory_id) -> Memory | None
client.update(memory_id, *, content=None, metadata=None) -> Memory
client.delete(memory_id) -> bool

# Consolidation + apoptosis
client.consolidate(*, user_id=None, max_samples=32,
                   retriever=None) -> ConsolidationReport                  # retriever= 1.4
client.prune(*, user_id=None, now=None, caspase_window_days=7) -> ApoptosisReport

# Outcome-conditioned learning (1.2 — opt in via MYELIN_CITE_LEARNING=1)
client.cite(memory_id, *, answer_correct: bool) -> None
client.abstain_event(query, *, at=None) -> None
client.cite_telemetry() -> dict

# Session-scoped working memory (1.2 — opt in via MYELIN_WORKING_MEMORY=1)
client.observe(content, *, session_id, metadata=None, user_id=None) -> ObservedTurn
client.commit(turn_ids, *, session_id) -> list[Memory]
client.attend(memory_id, *, session_id) -> None
client.session(session_id) -> SessionHandle
client.search_telemetry() -> SearchTelemetry

# Constructor (1.4 additions noted)
MemoryClient(
    user_id=None, backend=None, llm_client=None,
    salience_mode="off",                # "off" | "heuristic" | "dmem"
    salience_tau=0.40,
    salience_require_reward=False,
    salience_embedder=None,             # 1.4 — override default embedder
    salience_neighbor_window=32,        # 1.4 — rolling per-scope window
    dmem_gate=None,
    cite_learning=None, working_memory=None,
)
```

```python
# Agent orchestration (1.3)
from myelin.agent import OrchestratedAgent
agent = OrchestratedAgent(client, model="openai/gpt-4o" | "google/gemini-3.5-flash")
agent.classify(question) -> "factual" | "synthesis"
agent.answer(question, *, options=None, user_id=None) -> str
agent.last_usage  # {"input_tokens", "output_tokens", "cost_usd", ...}
```

---

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

All tests are stdlib + pytest only — no internet, no GPU, no LLM calls.
Dream-synthesis tests use trivial mock LLMs. The suite runs on every pull
request (Python 3.11 and 3.13).

---

## Project

- **Source / Issues:** https://github.com/myelinagent/Myelin-AI
- **Changelog:** [`CHANGELOG.md`](https://github.com/myelinagent/Myelin-AI/blob/main/CHANGELOG.md)

## License

MIT. See [`LICENSE`](https://github.com/myelinagent/Myelin-AI/blob/main/LICENSE).
