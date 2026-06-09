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

```python
from datetime import UTC, datetime, timedelta

client = MemoryClient(user_id="scott")
client.add("Scott prefers tea.", confidence=0.9, importance=0.9)
client.add("Scott uses Cursor.", confidence=0.9, importance=0.8)
poison = client.add(
    "Scott pays $1M monthly for cloud.",   # absurd; low signal
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

assert client.get(poison.id) is None    # gone from search
```

Tombstoned facts are invisible to `search()` but persist as an audit trail.

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
           confidence=0.5, importance=0.5, outcome=None) -> Memory        # outcome= 1.4
client.search(query, *, user_id=None, top_k=10, as_of=None,
              session_id=None, modes=("working","session","long_term")) -> list[SearchResult]
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
client.observe(content, *, session_id, metadata=None) -> ObservedTurn
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
Dream-synthesis tests use trivial mock LLMs.

---

## Project

- **Source / Issues:** https://github.com/myelinagent/Myelin-AI
- **Changelog:** [`CHANGELOG.md`](https://github.com/myelinagent/Myelin-AI/blob/main/CHANGELOG.md)

## License

MIT. See [`LICENSE`](https://github.com/myelinagent/Myelin-AI/blob/main/LICENSE).
