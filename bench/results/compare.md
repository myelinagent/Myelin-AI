# LongHAUL's static arm for Mem0, Graphiti and Letta

`python -m bench.compare`, 2026-11-16. The static arm of
[LongHAUL](lifetime.md) (store each session on its date, ask each
question some sessions after its evidence, forget nothing, feed nothing
back) run through three other open-source memory systems, with
Myelin-AI's static arm beside them on the same conversations and the
same question schedules.

**Scope.** The first three LoCoMo conversations (`conv-26`, `conv-30`,
`conv-41`): 1,451 turns, 382 questions with evidence, three schedules
(seeds 0 to 2, questions asked up to 6 sessions later). The full ten
were not run: the budget for model calls was $10 and writing is slow
(see cost below). ± is the spread between schedules, not between
conversations; one conversation differs from another by more than that
(Myelin-AI's top 5 is 43.0 on the first conversation and 47.6 on the
three).

**The measure** is the one in `lifetime.md`: how often all of a
question's evidence turns are in the top k. These systems return facts,
graph edges or passages, not turns, so an evidence turn counts as
retrieved when a returned item derives from it. k counts returned
items. How each system's items were traced back to turns is under
"What each system was" and matters as much as the numbers.

## Same embedder as Myelin-AI's runs

Every system ranks with `minishlab/potion-retrieval-32M`, the static
embedder Myelin-AI's benchmark runs use, served to all of them from one
local endpoint. Extraction model for the systems that have one:
`openai/gpt-4o-mini`.

| | All evidence in top 5 | top 10 | top 20 | top 10, asked 3+ sessions later | Evidence all has a stored item |
|---|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 47.6 ± 1.1 | 57.9 ± 0.7 | 67.5 ± 0.8 | 59.3 ± 0.8 | 100.0 |
| Mem0 OSS, `infer=True`, one `add` per turn; recorded mapping | 48.7 ± 0.5 | 62.6 ± 0.5 | 71.6 ± 0.7 | 63.8 ± 1.7 | 99.2 |
| the same run, also crediting a context turn the memory restates | 60.0 ± 0.2 | 68.2 ± 0.6 | 75.5 ± 0.7 | | |
| Mem0 OSS, `infer=True`, one `add` per session; turn guessed from wording | 58.7 ± 0.7 | 64.6 ± 0.5 | 68.8 ± 0.3 | 65.0 ± 1.5 | 78.8 |
| Graphiti, `add_episode` per turn, Kuzu; recorded mapping | 48.4 ± 0.7 | 54.5 ± 0.2 | 58.9 ± 0.3 | 55.4 ± 1.6 | 100.0 |
| the same run, also crediting a context turn the edge restates | 49.7 ± 0.8 | 55.1 ± 0.2 | 59.3 ± 0.4 | | |
| Letta archival passages, direct insert, no agent, no model | 31.1 ± 0.3 | 39.8 ± 0.0 | 47.6 ± 0.3 | 38.9 ± 1.3 | 100.0 |

"Evidence all has a stored item" is the share of questions for which
every evidence turn produced at least one item. Below 100, the rest
cannot be counted however the system ranks.

For context, `bench.lifetime --limit 3` on the same conversations:
Myelin-AI's feedback arm (no model; evidence labels as the feedback)
53.3 ± 0.7, 63.4 ± 1.2, 67.4 ± 0.8, and decay + feedback 53.8, 63.7,
68.0. The other systems have no such arms (`lifetime.md`).

### What writing cost

| | Items stored | Model calls to write | Prompt tokens | Completion tokens | Cost to write | Writing, wall clock | per turn | Search, mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 1,451 | 0 | 0 | 0 | $0.00 | 2.3 s | 0.002 s | 6 ms |
| Mem0, one `add` per turn | 2,590 | 1,451 | 12,177,898 | 129,790 | $1.14 | 63 min | 2.6 s | 130 ms |
| Mem0, one `add` per session | 653 | 70 | 637,900 | 31,997 | $0.08 | 8.6 min | 0.36 s | 120 ms |
| Graphiti | 1,287 | 4,839 | 10,317,710 | 216,068 | $1.68 | 124 min | 5.1 s | 55 ms |
| Letta archival passages | 1,451 | 0 | 0 | 0 | $0.00 | 15 min | 0.64 s | 470 ms |

For 1,451 turns. Cost is what OpenRouter billed for `gpt-4o-mini`
($0.15 and $0.60 per million tokens, less for cached prompt tokens).
Mem0 sends a prompt of about 8,400 tokens with every `add`; Graphiti
makes 3.3 calls per turn. No system called a model to search. Wall
clock is from one machine with up to eleven runs sharing four cores and
one Letta server, so read it to within a factor of two: the first
conversation alone took Letta 0.21 s per turn to write and 195 ms to
search.

By conversation, top 5 and top 10 (static embedder, mean of three
schedules):

| | `conv-26`, 149 questions | `conv-30`, 81 | `conv-41`, 152 |
|---|---:|---:|---:|
| Myelin-AI, static arm | 43.0, 52.3 | 53.5, 64.6 | 49.1, 59.6 |
| Mem0 per turn; recorded mapping | 45.6, 60.4 | 53.5, 64.2 | 49.1, 63.8 |
| also crediting a restated context turn | 57.0, 64.2 | 60.1, 69.1 | 62.9, 71.7 |
| Mem0 per session; turn guessed | 49.4, 54.4 | 61.7, 67.9 | 66.2, 72.8 |
| Graphiti; recorded mapping | 46.1, 51.0 | 45.7, 55.1 | 52.2, 57.7 |
| Letta archival passages | 25.1, 31.5 | 37.9, 48.1 | 33.3, 43.4 |

## With the others' default embedder

`minishlab/potion-retrieval-32M` is Myelin-AI's choice, not theirs.
Mem0 and Graphiti default to OpenAI's `text-embedding-3-small`, and a
Letta agent is normally given a model of that kind. The same runs with
`openai/text-embedding-3-small` for everyone, Myelin-AI included
(`MemoryClient(embedder=...)` takes any embedder; this is not a
configuration Myelin-AI ships):

| Three conversations, 382 questions | top 5 | top 10 | top 20 | top 10, asked 3+ sessions later |
|---|---:|---:|---:|---:|
| Myelin-AI, static arm | 62.8 ± 1.1 | 73.5 ± 0.3 | 80.8 ± 0.5 | 74.1 ± 1.5 |
| Mem0, one `add` per turn; recorded mapping | 57.5 ± 0.3 | 72.3 ± 0.5 | 78.8 ± 0.3 | 72.3 ± 1.8 |
| the same run, also crediting a restated context turn | 65.1 ± 0.2 | 75.7 ± 0.5 | 81.3 ± 0.3 | |
| Letta archival passages | 62.7 ± 0.3 | 73.2 ± 0.3 | 79.7 ± 1.0 | 74.7 ± 2.4 |

Graphiti was run with this embedder on the first conversation only
(149 questions), to keep within the budget:

| First conversation, 149 questions | top 5 | top 10 | top 20 | the static embedder: top 5 | top 10 | top 20 |
|---|---:|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 58.6 | 67.1 | 76.5 | 43.0 | 52.3 | 61.7 |
| Mem0, one `add` per turn; recorded mapping | 54.8 | 68.5 | 75.8 | 45.6 | 60.4 | 67.6 |
| the same run, also crediting a restated context turn | 61.5 | 71.1 | 77.9 | 57.0 | 64.2 | 71.1 |
| Graphiti; recorded mapping (restated context adds nothing here) | 44.1 | 50.8 | 59.1 | 46.1 | 51.0 | 56.6 |
| Letta archival passages | 57.7 | 70.5 | 76.7 | 25.1 | 31.5 | 38.7 |

Embedding 1,451 turns and the questions cost under half a cent. Mem0's
write cost was the same $1.14; Graphiti's first conversation $0.50.

## What each system was

Python 3.12, each system in its own virtualenv, none of them a
dependency of this project. All model and embedding calls went through
`bench.gateway`, which counted them.

**Myelin-AI.** This repository at the commit before this file, through
`bench.compare`'s own adapter: one memory per turn, a session per
`add_many`, default `search`, a fresh store per schedule. That is what
`bench.lifetime` does, and `bench.lifetime --limit 3` gives 47.5, 58.0,
67.5 against 47.6, 57.9, 67.5 here. They are not identical because
Myelin-AI's search is not repeatable to the last question: two runs of
`bench.lifetime` itself differ by one or two questions per schedule.

**Mem0, open source.** `mem0ai` 2.2.1 with its `nlp` extra, local
Qdrant (`qdrant-client` 1.20.0, on disk), BM25 through `fastembed`
0.9.1 and entities through `spacy` 3.8.18 with `en_core_web_sm`, so its
hybrid search had every part it documents. Telemetry off.
- Write: `Memory.add(messages, user_id=..., metadata=...)` with the
  default `infer=True`: one model call that extracts facts, with the
  last ten messages and the ten most similar memories as context. The
  first speaker is sent as `user` and the second as `assistant`, each
  message beginning with the speaker's name, as in the multi-speaker
  example in Mem0's own prompt.
- Search: `Memory.search(question, filters={"user_id": ...}, top_k=20)`,
  default threshold, no reranker.
- Mapping, one `add` per turn: the turn's id is passed as `metadata`.
  Mem0 stores it on each memory that call creates and returns it from
  `search`. So each memory is tied to exactly one `add` call by Mem0
  itself; the ids were also logged per call.
- Its weakness, found by reading results: Mem0 shows the model the last
  ten messages, and `gpt-4o-mini` often writes a memory from one of
  those while adding a later message. The memory is then stored under
  the later turn. In five of ten questions read by hand, a correct
  memory near the top was filed under a turn one to nine messages after
  the one that said it. The recorded mapping counts that as a miss. The
  second row credits such a memory also to the one of the ten preceding
  turns that holds more of its words than its own turn does
  (`bench.compare.with_context`). Every case of it read by hand was
  right, but it is a rule about wording, so the truth is between the
  two rows and nearer the second.
- The same habit fills the store with repeats: 2,590 memories from
  1,451 turns, and 20% of them are near-copies of a recent one. They
  take up places in the top k.
- A turn Mem0 extracts nothing from, or whose fact it judges already
  stored, has no memory, and an earlier memory is not credited to it.
  0.8% of questions had such an evidence turn.
- One `add` per session is the cheaper way to use it (70 calls, not
  1,451). Mem0 then does not say which message a memory came from. The
  session is recorded exactly; the turn is the one in that session
  sharing the most telling words with the memory (`closest_turns`),
  or all that tie (11% of memories). That mapping is a guess. With it,
  21% of questions have an evidence turn nothing was attributed to, and
  the questions that remain are answered at a high rate.
- Mem0's open-source `add` takes no timestamp, so it resolved "last
  week" against the day of the run. This measure does not see that; an
  answering model would.

**Graphiti.** `graphiti-core` 0.31.2 on the embedded Kuzu driver
(`kuzu` 0.11.3), because there was no Docker for Neo4j.
- Write: `Graphiti.add_episode(source=EpisodeType.message)`, one
  message per episode, the session date as `reference_time`. The model
  client was `OpenAIGenericClient`, the one for OpenAI-compatible
  endpoints.
- Search: `Graphiti.search(question, num_results=20)`, the default,
  which returns edges (facts) by hybrid search with reciprocal rank
  fusion.
- Mapping: each episode is named after its turn. Every edge lists the
  episodes it was extracted from, and Graphiti adds an episode to an
  existing edge when a later message states the same fact. An edge
  derives from every turn in that list: 12% of edges had more than one,
  the largest 22 (a pleasantry repeated through a conversation).
- Weaknesses. The Kuzu driver is deprecated upstream and needed two
  workarounds: its full-text indexes are not built by
  `Graphiti.build_indices_and_constraints` in this version, so the
  adapter calls the driver's own index setup, and passing a `group_id`
  fails, so the default group is used. Neo4j is the documented backend
  and may rank differently. As a check that Kuzu's hybrid search was
  not simply broken, the same edges ranked by plain cosine similarity
  to the question give 49.0, 54.5, 63.4, against Graphiti's own 48.4,
  54.5, 58.9.
- What limits it here is what an edge keeps. It stored fewer facts than
  turns (1,287 for 1,451), each a short relation between two entities,
  and the detail a question asks for is often not in the edge that
  derives from the evidence turn. A better embedder did not help it.

**Letta.** Two different things, and only the first is in the tables.
- *Archival passages, direct insert.* `letta` 0.16.8 server with
  `letta-client` 1.11.0, on PostgreSQL 16.2 with `pgvector` 0.6.2. Each
  turn is written with `agents.passages.create(text=..., tags=[turn
  id])` and searched with `agents.passages.search(query, top_k=20)`.
  No agent reads anything and no model is called. The mapping is the
  tag, returned by the search, one passage per turn. This is Letta's
  vector store over the raw turns: on all 149 questions of the first
  conversation its top 20 was identical, item for item, to ranking the
  turns by cosine similarity with the same embedder. It is not what
  Letta does with a conversation.
- *A Letta agent*, the documented way to use it, was tried and is not
  rankable. The default agent (`letta_v1_agent`) has three memory
  tools: `memory_insert` and `memory_replace`, which edit blocks that
  are always in its prompt, and `conversation_search` over the message
  history, which on a self-hosted server is an SQL text match. There is
  no store to take a top k from. Sent the first eight sessions of the
  first conversation one message at a time (174 turns), the agent
  replied to every message and called no memory tool at all: its blocks
  were unchanged at the end. With `archival_memory_insert` and
  `archival_memory_search` attached as well (listed as deprecated in
  this version), the same: no passage written in 174 turns. Each run
  made 175 to 180 model calls and used 2.0 to 2.1 million prompt
  tokens, $0.16 and $0.17, because the whole history is sent with each
  message (20,000 tokens per call by the end). So in its default mode
  Letta kept this conversation only as message history.
- Set-up, since it took most of the time spent on Letta. The `letta`
  package on PyPI is now a different product (Letta Code) from version
  0.32; 0.16.8 is the last server release, and it needs its
  dependencies resolved as of its release date (`uv pip install
  --exclude-newer 2026-05-20 "letta[server,sqlite]==0.16.8" asyncpg
  pgvector pg8000 psycopg2-binary`). It no longer runs on SQLite: the
  engine is always PostgreSQL. PostgreSQL came from the `pgserver`
  wheel, run as an unprivileged user on a TCP port. The wheel ships no
  migrations, so the tables were made with
  `letta.orm.Base.metadata.create_all`, plus `CREATE EXTENSION vector`
  and a sequence as the default of `messages.sequence_id`. Then
  `LETTA_PG_URI=... letta server`.

## Checks on the harness

- **Myelin-AI under this harness equals `bench.lifetime`** (above), and
  an offline test asserts the same schedule and the same score on a
  fixture.
- **Read by hand**, with the evidence turns beside what was returned:
  16 questions for Mem0, 10 for Graphiti, 6 for Letta passages, 4 for
  Mem0 per session. The recorded mapping credited the right turn every
  time. The one systematic error was the Mem0 one described above, and
  it runs against Mem0. Graphiti's lists show the reverse of Mem0's
  repeats: a few generic edges (a greeting, a thank-you) return for
  many unrelated questions and take places in the top 10.
- **Letta passages against plain cosine**: identical, as above.
- **The evidence labels have errors.** In the questions read, one
  listed a turn with no bearing on the answer and one listed the
  question that prompted the answer, not the answer. These cost every
  system the same.

## What this shows

- **With the same static embedder, Mem0 retrieves more of the evidence
  than Myelin-AI's static arm**: about level in the top 5 by the
  recorded mapping (48.7 against 47.6), ahead by 5 in the top 10 and 4
  in the top 20, and ahead by 8 to 12 at every depth once a memory is
  credited to the context turn it restates. Extracted facts name the
  speaker and resolve references, which a question about a named person
  matches better than the raw message does. It pays for that at write
  time: one model call, 8,400 prompt tokens and 2.6 seconds per turn,
  where Myelin-AI makes no call.
- **Graphiti is level with Myelin-AI in the top 5 and behind at 10 and
  20** (54.5 against 57.9, 58.9 against 67.5), at 3.3 model calls and 5
  seconds per turn.
- **The embedder matters more than the system.** Changing only the
  embedder moved Myelin-AI by 15 points in the top 5 and Letta's
  passages by 32. With `text-embedding-3-small`, plain vector search
  over raw turns (Letta's passages) equals Myelin-AI's hybrid search
  (62.7 against 62.8), and both are level with Mem0 between its two
  readings (57.5 to 65.1). With the static embedder, Myelin-AI's
  lexical channel is what keeps it 16 points above vector search alone.
  So the first table is partly a comparison of how well each system
  copes with a weak embedder, and Myelin-AI's default embedder is its
  weakest part on this data.
- **Myelin-AI's feedback arm, which needs no model, falls inside
  Mem0's range** on the same conversations with the static embedder
  (53.3 and 63.4 in the top 5 and 10; Mem0 48.7 to 60.0 and 62.6 to
  68.2). That is not the static-arm comparison asked for here, and its
  feedback is the evidence labels.

## What it does not show

- **Answers.** Evidence in the top k is not a correct answer. A fact
  written by a model may serve an answering model better than the raw
  turn, or worse (Mem0's dates here are wrong). Not measured.
- **Ten conversations.** Three were run, and they differ (table
  below). On each, Mem0 by its second reading was ahead of Myelin-AI's
  static arm and Letta's passages were last. By the recorded mapping
  Mem0 and Myelin-AI tied in the top 5 on two of the three, and
  Graphiti was ahead of Myelin-AI in the top 5 on two and behind on
  one.
- **Letta as an agent**, for the reason given.
- **Zep's hosted service, Mem0's hosted platform.** Only the open
  source.
- **The other three arms of LongHAUL.** Decay and feedback do not exist
  in these systems in a form a simulated clock can drive.

## Unfair, and in which direction

Against the other systems:
- The static embedder is not their default (second set of tables).
- `gpt-4o-mini` is not their default extraction model either: Mem0's is
  `gpt-5-mini`, Graphiti's `gpt-6-luna`. A stronger model may extract
  more, and may restate context less.
- One message per write is this harness's choice, made so that each
  item has one known source. It is what makes Mem0 restate context.
- The recorded mapping undercounts Mem0 (first Mem0 row).
- Graphiti ran on a deprecated backend with two workarounds.
- Letta is represented by its vector store, not its agent.

Against Myelin-AI:
- An item that merges several turns covers them all with one place in
  the top k (12% of Graphiti's edges; Myelin-AI's items are single
  turns).
- The second Mem0 row and the per-session row rest on wording, and
  could credit a turn the memory was not written from.
- Myelin-AI's optional write-time fact extraction (`myelin.enrich`,
  `bench.locomo run --facts`), the nearest thing it has to what Mem0
  does, was off, as it is in LongHAUL.

Either way:
- Mem0 with one `add` per session leaves 21% of questions uncountable,
  and it is not known how many of those are the mapping's failures and
  how many Mem0's.

## Not run, and why

- **All ten conversations**: the other 4,431 turns would cost about
  $3.50 and 3 hours for Mem0 and about $5 and 6 hours for Graphiti at
  the rates above. Not within what was allowed.
- **Graphiti on Neo4j**: no Docker.
- **Graphiti with `text-embedding-3-small` on three conversations**:
  budget; the first conversation is reported.
- **Letta agent as a ranked system**: it wrote nothing to rank.

## Reproducing

```bash
export MYELIN_BENCH_BASE_URL=https://openrouter.ai/api/v1
python -m bench.gateway --port 8765 --log usage.jsonl &

# one virtualenv with: "mem0ai[nlp]==2.2.1" fastembed en_core_web_sm
#                      "graphiti-core[kuzu]==0.31.2" model2vec numpy
python -m bench.compare --system myelin mem0 mem0-session graphiti \
    --data locomo10.json --out bench/runs/compare --limit 3 --max-minutes 25
# exit code 3 means out of time: run the same command again.

# another with letta==0.16.8 (see Letta above) and its server on port 8283
python -m bench.compare --system letta-passages \
    --data locomo10.json --out bench/runs/compare --limit 3

python -m bench.compare --report --system myelin mem0 mem0-session graphiti letta-passages \
    --data locomo10.json --out bench/runs/compare --limit 3
```

Add `--embedder openai/text-embedding-3-small` and a new `--out` for the
second set of tables. `--skip N` leaves out the first N conversations,
so conversations can be written by separate processes at once. Model
calls for everything here, probes and a repeated run included: $4.99.
