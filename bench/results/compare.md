# LongHAUL's static arm for Mem0, Graphiti and Letta

`python -m bench.compare`, 2026-11-16. The static arm of
[LongHAUL](lifetime.md) (store each session on its date, ask each
question some sessions after its evidence, forget nothing, feed nothing
back) run through three other open-source memory systems, with
Myelin-AI's static arm beside them on the same conversations and the
same question schedules.

**Scope.** All ten LoCoMo conversations: 5,882 turns, 1,531 questions
with evidence, three schedules (seeds 0 to 2, questions asked up to 6
sessions later). Graphiti finished eight of the ten (4,513 turns, 1,203
questions; `conv-43` and `conv-47` are missing) before the money for
model calls ran out, so it has its own table with every other system on
the same eight. The embedder-swap tables further down are still from
the first three conversations. ± is the spread between schedules, not
between conversations; conversations differ by far more (table below).

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

All ten conversations, 1,531 questions:

| | All evidence in top 5 | top 10 | top 20 | top 10, asked 3+ sessions later | Evidence all has a stored item |
|---|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 46.4 ± 0.3 | 55.8 ± 0.2 | 63.9 ± 0.2 | 57.8 ± 1.4 | 100.0 |
| Mem0 OSS, `infer=True`, one `add` per turn; recorded mapping | 47.2 ± 0.2 | 60.2 ± 0.2 | 68.8 ± 0.2 | 62.3 ± 0.7 | 99.0 |
| the same run, also crediting a context turn the memory restates | 57.9 ± 0.2 | 67.1 ± 0.2 | 73.9 ± 0.2 | | |
| Mem0 OSS, `infer=True`, one `add` per session; turn guessed from wording | 53.6 ± 0.3 | 58.5 ± 0.2 | 62.4 ± 0.1 | 59.9 ± 0.7 | 73.4 |
| Letta archival passages, direct insert, no agent, no model | 29.5 ± 0.2 | 36.9 ± 0.2 | 45.7 ± 0.4 | 37.5 ± 2.0 | 100.0 |

The eight conversations Graphiti finished, 1,203 questions:

| | All evidence in top 5 | top 10 | top 20 | top 10, asked 3+ sessions later | Evidence all has a stored item |
|---|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 45.7 ± 0.4 | 56.0 ± 0.2 | 64.6 ± 0.1 | 58.1 ± 1.3 | 100.0 |
| Graphiti, `add_episode` per turn, Kuzu; recorded mapping | 44.5 ± 0.3 | 51.5 ± 0.1 | 57.4 ± 0.2 | 53.3 ± 1.2 | 99.4 |
| the same run, also crediting a context turn the edge restates | 45.7 ± 0.2 | 52.5 ± 0.2 | 58.5 ± 0.2 | | |
| Mem0, one `add` per turn; recorded mapping | 48.5 ± 0.2 | 61.0 ± 0.3 | 68.8 ± 0.3 | 63.1 ± 0.9 | 98.8 |
| the same run, also crediting a restated context turn | 58.7 ± 0.2 | 67.3 ± 0.4 | 73.4 ± 0.2 | | |
| Mem0, one `add` per session; turn guessed from wording | 53.7 ± 0.4 | 58.7 ± 0.3 | 62.5 ± 0.1 | 59.8 ± 0.7 | 74.1 |
| Letta archival passages | 27.8 ± 0.2 | 35.3 ± 0.2 | 44.0 ± 0.5 | 36.2 ± 1.6 | 100.0 |

"Evidence all has a stored item" is the share of questions for which
every evidence turn produced at least one item. Below 100, the rest
cannot be counted however the system ranks.

For context, `bench.lifetime` on all ten: Myelin-AI's feedback arm (no
model; evidence labels as the feedback) 52.8 ± 0.2, 60.6 ± 0.4,
63.9 ± 0.2, and decay + feedback 52.8, 60.9, 64.8. The other systems
have no such arms (`lifetime.md`).

### What writing cost

| | Turns | Items stored | Model calls to write | Prompt tokens | Completion tokens | Cost to write | Writing, wall clock | per turn | Search, mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Myelin-AI, static arm | 5,882 | 5,882 | 0 | 0 | 0 | $0.00 | 20 s | 0.003 s | 6 ms |
| Mem0, one `add` per turn | 5,882 | 10,236 | 5,882 | 49,271,627 | 499,097 | $4.58 | 216 min | 2.2 s | 137 ms |
| Mem0, one `add` per session | 5,882 | 2,445 | 272 | 2,479,776 | 120,849 | $0.30 | 30 min | 0.31 s | 121 ms |
| Graphiti, eight conversations | 4,513 | 4,177 | 15,103 | 31,982,361 | 637,571 | $5.18 | 358 min | 4.8 s | 56 ms |
| Letta archival passages | 5,882 | 5,882 | 0 | 0 | 0 | $0.00 | 30 min | 0.31 s | 259 ms |

Cost is what OpenRouter billed for `gpt-4o-mini` ($0.15 and $0.60 per
million tokens, less for cached prompt tokens). Mem0 sends a prompt of
about 8,400 tokens with every `add`; Graphiti makes 3.3 calls per turn.
No system called a model to search. Wall clock is the sum over
conversations, measured on one machine with up to eleven runs sharing
four cores, so read it to within a factor of two.

By conversation, top 5 and top 10 (static embedder, mean of three
schedules):

| | Questions | Myelin-AI, static arm | Mem0 per turn; recorded mapping | also crediting a restated context turn | Mem0 per session; turn guessed | Graphiti; recorded mapping | Letta archival passages |
|---|---:|---:|---:|---:|---:|---:|---:|
| `conv-26` | 149 | 43.0, 52.3 | 45.6, 60.4 | 57.0, 64.2 | 49.4, 54.4 | 46.1, 51.0 | 25.1, 31.5 |
| `conv-30` | 81 | 53.5, 64.6 | 53.5, 64.2 | 60.1, 69.1 | 61.7, 67.9 | 45.7, 55.1 | 37.9, 48.1 |
| `conv-41` | 152 | 49.1, 59.6 | 49.1, 63.8 | 62.9, 71.7 | 66.2, 72.8 | 52.2, 57.7 | 33.3, 43.4 |
| `conv-42` | 199 | 44.7, 57.8 | 49.7, 60.8 | 57.1, 66.7 | 55.9, 61.0 | 44.6, 50.8 | 25.0, 33.5 |
| `conv-43` | 178 | 50.2, 54.9 | 39.3, 54.7 | 53.4, 65.4 | 54.1, 57.1 | not run | 36.9, 45.1 |
| `conv-44` | 123 | 44.7, 55.3 | 42.5, 56.1 | 59.9, 63.4 | 41.7, 45.0 | 35.0, 44.4 | 37.9, 44.2 |
| `conv-47` | 150 | 47.3, 54.9 | 46.9, 60.4 | 57.3, 67.1 | 52.2, 58.2 | not run | 34.0, 40.2 |
| `conv-48` | 191 | 42.2, 52.9 | 50.3, 64.6 | 60.0, 72.9 | 55.1, 59.9 | 44.9, 55.1 | 15.5, 21.5 |
| `conv-49` | 153 | 49.0, 55.6 | 54.0, 62.1 | 58.2, 64.7 | 51.0, 51.9 | 42.9, 46.4 | 28.3, 34.0 |
| `conv-50` | 155 | 43.9, 54.0 | 43.2, 55.7 | 55.3, 64.7 | 48.8, 57.8 | 43.7, 50.8 | 30.3, 38.1 |

## With the others' default embedder (three conversations)

These tables were not extended: they are from the first three
conversations (`conv-26`, `conv-30`, `conv-41`; 1,451 turns), where the
static embedder gives Myelin-AI 47.6, 57.9, 67.5, Mem0 per turn 48.7,
62.6, 71.6 (60.0, 68.2, 75.5 by the second reading) and Letta's
passages 31.1, 39.8, 47.6.

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
write cost for the three was $1.14, as with the static embedder;
Graphiti's first conversation $0.50.

## What each system was

Python 3.12, each system in its own virtualenv, none of them a
dependency of this project. All model and embedding calls went through
`bench.gateway`, which counted them.

**Myelin-AI.** This repository at the commit before this file, through
`bench.compare`'s own adapter: one memory per turn, a session per
`add_many`, default `search`, a fresh store per schedule. That is what
`bench.lifetime` does, and `bench.lifetime` gives 46.4, 55.7, 63.9
against 46.4, 55.8, 63.9 here. They are not identical because
Myelin-AI's search is not repeatable to the last question: two runs of
`bench.lifetime` itself differ by one or two questions per schedule.
The cause is the tie-break. Results with equal scores are ordered by
memory id, and ids are random (`uuid4`), so ties fall differently each
run. With `uuid.uuid4` replaced by a seeded generator, two runs of the
static arm gave identical outcomes for every question.

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
- The same habit fills the store with repeats: 10,236 memories from
  5,882 turns, and 21% of them are near-copies of a recent one. They
  take up places in the top k.
- A turn Mem0 extracts nothing from, or whose fact it judges already
  stored, has no memory, and an earlier memory is not credited to it.
  1.0% of questions had such an evidence turn.
- One `add` per session is the cheaper way to use it (272 calls, not
  5,882). Mem0 then does not say which message a memory came from. The
  session is recorded exactly; the turn is the one in that session
  sharing the most telling words with the memory (`closest_turns`),
  or all that tie (11% of memories). That mapping is a guess. With it,
  27% of questions have an evidence turn nothing was attributed to, and
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
  derives from every turn in that list: 11% of edges had more than one,
  the largest 22 (a pleasantry repeated through a conversation).
- Weaknesses. The Kuzu driver is deprecated upstream and needed two
  workarounds: its full-text indexes are not built by
  `Graphiti.build_indices_and_constraints` in this version, so the
  adapter calls the driver's own index setup, and passing a `group_id`
  fails, so the default group is used. Neo4j is the documented backend
  and may rank differently. As a check that Kuzu's hybrid search was
  not simply broken, the same edges ranked by plain cosine similarity
  to the question give 44.7, 51.8, 58.7, against Graphiti's own 44.5,
  51.5, 57.4.
- A model reply Graphiti could not parse made `add_episode` raise once
  in 4,513 turns. Nothing had been saved, so the adapter makes the
  write again (up to three times).
- What limits it here is what an edge keeps. It stored fewer facts than
  turns (4,177 for 4,513), each a short relation between two entities,
  and the detail a question asks for is often not in the edge that
  derives from the evidence turn. A better embedder did not help it on
  the one conversation where that was tried.

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
  recorded mapping (47.2 against 46.4), ahead by 4 in the top 10 and 5
  in the top 20, and ahead by 10 to 12 at every depth once a memory is
  credited to the context turn it restates. Extracted facts name the
  speaker and resolve references, which a question about a named person
  matches better than the raw message does. It pays for that at write
  time: one model call, 8,400 prompt tokens and about 2 seconds per
  turn, where Myelin-AI makes no call.
- **Graphiti is behind Myelin-AI at every depth** on the eight
  conversations it finished: by 1 in the top 5 (44.5 against 45.7), 4.5
  in the top 10 and 7 in the top 20, at 3.3 model calls and 5 seconds
  per turn. On the first three conversations alone it had looked level
  in the top 5.
- **Letta's vector store over raw turns is 17 points behind** in the
  top 5 with this embedder (29.5 against 46.4).
- **The embedder matters more than the system** (three conversations).
  Changing only the
  embedder moved Myelin-AI by 15 points in the top 5 and Letta's
  passages by 32. With `text-embedding-3-small`, plain vector search
  over raw turns (Letta's passages) equals Myelin-AI's hybrid search
  (62.7 against 62.8), and both are level with Mem0 between its two
  readings (57.5 to 65.1). With the static embedder, Myelin-AI's
  lexical channel is what keeps it 16 to 17 points above vector search
  alone.
  So the first table is partly a comparison of how well each system
  copes with a weak embedder, and Myelin-AI's default embedder is its
  weakest part on this data.
- **Myelin-AI's feedback arm, which needs no model, falls inside
  Mem0's range** on the ten conversations with the static embedder
  (52.8 and 60.6 in the top 5 and 10; Mem0 47.2 to 57.9 and 60.2 to
  67.1). That is not the static-arm comparison asked for here, and its
  feedback is the evidence labels.

## What it does not show

- **Answers.** Evidence in the top k is not a correct answer. A fact
  written by a model may serve an answering model better than the raw
  turn, or worse (Mem0's dates here are wrong). Not measured.
- **The same order everywhere.** Conversations differ (table above).
  Mem0 by its second reading was ahead of Myelin-AI's static arm on all
  ten. By the recorded mapping it was ahead in the top 5 on four, level
  on two and behind on four. Graphiti was ahead of Myelin-AI in the top
  5 on three of its eight, level on two and behind on three. Letta's
  passages were last on every conversation but one.
- **Graphiti on `conv-43` and `conv-47`**, and any system with another
  embedder beyond the first three conversations.
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
  the top k (11% of Graphiti's edges; Myelin-AI's items are single
  turns).
- The second Mem0 row and the per-session row rest on wording, and
  could credit a turn the memory was not written from.
- Myelin-AI's optional write-time fact extraction (`myelin.enrich`,
  `bench.locomo run --facts`), the nearest thing it has to what Mem0
  does, was off, as it is in LongHAUL.

Either way:
- Mem0 with one `add` per session leaves 27% of questions uncountable,
  and it is not known how many of those are the mapping's failures and
  how many Mem0's.

## Not run, and why

- **Graphiti on two of the ten conversations** (`conv-43`,
  `conv-47`): the account's money for model calls ran out. Each would
  have cost about $0.80; a conversation was started only if it could
  finish.
- **The embedder-swap tables on all ten**: the same reason.
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
    --data locomo10.json --out bench/runs/compare --max-minutes 25
# exit code 3 means out of time: run the same command again.

# another with letta==0.16.8 (see Letta above) and its server on port 8283
python -m bench.compare --system letta-passages \
    --data locomo10.json --out bench/runs/compare

python -m bench.compare --report --system myelin mem0 mem0-session graphiti letta-passages \
    --data locomo10.json --out bench/runs/compare
```

Add `--embedder openai/text-embedding-3-small` and a new `--out` for the
second set of tables. `--skip N` leaves out the first N conversations,
so conversations can be written by separate processes at once. The
report covers the conversations each system finished, so a system that
stopped early is summarised over fewer. Model calls for everything
here, probes and a repeated run included: $12.12.
