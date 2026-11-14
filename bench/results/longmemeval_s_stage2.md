# LongMemEval-S after stage 2a (hybrid search)

Adds an embedding channel beside BM25: `MemoryClient(embedder=...)` with
`minishlab/potion-retrieval-32M`, a static model that runs on a CPU. Both
rankings are fused by reciprocal rank (vector weight 2, lexical 1).

Everything else is as in `longmemeval_s_accuracy.md`: all 500 questions,
6,000-token budget, 300-token excerpts with the three best-ranked turns
whole, prompt `myelin-reasoned-v2`, judge `gpt-4o`, one run each,
2026-11-13.

## Answer accuracy

| | `gpt-4o`, lexical | `gpt-4o`, hybrid | `gpt-4o`, oracle | `gpt-4o-mini`, lexical | `gpt-4o-mini`, hybrid | `gpt-4o-mini`, oracle |
|---|---:|---:|---:|---:|---:|---:|
| **Accuracy (overall)** | 82.8 | **87.4** | 93.6 | 77.2 | **78.0** | 79.4 |
| Accuracy (task-averaged) | 82.9 | 86.8 | 93.9 | 77.5 | 78.3 | 81.3 |
| single-session-user | 95.7 | 98.6 | 98.6 | 97.1 | 95.7 | 91.4 |
| single-session-assistant | 96.4 | 96.4 | 100.0 | 96.4 | 91.1 | 98.2 |
| single-session-preference | 60.0 | 66.7 | 86.7 | 50.0 | 56.7 | 70.0 |
| multi-session | 69.2 | 81.2 | 88.7 | 62.4 | 65.4 | 65.4 |
| temporal-reasoning | 82.7 | 85.7 | 93.2 | 75.9 | 77.4 | 78.2 |
| knowledge-update | 93.6 | 92.3 | 96.2 | 83.3 | 83.3 | 84.6 |
| abstention subset | 70.0 | 66.7 | 83.3 | 66.7 | 63.3 | 66.7 |

## Retrieval (no model involved, 470 questions with evidence)

| | Lexical | Hybrid |
|---|---:|---:|
| All evidence turns in the answerer's context | 85.3 | 92.3 |
| Recall, all evidence sessions @10 | 86.0 | 89.6 |
| all @10: multi-session | 72.7 | 86.0 |
| all @10: single-session-preference | 86.7 | 93.3 |
| all @10: temporal-reasoning | 78.7 | 78.7 |
| Ingest per question (s), about 494 turns | 0.10 | 2.37 |
| Retrieve per question (ms) | 4 | 60 |

## What this says

- **With `gpt-4o`, hybrid search is worth 4.6 points** (82.8 to 87.4), well
  outside the 1.7-point standard error of a single run. Twelve of those
  points are on multi-session questions (69.2 to 81.2), which is where the
  embedding channel finds evidence worded unlike the question.
- **With `gpt-4o-mini` it is worth 0.8, which is noise.** That model was
  already within 2.2 points of its own perfect-retrieval ceiling, so there
  was little for better retrieval to add.
- **The remaining gap to perfect retrieval with `gpt-4o` is 6.2 points**,
  in preference (66.7 against 86.7), temporal (85.7 against 93.2) and
  multi-session (81.2 against 88.7) questions.
- **Abstention is the weakest category in every configuration**, including
  the oracle (83.3). More retrieved context makes it slightly worse: the
  model finds something nearby to answer from.
- For orientation only: 87.4 with `gpt-4o` is above the roughly 80 reported
  for plain retrieval and in the range vendors self-report for `gpt-4o`
  (mid-80s). Harnesses and prompts differ, so this is not a ranking.

## Model comparison (prototype, evidence coverage only)

| Embedding model | Vector only | Fused 1:1 | Fused, vector weight 2 |
|---|---:|---:|---:|
| none (BM25) | — | — | 85.3 |
| `potion-base-8M` | 91.5 | 90.0 | 90.9 |
| `potion-retrieval-32M` | 93.4 | 90.9 | 92.6 |

The vector ranking alone covers slightly more evidence than the fusion,
but it loses exact matches: single-session-assistant coverage was 94.6
vector-only against 100.0 fused 1:1. `BAAI/bge-small-en-v1.5` through
ONNX took over seven minutes to embed one question's history on this
4-core machine and was not evaluated further.

## Context budget: 6,000 tokens is the sweet spot for `gpt-4o`

Evidence coverage rises steadily with the budget (hybrid search, 470
questions with evidence, no model involved):

| Budget (tokens) | Turns shown | All evidence turns in context |
|---:|---:|---:|
| 4,000 | 29 | 87.7 |
| 6,000 | 45 | 92.3 |
| 8,000 | 62 | 95.7 |
| 10,000 | 79 | 96.8 |
| 12,000 | 95 | 97.9 |

Accuracy does not follow it. `gpt-4o` on a random sample of 250 questions
(seed 20261114), compared question by question on the 222 that have a
judged answer at all three budgets:

| | 4,000 | 6,000 | 8,000 |
|---|---:|---:|---:|
| **Accuracy (overall)** | 84.7 | **88.7** | 86.0 |
| multi-session (71) | 71.8 | 78.9 | 74.6 |
| temporal-reasoning (61) | 85.2 | 91.8 | 88.5 |
| knowledge-update (38) | 92.1 | 92.1 | 92.1 |
| single-session-user (22) | 100.0 | 100.0 | 95.5 |
| single-session-assistant (24) | 95.8 | 95.8 | 95.8 |
| abstention subset (12) | 7 right | 7 right | 9 right |
| Prompt tokens per question | 3,906 | 5,681 | 7,444 |

Against 6,000, the 4,000-token context fixed 7 answers and broke 16; the
8,000-token context (on its 240 judged questions) fixed 5 and broke 14.
Too little context misses evidence; too much distracts the answerer even
though the evidence is there. The default `token_budget=6000` stays.

Caveats: one sample, one run per budget; 28 of the 750 calls failed
when the API account ran low on credit, which is why 222 questions and
not 250 are compared. The differences are suggestive, not conclusive.

What it implies: the remaining 6-point gap to perfect retrieval is about
precision, not coverage. The next gains should come from ranking evidence
higher inside the same budget (a reranker, facts extracted at write time)
and not from showing more.

## With a current answer model: `gpt-5-mini`

The 94 to 95% scores other systems publish use answer models newer than
`gpt-4o`. Same setup as above (hybrid search, 6,000-token budget,
300-token excerpts, top three turns whole, prompt `myelin-reasoned-v2`,
judge `gpt-4o`), with `openai/gpt-5-mini` answering and
`--max-answer-tokens 4000`, since this model spends output tokens on
hidden reasoning. All 500 questions, one run each, 2026-11-14.

| | `gpt-4o`, hybrid | `gpt-5-mini`, hybrid | `gpt-5-mini`, oracle |
|---|---:|---:|---:|
| **Accuracy (overall)** | 87.4 | **89.2** | 93.4 |
| Accuracy (task-averaged) | 86.8 | 90.2 | 94.0 |
| single-session-user | 98.6 | 97.1 | 95.7 |
| single-session-assistant | 96.4 | 96.4 | 100.0 |
| single-session-preference | 66.7 | 83.3 | 90.0 |
| multi-session | 81.2 | 78.2 | 86.5 |
| temporal-reasoning | 85.7 | 90.2 | 95.5 |
| knowledge-update | 92.3 | 96.2 | 96.2 |
| abstention subset | 66.7 | 90.0 | 86.7 |

- **89.2% overall, 90.2% task-averaged.** Published `gpt-5-mini` results
  on this benchmark are around 94 to 95% (Mastra reports 94.87%); those
  figures are self-reported from other harnesses and prompts.
- **The model swap is worth 1.8 points here**, not the ten it was worth to
  a system whose `gpt-4o` score was 84. Most of what a stronger reader
  buys was already bought by the prompt and hybrid retrieval.
- **It helps where judgement matters**: preference 66.7 to 83.3,
  abstention 66.7 to 90.0, temporal 85.7 to 90.2. Multi-session slipped
  from 81.2 to 78.2, inside the noise of 133 questions.
- **Perfect retrieval gives 93.4**, the same ceiling as `gpt-4o` (93.6).
  So about 6 points of the benchmark are beyond this prompt whatever is
  retrieved, and Myelin is 4.2 points under its own ceiling. That gap is
  in multi-session (78.2 against 86.5), preference (83.3 against 90.0) and
  temporal (90.2 against 95.5) questions.
- No answer was empty; one of 500 reached the 4,000-token cap.

## Next

1. ~~Batch embedding at ingest~~ Done: `add_many` embeds a batch in one
   call, 2.37 s to 0.88 s per 494 turns, with identical retrieval results.
2. Preference and temporal questions: facts and event dates extracted
   when a memory is written.
3. Abstention.
4. Repeat runs for error bars.
