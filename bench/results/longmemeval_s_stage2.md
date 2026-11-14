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

## Next

1. ~~Batch embedding at ingest~~ Done: `add_many` embeds a batch in one
   call, 2.37 s to 0.88 s per 494 turns, with identical retrieval results.
2. Preference and temporal questions: facts and event dates extracted
   when a memory is written.
3. Abstention.
4. Repeat runs for error bars.
