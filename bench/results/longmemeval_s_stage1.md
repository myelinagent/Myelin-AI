# LongMemEval-S after stage 1 (core retrieval)

Same benchmark and setup as `longmemeval_s_baseline.md`: all 500 questions,
one memory per conversation turn, a fresh store per question, no model
involved. Recall is measured on the 470 questions that have evidence.

Stage 1 changed two things:

1. **Search is BM25 over a SQLite full-text index with word stemming**,
   replacing the unweighted token-overlap scan.
2. **Context is sized by a token budget** (`client.recall()`), replacing a
   fixed 10 results.

## Result

The number that matters for answering is whether the context handed to the
answering model contains every evidence session.

| | Baseline | Stage 1, top 10 | Stage 1, 6,000-token budget | BM25 control, same budget |
|---|---:|---:|---:|---:|
| **All evidence in the answerer's context** | 80.9 | 86.0 | **90.9** | 90.0 |
| Turns in context | 10.0 | 10.0 | 18.6 | 20.5 |
| Context size (est. tokens) | — | 3,075 | 5,700 | 5,675 |
| Recall, all evidence sessions @5 | 69.1 | 74.9 | 74.9 | 74.5 |
| Recall, all evidence sessions @10 | 80.9 | 86.0 | 86.0 | 82.3 |
| Recall, all evidence sessions @50 | 95.3 | 97.0 | 97.0 | 96.0 |
| Recall, any evidence session @10 | 94.7 | 97.9 | 97.9 | 96.8 |
| Retrieve per question (ms) | 293 | 7 | 14 | 3 |

By question type, all evidence in the top 10:

| | Baseline | Stage 1 |
|---|---:|---:|
| single-session-user | 98.4 | 100.0 |
| single-session-assistant | 100.0 | 100.0 |
| single-session-preference | 63.3 | 86.7 |
| multi-session | 59.5 | 72.7 |
| temporal-reasoning | 78.7 | 78.7 |
| knowledge-update | 97.2 | 97.2 |

## What this says

- Ten points of the gap closed: 80.9 to 90.9. About half came from better
  ranking and half from handing over more context.
- Stemming is what lifted Myelin above the plain BM25 control at the same
  depth (86.0 against 82.3 at 10): it matches "graduate" to "graduated".
- Search is about 40 times faster at this store size (293 ms to 7 ms).
- Temporal-reasoning recall did not move. Those questions need date
  handling, not better word matching.
- 97.0% of questions have all their evidence within the top 50 turns, so
  the remaining gap at this budget is about what fits, which points at
  ranking quality per token (an embedding channel, a reranker) and at long
  assistant turns using up the budget.

## Tried and dropped

Prefixing the query with `user:` to match the stored turn format changed
nothing at any budget (79.4 / 87.4 / 90.9 / 93.2 at 2K / 4K / 6K / 8K tokens,
with and without it), so it was not adopted.

## Excerpts of long turns

Added after the table above. Long assistant replies (median 419 tokens)
were using most of the budget while 94% of evidence sits in short user
turns (median 72). Shortening turns over 300 tokens to the passages that
mention the query lets about twice as many fit.

Share of questions where every turn marked as holding the answer is in the
answerer's context:

| Original search, top 10 | Stage 1, top 10 | + 6,000-token budget | + 300-token excerpts |
|---:|---:|---:|---:|
| 61.1 | 70.6 | 77.7 | 86.0 |

Session-level recall alone was misleading here: spreading results across
sessions raised it to 96% while turn-level recall fell to 42%, so that
idea was dropped. Favouring memories dated inside a time window read from
the question ("last month") was also dropped: it lowered recall on the
questions that have such a phrase, from 64.3% to 59.5%.

## Answer accuracy

See `longmemeval_s_accuracy.md`.
