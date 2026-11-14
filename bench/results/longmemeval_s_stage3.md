# LongMemEval-S: excerpt length, and what was tried for multi-session questions

All 500 questions, hybrid search (`minishlab/potion-retrieval-32M`),
6,000-token budget, the three best-ranked turns whole, prompt
`myelin-reasoned-v2`, `gpt-5-mini` answering, `gpt-4o` judging,
2026-11-14.

## Shorter excerpts

A turn longer than the excerpt length is shown as the passages that
share words with the question. Evidence is mostly in short user
messages, and 300-token excerpts of long assistant replies were using
the budget: the context held about 45 turns of roughly 480, and missed
evidence typically ranked just below the cut.

| | 300 tokens, run 1 | 300, run 2 | 150 | 100, run 1 | 100, run 2 | 100, six whole | Perfect retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Accuracy (overall)** | 89.2 | 88.8 | 89.6 | **90.2** | **90.4** | 89.8 | 93.4 |
| single-session-user | 97.1 | 95.7 | 95.7 | 94.3 | 94.3 | 95.7 | 95.7 |
| single-session-assistant | 96.4 | 96.4 | 96.4 | 94.6 | 92.9 | 94.6 | 100.0 |
| single-session-preference | 83.3 | 86.7 | 86.7 | 86.7 | 90.0 | 83.3 | 90.0 |
| multi-session | 78.2 | 78.9 | 76.7 | 80.5 | 81.2 | 77.4 | 86.5 |
| temporal-reasoning | 90.2 | 89.5 | 94.0 | 94.7 | 94.7 | 94.7 | 95.5 |
| knowledge-update | 96.2 | 93.6 | 94.9 | 93.6 | 93.6 | 96.2 | 96.2 |
| All evidence turns in context | 92.3 | 92.3 | 94.7 | 97.0 | 97.0 | 96.4 | — |
| Turns in context | 45.5 | 45.5 | 61.7 | 75.5 | 75.5 | 68.0 | 21.9 |

- 100-token excerpts are now the `recall()` default: 90.3 against 89.0,
  two runs each. Temporal questions gain about 5 points in every run.
- About 40 answers change between two runs of the same setting, so a
  single run cannot resolve a difference under about 2 points.
- Questions about what the assistant said lose 2 to 3 points: a long
  reply holding the answer is sometimes cut. Keeping six turns whole
  instead of three did not recover it.
- On LoCoMo the change makes no difference (utterances are short).

## Multi-session questions

The remaining gap to perfect retrieval is here: 80% against 86.5%, with
all evidence in context for 93% of them. The reader misses or
miscounts items among many near-misses. Tried on the 133 multi-session
questions, none adopted:

| Change | Result |
|---|---|
| Extra search queries written by `gpt-4o-mini`, fused with the original | Evidence coverage 92.5 to 93.9 overall, multi-session unchanged at 88.8 |
| Larger budget (9,000 and 12,000 tokens) | 79.7 to 82.7; evidence coverage no better than excerpts at 6,000 |
| Whole turns at 6,000 tokens | 68.4 and 71.4 (evidence coverage falls to 75%) |
| `gpt-4o-mini` picks the relevant turns, answerer sees only those | 62.4; the filter discards evidence (coverage 97 to 55) |
| Facts extracted at write time (`longmemeval_s_stage2.md`) | 75.9 indexed, 76.7 shown |
