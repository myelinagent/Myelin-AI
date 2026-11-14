# Hybrid fusion: the top of the list, and what cite feedback adds

No model is involved in the retrieval figures here. "All evidence in
top k" is the share of questions whose every evidence turn is among the
first k results; "at budget" is the same for a 6,000-token context.
Embedder `minishlab/potion-retrieval-32M`, 2026-11-14.

## The problem

Hybrid search fused the lexical and vector rankings with the vector
ranking weighted double and a rank constant of 60. That was chosen on
LongMemEval-S for evidence within a token budget. On LoCoMo's short
messages it was worse than lexical search alone at the top of the list:

| LoCoMo, all evidence in | top 5 | top 10 | top 20 | top 50 | at budget |
|---|---:|---:|---:|---:|---:|
| Lexical only | 47.7 | 54.9 | 61.1 | 68.6 | 77.0 |
| Vector-heavy fusion (before) | 33.5 | 39.6 | 46.9 | 61.9 | 80.3 |

Equal weights with a small rank constant fix the top of the list but
lose evidence deeper down (LongMemEval-S at budget 96.5 to 95.8; LoCoMo
with facts 87.9 to 87.2).

## The change

The first 20 places are ordered by equal weights with a rank constant
of 10; everything below keeps the vector-heavy fusion.

| All evidence in | top 5 | top 10 | top 20 | top 50 | at budget |
|---|---:|---:|---:|---:|---:|
| LoCoMo, before | 33.5 | 39.6 | 46.9 | 61.9 | 80.3 |
| LoCoMo, after | 42.5 | 51.8 | 58.8 | 64.5 | 80.3 |
| LoCoMo with facts indexed, before | 50.4 | 57.3 | 63.2 | 72.6 | 87.9 |
| LoCoMo with facts indexed, after | 55.1 | 62.6 | 69.3 | 74.7 | 87.9 |
| LongMemEval-S, before | 62.8 | 77.5 | 85.4 | 94.4 | 96.5 |
| LongMemEval-S, after | 64.5 | 77.5 | 85.6 | 94.4 | 96.5 |

This matters to `search(top_k=10)` and to anything that reads only the
first few results. It does not change what a 6,000-token `recall()`
holds, and answer accuracy there did not move beyond run-to-run noise:

| | Before | After |
|---|---:|---:|
| LongMemEval-S, `gpt-5-mini` | 91.3 (four runs, 89.6 to 93.0) | 90.6 (90.2, 91.0) |
| LoCoMo with facts, `gpt-4o-mini` | 82.4 | 81.4 |

On LoCoMo, 45 answers were gained and 60 lost between the two runs; two
runs of the same setting differ by about 50 each way.

## Cite feedback on held-out questions

`python -m bench.cite_feedback`: for each LoCoMo conversation, a random
half of the questions give cite feedback on their top 20 results and
the other half are scored. Five random splits, mean and standard
deviation.

| Fusion | Feedback | top 5 | top 10 | top 20 | top 50 |
|---|---|---:|---:|---:|---:|
| Vector-heavy (before) | none | 33.9 ± 0.9 | 39.7 ± 1.2 | 46.7 ± 1.5 | 62.2 ± 1.7 |
| | per question | 35.0 ± 1.0 | 41.3 ± 1.3 | 48.6 ± 1.6 | 63.2 ± 1.7 |
| | per memory | 39.5 ± 1.4 | 45.1 ± 1.1 | 53.1 ± 1.8 | 65.3 ± 1.3 |
| Two-part (after) | none | 42.1 ± 1.2 | 51.9 ± 0.8 | 59.0 ± 1.2 | 64.8 ± 1.2 |
| | per question | 41.8 ± 1.3 | 51.0 ± 0.8 | 59.0 ± 1.2 | 64.4 ± 0.9 |
| | per memory | 47.6 ± 0.9 | 55.0 ± 1.1 | 58.9 ± 1.2 | 67.0 ± 1.6 |
| Lexical only | none | 48.0 ± 1.1 | 55.2 ± 1.1 | 61.0 ± 0.8 | 68.3 ± 0.9 |
| | per question | 47.3 ± 1.2 | 55.0 ± 0.9 | 61.0 ± 0.6 | 68.5 ± 1.2 |
| | per memory | 48.2 ± 1.3 | 54.4 ± 1.2 | 61.1 ± 1.3 | 67.9 ± 0.8 |

"Per question" is what an application can do without labels: every
memory shown for a question gets one verdict (here, whether all the
evidence was in view). "Per memory" cites a memory as useful only if it
is evidence, an upper bound on feedback quality.

- Per-question feedback helped a little (1 to 2 points) while the
  ranking was weak at the top, and not at all once it was fixed.
- Precise per-memory feedback is worth about 5 points in the top 5 and
  3 in the top 10 with hybrid search, and nothing with lexical search.
- So cite-informed ranking pays when the feedback says which memory was
  useful, not only whether the answer was right. That is the direction
  for the feature: have the answering model name the memories it used.

This is one dataset, and feedback carries over because questions about
the same events share evidence.

## `cite_used`: naming what an answer relied on

Three follow-up measurements (three splits each, two-part fusion):

| Feedback | top 5 | top 10 | top 20 | top 50 |
|---|---:|---:|---:|---:|
| none | 42.4 ± 1.0 | 51.9 ± 0.3 | 59.3 ± 0.0 | 64.8 ± 0.3 |
| evidence cited as useful, nothing recorded for the rest | 42.9 | 51.9 | 59.3 | 65.5 |
| `cite_used` with the evidence (upper bound) | 47.7 ± 0.4 | 55.1 ± 0.6 | 59.3 ± 0.1 | 67.6 ± 0.6 |
| `cite_used` with what `gpt-4o-mini` says it used | 47.8 ± 0.5 | 55.2 ± 0.2 | 59.2 ± 0.2 | 68.1 ± 0.9 |

- Praising the useful memories alone does nothing. The gain comes from
  moving down the memories that keep being retrieved and not used.
- The existing "incorrect" cite would do that too, but it also counts
  heavily against a memory's health and so towards forgetting, which is
  wrong for a memory that was simply not needed. `cite_used` records
  those as shown-but-unused (`Memory.shown_unused_count`), which only
  ranking reads.
- A small model's own account of what it used is as good as the
  benchmark's evidence labels here (`--model` in
  `python -m bench.cite_feedback`: the model sees the numbered top 20,
  answers, and ends with a `Used:` line).
- The top 20 does not change because feedback re-orders the first 20
  results and does not bring new ones in.
