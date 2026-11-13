# LongMemEval-S baseline

The starting point for the retrieval work: Myelin 1.4.0 plus the stability
fixes, default search, measured before any retrieval change.

- **Benchmark:** LongMemEval-S, all 500 questions (`longmemeval_s_cleaned.json`,
  sha256 `d6f21ea9…c3a442`), about 494 turns of history per question.
- **Setup:** one memory per conversation turn, stored at the session's date;
  a fresh store per question; the question text used as the query.
- **Date:** 2026-11-13, base commit `6c2a445`.
- **Control:** textbook BM25 (k1=1.5, b=0.75) over the same turns.

## Retrieval (no model involved)

"Evidence sessions" are the sessions the benchmark marks as containing the
answer. Recall is measured over the top-k retrieved turns; the 30 abstention
questions have no evidence and are excluded, leaving 470.

| | myelin | bm25 control |
|---|---:|---:|
| Recall, any evidence session @5 | 91.1 | 94.5 |
| Recall, any evidence session @10 | 94.7 | 96.8 |
| Recall, any evidence session @50 | 98.9 | 99.6 |
| Recall, all evidence sessions @5 | 69.1 | 74.5 |
| Recall, all evidence sessions @10 | 80.9 | 82.3 |
| Recall, all evidence sessions @50 | 95.3 | 96.0 |
|   all @10: single-session-user | 98.4 | 100.0 |
|   all @10: single-session-assistant | 100.0 | 100.0 |
|   all @10: single-session-preference | 63.3 | 80.0 |
|   all @10: multi-session | 59.5 | 63.6 |
|   all @10: temporal-reasoning | 78.7 | 76.4 |
|   all @10: knowledge-update | 97.2 | 95.8 |
| Ingest per question (s) | 0.29 | 0.09 |
| Retrieve per question (ms) | 293 | 3 |

## What this says

- Finding *some* evidence is already easy; finding *all* of it is not. At
  the default of 10 results Myelin reaches every evidence session for 80.9%
  of questions, and 95.3% at 50. Retrieving more is the largest single lever.
- The weak categories are multi-session (59.5) and preference (63.3)
  questions, where the evidence is spread out or phrased unlike the question.
- BM25 is only 1.4 points ahead on the headline number, so swapping the
  scoring function alone will not move much. It is far ahead on speed:
  Myelin's search re-reads and re-tokenizes every stored turn per query
  (293 ms against 3 ms at about 500 turns).

## Not yet measured

Answer accuracy (the number leaderboards quote) needs an answering model and
the gpt-4o judge. The harness supports it (`--answerer`, `--judge`), but the
run was not possible on 2026-11-13 because the API key available had
exhausted its credit limit. Rerun with:

```bash
python -m bench.longmemeval run --system myelin --data longmemeval_s_cleaned.json --out bench/runs/lme_s_myelin --answerer gpt-4o-mini --judge gpt-4o
```

Machine-readable summaries are alongside this file.
