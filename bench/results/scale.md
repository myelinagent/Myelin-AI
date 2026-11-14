# Scale

Write throughput and `recall()` latency as one user's store grows, from
`python -m bench.scale`. Synthetic text, 6,000-token recall, 50 queries
per size, a single SQLite file, one process, 2026-11-14. Machine-dependent:
read the shape, not the absolute numbers.

## Lexical (no embedder)

| Memories | Adds per second | Recall p50 (ms) | Recall p95 (ms) | Database (MB) |
|---:|---:|---:|---:|---:|
| 5,000 | 21,538 | 3 | 13 | 4 |
| 50,000 | 20,769 | 15 | 62 | 42 |
| 200,000 | 16,625 | 45 | 190 | 168 |

Before the full-text shortlist in this change, 50,000 memories took 29 ms
(p95 142) and 200,000 took 119 ms (p95 600): every match was joined to
its memory row before ranking, which is slow when the query's words are
common.

## Hybrid (`minishlab/potion-retrieval-32M`)

| Memories | Adds per second | Recall p50 (ms) | Recall p95 (ms) | Database (MB) |
|---:|---:|---:|---:|---:|
| 5,000 | 2,331 | 8 | 17 | 25 |
| 50,000 | 3,726 | 30 | 78 | 250 |
| 200,000 | 3,620 | 83 | 367 | 1,002 |

The 200,000 row is after picking the best vector matches without sorting
every score; before that the median was 116 ms. The 95th percentile
varies from run to run by about 100 ms.

## What this says

- Recall time grows in proportion to the store. Up to tens of thousands
  of memories per user it is a few tens of milliseconds.
- Hybrid search compares the query with every stored vector, which is
  what sets its time at 200,000 memories. An approximate index would be
  the next step for stores of that size.
- Vectors are about 5 KB per memory (512 float32 values plus overhead),
  six times the text itself.
