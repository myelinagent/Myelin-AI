# Memory over time: forgetting and cite feedback

`python -m bench.lifetime`. Each LoCoMo conversation (ten of them, 19
to 32 sessions over several months) is replayed on a simulated clock:
a session is stored on its date, and each question is asked a random
number of sessions after the last message that answers it. No model.
Hybrid search (`minishlab/potion-retrieval-32M`), three random question
schedules, 1,531 questions each, 2026-11-14.

- **decay**: `prune()` after every session. Memories whose health has
  fallen (mostly through age) are quarantined and leave search.
- **feedback**: after each question, `cite_used` with the question's
  evidence as the used memories. This is an upper bound on feedback
  quality; a small model naming what it used matched it in
  `fusion.md`.

## Questions asked up to 6 sessions later

| | All evidence in top 5 | top 10 | top 20 | top 10, asked 3+ sessions later | Needed a memory decay had hidden | Store still searchable at the end |
|---|---:|---:|---:|---:|---:|---:|
| static | 46.4 ± 0.2 | 55.8 ± 0.1 | 64.0 ± 0.3 | 57.9 ± 1.3 | 0.0 | 100.0 |
| decay | 48.9 ± 0.2 | 57.5 ± 0.3 | 63.8 ± 0.1 | 59.2 ± 1.4 | 8.6 | 66.2 |
| feedback | 52.8 ± 0.4 | 60.6 ± 0.3 | 64.1 ± 0.2 | 61.4 ± 1.9 | 0.0 | 100.0 |
| decay + feedback | 53.5 ± 0.3 | 60.9 ± 0.4 | 65.6 ± 0.4 | 61.8 ± 1.6 | 7.1 ± 0.2 | 68.7 |

## Questions asked up to 12 sessions later

| | All evidence in top 5 | top 10 | top 20 | top 10, asked 3+ sessions later | Needed a memory decay had hidden | Store still searchable at the end |
|---|---:|---:|---:|---:|---:|---:|
| static | 44.8 ± 0.2 | 54.8 ± 0.2 | 62.8 ± 0.3 | 56.3 ± 0.1 | 0.0 | 100.0 |
| decay | 46.9 ± 0.3 | 55.6 ± 0.3 | 61.4 ± 0.4 | 56.5 ± 0.4 | 11.8 ± 0.3 | 66.2 |
| feedback | 51.0 ± 0.3 | 58.9 ± 0.5 | 62.9 ± 0.2 | 59.3 ± 0.4 | 0.0 | 100.0 |
| decay + feedback | 51.0 ± 0.1 | 58.7 ± 0.5 | 64.1 ± 0.0 | 58.8 ± 0.6 | 9.3 ± 0.1 | 68.6 |

## What it shows

- **Forgetting sharpens the top of the list.** With a third of the store
  out of search, all of a question's evidence is in the top 5 more
  often (46.4 to 48.9), because old, unused messages stop competing.
- **And it has a price.** For 8.6% of questions (11.8% at the longer
  lag) decay had hidden a memory the question needed. Those memories
  are quarantined, not deleted, and `search(include_quarantined=True)`
  still reaches them; nothing is lost for good. The price grows with
  how long ago the answer was said, and at the longer lag decay alone
  is a small net loss in the top 20.
- **Feedback is the larger effect**: about 6 points in the top 5 and 5
  in the top 10, at both lags.
- **Together.** Use keeps a memory healthy, so feedback cuts what decay
  wrongly hides (8.6 to 7.1, 11.8 to 9.3) and the store keeps a little
  more. At the shorter lag the combination is best on every ranking
  column; at the longer lag it matches feedback alone in the top 10
  and is ahead in the top 20, with a third less to search.

## Limits

- One dataset. LoCoMo asks each question once, so use of a memory before
  it is needed again comes only from questions that share evidence.
  Workloads that return to the same facts more often would favour both
  mechanisms more; ones that ask about old, never-revisited details
  would favour them less.
- Decay runs on the default thresholds, which were not tuned here. The
  8.6% is a property of those defaults on months-long timelines.
- Retrieval only. Whether a smaller, sharper store changes answers was
  not measured.
