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

## Tuning the decay parameters

Swept on five conversations and checked on the other five (three
schedules): how fast health falls with age (`W_AGE_OVERFLOW`), the
health below which a memory is quarantined (`QUARANTINE_THRESHOLD`),
and how much a useful cite lifts health (`W_USE_LIFT`). Held-out
conversations, decay + feedback:

| Setting | top 5 | top 10 | top 20 | Needed a memory decay had hidden | Store still searchable |
|---|---:|---:|---:|---:|---:|
| no decay (feedback only) | 51.9 | 59.7 | 62.3 | 0.0 | 100.0 |
| defaults before tuning | 52.5 | 59.9 | 64.9 | 7.7 | 64.0 |
| use lift 1.0 | 52.5 | 60.2 | 65.2 | 6.0 | 66.4 |
| use lift 2.0 | 52.7 | 60.3 | 65.2 | 5.7 | 67.5 |
| quarantine threshold 0.3, use lift 1.0 | 52.2 | 59.8 | 63.6 | 3.6 | 81.2 |
| half-speed decay, use lift 1.0 | 52.1 | 59.7 | 63.1 | 2.3 | 87.8 |

- **Ranking hardly depends on these parameters.** Over every sensible
  setting the top-5 and top-10 figures move by about half a point.
  What decay buys is a smaller store.
- **Its cost is proportional to what it removes.** Whichever parameter
  is turned, about one question in four hundred loses a needed memory
  for each 1% of the store hidden. Without cites, health is a function
  of age alone, so decay takes the oldest memories whether or not they
  matter. Speed and threshold are therefore a dial, not something with
  an optimum: the defaults hide about a third of a months-old store.
- **A larger use lift is the one free gain**: fewer needed memories
  hidden at the same ranking, because used memories stay healthy for
  longer. The lift is capped by the rule that one contradiction must
  outweigh 100 useful cites, which allows at most about 0.75. The
  default is now 0.7 (was 0.5). On all ten conversations: needed
  memory hidden 7.1% to 6.5%, top 5 53.4 to 53.5, top 10 60.9 to 61.0,
  store searchable 68.7% to 69.6%.

To forget less, lower `QUARANTINE_THRESHOLD` or halve `W_AGE_OVERFLOW`
in `myelin/_internal/health.py`; the table shows what each buys.
Telling needed from unneeded memories better than age does would take
a signal decay does not use yet, such as how often a memory is
retrieved and not used.

## Limits

- One dataset. LoCoMo asks each question once, so use of a memory before
  it is needed again comes only from questions that share evidence.
  Workloads that return to the same facts more often would favour both
  mechanisms more; ones that ask about old, never-revisited details
  would favour them less.
- The headline tables above were measured with the use lift at 0.5,
  before the tuning below.
- Retrieval only. Whether a smaller, sharper store changes answers was
  not measured.
