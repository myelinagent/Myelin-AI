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

## Grading memories together

Health now weighs four things in one grade: age, a memory's own use,
the use of related memories (close in the same conversation, or similar
by embedding; `myelin/_internal/relations.py`), and how often it was
retrieved for a question and not used. Weights (`W_RELATED` 2.0,
`W_UNUSED` -0.3) were swept on five conversations and checked on the
other five.

Held-out conversations, decay + feedback, three schedules:

| | top 5 | top 10 | top 20 | Needed a memory decay had hidden | Store still searchable |
|---|---:|---:|---:|---:|---:|
| use and age only | 52.3 | 59.9 | 64.9 | 6.8 | 65.0 |
| **combined grade** | 52.8 | 60.1 | 65.8 | 6.1 | 70.4 |
| relationships only | 51.8 | 59.8 | 63.6 | 2.9 | 83.2 |
| unused penalty only | 52.8 | 60.8 | 67.4 | 11.7 | 48.7 |

All ten conversations with the combined grade as the default, decay +
feedback (compare the tables at the top, measured before it):

| Questions asked | top 5 | top 10 | top 20 | Needed a memory decay had hidden | Store still searchable |
|---|---:|---:|---:|---:|---:|
| up to 6 sessions later, before | 53.5 | 60.9 | 65.6 | 7.1 | 68.7 |
| up to 6 sessions later, combined | 53.6 ± 0.6 | 61.0 ± 0.3 | 66.6 ± 0.5 | 6.2 ± 0.2 | 73.3 |
| up to 12 sessions later, before | 51.0 | 58.7 | 64.1 | 9.3 | 68.6 |
| up to 12 sessions later, combined | 50.7 ± 0.2 | 58.1 ± 0.2 | 63.8 ± 0.6 | 9.9 ± 0.9 | 71.7 |

- **Relationships protect.** Neighbours of used memories stay healthy,
  so fewer needed memories are hidden, and fewer memories are pruned,
  at about the exchange rate of simply decaying more slowly.
  Similarity links did slightly better than conversation links.
- **The unused penalty prunes.** It removes memories that keep being
  retrieved without helping and lifts ranking by a point or two, and it
  hides needed memories in proportion. A stronger penalty (-0.6)
  collapsed the store.
- **Together** they give a small gain when questions follow within a
  few sessions (top 20 up a point, slightly fewer needed memories
  hidden) and a small loss when questions come much later (top 10 down
  about half a point). The differences are a few tenths to one point,
  against a spread of about half a point between schedules.
- The combined grade is a sounder design than age and own use alone,
  and it is what makes the decision explainable (see the README). On
  this benchmark it is not a step change: each question is asked once,
  so a used memory's neighbours are only a little more likely to be
  needed later than any other memory.

## As shipped: relationships on, no penalty for unused showings

The unused penalty (`W_UNUSED`) was then set to zero: being retrieved
and not used lowers a memory's rank, not its health. All ten
conversations, three schedules, decay + feedback:

| Questions asked | | top 5 | top 10 | top 20 | Needed a memory decay had hidden | Store still searchable |
|---|---|---:|---:|---:|---:|---:|
| up to 6 sessions later | with the penalty | 53.6 | 61.0 | 66.6 | 6.2 | 73.3 |
| | **as shipped** | 52.8 ± 0.4 | 60.8 ± 0.2 | 64.9 ± 0.2 | 2.5 ± 0.1 | 86.2 |
| up to 12 sessions later | with the penalty | 50.7 | 58.1 | 63.8 | 9.9 | 71.7 |
| | **as shipped** | 51.0 ± 0.4 | 58.8 ± 0.3 | 63.5 ± 0.1 | 4.0 ± 0.6 | 85.0 |

Needed memories hidden fall by more than half and the loss at the
longer lag goes away. Decay prunes about 14% of the store instead of
27%, and the top 20 at the shorter lag gives back a point or two.

## Revisited questions

`--revisits 2`: every question is asked three times, each asking some
sessions after the last. This is the workload LoCoMo lacks, a user
returning to the same facts. All ten conversations, three schedules;
figures are for the repeat askings only.

| | top 5 | top 10 | Needed a memory decay had hidden | Store still searchable at the end |
|---|---:|---:|---:|---:|
| static | 44.1 ± 0.1 | 54.0 ± 0.2 | 0.0 | 100.0 |
| decay | 45.9 ± 0.1 | 54.0 ± 0.2 | 14.1 ± 0.2 | 66.2 |
| feedback | 53.7 ± 0.5 | 59.5 ± 0.3 | 0.0 | 100.0 |
| decay + feedback | 53.8 ± 0.5 | 59.6 ± 0.2 | 2.0 ± 0.1 | 89.6 |

- **Promotion by use is worth more when facts come back**: 9.6 points
  in the top 5 on repeat askings (44.1 to 53.7), against 6.4 when each
  question is asked once.
- **Use protects what is asked about again.** Decay on its own had
  hidden a needed memory on 14.1% of repeat askings. With feedback it
  is 2.0%, while a tenth of the store is still pruned.
- This is where the mechanisms show that they belong together, and it
  is invisible in a benchmark that asks each question once.

## The same grade in ranking: measured, left off

`prune()` stores each memory's related support (`Memory.related_support`),
and cite-informed ranking can count it towards a memory's own record
(`RELATED_SHARE` in `myelin/_internal/cite_ranking.py`). Five training
conversations, decay + feedback:

| Share of a useful cite per unit of related support | top 5 | top 10 | top 20 | Repeat askings: top 5 | top 10 |
|---|---:|---:|---:|---:|---:|
| 0 (off) | 54.2 | 61.3 | 65.9 | 53.8 | 60.3 |
| 0.25 | 54.0 | 61.4 | 65.7 | 54.0 | 60.5 |
| 0.5 | 54.0 | 61.3 | 65.6 | 53.6 | 60.1 |
| 1 | 54.4 | 61.5 | 65.6 | 53.6 | 60.3 |
| 2 | 53.6 | 61.3 | 65.7 | 53.5 | 60.1 |

No setting differs from off by more than the spread between schedules,
with or without revisits, so the share stays at 0. Being related to
useful memories is a reason to keep a memory; on this data it is not a
reason to rank it higher for a particular question. The stored support
remains available for explaining why a memory was kept.

## Limits

- One dataset. LoCoMo asks each question once, so use of a memory before
  it is needed again comes only from questions that share evidence.
  Workloads that return to the same facts more often would favour both
  mechanisms more; ones that ask about old, never-revisited details
  would favour them less.
- The tables at the top were measured with the use lift at 0.5 and
  without relationships; "As shipped" below has the current defaults.
- Retrieval only. Whether a smaller, sharper store changes answers was
  not measured.
