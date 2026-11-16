# LongHAUL: memory over time, forgetting and cite feedback

LongHAUL (Long-horizon Assessment of Use and Loss) is the name of this
protocol: what a memory gains from being used, and what forgetting loses.

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

## A model in the loop

`--model openai/gpt-4o-mini` adds arms in which the used set is what the
answering model names from the numbered top 20, not the evidence labels.
LoCoMo, ten conversations, two schedules, hybrid search.

Each question asked once (1,531 askings per schedule):

| | top 5 | top 10 | top 20 | Needed a hidden memory | Store still searchable |
|---|---:|---:|---:|---:|---:|
| static | 46.4 | 55.9 | 63.9 | 0.0 | 100.0 |
| feedback (labels) | 52.7 | 60.5 | 63.8 | 0.0 | 100.0 |
| feedback (model) | 52.7 | 60.5 | 63.8 | 0.0 | 100.0 |
| decay + feedback (labels) | 52.7 | 60.6 | 64.7 | 2.5 | 86.2 |
| decay + feedback (model) | 52.9 | 60.6 | 64.1 | 1.0 | 93.2 |

Two revisits, repeat askings only:

| | top 5 | top 10 | Needed a hidden memory | Store still searchable |
|---|---:|---:|---:|---:|
| static | 44.0 | 53.9 | 0.0 | 100.0 |
| feedback (labels) | 53.4 | 59.3 | 0.0 | 100.0 |
| feedback (model) | 53.3 | 59.0 | 0.0 | 100.0 |
| decay + feedback (labels) | 53.4 | 59.4 | 2.0 | 89.6 |
| decay + feedback (model) | 53.2 | 59.1 | 0.3 | 96.9 |

- **The ranking gain does not depend on the labels.** The model's own
  account gives the same top 5 and top 10 as the labels, asked once or
  revisited.
- **The model names more.** It named about 1.8 memories per reply, of
  which 43% were labelled evidence, and it named 87% of the labelled
  evidence it was shown. With decay that protects more and prunes less:
  fewer needed memories are hidden (1.0% against 2.5%) and less of the
  store leaves search (7% against 14%).
- Every one of the 12,248 askings got a readable reply. Cost: about
  $2.85 for both runs.

## A second dataset: LongMemEval-S

`--dataset longmemeval` replays each LongMemEval-S entry as its own
timeline with one question. Two revisits, three schedules, hybrid
search, no model. 479 of the 500 entries have flagged evidence turns.

All 500 entries:

| | top 5 | top 10 | top 20 | Needed a hidden memory | Store still searchable | Repeat askings: top 5 | top 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | 65.1 | 77.8 | 86.6 | 0.0 | 100.0 | 64.9 | 77.7 |
| decay | 62.5 | 75.3 | 83.9 | 2.7 | 98.9 | 62.3 | 75.2 |
| feedback | 69.4 | 81.2 | 86.6 | 0.0 | 100.0 | 71.3 | 82.7 |
| decay + feedback | 67.2 | 79.0 | 84.2 | 2.4 | 98.9 | 69.3 | 80.7 |

The 120 entries whose history spans 31 days or more
(`--min-span-days 31`; knowledge-update 74, temporal-reasoning 46):

| | top 5 | top 10 | top 20 | Needed a hidden memory | Store still searchable | Repeat askings: top 5 | top 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| static | 66.7 | 75.9 | 86.2 | 0.0 | 100.0 | 66.5 | 75.9 |
| decay | 56.3 | 65.5 | 75.0 | 11.2 | 95.5 | 56.2 | 65.5 |
| feedback | 70.3 | 79.0 | 86.2 | 0.0 | 100.0 | 72.0 | 80.6 |
| decay + feedback | 61.4 | 70.1 | 76.4 | 9.8 | 95.5 | 63.8 | 72.4 |

- **Feedback carries over.** Top 5 rises 4.3 points over all entries
  and 6.4 on repeat askings, as on LoCoMo.
- **Decay does nothing on most of this dataset.** 70% of the histories
  span 20 days or less (median 10), and nothing is pruned in a history
  under 90 days. On the first 150 entries, none of which spans more than
  30 days, the decay rows equal the static rows exactly.
- **Where decay acts here, it hurts, and feedback does not rescue it.**
  On the long histories decay costs about ten points at every depth and
  hides a needed memory on 11.2% of askings while pruning only 4.5% of
  the store. Feedback brings that to 9.8%, against 8.6% to 2.5% on
  LoCoMo. With one question per timeline the old evidence is already
  hidden the first time it is asked for, so it is never shown and never
  cited. Use cannot protect a memory that has not been used yet.
- **How to read the damage.** In a decay-only replay, 13 of the 16
  entries spanning 150 days or more lost a needed memory, and 12 of
  those 13 are knowledge-update questions. Their labelled evidence
  includes the turn with the old, superseded value. The measure counts
  that turn as needed, although the answer may need only the latest
  value. Answer accuracy was not measured, so part of this cost may be
  the store forgetting what it should.

With `gpt-4o-mini` naming what it used (one schedule), top 5 on repeat
askings:

| | static | feedback (model) | feedback (labels) |
|---|---:|---:|---:|
| First 150 entries | 56.5 | 59.4 | 61.5 |
| Histories of 31+ days | 66.4 | 66.8 | 72.0 |

On LongMemEval the model's account is weaker than the labels: it named
62 to 80% of the labelled evidence it was shown, against 87% on LoCoMo.
The reason was not tested. Long turns are shown to the model as
200-token excerpts, which may cut the evidence out, and the model may
reasonably ignore a turn holding a superseded value. One schedule, so
treat the sizes as rough.

## Other systems under this protocol: assessed, not run

Read from the current packages (`mem0ai` 2.2.1, `graphiti-core` 0.31.2,
`letta-client` 1.12.1) and vendor documentation. Effort and cost are
estimates.

| | Mem0 (open source) | Graphiti / Zep | Letta |
|---|---|---|---|
| Model call at write time | By default (extraction) | Always | Not for direct passage inserts |
| Forgetting by age | No (hosted platform re-ranks by recency of access; hides nothing) | No; contradicted facts are marked invalid | No |
| Use feedback | Hosted client only; effect on ranking undocumented | No | No |
| Simulated clock | No | Partly (caller-supplied valid time) | Partly (caller-supplied creation time) |
| Adapter effort | about 1 day | 2 to 3 days | about 2 days |
| Cost for LoCoMo | $2 to 5 | $15 to 40 | near zero in passages mode |

None of the three has both age-based forgetting and use feedback in a
form that can be driven on a simulated clock, so for them the decay and
feedback arms do not exist and "needed a hidden memory" is zero by
construction. What can be compared fairly is the static arm: the same
timelines, lags and revisits, each system in its documented default
write mode, with an evidence turn counted as retrieved when any
returned item derives from it. That is about one to one and a half
weeks of work and $50 to 200, and is not done here.

Since done for the static arm on three conversations:
[`compare.md`](compare.md).

## Limits

- Two datasets, and they disagree about decay. LoCoMo asks many
  questions of each conversation, so use protects memories before decay
  reaches them. LongMemEval asks one question per history, so it cannot.
  Real workloads sit somewhere between.
- The tables at the top were measured with the use lift at 0.5 and
  without relationships; "As shipped" below has the current defaults.
- Retrieval only. Whether a smaller, sharper store changes answers was
  not measured.
