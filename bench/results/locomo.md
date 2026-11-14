# LoCoMo

Ten long two-person conversations (5,882 utterances), 1,540 questions —
all but the adversarial category, which has no gold answer and is left
out of every published comparison. Each conversation is ingested once,
one memory per utterance dated by its session, and all of its questions
are asked against that store.

Settings match the LongMemEval runs: 6,000-token budget, 300-token
excerpts with the three best-ranked utterances whole, prompt
`myelin-reasoned-v2`, hybrid search with `minishlab/potion-retrieval-32M`.
The judge is `gpt-4o-mini` with the prompt from Mem0's evaluation code,
which is what published LoCoMo "LLM-judge" scores use. One run each,
2026-11-14.

## Answer accuracy (Myelin, hybrid search)

| | Questions | `gpt-4o-mini` answering | `gpt-5-mini` answering |
|---|---:|---:|---:|
| **Judged correct (overall)** | 1,540 | 78.7 | **89.0** |
| single-hop | 841 | 88.6 | 93.2 |
| multi-hop | 282 | 77.7 | 84.8 |
| temporal | 321 | 59.8 | 86.6 |
| open-domain | 96 | 58.3 | 71.9 |
| Context tokens per question | | 6,011 | 6,009 |

One standard error on the overall figure is about 1 point (0.8 at 89%).

## Retrieval (no model involved)

Share of questions whose every evidence utterance is in the 6,000-token
context, and the mean share of evidence utterances shown.

| | BM25 control | Myelin, lexical | Myelin, hybrid |
|---|---:|---:|---:|
| All evidence utterances in context | 71.2 | 76.9 | 80.3 |
| Share of evidence utterances in context | 78.4 | 83.2 | 86.9 |
| Utterances in context | 184.5 | 177.0 | 190.6 |
| Retrieve per question (ms) | 1 | 3 | 5 |

## What the numbers say

- **The answering model matters most on dates.** With `gpt-4o-mini`, 129
  of 321 temporal questions were wrong, and in 103 of those every
  evidence utterance was in the context. The usual error is off by one:
  a speaker says "yesterday" and the answer gives the conversation's date.
  `gpt-5-mini` reads the same context and gets 86.6%. Resolving relative
  dates when a memory is written would take this burden off the reader.
- **Retrieval has room.** One question in five is answered without all of
  its evidence in view; multi-hop questions need several utterances from
  different sessions.
- **Open-domain questions** ask for inference beyond what was said
  ("would she enjoy…"), and are the smallest category.

## Caveats

- The judge prompt tells the judge to be generous, and an independent
  audit found roughly 6% of the answer key wrong. Scores are comparable
  with other LoCoMo numbers produced the same way, and are not a measure
  of correctness.
- Systems publish LoCoMo scores with different answer models, context
  sizes and prompts, so cross-system comparison is orientation only.
- The whole of a LoCoMo conversation is about 22,000 tokens (13,000 to 26,000), so a model
  given everything is a strong baseline; this run shows it 6,000.
- The dataset is CC BY-NC 4.0 and is not redistributed here.
