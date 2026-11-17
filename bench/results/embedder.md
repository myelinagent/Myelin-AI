# Embedder pilot: a small neural embedder, off the shelf and fine-tuned

Run on 2026-11-16 and 17. Scripts and data: `bench/embedder/`.

The question: does a small neural embedder retrieve better than the
static default (`minishlab/potion-retrieval-32M`), and does fine-tuning
one on synthetic conversations help further?

## What was run

- **Off the shelf:** `BAAI/bge-small-en-v1.5` (33M parameters) and
  `sentence-transformers/all-MiniLM-L6-v2` (22M), no prefixes.
- **Fine-tuned:** `bge-small-en-v1.5`, one pass over 12,326 pairs built
  from 450 of the 500 synthetic conversations in
  `bench/embedder/data/`, batch 16, three near-miss wrong turns per
  question, about 35 minutes on a 4-core CPU. On the 50 held-out
  synthetic conversations the answering turn was the top result for
  55.2% of questions after training against 38.3% before, and in the
  top 5 for 87.5% against 74.1%.
- **Measured with** `bench.lifetime` (LongHAUL), hybrid search, no model.

## LoCoMo, all ten conversations, three schedules

Share of questions with all their evidence in the top results (%).

| Embedder | Arm | top 5 | top 10 | top 20 |
|---|---|---:|---:|---:|
| Static default | static | 46.4 ± 0.3 | 55.8 ± 0.2 | 63.9 ± 0.2 |
| Static default | feedback | 52.7 | 60.6 | 64.0 |
| `all-MiniLM-L6-v2` | static | 52.4 ± 0.0 | 63.1 ± 0.1 | 70.4 ± 0.1 |
| `all-MiniLM-L6-v2` | feedback | 57.1 ± 0.3 | 66.6 ± 0.5 | 70.5 ± 0.1 |
| `bge-small-en-v1.5` | static | 56.5 ± 0.2 | 66.1 ± 0.3 | 74.4 ± 0.1 |
| `bge-small-en-v1.5` | feedback | 60.0 ± 0.4 | 69.8 ± 0.1 | 74.4 ± 0.2 |
| `bge-small-en-v1.5`, fine-tuned | static | 66.0 ± 0.2 | 75.4 ± 0.3 | 81.3 ± 0.4 |
| `bge-small-en-v1.5`, fine-tuned | feedback | 67.5 ± 0.3 | 75.5 ± 0.2 | 81.3 ± 0.4 |

The MiniLM rows were embedded through `fastembed`; the others through
`sentence-transformers` (`bench.embedder.evaluate`). `bge-small` gave
the same figures through both, to within 0.2.

## LongMemEval-S, first 100 entries, one schedule

94 of these entries have flagged evidence, so one question is about
one point.

| Embedder | top 5 | top 10 | top 20 |
|---|---:|---:|---:|
| Static default | 70.2 | 78.7 | 84.0 |
| `bge-small-en-v1.5` | 70.2 | 76.6 | 89.4 |
| `bge-small-en-v1.5`, fine-tuned | 71.3 | 80.9 | 87.2 |

## What it shows

- **On LoCoMo a small neural embedder is worth about ten points** over
  the static default at every depth, with no training, and fine-tuning
  adds about nine more.
- **None of that carries to LongMemEval.** There the three embedders
  are within what one schedule of 94 questions can resolve. Its
  messages are long user and assistant turns (about 1,000 characters on
  average); LoCoMo's, and the synthetic ones, are short messages
  between two people.
- **So the fine-tune's gain is, as far as this shows, a gain on
  LoCoMo-like data.** The synthetic conversations were written to
  resemble that kind of data, and the instructions for their questions
  were written with LoCoMo's question types in mind. No LoCoMo text was
  used, but the result should not be read as a general improvement.
- **Feedback adds less as the embedder improves.** In the top 5 on
  LoCoMo it adds 6.3 points with the static embedder, 3.5 with
  `bge-small` and 1.5 with the fine-tuned one. At top 20 it adds
  nothing with any of them.
- **Cost.** `all-MiniLM-L6-v2` took 4.6 ms per text on a 4-core CPU.
  `bge-small` embedded about 21 LongMemEval messages a second on the
  same machine. The static embedder is far faster and needs no runtime
  beyond numpy.

## Not done

- LongMemEval beyond the first 100 entries, or more than one schedule.
- Training data that looks like LongMemEval (long assistant turns,
  first-person questions; only about 10% of the synthetic questions are
  first-person).
- More than one pass of training, a larger base model, or distilling
  the tuned model back into a static one.
- The fine-tuned weights are not published; they can be rebuilt with
  the commands in `bench/embedder/README.md`.
