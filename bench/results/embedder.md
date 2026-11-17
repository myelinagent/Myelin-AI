# Embedder pilot: a small neural embedder, off the shelf and fine-tuned

Run on 2026-11-16 and 17, in two rounds. Scripts and data: `bench/embedder/`.

The question: does a small neural embedder retrieve better than the
static default (`minishlab/potion-retrieval-32M`), and does fine-tuning
one on synthetic conversations help further?

## What was run

- **Off the shelf:** `BAAI/bge-small-en-v1.5` (33M parameters) and
  `sentence-transformers/all-MiniLM-L6-v2` (22M), no prefixes.
- **Fine-tuned, round one:** `bge-small-en-v1.5`, one pass over 12,326
  pairs built from 450 of the 500 two-person synthetic conversations in
  `bench/embedder/data/`, batch 16, three near-miss wrong turns per
  question, about 35 minutes on a 4-core CPU. On the 50 held-out
  conversations the answering turn was the top result for 55.2% of
  questions after training against 38.3% before, and in the top 5 for
  87.5% against 74.1%.
- **Fine-tuned, round two:** the same recipe over 23,781 pairs from
  both data files: the 500 conversations above plus 260 histories
  between a person and an assistant and 240 sets of agent notes. On its
  held-out tenth the answering turn was the top result for 59.3% of
  questions against 43.8% before, and in the top 5 for 92.2% against
  80.6%. The held-out sets differ, so these do not compare with round
  one's.
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
| Fine-tuned, round one | static | 66.0 ± 0.2 | 75.4 ± 0.3 | 81.3 ± 0.4 |
| Fine-tuned, round one | feedback | 67.5 ± 0.3 | 75.5 ± 0.2 | 81.3 ± 0.4 |
| Fine-tuned, round two | static | 66.7 ± 0.1 | 75.7 ± 0.3 | 82.0 ± 0.2 |

The feedback arm was not measured for round two. The MiniLM rows were embedded through `fastembed`; the others through
`sentence-transformers` (`bench.embedder.evaluate`). `bge-small` gave
the same figures through both, to within 0.2.

## LongMemEval-S, 250 entries, static arm

A sample of 250 of the 500 entries, drawn evenly across the six
question types. The static default was run on three schedules; each
neural embedder on one, so differences of a point or two between them
are not resolved.

| Embedder | top 5 | top 10 | top 20 |
|---|---:|---:|---:|
| Static default | 64.9 ± 1.7 | 77.9 ± 0.5 | 86.6 ± 0.5 |
| `bge-small-en-v1.5` | 62.7 | 80.1 | 89.2 |
| Fine-tuned, round one | 65.1 | 82.6 | 90.0 |
| Fine-tuned, round two | 72.2 | 85.5 | 92.1 |

Decay costs each embedder three to four points at every depth, and
feedback changes nothing here, as with the static default.

An earlier version of this page reported the first 100 entries of the
file. The file is sorted by question type, so that slice held only two
of the six types, and its figures are withdrawn.

## What it shows

- **On LoCoMo a small neural embedder is worth about ten points** over
  the static default at every depth, with no training, and fine-tuning
  adds about ten more. Doubling the training data added under a point
  there.
- **On LongMemEval the off-the-shelf model is no better than the static
  default in the top 5**, and two to three points better at 10 and 20.
  Round one, trained only on short two-person messages, adds two to
  three points to that.
- **Training data shaped like the target matters.** Round two added
  assistant histories and first-person questions, and gained seven
  points in the top 5 over round one on LongMemEval while leaving
  LoCoMo where it was. Against the static default it is 7.3 points
  ahead in the top 5 and 5.5 to 7.6 ahead at 10 and 20.
- **The synthetic data was written with both benchmarks in mind.** No
  benchmark text was used, but the briefs describe the kinds of
  conversation and question each benchmark holds, so these gains are
  not evidence about data of another kind.
- **Feedback adds less as the embedder improves.** In the top 5 on
  LoCoMo it adds 6.3 points with the static embedder, 3.5 with
  `bge-small` and 1.5 with the fine-tuned one. At top 20 it adds
  nothing with any of them.
- **Cost.** `all-MiniLM-L6-v2` took 4.6 ms per text on a 4-core CPU.
  `bge-small` embedded about 21 LongMemEval messages a second on the
  same machine. The static embedder is far faster and needs no runtime
  beyond numpy.

## Not done

- The full 500 LongMemEval entries, or more than one schedule for the
  neural embedders there.
- The feedback and decay arms on LoCoMo for round two.
- More than one pass of training, a larger base model, or distilling
  the tuned model back into a static one.
- The fine-tuned weights are not published; they can be rebuilt with
  the commands in `bench/embedder/README.md`.
