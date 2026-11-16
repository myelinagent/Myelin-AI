# A retrieval embedder tuned for conversational memory (pilot)

A pilot to find out whether a small sentence embedder, fine-tuned on
synthetic conversations, retrieves better on LongHAUL than the same
model off the shelf.

## What is here

- `data/conversations.jsonl.gz`: 500 invented two-person conversations
  (49,210 turns over 5 to 9 dated sessions each) with 9,970 questions,
  each tagged with the turn or turns that answer it. All of it was
  written by a language model from the seeded briefs in `briefs.py`;
  none of it comes from LoCoMo, LongMemEval or any other dataset.
- `briefs.py`: writes the briefs.
- `validate.py`: checks a conversation's format.
- `pairs.py`: turns conversations into training pairs.
- `train.py`: fine-tunes `BAAI/bge-small-en-v1.5`.
- `evaluate.py`: runs LongHAUL with a sentence-transformers model.

## Run it

```bash
pip install -e ".[static]" sentence-transformers datasets accelerate
```

```bash
python -m bench.embedder.pairs bench/embedder/data/conversations.jsonl.gz --out pairs/
```

```bash
python -m bench.embedder.train --pairs pairs/ --out tuned/ --epochs 2
```

```bash
python -m bench.embedder.evaluate tuned/ --data locomo10.json
```

`train` uses the Mac GPU (MPS) or CUDA when there is one. For the
baseline, give `evaluate` the model id `BAAI/bge-small-en-v1.5` in place
of `tuned/`. `locomo10.json` is the LoCoMo file the other benchmarks
use; it is not in this repository.

## What is known about the data

- Every conversation passes `validate.py`.
- For single-fact questions whose answer has a name or a number, the
  cited turn contains it in 96% of cases; the misses sampled were
  correct turns that spell a number as a word.
- About 10% of questions are in the first person; a quarter was asked
  for.
- About a fifth of questions share an evidence turn with another.
- One model wrote all of it, so its conversations may be more alike
  than real ones. LoCoMo and LongMemEval are the test of whether
  training on it transfers.
