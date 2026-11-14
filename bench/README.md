# Benchmark harness

Runs a memory system through a benchmark from one command, with no UI and no
prompts. It lives outside the `myelin` package and has no dependencies beyond
Myelin itself.

## LongMemEval

Get the data (about 280 MB):

```bash
curl -L -o longmemeval_s_cleaned.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Retrieval metrics only — no model, no key, about a minute:

```bash
python -m bench.longmemeval run --system myelin --data longmemeval_s_cleaned.json --out bench/runs/myelin
```

Full run with answers and grading:

```bash
python -m bench.longmemeval run --system myelin --data longmemeval_s_cleaned.json --out bench/runs/myelin-qa --answerer gpt-4o-mini --judge gpt-4o
```

Size the answerer's context by tokens instead of a fixed 10 turns:

```bash
python -m bench.longmemeval run --system myelin --data longmemeval_s_cleaned.json --out bench/runs/myelin-6k --token-budget 6000
```

Add `--max-turn-tokens 300` to shorten longer turns to a query-focused
excerpt before they are counted against the budget.

`--whole-turns 3` keeps the three best-ranked turns complete. `--prompt`
selects the answer prompt: `reasoned` (default) or `direct` from
`myelin.build_answer_prompt`, or `v1`, the plain prompt the first baselines
used.

For answer models that reason before answering (for example `gpt-5-mini`),
raise the output cap with `--max-answer-tokens 4000`.

`--embedder minishlab/potion-retrieval-32M` turns on hybrid search with a
static embedding model (needs `pip install model2vec`).

Compare runs:

```bash
python -m bench.longmemeval report bench/runs/myelin bench/runs/bm25
```

## LoCoMo

Get the data (CC BY-NC 4.0, about 3 MB):

```bash
curl -L -o locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

```bash
python -m bench.locomo run --system myelin --data locomo10.json \
    --out bench/runs/locomo --embedder minishlab/potion-retrieval-32M \
    --answerer gpt-4o-mini --judge gpt-4o-mini
python -m bench.locomo report bench/runs/locomo
```

Each conversation is ingested once, one memory per utterance, and its
questions are asked against that store; the adversarial category is
skipped. Omit `--answerer` for retrieval metrics only. The judge prompt is
the one published LoCoMo scores use, with `gpt-4o-mini`. The options
shared with the LongMemEval runner mean the same here. Results:
[`results/locomo.md`](results/locomo.md).

The sections below apply to both runners.

### Systems

| `--system` | What it is |
|---|---|
| `myelin` | Myelin as shipped: one memory per turn, default `search()` ranking |
| `bm25` | Control: textbook BM25 over the same turns |
| `oracle` | Control: the answerer is handed exactly the evidence sessions (reader ceiling) |

### Models

Any OpenAI-compatible endpoint works:

| Variable | Meaning |
|---|---|
| `MYELIN_BENCH_BASE_URL` | Endpoint root, default `https://api.openai.com/v1` |
| `MYELIN_BENCH_API_KEY` | Bearer token; falls back to `OPENAI_API_KEY`; if neither is set no `Authorization` header is sent |

The benchmark's official judge is `gpt-4o`. Model ids are passed through
unchanged, so use the endpoint's own names (for example `openai/gpt-4o` on
OpenRouter).

### What a run writes

| File | Contents |
|---|---|
| `manifest.json` | Dataset hash, system, top-k, models, prompt version, git commit |
| `answers.jsonl` | One row per question: retrieval recall, timings, tokens, answer |
| `hypotheses.jsonl` | `question_id` + `hypothesis`, the input format of the official `evaluate_qa.py` |
| `judged.jsonl` | Judge verdict per question, in the official log format |
| `summary.json` | Accuracy (overall, task-averaged, per type, abstention), recall, cost |

Runs are checkpointed per question: rerunning the same command continues an
interrupted run and retries failed calls. A run directory is tied to its
settings; changing the system, top-k, token budget, excerpt size, answerer or
dataset needs a new `--out`.

### Rules for comparable numbers

- Keep the answerer, judge, prompt version and top-k fixed across the runs
  you compare; the manifest records them.
- Report overall and task-averaged accuracy, and the control runs beside
  the system's.
- State whether `cite()` learning was on. It is off in this harness: feeding
  the scorer's verdict back before scoring would leak the answer key.

Recorded results are in `bench/results/`. On LongMemEval-S a 6,000-token
budget scored higher with `gpt-4o` than 4,000 or 8,000; see
`longmemeval_s_stage2.md`.
