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

Compare runs:

```bash
python -m bench.longmemeval report bench/runs/myelin bench/runs/bm25
```

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
settings; changing the system, top-k, token budget, answerer or dataset needs
a new `--out`.

### Rules for comparable numbers

- Keep the answerer, judge, prompt version and top-k fixed across the runs
  you compare; the manifest records them.
- Report overall and task-averaged accuracy, and the control runs beside
  the system's.
- State whether `cite()` learning was on. It is off in this harness: feeding
  the scorer's verdict back before scoring would leak the answer key.

Recorded results are in `bench/results/`.
