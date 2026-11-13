# LongMemEval-S answer accuracy: baseline, stage 1, oracle

First end-to-end numbers. All 500 questions, 2026-11-13.

- **Answerer:** `gpt-4o-mini` (via OpenRouter as `openai/gpt-4o-mini`),
  temperature 0, prompt `lme-answer-v1`.
- **Judge:** `gpt-4o` with the benchmark's own judge prompts.
- **Runs:** one each; no repeat runs, so differences of a point or two are
  within noise.

| | Baseline | Stage 1 | Oracle control |
|---|---:|---:|---:|
| What the answerer sees | top 10 turns, original token-overlap search | BM25 search, 6,000-token budget, 300-token excerpts | exactly the evidence sessions |
| **Accuracy (overall)** | 60.4 | **64.8** | 75.2 |
| Accuracy (task-averaged) | 60.4 | 62.9 | 72.6 |
| single-session-user (70) | 90.0 | 87.1 | 95.7 |
| single-session-assistant (56) | 92.9 | 87.5 | 100.0 |
| single-session-preference (30) | 6.7 | 13.3 | 20.0 |
| multi-session (133) | 36.1 | 54.1 | 66.2 |
| temporal-reasoning (133) | 54.9 | 59.4 | 71.4 |
| knowledge-update (78) | 82.1 | 75.6 | 82.1 |
| abstention subset (30) | 90.0 | 83.3 | 83.3 |
| All evidence turns in context | 61.1 | 86.0 | — |
| Context tokens per question (measured) | 3,141 | 5,206 | 5,735 |

## What this says

- **Stage 1 is worth 4.4 points overall**, almost all of it from
  multi-session questions (36.1 to 54.1), where more of the scattered
  evidence now reaches the answerer.
- **It costs a few points where one turn was enough.** Single-session and
  knowledge-update questions were already well served by 10 turns; four
  times as many turns, some shortened, gave the small answerer more to get
  wrong. Knowledge-update (82.1 to 75.6) is the clearest case: more context
  means more stale values next to the current one.
- **The reader is now the larger gap.** With perfect retrieval this
  answerer and prompt reach 75.2. Stage 1 is 10.4 points below that ceiling
  while 86% of questions already have every evidence turn in context.
- **Preference questions fail everywhere (6.7 / 13.3 / 20.0)**, including
  with perfect retrieval. The prompt tells the model to say it does not
  know when the history lacks the answer; these questions want a
  suggestion informed by the user's history. This is a prompt fault.

## Against published numbers

Not directly comparable: published LongMemEval-S results mostly use
`gpt-4o` or newer as the answerer, and the answerer alone moves a fixed
memory system by about ten points. For orientation only, plain retrieval
with `gpt-4o` is reported at about 80 and the leaders at 90 to 96 with
newer models.

## Next

1. Fix the answer prompt for preference questions and for conflicting
   values (prefer the most recent), then re-measure with the oracle control.
2. Re-check whether excerpts cost single-session-assistant accuracy by
   running the stage 1 setup with `--max-turn-tokens 0`.
3. Run the same three with `gpt-4o` as answerer for comparability.

Machine-readable summaries: `longmemeval_s_qa_baseline.json`,
`longmemeval_s_qa_stage1.json`, `longmemeval_s_qa_oracle.json`.
