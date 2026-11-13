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

## With Myelin's answer prompt

Added later the same day. The table above used a plain prompt (`v1`).
`build_answer_prompt` (style `reasoned`) tells the answerer how to handle
changed facts, totals and orderings, dates relative to a conversation,
requests for advice, and questions about something the history does not
cover. Only what follows the final `Answer:` line is graded.

Stage 1 here is BM25 search, a 6,000-token budget, 300-token excerpts
with the three best-ranked turns kept whole (`--whole-turns 3`), prompt
`myelin-reasoned-v2`. Judge `gpt-4o`. One run each.

| | `gpt-4o-mini`, stage 1 | `gpt-4o-mini`, oracle | `gpt-4o`, stage 1 | `gpt-4o`, oracle |
|---|---:|---:|---:|---:|
| **Accuracy (overall)** | **77.2** | 79.4 | **82.8** | 93.6 |
| Accuracy (task-averaged) | 77.5 | 81.3 | 82.9 | 93.9 |
| single-session-user | 97.1 | 91.4 | 95.7 | 98.6 |
| single-session-assistant | 96.4 | 98.2 | 96.4 | 100.0 |
| single-session-preference | 50.0 | 70.0 | 60.0 | 86.7 |
| multi-session | 62.4 | 65.4 | 69.2 | 88.7 |
| temporal-reasoning | 75.9 | 78.2 | 82.7 | 93.2 |
| knowledge-update | 83.3 | 84.6 | 93.6 | 96.2 |
| abstention subset | 66.7 | 66.7 | 70.0 | 83.3 |
| All evidence turns in context | 85.3 | — | 85.3 | — |

### How it got there (`gpt-4o-mini`, stage 1 retrieval)

| Step | Overall |
|---|---:|
| 1.4.0 search, top 10, plain prompt | 60.4 |
| Stage 1 retrieval, plain prompt | 64.8 |
| + Myelin prompt, first version | 74.4 |
| + top three turns whole, stricter abstention rule | 77.2 |

### What the revisions did

The first version of the prompt was measured with excerpts on every long
turn. It showed two weaknesses, and the second version addresses both:

| | `gpt-4o-mini` before → after | `gpt-4o` before → after |
|---|---:|---:|
| Overall | 74.4 → 77.2 | 84.0 → 82.8 |
| single-session-assistant | 82.1 → 96.4 | 91.1 → 96.4 |
| abstention subset | 53.3 → 66.7 | 73.3 → 70.0 |
| single-session-preference | 56.7 → 50.0 | 70.0 → 60.0 |
| multi-session | 62.4 → 62.4 | 72.9 → 69.2 |

- Keeping the best-ranked turns whole recovered the answers that sit
  inside long assistant replies, for both models.
- The stricter abstention rule helped the small model and did nothing
  measurable for `gpt-4o`.
- The overall change is +2.8 for `gpt-4o-mini` and -1.2 for `gpt-4o`. With
  500 questions and single runs, one standard error is about 1.7 points,
  so neither overall change is established; the category moves of 5 points
  or more on small categories (30 preference questions) are not either.
  Repeat runs are needed before tuning further on differences this size.

### What this says

- **With `gpt-4o-mini`, stage 1 is 2.2 points under perfect retrieval.**
  For a small answerer the reader is the limit, not the memory.
- **With `gpt-4o`, retrieval is the limit**: 82.8 against 93.6. The gap is
  in multi-session (69.2 against 88.7), preference (60.0 against 86.7) and
  temporal (82.7 against 93.2) questions, which is where evidence is
  spread across sessions or worded unlike the question.
- For orientation only (different harnesses and prompts): plain retrieval
  with `gpt-4o` is reported at about 80 on this benchmark and the best
  self-reported `gpt-4o` results are in the mid-80s.

## Next

1. Close the retrieval gap for `gpt-4o`: an embedding channel beside BM25
   and facts extracted at write time appended to the indexed text.
2. Repeat runs (three seeds) to put error bars on these numbers.

Machine-readable summaries: `longmemeval_s_qa_baseline.json`,
`longmemeval_s_qa_stage1.json`, `longmemeval_s_qa_oracle.json` (plain
prompt) and `longmemeval_s_qa_v2_*.json` (Myelin prompt).
