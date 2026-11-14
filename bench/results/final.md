# Results on `main` (2026-11-14)

Runs of the code as merged, after the retrieval changes in
`longmemeval_s_stage3.md` and `fusion.md`. Hybrid search
(`minishlab/potion-retrieval-32M`), 6,000-token `recall()` context,
prompt `myelin-reasoned-v2`.

## LongMemEval-S (500 questions, judge `gpt-4o`, the benchmark's prompts)

| | `gpt-5-mini`, run 1 | run 2 | run 3 | run 4 | Mean | `gpt-4o`, one run |
|---|---:|---:|---:|---:|---:|---:|
| **Accuracy (overall)** | 90.2 | 91.0 | 89.4 | 89.8 | **90.1** | **88.6** |
| Accuracy (task-averaged) | 91.0 | 91.9 | 90.5 | 90.2 | 90.9 | 88.7 |
| single-session-user | 97.1 | 98.6 | 94.3 | 95.7 | 96.4 | 97.1 |
| single-session-assistant | 96.4 | 96.4 | 96.4 | 96.4 | 96.4 | 94.6 |
| single-session-preference | 83.3 | 86.7 | 86.7 | 83.3 | 85.0 | 76.7 |
| multi-session | 78.9 | 81.2 | 78.2 | 81.2 | 79.9 | 80.5 |
| temporal-reasoning | 92.5 | 92.5 | 92.5 | 93.2 | 92.7 | 89.5 |
| knowledge-update | 97.4 | 96.2 | 94.9 | 91.0 | 94.9 | 93.6 |
| abstention subset | 93.3 | 86.7 | 86.7 | 83.3 | 87.5 | 70.0 |

All evidence turns are in the context for 96.4% of questions. With the
evidence sessions handed over directly, `gpt-5-mini` scores 93.4% and
`gpt-4o` 93.6% (`longmemeval_s_accuracy.md`, `longmemeval_s_stage2.md`).

The four `gpt-5-mini` runs before the fusion change averaged 91.3
(89.6 to 93.0). The difference from 90.1 is about one and a half
standard errors: the eight runs together average 90.7, and that is the
fairest single figure for this configuration.

Where it started: 60.4% with `gpt-4o-mini` on release 1.4.0; 89.0% with
`gpt-5-mini` and 87.4% with `gpt-4o` before the changes of 2026-11-14.

## LoCoMo (1,540 questions, judge `gpt-4o-mini`, the prompt published scores use)

| | `gpt-5-mini`, facts indexed |
|---|---:|
| **Judged correct (overall)** | **92.6** |
| single-hop | 95.7 |
| multi-hop | 90.8 |
| temporal | 91.0 |
| open-domain | 76.0 |

One run; the same setting scored 91.9 before the excerpt and fusion
changes, and two runs of one setting differ by about a point. Without
write-time facts: 89.0 (`locomo.md`).

## Published scores, for orientation

Self-reported unless noted; answering model, judge and prompts differ,
so none of these is a like-for-like comparison.

| LongMemEval-S | Score | Answering model |
|---|---:|---|
| Mastra Observational Memory | 94.9 | gpt-5-mini |
| Zep | 94.7 | gpt-5.4 |
| Mem0 | 92.5 | not stated |
| Hindsight | 91.4 | Gemini 3 Pro |
| **Myelin** | **90.1** (eight runs: 90.7) | gpt-5-mini |
| Hindsight | 89.0 | open 120B model |
| **Myelin** | **88.6** | gpt-4o |
| Mastra | 84.2 | gpt-4o |
| Zep, measured by others | 71.2 | gpt-4o |
| Mem0, measured by others | 64.2 | gpt-4.1-mini |

| LoCoMo | Score |
|---|---:|
| Zep | 94.7 (58 to 79 when measured by others) |
| **Myelin** | **92.6** |
| Mem0 | 92.5 (58 to 66 when measured by others) |
| MemMachine | 91.7 |
| Hindsight | 85.7 |
| Letta, filesystem | 74.0 |

### Open source or paid

As far as public sources say on 2026-11-14; check each project's licence
before relying on this.

| System | Code | Paid offering | What the published score was run on |
|---|---|---|---|
| **Myelin** | MIT, all of it; runs locally on SQLite | none | this repository |
| Mem0 | Apache-2.0 library | hosted platform | vendor's own evaluation |
| Zep | Graphiti engine is Apache-2.0; the self-hosted Zep Community Edition was retired in 2025 | Zep Cloud | Zep Cloud |
| Mastra | Apache-2.0 framework, with some enterprise-licensed parts | Mastra Cloud and enterprise licence | the open framework's Observational Memory |
| Hindsight | MIT, self-hostable | Vectorize cloud | the open project |
| Letta | Apache-2.0 | Letta Cloud | the open project |
| MemMachine | reported as Apache-2.0 (not checked here) | vendor cloud | own preprint |

Zep is the only entry whose published scores cannot be reproduced with
freely available code. The answering and judging models are paid APIs
for every system, this one included.

An independent audit of LoCoMo found 6.4% of the answer key wrong and
a judge that accepts many wrong answers; a perfect system would score
about 93.6.
