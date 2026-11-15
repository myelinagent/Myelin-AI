# Third-party notices

The `myelin` package itself contains no third-party code. The benchmark
harness under `bench/` and the tests reuse a few short pieces of other
projects, listed here. No dataset is redistributed in this repository.

## Mem0 — LoCoMo judge prompt

`bench/locomo.py` (`JUDGE_PROMPT`) is the `ACCURACY_PROMPT` from
`evaluation/metrics/llm_judge.py` in [mem0ai/mem0](https://github.com/mem0ai/mem0)
(tag v1.0.0), with whitespace-only changes.

Copyright 2023 Taranjeet Singh. Licensed under the Apache License,
Version 2.0: <https://www.apache.org/licenses/LICENSE-2.0>.

## LongMemEval — judge prompts and one test item

The judge prompts in `bench/longmemeval.py` are ported from
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(`src/evaluation/evaluate_qa.py`), and the tests use one question from
that dataset as a fixture.

MIT License. Copyright (c) 2024 Di Wu.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Datasets used for evaluation (not included)

- **LoCoMo** (Maharana et al., ACL 2024,
  <https://github.com/snap-research/locomo>): CC BY-NC 4.0,
  <https://creativecommons.org/licenses/by-nc/4.0/>. Non-commercial use
  only.
- **LongMemEval** (Wu et al., ICLR 2025,
  <https://github.com/xiaowu0162/LongMemEval>): MIT.

Scores computed on them are reported for research comparison. Run
outputs (`bench/runs/`), which contain dataset text, are not tracked.
