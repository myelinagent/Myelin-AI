# Paper draft

`main.tex` is a draft for arXiv: the memory system, the memory-over-time
evaluation protocol, and the case for longitudinal benchmarks of agent
memory. `main.pdf` is the compiled draft (8 pages).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs a TeX distribution with `booktabs`, `natbib`, `enumitem`,
`hyperref` and Latin Modern (on Debian or Ubuntu: `texlive-latex-base
texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
lmodern`).

## Where each number comes from

| In the paper | Source |
|---|---|
| Table 1 and the static-benchmark discussion | `bench/results/final.md`, `longmemeval_s_stage3.md`, `longmemeval_s_accuracy.md`, `locomo.md` |
| Fusion and top-k figures; held-out cite feedback (42.4, 47.7, 47.8) | `bench/results/fusion.md` |
| Excerpt length and evidence coverage (92.3, 97.0, 96.4) | `bench/results/longmemeval_s_stage3.md` |
| Tables 2 and 3, the decay sweep, the unused penalty, ranking with related support | `bench/results/lifetime.md` |
| Latency at 200,000 memories | `bench/results/scale.md` |
| Published scores of other systems | `bench/results/final.md`, with the sources cited in `refs.bib` |

Every table can be regenerated with the commands in `bench/README.md`.

## Before submitting

- [ ] Authors, affiliation and contact are placeholders.
- [ ] The protocol has no name yet; the code calls it `bench.lifetime`.
- [ ] `refs.bib` was checked against primary sources on 2026-11-14.
      Two details are still to confirm by hand (noted at the top of the
      file): the BM25 survey's volume and pages, and the exact title of
      the LoCoMo audit post.
- [ ] Other systems' scores were re-checked against their sources on
      2026-11-14 (`bench/results/final.md`); they change often, so check
      again on the day of submission and read the decimals by eye.
- [ ] LoCoMo is CC BY-NC 4.0. Reporting its scores in a preprint is
      the use it was released for. Using them in the README of a
      company-owned library is a grey area, and in marketing for a paid
      product the clearest risk: ask counsel, or Snap Research, first.
      LongMemEval is MIT and carries no such limit.
- [ ] Consider adding a figure of the protocol's timeline and a plot of
      needed-but-hidden against store pruned from the decay sweep.
- [ ] The limitations section lists what a reviewer is likely to ask
      for first: a second dataset, a model in the loop, and other
      systems run under the same protocol.
