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
- [ ] `refs.bib` was written from memory and from pages seen while
      drafting: check every title, author list, venue and year. The
      Hindsight entry has no author list.
- [ ] Other systems' scores are quoted as their authors report them;
      re-check them against the sources on the day of submission.
- [ ] LoCoMo is CC BY-NC 4.0: check that the intended use of the
      results is compatible with it.
- [ ] Consider adding a figure of the protocol's timeline and a plot of
      needed-but-hidden against store pruned from the decay sweep.
- [ ] The limitations section lists what a reviewer is likely to ask
      for first: a second dataset, a model in the loop, and other
      systems run under the same protocol.
