# Paper draft

Two drafts for arXiv that cite each other.

- `main.tex` / `main.pdf` (11 pages): LongHAUL, the memory-over-time
  evaluation protocol, and the case for longitudinal benchmarks of agent memory.
- `myelin.tex` / `myelin.pdf` (8 pages): the system itself, its design,
  interface, benchmark results, cost, and where it fails. Its numbers
  come from the same result files listed below, plus the write
  throughput and latency table from `bench/results/scale.md`.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex myelin && bibtex myelin && pdflatex myelin && pdflatex myelin
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

- [x] Author and affiliation set (Scott Benson, Apex Insights). No
      contact address is printed; add one if wanted.
- [ ] The system paper cites the benchmark paper as "in preparation";
      replace with its arXiv identifier once it has one, and cite the
      system paper from the benchmark paper likewise.
- [x] The protocol is named LongHAUL (Long-horizon Assessment of Use and
      Loss); the code runs it as `bench.lifetime`. A web search on
      2026-11-15 found no other benchmark with that name; search again
      before submitting.
- [ ] The three benchmark entries added on 2026-11-16 (MemoryAgentBench,
      AMA-Bench, MemRiskBench) were taken from their arXiv pages by an
      automated reader; check authors, titles and the paragraph's
      description of each against the papers by hand.
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
- [x] A second dataset (LongMemEval-S) and a model in the loop
      (`gpt-4o-mini`) have been run; see `bench/results/lifetime.md`.
- [ ] Other systems have not been run under the protocol. The
      assessment in `bench/results/lifetime.md` puts a static-arm
      comparison with Mem0, Graphiti and Letta at one to two weeks and
      $50 to 200.
- [ ] The model arms on LongMemEval-S are one schedule; repeat them
      before relying on the sizes.
