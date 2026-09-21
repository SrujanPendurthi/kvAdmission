# LaTeX submission source

Two framings of the same evidence, sharing one figure and one bibliography.

```
paper.tex               specification-first. Leads with the oracle, the
                        specification it yields, and the screening test.
                        The recommended primary. ~2 columns shorter.
paper-measurement.tex   measurement-first. Leads with the dead zone; keeps
                        the replication detail and an errata appendix.
figure.tex              the horizon figure, as TikZ. Shared. Regenerate:
                          python3 scripts/make_latex.py
refs.bib                12 references. Shared.
```

## Build

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
pdflatex paper-measurement && bibtex paper-measurement && \
  pdflatex paper-measurement && pdflatex paper-measurement
```

## Which one

`paper.tex` survives the literature better: position-resolved evaluation of
compression and middle under-retention both have prior art (see the related
work section), so leading with the dead zone is partly a restatement, while
the oracle-derived specification and the screening test have none. It also
makes the absence of a proposed method structural rather than a gap — a paper
that specifies a target and supplies a rejection test is not expected to ship
a signal meeting it.

`paper-measurement.tex` is the better read if the audience cares more about
the phenomenon than the tooling, and it carries the errata appendix.

## Switching to a venue style

Replace the `\documentclass` / `\usepackage[margin=1in]{geometry}` block with
the venue's package (e.g. `\usepackage{neurips_2026}`) and drop the `geometry`
line. Everything else is standard: `booktabs`, `tikz`, `natbib`, `amsmath`.

The figure is TikZ rather than an included image deliberately — there is no
SVG-to-PDF conversion step, and it regenerates from
`results/gpu-distance-curve.json` if the numbers change.

## NOT COMPILED

No LaTeX toolchain was available on the machine where this was written, so
**this source has never been run through `pdflatex`**. It passes static checks
(environments balanced, braces balanced, all `\cite` keys resolve against
`refs.bib`, all `\ref` targets have labels, TikZ coordinates numeric) but
expect to fix small things on the first real build — most likely table column
widths and figure placement.

All 12 references now carry full author lists, recovered from the arXiv
listings. One entry had been wrong rather than merely incomplete: arXiv
2512.17452 was carried through earlier drafts as "WG-KV: Write-Gated KV Cache
Admission", which is not its title. It is *KV Admission: Learning What to Write
for Efficient Long-Context LLM Inference* (Huang, Hsiu, Fang, Chen). The bib key is
now `kvadmission2025` and the prose in both papers is corrected.
