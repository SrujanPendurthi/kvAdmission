# LaTeX submission source

```
paper.tex     the paper. Self-contained `article` class - no venue style needed.
figure.tex    the horizon figure, as TikZ. Regenerate with:
                python3 scripts/make_latex.py
refs.bib      12 references.
```

## Build

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

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

The four arXiv-only references (`indexmem2026`, `wgkv2025`,
`learningtoevict2026`, `triroute2026`) have partial metadata: author lists were
not fully recovered during the literature search and should be completed from
the arXiv listings before submission.
