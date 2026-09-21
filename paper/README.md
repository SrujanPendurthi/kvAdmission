# LaTeX submission source

```
paper.tex               the paper. Specification-first framing.
tables.tex              GENERATED: every table and experimental number, as macros
figure.tex              GENERATED: the retention-horizon figure
figure-crowding.tex     GENERATED: the crowding-law figure
figure-screen.tex       GENERATED: the candidate-screen figure
refs.bib                references, each checked against its arXiv listing
paper-measurement.tex   NOT MAINTAINED. The earlier measurement-first framing,
                        frozen at 65777a5 with hand-typed numbers.
```

Regenerate everything marked GENERATED from `results/*.json`:

```bash
python3 scripts/make_latex.py
```

Never edit a number in `paper.tex` or a generated file by hand. Earlier drafts
did, and a review found a table stitched from three runs, a stale count and a
wrong span tally; generating them removes that class of error.

## Build

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
# or: tectonic paper.tex
```

## arXiv

arXiv does not run BibTeX. Upload `paper.tex`, `tables.tex`, the three
`figure*.tex` files and the generated `paper.bbl`. Upload them all to Overleaf
as well: a missing `\input` file is a fatal error there.

## Switching to a venue style

Replace the `\documentclass` / `\usepackage[margin=1in]{geometry}` block with
the venue's package (e.g. `\usepackage{neurips_2026}`). Everything else is
standard: `booktabs`, `tikz`, `natbib`, `amsmath`.
