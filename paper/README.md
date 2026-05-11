# Paper — Hyperbolic JEPA

NeurIPS 2024 workshop submission. LaTeX source + bibliography + style file.

## Build

```
make            # pdflatex + bibtex + 2x pdflatex -> main.pdf
make clean      # remove intermediates
make distclean  # also remove main.pdf
```

## Files

- `main.tex` — paper source
- `references.bib` — bibliography (plainnat style)
- `neurips_2024.sty` — official NeurIPS 2024 style file
- `outline.md` — author-facing outline with per-table status, updated as results landed
- `main.pdf` — compiled output (10 pages: 6.5 body + references)

## Submission checklist

- [ ] Switch `\usepackage[preprint]{neurips_2024}` to `\usepackage{neurips_2024}` for anonymous submission
- [ ] Set author field to `Anonymous Author(s) \\ Affiliation`
- [ ] Fill in workshop-specific reviewer/checklist additions if required
- [ ] Replace repo URL placeholder `[anonymized]` after de-anonymization
