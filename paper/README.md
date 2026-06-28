# SpectraForge paper — build instructions

This folder is a self-contained LaTeX + Word package for the paper
*"SpectraForge: A Physics-Grounded Synthetic Benchmark for Band Selection in Multi-Excitation
Hyperspectral Imaging, and an Honest Appraisal of Autoencoder-Based Selectors."*

## Contents

- `paper.md` — the canonical source (single source of truth for both .tex and .docx).
- `paper.tex` — LaTeX paper (generated from `paper.md`).
- `SpectraForge-paper.docx` — identical, more readable Word version (generated from `paper.md`).
- `figures/` — all figures (PNG) + `params.json` (the exact config/parameters used to generate each).
- `SpectraForge-summary.docx` / `summary.md` — one-page summary of findings.

## Build the PDF

```
cd paper
pdflatex paper.tex
pdflatex paper.tex      # run twice for references/captions
```

Requires a LaTeX distribution (TeX Live / MiKTeX) with `graphicx`, `booktabs`, `amsmath`, `hyperref`,
`caption`, `float`, `xcolor` (all standard). The figures are referenced relatively (`figures/...`), so
compile from inside this folder.

## Regenerate everything from source

```
# from the repo root, with the project venv:
python reports/paper_figures.py                                   # re-run sims -> figures/ + params.json
python reports/md_to_latex.py paper/paper.md paper/paper.tex      # -> LaTeX
python reports/md_to_docx.py  paper/paper.md paper/SpectraForge-paper.docx   # -> Word
```

All figures are produced by **re-running the simulations** (`paper_figures.py`); `figures/params.json`
records the exact configuration and parameters behind every figure.
