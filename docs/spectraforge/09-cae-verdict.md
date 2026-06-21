# 09 — Verdict: did the published CAE work?

**Question asked before pushing:** did the convolutional autoencoder (the published method, `C0`
`HyperspectralCAEWithMasking`) work as expected? **Answer: no — robustly, on every synthetic regime,
and the one remaining excuse for it ("real data is dense, so it works there") is now refuted.**

## The evidence, in one place

| experiment | CAE F1 | recon R | infl–signal corr | CAE picks informative bands? |
|------------|-------:|--------:|-----------------:|:----------------------------:|
| clean, bright separated peaks (doc 05) | 0.329 (< random) | +0.001 | −0.810 | no (peak_recovery 0.00) |
| realistic confounded (doc 06) | 0.364 | ≈0 | <0 | no (17% disc) |
| extended suite, realistic (doc 08) | 0.320 | +0.842* | −0.397 | no (14% disc) |
| **density sweep (this doc)** | **0.32–0.34 at every density** | **≈0 at every density** | **−0.95 → −0.42** | **no — 0% at every density** |

*the doc-08 number is the *enlarged* deep MLP C5; the published spatial CAE's R is ≈0.

Across four independent setups the CAE classifies at **chance or below**, reconstructs essentially
**nothing** (R ≈ 0), and its perturbation-influence is **anti-correlated with signal** — so it
systematically selects the *least* informative bands.

## The decisive new test: does density rescue it?

doc 02 explained the synthetic failure as a **sparsity** artifact: after global min-max normalization
the synthetic cube is "mostly ≈0", so "predict ~0" is a cheap optimum; **real data is dense**, so the
CAE was assumed to learn there and the published real-data parity was attributed to that. We tested it
directly (`reports/cae_density_study.py`): **bright, well-separated, easy-to-find dyes** (the case the
CAE *should* ace) plus a controllable broadband autofluorescence background, sweeping the cube from
sparse (density 0.37) to dense (0.90).

| bg_amp | density | **CAE R** | CAE infl-corr | **CAE F1** | **CAE disc%** | C2 R | C2 F1 | var-rank F1 | oracle |
|-------:|--------:|----------:|--------------:|-----------:|--------------:|-----:|------:|------------:|-------:|
| 0.0 | 0.37 | +0.003 | −0.951 | 0.332 | 0% | +0.323 | 0.647 | 0.726 | 0.719 |
| 0.5 | 0.54 | −0.001 | −0.946 | 0.340 | 0% | +0.360 | 0.636 | 0.718 | 0.717 |
| 1.0 | 0.66 | −0.000 | −0.925 | 0.330 | 0% | +0.432 | 0.624 | 0.702 | 0.704 |
| 2.0 | 0.79 | +0.003 | −0.788 | 0.324 | 0% | +0.550 | 0.657 | 0.644 | 0.684 |
| 4.0 | 0.90 | +0.010 | −0.422 | 0.327 | 0% | +0.695 | 0.567 | 0.429 | 0.626 |

**The CAE's reconstruction R never leaves ≈0** — not even at 90% density. It **never once selects a dye
band** (disc% = 0% at every level) and stays at chance F1. The per-pixel spectral AE (C2), by contrast,
reconstructs **better and better as density rises** (R +0.32→+0.70) and keeps classifying (~0.6). So:

- **Q1 (density → reconstruction): NO.** Density does not move the CAE off the degenerate solution.
- **Q2 (reconstruction → selection): moot** — the CAE never reconstructs here regardless.

**The sparsity explanation is wrong.** The CAE's failure is not about the data being mostly-zero; it is
the **spatial convolutional architecture + objective** itself, which (even with the band-collapse
removed — C1, doc 05) cannot encode the per-pixel spectral structure that the selection needs. The
right inductive bias for this problem is **per-pixel spectral**, not spatial — exactly what C2/C3/C7 do.

## What this means for the published real-data claim

The papers claim the CAE achieves classification parity on real cubes. That claim now rests on **no
validated mechanism**: the perturbation-influence it is built on is broken at every density we can
produce, and the one physical reason offered for why real data would differ (density) does not hold.
Two possibilities remain, and only real data can separate them:

1. the CAE's real-data "parity" was **never benchmarked against a trivial variance baseline** and may
   itself be an artifact (the *Worse than Random* failure mode — doc 05 refs); or
2. real tissue has some structure (beyond density) that the spatial CAE exploits and our synthetic
   model lacks.

**Either way, the CAE has not been shown to work, and the burden of proof is now on the real-data
experiment** — run the CAE **and variance-ranking and a spectral AE** on Lichens/Collagen at equal
budget. That is the outstanding gate (blocked here: the processed cubes are not on this machine).

## Bottom line

- **The CAE did not work as expected.** It is degenerate on every synthetic regime; density does not
  save it; it actively avoids the informative bands.
- **The principle (perturbation-based selection) is sound** — a per-pixel spectral AE makes it work.
- **The default stays `"standard"`** until real data is run, but the recommendation is unchanged and
  now strongly evidenced: replace the spatial CAE with a **masked per-pixel spectral AE** (C3/C7), and
  **re-test the published real-data claim against variance-ranking** before trusting it.
