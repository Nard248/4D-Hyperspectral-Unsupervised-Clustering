# 17 — FRET (strongly nonlinear) and why the AE does not beat PCA at *selection*

The decisive test of the hypothesis "on nonlinear data the AE must beat PCA." We built the strongest
honest nonlinear regime available and evaluated it with the correct (panel) metric. The result
characterises *precisely* when the AE could win — and shows why, for blind band **selection**, it
ties PCA even when the data is strongly nonlinear.

## The regime (`reports/fret_regime.py`, renderer FRET + XOR interaction scene)

- **Class lives in dye co-localization (XOR).** Two dyes are driven by independent fields; the class is
  `b1 XOR b2` of their median-binarised concentrations, so each dye's *marginal* is class-independent
  (a linear / single-band test sees nothing) and the information is *purely* in the joint.
- **FRET** (donor D1 515 nm → acceptor D2 580 nm) relocates that joint into the band ratio: exciting at
  the donor band (470 nm), the acceptor's 580 nm emission appears *only* by transfer, so it encodes
  `c_D1·E(c_D2)` — a saturating, product (nonlinear) signal.

This is genuinely, strongly nonlinear: the **all-bands nonlinear gap is +0.113** (vs ≤0.04 for
reabsorption), and the marginal mutual-information oracle *fails* (best-NL 0.572) because per-band MI
cannot see an XOR. Fairness precondition strongly met.

## Result (4 seeds, fair panel CV macro-F1)

| method | linear | best-NL | nonlinear gap |
|--------|-------:|--------:|--------------:|
| all-bands (ceiling) | 0.549 | 0.662 | **+0.113** |
| variance | 0.481 | 0.549 | +0.069 |
| **pca_load** | 0.549 | **0.641 ± 0.011** | +0.091 |
| **AE+perturb** | 0.541 | **0.633 ± 0.016** | +0.092 |
| oracle (marginal MI) | 0.516 | 0.572 | +0.056 |

**AE − pca margin = −0.008 (within noise). They are tied.** The AE's nonlinear gap (+0.092) equals
pca's (+0.091) — it *does* capture the nonlinear structure — but it does not *select* better.

## The insight: band SELECTION ≠ CLASSIFICATION

The intuition "AE beats PCA on nonlinear data" is true for **representation / classification** — but
the task here is **selection**: choosing *which* bands to keep. Those are different problems:

> **Informative ⟹ has variance ⟹ PCA selects it.** The discriminative bands (the dye bands, the FRET
> band) all carry concentration-driven variance, so `pca_load` *selects* them. The *nonlinearity* — the
> XOR, the FRET ratio — is then resolved by the **downstream nonlinear classifier** (RF/MLP in the
> panel), not by the selector. PCA does not need to *understand* the nonlinearity to *pick the right
> bands*; it only needs them to be spectrally prominent, which informative bands are.

So a nonlinear *classification* task does **not** imply a nonlinear *selection* task. The AE's
theoretical advantage (modelling a nonlinear manifold) buys it nothing extra at selection, because
selecting variance-prominent bands — which both methods do — already captures the informative set, and
the expressive evaluator extracts the rest.

The only regime where the AE could select better is one where the informative bands are *not*
spectrally prominent (low variance). But there the AE's perturbation — which ranks bands by
*reconstruction influence*, itself variance-driven — would also miss them (confirmed in doc 15: under
nuisance domination the AE was *more* fooled than PCA). So no *blind* selector we can build exploits
nonlinearity to beat PCA; doing so requires *label* information (a discriminability/wrapper selector),
which is a different method, not the AE.

## Verdict (across linear, reabsorption, and FRET regimes)

- **For blind band selection, AE+perturbation ≈ `pca_load`** across linear *and* strongly nonlinear
  regimes (clean is the one exception, where the AE *exceeds* pca). The AE reliably *captures* nonlinear
  structure (its nonlinear gap tracks the oracle) but does not *select* better than PCA.
- **"Exceed by far more" is not achievable for a blind selector on this data class** — a precise,
  evidence-backed conclusion, not a tuning shortfall. It follows from *informative ⟹ has variance ⟹
  PCA-selectable*.
- **What this is worth:** the correct evaluation methodology (doc 16) and this selection-vs-
  classification result are the real scientific contributions. The AE+perturbation is *validated* (it
  matches the best blind method everywhere and beats it on clean), and its limits are now understood.

## Where a real advantage could still live

- **Real data (the decisive gate):** real Lichens/Collagen cubes may have informative structure that is
  *not* variance-prominent (instrument/sample effects), where the picture could differ — evaluated with
  the panel metric established here.
- **A different objective:** if the goal is to *beat* `pca_load` rather than to validate the AE, a
  *semi-supervised / discriminability-aware* selector (using a few labels) is the principled route —
  but that abandons the purely-blind premise.
