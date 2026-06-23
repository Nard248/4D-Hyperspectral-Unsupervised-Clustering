# 15 — The nonlinear-regime test (an honest negative, and what it means)

Hypothesis (the principled route to *exceeding* `pca_load` by a wide margin): the realistic regime is
near-saturated because its structure is essentially **linear**, where PCA is near-optimal. An
autoencoder's real advantage is **nonlinear** manifolds — and fluorescence supplies them (inner-filter
effect, reabsorption, concentration quenching, detector saturation). So on a genuinely nonlinear scene
the AE+perturbation *should* beat the linear baselines by more.

**We tested it (`reports/nonlinear_regime.py`):** render the realistic scene, then apply a per-pixel
**inner-filter attenuation** (coupled to total brightness — i.e. the bright nuisances) plus a
**saturating quench**, and compare blind selectors against both a linear (`f_classif`) and a nonlinear
(`mutual_info_classif`) top-k oracle. Three seeds, 12-band budget, KNN macro-F1.

| method | LINEAR control | NONLINEAR (IFE+sat) |
|--------|---------------:|--------------------:|
| variance | 0.351 | 0.361 |
| **pca_load[k6]** | **0.486** | **0.460** |
| **AE+perturb (conv)** | 0.461 | **0.368** |
| oracle `f_classif` (linear) | 0.493 | 0.472 |
| oracle `mutual_info` (nonlinear) | 0.480 | 0.450 |

## Result: the hypothesis was **not** supported — and the reason is instructive

1. **The nonlinearity I applied was *destructive*, not *information-relocating*.** Both oracles
   *dropped* (f_classif 0.493→0.472, MI 0.480→0.450) and the nonlinear MI-oracle stayed *below* the
   linear one. So the inner-filter+saturation transform mostly **reduced** the total discriminative
   information (uniform per-pixel dimming dominated by the bright nuisances) rather than encoding it in
   nonlinear structure. It is therefore not a fair test of "nonlinear info the AE can exploit."
2. **pca_load remained the strongest blind method even so** (0.460), degrading *gracefully*.
3. **The AE+perturbation collapsed harder (0.461→0.368)** and its hit-rate on the discriminative window
   fell from 69% to 22%. The diagnostic: the perturbation ranks bands by **reconstruction influence**,
   which the bright nuisance-driven attenuation pattern now dominates — so the selector follows the
   nuisances. **The perturbation method is *more* nuisance-sensitive than PCA**, not less.

## What this means (honest verdict)

- On **additive-fluorophore synthetic data**, `pca_load` is a remarkably **strong and robust** baseline.
  Both PCA and a reconstruction-AE are fundamentally **variance-driven**, and PCA selects the
  spectrally-prominent informative bands near-optimally; a destructive nonlinearity hurts the
  variance-following AE *more*, not less.
- A fair test of the nonlinear hypothesis needs a generator where nonlinear mixing **relocates**
  discriminative information into band *shape/ratios* while **preserving** total information (so that
  `mutual_info` oracle > `f_classif` oracle). That requires changes **inside the renderer**
  (concentration-dependent reabsorption / self-absorption that reshapes emission bands), not a
  post-hoc multiply on the cube. That is the correct next experiment if we pursue this further — but it
  is a generator change, not a quick sweep.
- **The defensible, evidence-backed claims remain:** the fixed AE+perturbation **matches `pca_load`
  across regimes and exceeds it on clean data**; "exceeding by far more" is **not** supported by the
  synthetic evidence, and chasing it with post-hoc synthetic nonlinearities is not rigorous.

## Recommendation

The synthetic benchmark has been pushed to its honest ceiling. The decisive lever now is **real-data
validation (Lichens / Collagen)**, where the true task, real nonlinear photophysics, and real nuisance
structure may favor the nonlinear model in ways a fair synthetic proxy has not. If we do want one more
synthetic push, the *only* principled version is a **reabsorption-reshaping renderer** (above) — happy
to implement it, but it is a modelling change, not another sweep.
