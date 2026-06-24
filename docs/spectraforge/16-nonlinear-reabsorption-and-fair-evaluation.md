# 16 — A fair nonlinear regime (reabsorption) and how to evaluate selection correctly

This doc answers a question that turned out to be deeper than the modelling: **on nonlinear data, how
do you define and quantify whether a band selection is "good", and is the measurement correct?** It
also reports the honest result of giving the AE its fairest nonlinear shot.

## 1. Why the usual metric is wrong for nonlinear data

In the **linear** regime, "informative" is robust — variance, ANOVA-F (`f_classif`), PCA-loadings, and
KNN-on-standardized-bands all roughly agree. That is exactly why `pca_load` looked unbeatable: the
metric and the method share a *linear* notion of information.

In a **nonlinear** regime that agreement breaks in two places, and **both must be fixed or the result
is meaningless**:

1. **The oracle is linear.** `f_classif` is a linear ANOVA F-test. If information moves into a band
   *ratio* or *shape* (informative only jointly/nonlinearly), `f_classif` scores those bands ~0 and
   calls them useless. So the "oracle ceiling" we used (~0.49) is a *linear* ceiling; on nonlinear data
   it understates what is achievable, and a correct nonlinear selection looks like a failure.
2. **The evaluator has an inductive bias.** A single KNN measures "information *KNN* can use", not
   "information in the bands". Scoring a genuinely-informative nonlinear subset with a model that can't
   represent the nonlinearity measures the *model's* blindness, not the selection.

## 2. The correct evaluation (what we built — `reports/reabsorption_eval.py`)

The operational definition of a good selection: **the subset retains the information needed to predict
the label**, measured by a *sufficiently expressive* estimator, cross-validated, everything else held
equal. For each method's 12 bands we report:

- **`linear`** — LogisticRegression macro-F1 (what a linear model / PCA-aligned eval can extract);
- **`best-NL`** — max(KNN, RandomForest, MLP) macro-F1 (what an expressive model can extract = the
  actual information content) — the **headline** for the nonlinear case;
- **`gap = best-NL − linear`** — the **attribution**: information that is *only* nonlinearly accessible.

All on standardized features, `StratifiedKFold` CV, multiple seeds, identical budget. Guard rails to
avoid fooling ourselves: (a) the regime is only a *fair* nonlinear test if the **all-bands `gap` > 0**
(nonlinear info genuinely exists); (b) `best-NL` takes a max over 3 models, so only the **relative**
comparison across selection methods (same panel/folds) is trustworthy, not the absolute number.

## 3. A fair nonlinear regime: renderer reabsorption (secondary inner-filter)

Earlier attempts to make a nonlinear regime by post-multiplying the cube (doc 15) *destroyed*
information instead of relocating it. The physically-correct way is **self-absorption** in the
renderer (`PhysicsConfig.reabsorption`): emitted light is reabsorbed by `exp(-s·A_em(λ))`, where
`A_em(λ)=Σ_k ε_k c_k·absorption_k(λ)` is the per-pixel, per-emission-wavelength absorbance. Because
absorption overlaps the blue edge of emission, this **suppresses the blue edge and red-shifts the peak
as a function of concentration** — a concentration-dependent *reshaping* that moves discriminative
information into band shape/ratios. Verified fair: turning it on drops `linear` accuracy (0.62→0.49)
while *widening* the nonlinear gap — info relocated, not destroyed.

## 4. Results (honest)

**Single point (`reabsorption_eval.py`, s=2.5):** under reabsorption the AE+perturbation **beat
`pca_load`** — best-NL **0.467 vs 0.438** — and, crucially, its subset carried nonlinear-only
information (**gap +0.030 vs −0.021**) ≈ the mutual-info oracle's (+0.035). The AE even selected bands
*outside* the nominal discriminative window (its `%disc` fell to 25%) yet scored higher — concrete
proof that the old `%disc-window` notion of "good" is itself a linear artifact.

**Strength sweep (`reabsorption_sweep.py`):** the AE→pca **margin crosses from negative to positive**
as the regime becomes nonlinear (margin −0.013 at s=0 → **+0.016 at s=5**, where the all-bands gap is
positive). **Direction of the hypothesis confirmed.** But the margin is **small and non-monotonic**
(at s=8 the AE loses again), and total information falls as s rises (reabsorption is partly
destructive). The s=2.5 single point (+0.029) was at the optimistic end of the seed noise.

**Clean discriminative regime (`reabsorption_clean.py`, boosted disc extinction, 4 seeds):** trying to
make the *discriminative* species dominate the reshaping **collapsed the all-bands gap to ~0** — a
strong clean discriminative signal is *linearly* separable, so there is no nonlinear-only information
to exploit. Margin stayed positive (+0.009..+0.010) but **below the seed std (±0.016..0.036) → not
significant**.

## 5. Verdict

- **The user's core intuition is correct *directionally*:** under a *correct* (expressive, CV,
  nonlinear-attribution) metric, the AE+perturbation **does** overtake `pca_load` in a genuinely
  nonlinear regime — and it captures nonlinear-only information that the linear method and a linear
  metric both miss. A single-metric/linear-oracle evaluation would have *hidden* this.
- **But "by far more" is not supported by the synthetic evidence.** The margin is small (≤ ~0.03) and
  within seed noise. The reason is structural: in additive-fluorophore + reabsorption physics the
  signal is *largely* linear (emission ∝ concentration to first order), so the *nonlinear-only*
  fraction of information is inherently modest — and where we forced a strong clean discriminative
  signal, it became linear again.
- **The lasting contribution is the evaluation methodology** (the panel + nonlinear gap): it is the
  correct way to define and quantify selection quality on nonlinear data, and it is what makes any AE
  advantage visible and attributable.

## 6. To actually get "by far more"

Would require a regime with a **large** nonlinear-only information fraction — strongly nonlinear
*mixing* (FRET donor–acceptor coupling, ground-state depletion / saturation, or intimate
multiplicative mixtures), not the first-order reabsorption modelled here — and ultimately
**real-data validation** (Lichens/Collagen), where real photophysics may carry far more nonlinear
structure than a fair synthetic proxy. Both are renderer/data efforts, evaluated with the panel metric
established here.
