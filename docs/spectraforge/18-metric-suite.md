# 18 — The metric suite: every measure that makes sense for band selection on labelled ME-HSI

Goal: a **convergent, multi-axis** evidence battery so the standing of the AE+perturbation selector is
*diamond solid* — not one number anyone can dispute, but many complementary metrics that all point the
same way. Implemented in `reports/metric_suite.py` (results table → `reports/exp_records/metric_suite.csv`).

Each metric is listed with **what it proves**, **how it is computed**, and **its pitfall on this data**.
A good selection should look good on *all* axes at once; a metric that looks good in isolation but bad
elsewhere is a red flag (e.g. high variance-selection F1 but terrible ground-truth overlap = it got
lucky via a correlated band).

## Axis A — Downstream information-retention (the operational definition)

"A good selection lets a downstream model predict the label." Measured with a **classifier panel**
spanning inductive biases, **cross-validated**, on **standardised** selected bands.

| metric | what it proves | computation | pitfall here |
|--------|----------------|-------------|--------------|
| **macro-F1** | class-balanced predictive power | `f1_score(avg=macro)` via `cross_val_predict` | dominant-class bias if not macro |
| **balanced accuracy** | accuracy corrected for class priors | `balanced_accuracy_score` | — |
| **Matthews CC (MCC)** | correlation of pred vs truth, robust to imbalance | `matthews_corrcoef` | — |
| **Cohen's κ** | agreement above chance | `cohen_kappa_score` | — |
| **`linear` vs `best-NL`** | *where* the information lives | logreg vs max(knn/rf/mlp) | see Axis-A note |
| **nonlinear `gap`** | information only a nonlinear model can use | `best-NL − linear` | needs all-bands gap>0 to be a fair NL test (docs 16-17) |

*Axis-A note (the crux, docs 16-17):* a single classifier or a **linear oracle (`f_classif`) bakes in a
bias** and cannot credit a nonlinear selection. The panel + `gap` is what makes the comparison fair on
nonlinear data. Only the **relative** comparison across selectors (same panel, same folds) is trusted —
`best-NL` is a max over 3 models, so it is mildly optimistic in absolute terms.

## Axis B — Classifier-independent information (trust no single model)

| metric | what it proves | computation | pitfall |
|--------|----------------|-------------|---------|
| **relevance** | total label information in the set | Σ per-band `mutual_info_classif` | marginal MI misses pure interactions (XOR) — so report alongside Axis A |
| **redundancy** | how non-redundant the set is | mean \|corr\| among selected bands | low redundancy is good *only if* relevance stays high |
| **mRMR score** | relevance − redundancy (the classic trade-off) | `relevance/k − redundancy` | a criterion, not ground truth |

These quantify "informative **and** non-redundant" without any classifier — a check that Axis A's win
isn't a single-model artefact. (Marginal MI's blindness to XOR is *itself* informative: when the
`mutual_info` oracle underperforms the panel, the data is genuinely interaction-nonlinear.)

## Axis C — Ground-truth band fidelity (the synthetic advantage)

Because the data is synthetic we can ask **did it pick the physically-right bands?** The ground-truth
informative set = top `2·budget` bands by mutual information in a **near-noise-free render of the same
regime** (a data-driven, regime-appropriate definition — *not* the hand-drawn "discriminative window",
which doc 17 showed is a linear artefact).

| metric | what it proves | computation |
|--------|----------------|-------------|
| **precision** | fraction of picks that are truly informative | `|sel ∩ gt| / |sel|` |
| **recall** | fraction of informative bands captured | `|sel ∩ gt| / |gt|` |
| **Jaccard** | overall set overlap | `|sel ∩ gt| / |sel ∪ gt|` |

Pitfall: the ground-truth set is *necessary-but-not-sufficient* (a band can be informative jointly yet
have modest marginal MI), so treat C as corroborating A/B, not overriding them.

## Axis D — Stability (a trustworthy method is consistent)

| metric | what it proves | computation |
|--------|----------------|-------------|
| **selection stability** | the method picks the same bands across data draws | mean pairwise Jaccard of the per-seed selected sets |

A selector with high task-F1 but near-zero stability is fragile (its win won't transfer). Reported per
method per regime.

## Axis E — Statistical significance (is the difference real?)

For the headline claim (AE vs the best **blind** baseline `pca_load`), paired across seeds on `best-NL`:

| metric | what it proves | computation |
|--------|----------------|-------------|
| **mean margin** | direction & size of the effect | `mean(AE − pca)` over seeds |
| **bootstrap 95% CI** | uncertainty of the margin | 2000× resample of the paired diffs |
| **Wilcoxon p** | non-parametric significance | `scipy.stats.wilcoxon` (needs ≥6 seeds for power) |
| **Cohen's d** | standardised effect size | `mean(diff)/std(diff)` |

A claim is "solid" only when the CI excludes 0 *and* the effect size is non-trivial *and* it holds
across regimes — not on a single lucky seed.

## The slate of selectors (compare against everything, not just PCA)

- **Blind:** `random` (floor), `variance`, `pca_load` (best known blind), `laplacian` (manifold,
  He et al. 2005), **`AE-conv` / `AE-mlp`** (the proposed approach, two configs = the family).
- **Supervised references (★, upper bounds):** `mutual_info` top-k, `mRMR` (Peng 2005).
- **Ceiling:** `all-bands`.

The diamond-solid claim is: **the AE family matches the best *blind* baseline (`pca_load`) on every
axis, beats it on clean, captures nonlinear structure (Axis-A `gap`) that linear selection cannot, and
trails only the *supervised* references** — i.e. it is a sound, competitive, fully-unsupervised
selector whose limits are understood (selection ≠ classification, doc 17).

## Results

> Full run: 4 regimes (clean / realistic / reabsorption / FRET) × 6 seeds. Numbers land in
> `reports/exp_records/metric_suite.csv` and the headline tables are pasted here once the run
> completes. (See the bottom of this doc / the commit that follows.)
