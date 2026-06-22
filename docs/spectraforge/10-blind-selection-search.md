# 10 — Blind/unsupervised selection: the broad search

Goal: stop iterating on one architecture and instead **search broadly** for *any* configuration that,
**blind/unsupervised**, recovers the informative band subset on synthetic ME-HSI — converging toward
the (labels-using) discriminability **oracle**. Everything is recorded: per-config CSVs in
[`../../reports/exp_records/`](../../reports/exp_records/), this doc is the running log.

**Evaluation contract** (`reports/sweep_common.py`): a method is a blind
`select(X, colmap, n, seed, rng, spectra) -> cols`; scored by downstream **KNN macro-F1** across four
regimes (clean → mild → realistic → dense confound) × seeds, vs the oracle ceiling. Two engines:
a programmatic **grand sweep** over ~25 method families × hyperparameters (`reports/grand_sweep.py`,
`method_zoo.py`, `gate_zoo.py`), and **worktree-isolated exploration agents** that implement + measure
novel methods.

---

## Round 1 — grand sweep (66 configs, 4 regimes × 3 seeds; oracle ceiling mean-F1 = 0.494)

Top of the leaderboard (mean F1 / vs-oracle / per-regime clean·mild·realistic·dense):

| rank | method | mean F1 | vs oracle | per-regime |
|----:|--------|--------:|----------:|------------|
| 1 | **pca_load[k=8]** | **0.461** | **0.93** | 0.43 / 0.47 / 0.48 / 0.46 |
| 2 | ae_masked_conv (C7) | 0.440 | 0.89 | 0.49 / 0.42 / 0.43 / 0.42 |
| 3 | ae_masked (C3) | 0.425 | 0.86 | 0.48 / 0.40 / 0.41 / 0.40 |
| 4 | band_cluster[ward] | 0.417 | 0.84 | 0.42 / 0.41 / 0.43 / 0.41 |
| 5 | concrete_ae | 0.415 | 0.84 | **0.53** / 0.35 / 0.38 / 0.40 |
| … | nmf / sparsepca / mrmr / svd | ~0.41 | ~0.83 | strong-clean, weaker-confounded |
| ref | variance-ranking | 0.385 | 0.78 | — |

**Findings.**
- **`pca_load[k=8]` is the robust winner (93% of the oracle).** Summing |loadings| over the first ~8
  principal components and picking the top diverse bands recovers most of the discriminative subspace
  — and it is the *only* top method that holds up across **every** regime (0.43–0.48), not just clean.
- **Reconstruction-driven methods overfit the nuisances.** The AEs and especially the Concrete AE are
  best on *clean* data (concrete_ae 0.53) but **collapse on confounded data** (→0.35) — the same trap
  the CAE/C5 fell into: reconstructing the full nuisance-dominated spectrum makes the selection chase
  the loud, non-informative bands. Selection quality is *not* reconstruction quality (doc 08 redux).
- **The component count is a knob.** pca_load k=8 ≫ k=16/32 (too many comps re-admit nuisance
  directions) and > k=4 (too few miss structure).
- **The unsupervised-discriminability idea (`cluster_fratio`) underperforms (~0.39)** as-is: clustering
  on the raw confounded data forms clusters on the loud nuisances. Promising *if* the nuisance
  variance is stripped first — handed to the exploration agents.

<!-- ROUND1_AGENTS -->

## Round 2 — densify the winner (PCA-loadings × component-count × preprocessing) + consensus

`reports/round2_sweep.py` (oracle 0.494):

| rank | method | mean F1 | vs oracle |
|----:|--------|--------:|----------:|
| 1 | **pca_load[k=6]** | **0.477** | **0.97** |
| 2–4 | pca_load[k=5 / k=5,l2 / k=7] | 0.474 | 0.96 |
| 5–7 | pca_load[k=5,snv / k=6,snv / k=6,l2] | 0.467–0.470 | 0.95 |
| 8 | pca_load[k=8] (round-1 best) | 0.461 | 0.93 |
| 15 | consensus (rank-fusion) | 0.428 | 0.87 |

- **The sweet spot is k ≈ 5–7 components → 0.474–0.477 = 95–97% of the labels-using oracle.** ~6 PCA
  components capture the informative subspace (≈ the few real material factors); more re-admit
  nuisance directions, fewer miss structure.
- **Preprocessing (L2 shape-norm / SNV / derivative) does NOT beat raw** at the optimal k, and
  **derivative hurts** — the raw principal-subspace loading is already doing the right thing.
- **Rank-fusion consensus underperformed (0.428)** — averaging in weaker rankers dragged it down;
  the single best ranker wins.

**Interim convergence:** a *blind, deterministic, ~1-second* PCA-loadings selector (k≈6) reaches **97%
of the discriminability oracle, robustly across all four regimes** — a genuinely good
blind/unsupervised informative-subset selector on this synthetic data.

<!-- CONCLUSION -->
