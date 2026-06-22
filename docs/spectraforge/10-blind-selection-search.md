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

### Round 1b — 16 worktree-isolated exploration agents (each implements + measures one method)

Measured on clean+realistic × seeds 1,2 (the agents' brief). All 16 ran successfully:

| method | mean | clean | realistic | family |
|--------|-----:|------:|----------:|--------|
| **robust-discriminability** | **0.504** | 0.517 | 0.491 | strip loud PCA dirs → cluster → between-cluster F-ratio |
| barlow_twins | 0.487 | 0.497 | 0.476 | self-supervised (VICReg/Barlow) |
| contrastive-saliency | 0.480 | 0.493 | 0.467 | self-supervised (SimCLR) |
| graph-coverage | 0.472 | 0.501 | 0.443 | band-graph spectral coverage |
| derivative-pca | 0.463 | 0.447 | 0.480 | per-excitation derivative + PCA-loadings |
| dpp-diverse | 0.452 | **0.533** | 0.370 | determinantal point process (overfits clean) |
| shape-normalized / ica-kurtosis / consensus / mutual-info / vae / robust-pca | 0.42–0.45 | ~0.50 | 0.35–0.40 | — |
| dictionary-learning | 0.417 | | | sparse coding |
| transformer-attention | 0.386 | 0.434 | 0.338 | reconstruction (bottom) |
| residual-autoencoder | 0.381 | 0.383 | 0.378 | reconstruction (bottom) |

The "strip-the-loud-nuisance-directions then cluster-and-discriminate" recipe (the lead from round 1)
topped the agents and **matched the oracle on these two regimes** (VS_ORACLE 1.007). Self-supervised
methods came next. **Reconstruction-based methods ranked last again** — the nuisance trap, a third time.

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

## Round 3 — full validation + generalization (the tiebreaker)

The agents tuned on clean+realistic × 2 seeds; round 3 re-scores the winners on the **full** benchmark
(4 regimes × 3 seeds) and on **7 held-out regimes** (`regime_zoo`: 2–5 classes, low/high overlap,
faint signal, heavy nuisance, low noise) — does the winner *generalize* when the data-generation
assumptions change?

**Full benchmark (oracle 0.494):** pca_load[k6] **0.477** (0.97) · barlow 0.475 · robust_disc 0.472 ·
derivative_pca 0.464 · ae_masked(C3) 0.425 · variance 0.385. The three leaders are **tied at 96–97%**
— robust_disc fell from its 2-regime 0.504 to 0.472 (its 2-regime "matches oracle" was the easy subset).

**Generalization (mean over 7 held-out regimes; oracle 0.449):**

| method | MEAN F1 | % of oracle |
|--------|--------:|------------:|
| **pca_load[k=6]** | **0.431** | **96%** |
| derivative_pca | 0.405 | 90% |
| robust_disc (agent winner) | 0.400 | 89% |
| variance | 0.393 | 88% |

`pca_load[k=6]` is at/near the oracle on every regime with recoverable signal (g_2class 0.615/0.617,
g_highoverlap 0.456/0.451, g_lowoverlap 0.464/0.479, g_lownoise 0.499/0.515) and **generalizes best**.
The adaptive `robust_disc` and self-supervised `barlow` **overfit the tuning regimes** and degrade on
held-out ones (robust_disc g_lowoverlap 0.376 vs pca 0.464). On the near-chance regimes (g_5class,
g_faint, g_heavynuis — where even the oracle ≈ chance because the signal is essentially gone) all
methods are indistinguishable noise; variance edges ahead only there.

## Conclusion — the converged blind selector

After ~hundreds of evaluations (66 swept configs × 4 regimes × 3 seeds + 16 agent methods + 2
validation rounds + a 7-regime generalization battery):

> **The best blind/unsupervised informative-subset selector found is a PCA-loadings selector with
> k ≈ 6 components: rank bands by the summed |loading| over the top ~6 principal components, take a
> diverse top-n. It recovers 94–97% of the labels-using discriminability oracle on the main benchmark
> and 96% across 7 held-out regimes — the most robust AND most generalizable method tested.**

Why it wins, and what the whole search taught us:
1. **Capture the informative *subspace*, don't reconstruct the spectrum.** The few real material
   factors live in a low-dim subspace; ~6 PCA directions span it. Picking bands that load on it is
   enough. Every **reconstruction-based** method (CAE, deep AE, Concrete AE, transformer-AE,
   residual-AE) ranked at/near the **bottom** — reconstructing the nuisance-dominated spectrum makes
   the selection chase loud, non-informative bands (the nuisance trap, seen now a third time).
2. **Component count is the one knob that matters** (k≈6): too few miss structure, too many re-admit
   nuisance directions. It is stable across regimes — no per-regime tuning needed.
3. **Simplicity generalizes; cleverness overfits.** The adaptive "strip-then-discriminate" and the
   self-supervised (Barlow/contrastive) methods *matched* PCA-loadings on the regimes they were tuned
   on, but generalized worse. A 1-second deterministic linear method is the robust choice.
4. **It is genuinely good, not just relatively.** 96% of an oracle that *uses the true labels* is
   strong for a blind method — the residual gap is the inherent cost of not knowing the task.

**Caveats (honest):** all synthetic; the oracle ceiling itself is modest because the task (dim,
overlapping dyes under heavy nuisance) is hard; on regimes where signal ≈ 0 nothing works. The
outstanding gate is unchanged — **confirm `pca_load[k≈6]` on the real Lichens/Collagen cubes** vs the
CAE and vs variance-ranking. But on synthetic data, the search has converged on a simple,
robust, generalizable answer.
