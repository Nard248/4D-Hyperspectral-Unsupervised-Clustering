# 12 — CAE at real-data scale + conv-architecture experiments

Two follow-up experiments to the reconstruction audit (doc 11): (a) does the CAE reconstruct at
real-data resolution where the scene is *many* chunks (real batched training), and (b) what happens
to the conv architecture if we drop the spatial dimension (per-pixel) and vary the parallel
excitation branches.

## 1. High resolution / many chunks (`reports/cae_largescene.py`)

Hypothesis (and doc-02's own claim): the CAE fails on 64×64 only because that is a *single chunk*
(batch=1); at real-data resolution it becomes many chunks → real batched multi-sample training → it
should learn. Tested at 256px (25 chunks) and 512px (81 chunks):

| scene | chunks | config | recon R² | CAE F1 | oracle |
|-------|------:|--------|---------:|-------:|-------:|
| 256px clean | 25 | default | −0.008 | 0.325 | 0.485 |
| 256px clean | 25 | no-reg | −0.008 | 0.325 | 0.485 |
| 256px realistic | 25 | no-reg | −576 | 0.322 | 0.480 |
| 512px clean | 81 | no-reg | −0.145 | 0.330 | 0.501 |

**Refuted.** With 25–81 chunks of genuine batched training the CAE *still* does not reconstruct
(R² ≤ 0) and *still* selects at chance (F1 ≈ 0.33), exactly as at 64×64. So single-chunk/batch=1 was
**not** the cause, and the "real data is large/many-chunks so the CAE works there" assumption does
not hold even on synthetic data at that scale. The failure is the architecture, confirmed across
64 → 512 px.

## 2. Per-pixel + parallel-branch conv setups (`reports/conv_arch_zoo.py`, `conv_arch_search.py`)

Keep the CAE's per-excitation **parallel-branch** DNA but **drop the spatial convolution** (each pixel
is one sample) and sweep the branch conv: band-kernel (3/5/9), depth (1–3), merge (concat/mean), and
the band-**collapse** (the CAE's average-pool). reconR = mean per-band corr(input, recon).

| variant | clean reconR | clean F1 | real reconR | real F1 |
|---------|-------------:|---------:|------------:|--------:|
| pca_load[k=6] (ref) | – | 0.463 | – | **0.479** |
| C2 spectral-MLP (ref) | +0.290 | 0.417 | +0.541 | 0.448 |
| branch, **no collapse**, kb3–9 × d1–3 (9 configs) | **+0.39 … +0.43** | 0.33–0.40 | **+0.597 … +0.612** | 0.34–0.41 |
| branch kb5 d2, mean-merge | +0.347 | 0.338 | +0.587 | 0.350 |
| branch kb5 d2, **+COLLAPSE** concat | +0.169 | **0.503** | +0.448 | 0.326 |
| branch kb5 d2, +COLLAPSE mean | +0.166 | 0.486 | +0.366 | 0.358 |

**Findings.**
- **Per-pixel "makes sense" — for reconstruction.** Dropping the spatial conv lifts reconstruction
  from the spatial CAE's ≈0 to **+0.4 … +0.6**. The spatial convolution was actively harmful here;
  the information is per-pixel spectral, not spatial.
- **Conv knobs barely matter.** Band-kernel 3 vs 9 and depth 1 vs 3 change reconR/F1 only marginally;
  concat-merge reconstructs a bit better than mean-merge. The 1D-conv adds nothing over the plain MLP.
- **The reconstruction↔selection inversion is now stark and causal.** The *best* reconstructors
  (no-collapse, reconR +0.6) **select worst** (F1 0.33); the *worst* reconstructor among per-pixel
  (the **collapse** bottleneck, reconR +0.17) **selects best on clean (F1 0.503 — near the 0.486
  oracle!)**. A capacity bottleneck that throws away reconstruction detail *helps* selection on clean
  data by forcing the latent onto the dominant (= informative, when variance=info) structure.
- **But none beats the simple selectors on realistic data.** Best branch-conv realistic F1 = 0.408
  (kb5 d2) < pca_load 0.479. On confounded data the bottleneck latches onto the loud nuisances
  (collapse realistic F1 = 0.326), so its clean-data win does not generalize.

## Synthesis (both experiments)

1. **The spatial-CAE failure is resolution- and batching-independent** — it is the spatial-conv +
   band-collapse architecture, full stop (now shown across 64→512px and 25→81 chunks, on top of the
   doc-11 config × data sweeps).
2. **Going per-pixel fixes reconstruction** but **does not make perturbation-selection competitive**
   with the simple blind methods; the conv structure is not the lever.
3. **Reconstruction fidelity is anti-correlated with selection quality** — confirmed yet again, now
   within a single architecture family by toggling one bottleneck. *This is the core lesson of the
   whole investigation:* a band selector should be optimized for discriminative structure, not for
   faithful reconstruction.
4. The robust answer remains the simple **PCA-loadings (k≈6)** blind selector (~0.48, 96–97% of the
   oracle; doc 10). The one intriguing lead is the **collapse-bottleneck per-pixel AE on clean data**
   (0.503, near-oracle) — a "bottleneck-as-selector" idea worth a look if a clean regime is the target.
