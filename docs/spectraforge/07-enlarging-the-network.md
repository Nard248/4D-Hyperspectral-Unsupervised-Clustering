# 07 — Enlarging the network (bigger nets + the training bag-of-tricks)

Following the request to scale the autoencoders up "following the research guides, practices, tricks",
we built three **enlarged** per-pixel candidates and a modern training recipe, and tested them on both
the clean gate and the realistic confounded regime from doc 06. The result is a clear — and
**counterintuitive** — lesson about what actually matters for *perturbation-based* band selection.

## 1. What we added

**Enlarged candidates** (`spectral_select.architectures`, registry `LARGE_CANDIDATES`):

| # | class | enlargement | grounding |
|---|-------|-------------|-----------|
| **C5** | `DeepSpectralAE` | wide (256) **residual** MLP, **LayerNorm + GELU + dropout**, latent 16 | the standard deep-net "bag of tricks" |
| **C6** | `ConvSpectralAE` | **1D convolution along the emission-band axis** (excitations as channels), GroupNorm | HSI/Raman 1D-conv & operational AEs (SRL-SOA etc.) |
| **C7** | `MaskedConvSpectralAE` | C6 backbone **+ denoising band masking** | SS-MAE/SMAE: masking + spectral locality together |

**Modern training recipe** — added as *opt-in* knobs to `SpectralSelector` (defaults leave C2/C3/C4
byte-identical): **minibatch SGD**, **AdamW (weight decay)**, **cosine LR schedule**, **early
stopping** (restore best), optional grad clipping. C5/C6/C7 use latent 16, width 256/48, batch 512,
weight-decay 1e-4, cosine, ~200 epochs with early stopping.

The selection math is unchanged — every candidate still reuses `selection_core` for
perturbation→influence→diverse top-k. So the only variable is again **the model + objective**.

## 2. Results

`reports/enlarged_comparison.py`, mean over 3 scenes each. Small AEs (C2/C3) for reference.

**Clean gate** (well-separated bright peaks; KNN-F1 / peak_recovery):

| method | KNN-F1 | peak_rec |
|--------|-------:|---------:|
| discriminability-oracle | 0.514 | 0.67 |
| variance-ranking | 0.480 | 0.67 |
| all bands | 0.479 | – |
| **C7 masked-conv** | **0.490** | **0.89** |
| C3 masked-AE (small) | 0.484 | 0.89 |
| C2 spectral-AE (small) | 0.432 | 0.56 |
| C6 conv-AE | 0.415 | 0.56 |
| **C5 deep-AE** | **0.335** | 0.11 |
| random | 0.373 | 0.11 |

**Realistic confounded regime** (variance ≠ informativeness; KNN-F1 / % bands on the discriminative window):

| method | KNN-F1 | disc-band % |
|--------|-------:|------------:|
| all bands | 0.504 | 28% |
| discriminability-oracle | 0.491 | 100% |
| **C7 masked-conv** | **0.433** | 53% |
| C2 spectral-AE (small) | 0.431 | 69% |
| C3 masked-AE (small) | 0.414 | 61% |
| C6 conv-AE | 0.407 | 28% |
| **variance-ranking** | 0.344 | 33% |
| **C5 deep-AE** | **0.320** | 14% |
| random | 0.372 | 28% |

Single-seed diagnostics on the realistic data make the mechanism explicit: **C5 has the *highest*
reconstruction (R = +0.87) yet the *most negative* influence–signal correlation (−0.42)** and lands
only 8–14% of bands on the discriminative window — it reconstructs the bright nuisances/scatter so
well that its perturbation-influence is dominated by them.

## 3. The lesson: bigger ≠ better for perturbation selection

**Enlarging the network, on its own, does not help — and pure capacity actively hurts.** The
biggest, most heavily-regularized net (C5: wide residual MLP, LayerNorm, dropout, weight decay,
cosine LR, early stopping) is the **worst learned method on *both* benchmarks** (0.335 clean, 0.320
realistic — both ≈ random), and it is worst *because* it reconstructs best:

> For perturbation-based selection, influence ≈ "how much does the reconstruction move when I push a
> latent dimension." A high-capacity AE faithfully encodes the **highest-energy** content — which in
> realistic data is the **bright nuisances + scatter**, not the dim discriminative dyes. So its
> influence concentrates on exactly the non-informative bands. **Reconstruction fidelity here is
> *inversely* related to selection quality** (C5: R 0.87 → F1 0.32; C2: R 0.54 → F1 0.43).

What does help is the **training objective**, not the size:

- **Masking/denoising is the lever.** The two masked candidates — small C3 and the enlarged
  **C7 (masked-conv)** — are the top learned methods on the clean gate (0.484 / **0.490**, both with
  0.89 peak recovery, near the oracle), and C7 ties C2 for best on the realistic data. Masking forces
  the latent to *infer* bands from context, so it cannot just echo the loud nuisances.
- **Convolutional locality helps only with the right objective.** Plain conv (C6) beats the plain deep
  MLP (C5) — band-axis locality is a useful prior — but without masking it still drifts to the
  nuisances on realistic data (28% disc). Conv **+ masking** (C7) is the best overall architecture.
- **The small AEs were already near the ceiling.** C2/C3 (a 3-layer MLP, ~10k params) match or beat
  every enlarged variant except C7, and C7's gain is modest. There is little headroom to buy with size.

**Recommendation.** Do not scale capacity blindly. Adopt **C7 (masked 1D-conv)** where a bit more
capacity is wanted (best clean-gate result, robust on realistic), otherwise the **small masked AE
(C3)** is a near-equal, far cheaper choice. Treat reconstruction R as a *necessary-but-not-sufficient*
gate (R>0), never as the thing to maximize. This is the sharpest statement yet of the project thesis:
**what the latent is *forced to encode* (the objective) governs selection quality — network size and
reconstruction error do not.**

### References
- SRL-SOA / sparse 1D-operational autoencoder for HSI band selection — arXiv 2204.13651.
- Dropout Concrete Autoencoder for HSI band selection — arXiv 2401.16522.
- MAE (masking as denoising) — He et al., CVPR 2022; SS-MAE — arXiv 2505.05710.
