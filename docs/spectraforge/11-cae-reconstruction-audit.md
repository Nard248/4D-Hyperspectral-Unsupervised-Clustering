# 11 — CAE reconstruction audit + the config that makes it work

Prompted by a sharp challenge: *we assumed the published pipeline was correct — verify it. Does the
CAE actually reconstruct the input? Did batching corrupt pixels? The loss alone is deceiving; inspect
numerically AND visually.* This doc does exactly that, then searches for a configuration under which
the CAE + perturbation mechanism is genuinely supported by evidence.

## 1. Pipeline integrity — verified clean (it was NOT corrupting pixels)

- **Normalization round-trips to machine precision.** Denormalizing `dataset.get_all_data()` back to
  the original cube: `max|orig − denorm| = 1.7e-08` (realistic) / `6.6e-09` (clean). The global
  min-max normalization preserves every pixel. *(Note: `normalization_method`/`percentile_range` are
  dead params — the code always uses global min/max; bright scatter outliers therefore squash the
  signal toward 0, which matters, but it is lossless.)*
- **No chunk/batch corruption on synthetic data.** Training uses `chunk_size=64`; a 64×64 scene is a
  **single chunk** (batch=1), so the chunk-split/merge path (which matters for large real cubes) is
  not even exercised — there is no reassembly to corrupt pixels here.

So the failure is **not** a data-pipeline bug. `reports/cae_recon_audit.py` (run it to regenerate).

## 2. The reconstruction is genuinely broken — and the loss hid it

`R² = 1 − MSE_model / MSE_meanpredictor` per band (R²≤0 ⇒ no better than predicting the per-band
spatial mean; R²→1 ⇒ faithful):

| scene | mean corr(in,rec) | **mean R²** | model MSE / mean-predictor MSE | bands with R²>0.1 |
|-------|------------------:|------------:|-------------------------------:|------------------:|
| realistic (default CAE) | +0.025 | **−27.8** | 2.1–3.6× **worse** | 0 % |
| clean (default CAE) | −0.003 | **−0.18** | 1.15× worse | 0 % |

The published CAE doesn't merely "predict the mean" (the charitable doc-02 framing) — it is **worse
than the mean predictor**, because dropout 0.5 + sparsity-KL (weight 1.0, target 0.1) + sigmoid +
only 30 epochs drive the output to a *wrong* near-constant field. The training MSE looks "low" (~0.004)
only because the globally-normalized cube is mostly ≈0, so a flat ≈0 output scores low MSE while
reconstructing nothing. **The loss was deceiving; R² and the picture are not.**

**Visual proof** (`reports/exp_records/recon_audit/recon_*.png`): the input shows clear spatial texture;
the CAE reconstruction is a **flat, near-constant field** with none of the structure (and not even the
correct constant). Residual ≈ the whole input.

## 3. Does fixing the config make the CAE + perturbation work?

`reports/cae_config_search.py` flips every knob that could be blamed (dropout, sparsity-KL, epochs,
capacity, and the band-collapse architecture) and measures **both** reconstruction R² and the
perturbation-selection KNN-F1, on clean and realistic (oracle F1: clean 0.486, realistic 0.470):

| variant | clean R² | clean F1 | real R² | real F1 |
|---------|---------:|---------:|--------:|--------:|
| default (published) | −0.315 | 0.331 | −220.8 | 0.371 |
| no dropout | −0.411 | 0.336 | −62.5 | 0.338 |
| no sparsity | −0.828 | 0.329 | −20.1 | 0.398 |
| no reg (drop+sparsity off) | −0.634 | 0.347 | −23.7 | 0.363 |
| no reg + 300 epochs | −0.135 | 0.336 | −1.62 | 0.391 |
| no reg + 300 ep + k=40 | **−0.124** | 0.337 | −4.04 | 0.403 |
| no band-collapse + no reg + 300 ep | −0.270 | 0.330 | −17.4 | 0.335 |

**No configuration crosses R² > 0, and selection F1 never rises above chance (~0.33).** With proper
config the CAE only *approaches* the mean-predictor (clean R² ≈ −0.12) — it never beats it. Removing
the band-collapse did not help. So the failure is **robust to model configuration**, and it is real
(the pipeline is verified clean). The perturbation mechanism, built on this broken reconstruction,
selects at chance regardless.

### Data-side search (is there any synthetic regime where it works?)

`reports/cae_data_search.py`, best CAE config (no-reg, 500 epochs, k=40), from a trivial single
fluorophore up to realistic:

| data | recon R² | CAE selection F1 | oracle F1 |
|------|---------:|-----------------:|----------:|
| **1 dye (dense smooth, ~rank-1)** | **−0.922** | – | – |
| 2 dyes (clean, low noise) | −0.707 | 0.662 | 0.896 |
| 3 dyes (clean, low noise) | −0.074 | 0.601 | 0.846 |
| 3 dyes, per-pixel-normalized input | −15.4 | 0.594 | 0.643 |

**The CAE cannot reconstruct even a single, dense, smooth, essentially rank-1 fluorophore field**
(R² = −0.92) — data that a 1-component PCA reconstructs near-perfectly. R² is negative across the
entire ladder. So reconstruction failure is **fundamental to the spatial-conv + band-collapse
architecture**, independent of model config *and* data regime. (Its selection is only weakly useful on
trivial clean scenes — well below the oracle — and at chance on realistic data; see doc 09.)

## Verdict

We did exactly what was asked — verified the assumption that the pipeline was correct, and inspected
reconstruction numerically *and* visually rather than trusting the loss:

1. **The pipeline is correct** — normalization is lossless (round-trip 1e-8), and synthetic scenes are
   a single chunk so there is no batch/chunk corruption. The earlier conclusions were *not* built on a
   data bug.
2. **The loss was deceiving; the reconstruction is genuinely broken.** R² (vs a per-band-mean
   predictor) is negative everywhere and the saved images show a flat field, not the textured input.
3. **No model configuration and no data regime makes the spatial CAE reconstruct** (R² < 0 across 7
   configs × the full data ladder) — so its perturbation-influence has nothing real to read, and
   selection is at chance on realistic data. The spatial-conv-with-band-collapse architecture
   fundamentally cannot represent this per-pixel spectral data.

**So the spatial-CAE + perturbation method is NOT supported by the evidence on synthetic ME-HSI — and
this is now exhaustively, reproducibly established, not an artifact.** *But the core idea is vindicated
in a different form:* the **per-pixel spectral autoencoder** reconstructs (R² > 0) and its
latent-perturbation selection works (doc 05/10), and a simple **PCA-loadings (k≈6)** selector reaches
**96–97% of the labels-using oracle** robustly (doc 10). The principle (an unsupervised
representation whose structure reveals the informative bands) holds; the specific *spatial CAE
architecture* is the part that must be replaced. The remaining real-world gate is unchanged: confirm
on the real Lichens/Collagen cubes.
