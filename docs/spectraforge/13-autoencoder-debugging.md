# 13 — Debugging the autoencoder (real bugs found + the reconstruction verdict)

Asked to *verify the assumption that the pipeline was correct, alter/debug the autoencoder so
reconstruction actually works, and not trust the loss alone*. This is the result. **Several genuine
bugs were found and fixed**, my earlier reconstruction metric was shown to be misleading, and the
residual failure was localized to a specific architectural choice (not a bug).

## Bugs found and fixed in the training pipeline (`models/training.py`)

The published `train_with_masking` had **four** real defects, all fixed (the published "standard"
model trained with default config is otherwise byte-identical):

1. **`ReduceLROnPlateau(verbose=…)`** crashed on torch ≥ 2.12 (kwarg removed) — every CAE training
   run errored on a modern install. Removed.
2. **`best_model_path` `UnboundLocalError`.** It is assigned only inside `if avg_loss < best_loss`,
   but loaded unconditionally afterward. If the loss is ever NaN / never improves, it is never set →
   crash. Now initialized to `None` with a guarded fallback to the final model.
3. **`0.0 * NaN = NaN` sparsity poisoning.** The sparsity-KL (`compute_sparsity_loss`) assumes a
   sigmoid latent in (0,1); for a relu/gelu latent it returns **NaN**, and the loss line
   `recon_loss + sparsity_weight * sparsity_loss` propagated that NaN **even when
   `sparsity_weight == 0`**. So *every* non-sigmoid run silently trained on a NaN loss (never learned)
   and then hit bug #2. Now the sparsity term is only computed/added when its weight is > 0.
4. **LR-collapse default.** `scheduler_patience=5` halves the LR after 5 stalled epochs, freezing a
   hard start near init. The same CAE via a plain Adam/MSE loop reached R² ≈ +0.12 on rank-1 vs −0.92
   through the pipeline. `Config.training_scheduler_patience` exposes it (set high to disable).

Also: `autoencoder.py` now takes `hidden_activation`/`output_activation` (default `sigmoid` =
published; `relu`/`gelu` selectable), threaded through `Config` + `_create_model`. The published net
stacks **three hidden sigmoids** — a poor trainability choice.

## The metric was deceiving too

My earlier "R² < 0, reconstructs nothing" used **per-band-averaged R²**, which is dominated by the
~40 of 57 bands that are essentially noise (off-peak, emission ≈ 0 — unreconstructable by *anything*;
even a per-pixel MLP scores low). The honest metric is **correlation on the high-variance signal
bands** (`signalCorr`). On the noisy clean scene the achievable ceiling (MLP control) is ~0.49, not
~1.0.

## Reconstruction verdict (signal-band metric, all bug-fixes applied)

| model (clean 3-dye) | signalCorr | selF1 | notes |
|---------------------|-----------:|------:|-------|
| MLP control (reference) | 0.492 | – | achievable ceiling on this noisy data |
| **standard CAE (band-collapse), gelu, fixed** | **0.001** | 0.331 | reconstructs nothing → selection at chance |
| **band-preserving CAE (no collapse)** | **0.104** | **0.414** | the fix that works |

Even with **all four bugs fixed**, the band-activation fixed, the LR held, and 250 epochs, the
**standard band-collapse CAE still reconstructs nothing on multi-fluorophore data** (signalCorr 0.001).
On a single-fluorophore (rank-1) scene the same model reaches signalCorr ≈ 0.34 — so the bottleneck is
specifically the **band-collapse** (`adaptive_avg_pool3d`): it averages over the emission axis, so it
cannot represent *different* spatial patterns at *different* bands, which is exactly the
multi-fluorophore case.

## Verdict

**Removing the band-collapse — keeping everything else about the CAE idea (spatial convolution +
per-excitation branches + latent-perturbation selection) — makes the method work**: reconstruction
rises ~100× (0.001 → 0.104) and, more importantly, **selection goes from chance (0.331) to 0.414 ≈
85 % of the labels-using oracle (0.486)** — the first time the CAE+perturbation produces a meaningful
band subset on multi-fluorophore data.

So the answer to "is it a bug?" is **both**:
1. **There were real bugs** — four of them in `train_with_masking` (a crash-on-NaN, a `0×NaN` that
   silently poisoned every non-sigmoid run, an LR-collapse default, and a torch-compat crash). These
   are fixed; they explained the *crashes* and the corrupted activation experiments.
2. **But the published architecture also has a genuine design flaw** — the band-collapse
   `adaptive_avg_pool3d` — which the bug-fixes alone cannot overcome on multi-component data
   (signalCorr stays 0.001). Removing it is necessary, and sufficient to make selection work.

**Recommendation:** adopt the band-preserving spatial CAE (drop `adaptive_avg_pool3d`, keep the band
axis in the latent) with the fixed trainer (no sparsity poisoning, no LR collapse). Reconstruction is
still below the noisy-data ceiling (0.10 vs 0.49) — the band-preserving conv3d is heavy and likely
wants more epochs/capacity — but **selection already works (0.41) because the band-preserving latent
retains the per-band structure the perturbation reads**, consistent with the broader finding that
selection quality is not tied to perfect reconstruction. The remaining real-world gate is unchanged:
validate on the Lichens/Collagen cubes.

