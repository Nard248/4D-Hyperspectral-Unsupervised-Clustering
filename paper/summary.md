# SpectraForge — Summary of Findings

**One line:** A physics-grounded synthetic ME-HSI benchmark and a significance-tested verdict on
autoencoder-based band selection — honest about what works and what does not.

## What we built

- **SpectraForge generator:** renders multi-excitation excitation–emission cubes from a parametric
  fluorophore model obeying real photophysics (Kasha, Vavilov, Franck–Condon, Stokes shift), with
  optional inner-filter, **reabsorption**, **FRET**, scatter, photon/read noise, and **fixed-pattern
  clutter**. Scenes are built so **variance ≠ informativeness** — the highest-variance bands carry no
  class information — which makes the benchmark an honest test.
- **A full method suite:** the AE + latent-perturbation selector (and convolutional variant), classical
  baselines (variance, PCA-loadings, Laplacian), supervised references (mutual-info, RF-importance), the
  labels-using oracle, the full spectrum, and — crucially — a **random** control.

## The headline findings (≈32 experiments, 3 rounds, significance-tested)

1. **Debugging mattered.** The published CAE selected at chance due to four real trainer bugs + an
   emission-axis "band-collapse"; fixing these lifted it to ≈ 85% of the labels-using oracle.
2. **Blind selection does not beat *random* under clutter.** Under realistic fixed-pattern clutter,
   blind PCA *and* AE collapse to random; only **label-aware** selection beats random. Blind selection
   beats random only on clean, clutter-free data.
3. **Selection never beats the full spectrum.** Across label budgets, full-data ≥ every selection;
   supervised selection beats full only in a narrow corner (clutter + ≲10 labels/class). Selection is a
   **compression** tool, not an accuracy tool.
4. **The AE has no distinct advantage over PCA** — not as a selector (AE ≈ PCA; both ≈ random under
   clutter), nor as a denoiser (its clean-data gain is *linear* low-rank denoising that PCA matches).
   Its one significant edge (nonlinear reabsorption, one metric) did not reproduce at three seeds.
5. **The one structural win:** when discrimination is carried by **spatial texture** (identical
   per-pixel marginals), per-pixel selection is **provably blind**, and a **convolutional/spatial**
   model is *necessary* — vindicating the original spatial-CAE idea, **for spatial classification, not
   per-pixel band selection.**

## Why (the mechanisms)

- **Selection ≠ classification:** the AE's nonlinearity helps the *classifier*, not the band *choice*.
- **Unsupervised ≡ variance:** reconstruction/variance/PCA-loadings rank by variance, which clutter
  decouples from class → performance like random.
- **Extractable ⇔ variance-prominent:** a useful band is high-variance (simple methods find it); a
  non-prominent band is too weak to extract — no gap for a learned selector.

## Recommendation

For per-pixel spectral band selection on realistic data, use **supervised** selection (mutual-info for
clutter/linear, RF-importance for nonlinear) and **always benchmark against random and full-data**. Use
the **convolutional/spatial** model where discrimination has **spatial structure** — that is its real,
provable value. Validate on real data with the same controls (random + full).
