# selection_core: the perturbation-influence algorithm

## Purpose and status

`selection_core` is the single implementation of the "Perturbation-Based Autoencoder" selection algorithm. It is 193 lines of pure functions on torch tensors with no domain types: `spectral_select.Analyzer` (hyperspectral cubes) and `channel_select.run_selection` (grouped sensor channels) both call it. It was extracted in June 2026 ("Phase 7 engine unification", `docs/superpowers/plans/2026-06-16-phase7-engine-unification.md`) from the `Analyzer`, and `tests/test_analyzer_core_equivalence.py` pins that the extraction changed nothing.

Two axis conventions are the only assumptions: in every per-group reconstruction the **channel (band) axis is last**; in the latent the **batch axis is first**.

## API

Re-exported by `selection_core/__init__.py`:

| Function | Signature |
|---|---|
| `select_important_dimensions` | `(latent: Tensor, method: str, n: int) -> list[tuple[float, tuple[int, ...]]]` |
| `latent_statistics` | `(latent_flat: Tensor) -> dict` |
| `perturbation_amount` | `(coord, latent_dims, magnitude, sign, stats, baseline_latent, method) -> float` |
| `measure_influence` | `(decode_fn, groups, perturbed_latent, baseline_recon, weight=1.0) -> dict[group, np.ndarray]` |
| `accumulate_influence` | `(decode_fn, groups, channels_per_group, latent, baseline_recon, important_dims, *, magnitudes, directions, perturbation_method) -> dict[group, np.ndarray]` |
| `normalize_influence` | `(influence, data, method, *, variance_float64=True) -> dict[group, np.ndarray]` |

`decode_fn` is the model's decoder (`latent -> {group: reconstruction}`), `groups` the list of group keys (excitation wavelengths for HSI; sensor locations or brain regions for `channel_select`), `data` the `{group: tensor}` input used for variance normalisation.

## The algorithm, step by step

Input: a trained autoencoder, a batch of baseline inputs (patches of the cube, or windows of a time series), its latent `z` of shape `(batch, *latent_dims)` and its reconstruction.

1. **Pick the important latent coordinates** (`select_important_dimensions`). Flatten `z` to `(batch, -1)` and score each coordinate: `variance` = variance over the batch; `activation` = mean absolute activation; `pca` = StandardScaler + PCA with `n_comp = min(2n, n_features, n_samples - 1)`, score = column sum of `|components|`. Keep the top `n`, returned as `(score, coord)` with `coord` unravelled back to the latent shape.
2. **Latent statistics** (`latent_statistics`). Per coordinate: `std`, `min`, `max`, percentiles 5 / 10 / 25 / 50 / 75 / 90 / 95 (`torch.quantile`). With a batch of one, `std` is set to zero instead of NaN.
3. **Perturbation amount** (`perturbation_amount`) for a coordinate, a magnitude `m` and a sign `s`: `percentile` moves the coordinate to the stored percentile nearest `50 + s·m/2` (amount = target minus the batch mean); `standard_deviation` uses `s · (m/100) · std`; `absolute_range` uses `s · (m/100) · (max - min)`.
4. **Influence of one perturbation** (`measure_influence`). Add the amount to the coordinate in a clone of the latent, decode, and for each group compute `mean(|perturbed - baseline|)` over **every axis except the last**; for HSI that is over batch, height and width, leaving one number per emission band; for time series over batch and time, leaving one per channel. Multiply by `weight`.
5. **Accumulate** (`accumulate_influence`). For each important coordinate, each magnitude, each direction (`bidirectional` -> signs -1 and +1 with weight 0.5 each; `positive`, `negative` -> one sign with weight 1.0), call step 4 with `weight = importance score x sign weight` and sum. The result is one influence vector per group.
6. **Normalise** (`normalize_influence`). `none` copies; `max_per_group` divides each group by its maximum (if above 1e-10); `variance` divides each channel by the variance of the input data along all axes but the last, clamped at 1e-10. `variance_float64=False` keeps the computation in float32 so the HSI `Analyzer` stays byte-identical to its historical output.
7. **Selection** is **not** here by design: HSI ranks `(excitation, emission)` pairs in nanometres with `WavelengthBand` objects, the general engine ranks `(group, channel)` tuples in channel indices. Each package keeps its own top-N / MMR / minimum-distance step (`spectral_select/analyzer.py:_select_top_bands`, `channel_select/engine.py:select_channels`).

## How the two callers use it

| Step | `spectral_select.Analyzer` | `channel_select.run_selection` |
|---|---|---|
| model | `HyperspectralCAEWithMasking` (3D CNN, latent `(B, k3, 1, H, W)`) | `TemporalGroupedAutoencoder`, `BottleneckTemporalAutoencoder`, `SpatialBottleneckAutoencoder` |
| baseline batch | up to `n_baseline_patches` spatial patches with more than 50 % mask coverage | the whole dataset (subsampled by the caller) encoded in one forward pass |
| config | `Config` (`dimension_selection_method`, `n_important_dimensions`, `perturbation_*`, `normalization_method` in `variance / max_per_excitation / none`) | `SelectionConfig` (same names; normalisation in `variance / max_per_group / none`) |
| normalisation call | skipped entirely when `none`; `max_per_excitation` translated to `max_per_group`; `variance_float64=False` | always called (no-op on `none`); float64 |
| selection | `_select_top_bands` / `_select_bands_mmr` (cosine similarity of spatial profiles) / `_select_bands_min_distance` (nm) | `select_channels(influence, data, K, method, lambda_diversity, min_distance)` (cosine similarity of channel profiles; distance in channel index) |

`tests/test_analyzer_core_equivalence.py` builds a deterministic fake HSI model with a linear decoder and a fixed 5-D latent and asserts exact equality of dimension selection (variance and activation), influence accumulation (standard-deviation and percentile perturbation) and the float32 normalisation path. `tests/selection_core/test_engine.py` pins the primitives (ranking order, multi-dimensional coordinates, "reduces all but the last axis", weights, the precision switch, `max_per_group`, the batch-of-one guard).

## Gotchas

* `run_selection` in `channel_select` encodes the entire dataset in one pass; the experiment scripts subsample to 1 500 to 4 000 windows before calling it. `Analyzer` avoids the problem with baseline patches.
* `torch.quantile` has an input-size limit of about 16.7 million elements; a very large HSI latent could hit it. Nothing guards against it.
* The two MMR implementations differ in one detail: `channel_select` guards the relevance denominator with `max_inf or 1.0`, `Analyzer` with `if max_influence < 1e-10`. Both are inside the intentionally unshared step, so the equivalence tests do not cover them.
* The design spec's `adapters/spectral.py` regression test (the shared engine reproducing `Analyzer`'s selections end to end on a real cube) was never written; what exists is the math-level equivalence test with a fake model.
* `pytest.ini` measures coverage for `spectral_select` only; CI (`.github/workflows/test.yml`) covers all four packages.
