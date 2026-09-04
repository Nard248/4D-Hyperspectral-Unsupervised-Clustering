# spectral_select: the hyperspectral library

## Purpose and status

`spectral_select` is the core library: the data types every other component speaks (`SpectraData`, `ExcitationData`), the raw-data loader wrapper, the `Analyzer` that trains the masked convolutional autoencoder and runs perturbation-based band selection, the `Visualizer`, the `Validator` for scoring against ground truth, a desktop viewer (Tk) and a Jupyter ROI widget. It produced every published number on the hyperspectral datasets (Lichens, Collagen, Pepsin, Drop Data, Sponges). Version `0.1.0`, Python 3.11+, about 9 000 lines.

The perturbation algorithm itself was moved to `selection_core` in June 2026 (guide 04); `Analyzer` delegates to it and a characterisation test pins the two byte-for-byte. The selection step (top-N, MMR, minimum distance) and everything HSI-specific stays here.

## Files

```
src/spectral_select/
├── __init__.py       public API (28 names); importing it pulls torch, matplotlib, seaborn, sklearn, PIL and tkinter
├── types.py          LoadingOptions, ExcitationData, SpectraData, WavelengthBand, AnalysisMetrics, WavelengthResult,
│                     GroundTruth, ValidationMetrics
├── loader.py         DataLoader (wraps mehsi_preprocessor's HyperspectralDataLoader), DataLoadingError
├── config.py         Config dataclass (all knobs), YAML / JSON round trip, component registries
├── analyzer.py       Analyzer: prepare() / select() / fit() / transform() / save_results() / save_model()
├── models/
│   ├── autoencoder.py   HyperspectralCAEWithMasking (3D CNN, per-excitation heads, shared bottleneck)
│   ├── dataset.py       MaskedHyperspectralDataset (whole-image container, global min-max normalisation)
│   └── training.py      create_spatial_chunks, merge_chunk_reconstructions, train_with_masking, evaluate_model_with_masking
├── results.py        ResultsManager: results/<sample>/runs/<run_id>/{config, result, model, visualizations, layers}
├── validation.py     Validator (ARI, NMI, AMI, purity, per-class P/R/F1, Hungarian-free majority mapping), load_ground_truth_from_png
├── visualizer.py     Visualizer: influence heatmap, scatter, ranking, coverage, confusion matrix, per-class, accuracy map, dashboard
├── viewer.py         pure helpers (create_rgb_image, compose_false_color, extract_spectrum, statistics, histogram) + ViewerApp (tkinter)
├── widgets.py        ROIWidget (ipywidgets + matplotlib selectors), create_display_image, path_to_mask
└── protocols.py      ClassifierProtocol, ClusteringProtocol, AutoencoderProtocol, WavelengthRankerProtocol (documentation only)
tests/                test_types, test_config, test_loader, test_analyzer_*, test_results, test_validator_integration,
                      test_widgets, test_pipeline_integration, test_edge_cases, test_notebooks, ...
docs/                 USER_GUIDE.md, CONFIGURATION.md (some defaults are stale, see below), QUICKREF.md, guides/
```

## Data model

`ExcitationData(excitation_nm, cube, emission_wavelengths, exposure_time=None, laser_power=None)`: one cube of shape `(H, W, n_bands)`; validated on construction (3-D, `len(emission_wavelengths) == n_bands`, positive excitation). Properties `height`, `width`, `n_bands`, `shape`.

`SpectraData(excitations: {float: ExcitationData}, mask=None, sample_name="sample", loading_options=None, metadata={})`: the 4D container. Validated: non-empty, all excitations share `(H, W)`, mask matches. Properties `excitation_wavelengths` (sorted), `n_excitations`, `spatial_shape`; `get_excitation(ex_nm)` (float keys; `KeyError` lists what exists); `from_pickle(path)`, `to_pickle(path)`, `from_raw(data_path, metadata_path=None, mask=None, sample_name=None, loading_options=None, **loader_kwargs)`, `to_dict` / `from_dict` (metadata only).

Pickle schemas and their two dialects are specified in `../DATA_GUIDE.md`, section 5. Things that bite:

* `from_pickle` sets `sample_name = path.stem` and puts every unknown top-level key into `metadata` (for the loader dialect that includes the huge `raw_data`).
* `to_pickle` writes `data`, `excitation_wavelengths`, and optionally `mask`, `exposure_times`, `laser_powers`; it drops `sample_name`, `metadata` and `loading_options`.
* `exposure_times` / `laser_powers` are keyed by `str(ex)`; a hand-built pickle with float keys silently yields `None`.
* `LoadingOptions(cutoff_offset=30, apply_rayleigh_cutoff=True, apply_second_order_cutoff=True, normalize_exposure=True, normalize_laser_power=True, roi=None, downscale_factor=1)` is honoured only for `cutoff_offset` and the OR of the two cutoff flags on the `from_raw` path; the normalisation, ROI and downscale fields are documentation.

Result types: `WavelengthBand(rank, excitation_nm, emission_nm, emission_band_index, influence_score)`; `AnalysisMetrics(total_bands_available, bands_selected, compression_ratio, max/min/mean_influence_score)`; `WavelengthResult(sample_name, selected_bands, metrics, timestamp, config_snapshot, method_summary)` with `to_json` / `from_json` / `to_excel`; `GroundTruth(labels, color_mapping, class_names)` with **-1 = background**; `ValidationMetrics` (15 fields).

## Loading raw data

`DataLoader(data_path, metadata_path=None, cutoff_offset=30, verbose=True)` validates the path immediately, probes `import imagej` lazily (`imagej_available`), and builds a `mehsi_preprocessor.io.hyperspectral_loader.HyperspectralDataLoader` on first `load(apply_cutoff=True, pattern="*.im3")`. The return value is `{"data": {str(ex): {"cube", "wavelengths", "excitation"}}, "excitation_wavelengths": [...], "metadata": {...}}`. `SpectraData.from_raw` wraps it. The Java requirements and the one-JVM-per-process rule are in `../SETUP.md` and guide 02. Error messages mention a `SpectraData.from_raw_dict()` that does not exist; use `from_pickle` or the constructor.

## Config

`Config` is a dataclass with validation and YAML / JSON round trip. Defaults **from the code** (`config.py:116-180`; `docs/CONFIGURATION.md` lists several stale ones):

| Group | Fields and defaults |
|---|---|
| identity and paths | `sample_name="sample"`, `data_path=None`, `mask_path=None`, `model_path=None`, `output_dir=None` |
| latent dimensions | `dimension_selection_method="activation"` (`variance`, `activation`, `pca`), `n_important_dimensions=15` |
| perturbation | `perturbation_method="percentile"` (`percentile`, `standard_deviation`, `absolute_range`), `perturbation_magnitudes=[10, 20, 30]`, `perturbation_directions=["bidirectional"]` |
| normalisation | `normalization_method="variance"` (`variance`, `max_per_excitation`, `none`) |
| selection | `n_bands_to_select=30`, `n_layers_to_extract=10`, `use_diversity_constraint=False`, `diversity_method="mmr"`, `lambda_diversity=0.5`, `min_distance_nm=15.0` |
| outputs | `save_tiff_layers=True`, `save_visualizations=True` (never read), `save_detailed_results=True` |
| technical | `device="cuda"` (falls back to CPU), `n_baseline_patches=50`, `patch_size=32`, `patch_stride=16`, `random_seed=42` (never read) |
| model | `model_k1=20`, `model_k3=20`, `model_filter_size=5` (odd), `model_sparsity_target=0.1`, `model_sparsity_weight=1.0`, `model_dropout_rate=0.5` |
| training | `training_epochs=30`, `training_lr=0.001`, `training_chunk_size=64`, `training_chunk_overlap=8`, `training_early_stopping_patience=None`, `training_scheduler_patience=5` |
| pluggable | `classifier="knn"`, `clustering="kmeans"`, `autoencoder_architecture="standard"`, `wavelength_ranker="perturbation"` (only the autoencoder registry is wired) |

**The implicit model cache.** When `model_path` is `None`, `Config` sets it to `model_output/<sample_name>/model.pth` (relative to the working directory). The next `fit()` with the same `sample_name` loads that file instead of training. If the band counts differ (`size mismatch`), it retrains; otherwise it silently reuses. Set `model_path` explicitly in scripts and notebooks.

## Analyzer

```python
from spectral_select import Analyzer, Config, SpectraData, Visualizer
data = SpectraData.from_pickle("Data/processed/Lichens_2/data_dual_cutoff_40nm.pkl")
config = Config(sample_name="lichens_2", n_bands_to_select=30, training_epochs=100, device="mps",
                model_path="model_output/lichens_2/model.pth", output_dir="results/lichens_2")
analyzer = Analyzer(config).fit(data)          # prepare() + select()
bands = analyzer.get_wavelengths()             # [WavelengthBand], rank 1..N
reduced = analyzer.transform(data)             # SpectraData with only the selected bands
analyzer.save_results()                        # JSON + TIFF layers + text summary in output_dir
Visualizer.from_analyzer(analyzer).plot_all()
```

Pipeline, in order:

1. `prepare(data, progress_callback=None)`:
   * `_load_data`: `SpectraData` -> `MaskedHyperspectralDataset(normalize=True)` (global min-max over all excitations, NaN -> invalid).
   * `_load_or_train_model`: build `HyperspectralCAEWithMasking`; `torch.load(model_path)` if it exists (retrain on `Missing key(s)` / `size mismatch`); else `train_with_masking(...)` and save `model.pth`.
   * `_setup_baseline`: grid-scan patches of `patch_size` / `patch_stride` with more than 50 % mask coverage, up to `n_baseline_patches` (fallbacks: smaller patches, then one patch on the mask bounding box); encode them once -> `baseline_latent`, decode -> `baseline_reconstruction`.
2. `select(config=None)` (fast, repeatable with different selection settings):
   * `selection_core.select_important_dimensions(latent, method, n)`;
   * `selection_core.accumulate_influence(...)` -> `{excitation: per-band influence}`;
   * `selection_core.normalize_influence(..., variance_float64=False)` unless `normalization_method == "none"` (`max_per_excitation` maps to the engine's `max_per_group`);
   * `_select_top_bands`: plain top-N, or `_select_bands_mmr` (cosine similarity between per-band spatial profiles, score `relevance - λ · max_similarity`), or `_select_bands_min_distance` (minimum nm gap within the same excitation);
   * builds the `WavelengthResult` with `AnalysisMetrics`, config snapshot and method summary.
3. `fit(data)` = both; `transform(data)` keeps only the selected band indices per excitation (excitations with no selected band are **dropped**); `fit_transform`.

The GUI calls `prepare` in a thread with a progress callback (epoch, total, loss) and `select` per parameter change; that split is why the two methods exist.

### The autoencoder

`HyperspectralCAEWithMasking(excitations_data, k1=20, k3=20, filter_size=5, sparsity_target=0.1, sparsity_weight=1.0, dropout_rate=0.5)`: per-excitation `Conv3d(1 -> k1, kernel (F, F, min(5, n_bands)))` + sigmoid + adaptive average pool over the band axis, mean-merge across excitations, shared `Conv3d(k1 -> k3, (F, F, 1))` + sigmoid (the latent, `(B, k3, 1, H, W)`), dropout, shared `Conv3d(k3 -> k1)`, per-excitation `Conv3d(k1 -> n_bands)` + sigmoid. No spatial downsampling. Loss = masked MSE averaged over excitations + `sparsity_weight` x KL sparsity on the latent means. Because the band counts are baked into the layers, a saved model only fits a dataset with the same per-excitation band counts.

`train_with_masking(model, dataset, num_epochs, learning_rate, chunk_size=64, chunk_overlap=8, batch_size=1, device, early_stopping_patience=None, scheduler_patience=5, mask=None, output_dir="model_output", verbose=True, progress_callback=None)`: Adam + `ReduceLROnPlateau`; all overlapping spatial chunks are materialised on the device up front (memory scales with the whole cube); saves `best_hyperspectral_model.pth`, `final_hyperspectral_model.pth`, `training_losses.npy`, `training_curves.png` in `output_dir`; returns `(model, losses)`. `evaluate_model_with_masking` (MSE / MAE / PSNR per excitation) exists but is not wired into `Analyzer`.

### Outputs

`save_results(output_dir=None)` writes `wavelength_result.json`, `layers/layer_{NN}_ex{X}nm_em{Y}nm_inf{score}.tiff` (16-bit, min-max scaled, top `n_layers_to_extract`) with `layer_metadata.json`, `analysis_config.json`, `selected_bands.txt`. Training writes into `model_path.parent`. `ResultsManager` offers a run-per-directory layout (`results/<sample>/runs/<run_id>/...` with a `latest` symlink, git and environment metadata) used by the experiment drivers.

## Validation and visualisation

`Validator().fit(cluster_labels, ground_truth, valid_mask=None)` drops pixels where either array is negative, maps clusters to classes by majority vote, and computes ARI (`score()`), NMI, AMI, Fowlkes-Mallows, V-measure, homogeneity, completeness, purity, the confusion matrix and per-class precision / recall / F1. `get_metrics_dict()` keys are `N_Clusters, N_GT_Classes, Purity, ARI, NMI, AMI, V-Measure, Homogeneity, Completeness, FM-Score`. `compare({name: labels}, gt)` returns a DataFrame; `generate_report()` a Markdown string.

`load_ground_truth_from_png(png_path, background_colors=None, target_shape=None, class_colors=None, color_tolerance=50)`: each unique colour becomes a class unless `class_colors={name: (r, g, b)}` is given (then a per-pixel Python loop with tolerance; slow on large images). Default background colours are two greys; for the GUI's `class_mask.png` pass `background_colors=[(0, 0, 0, 255)]`.

`Visualizer(output_dir=None, dpi=300, figsize=(12, 8), style="seaborn-v0_8-whitegrid")` with factories `from_result`, `from_analyzer`, `from_results_manager`, `from_validation`. Plots: `plot_influence_heatmap`, `plot_wavelength_scatter`, `plot_excitation_distribution`, `plot_influence_ranking`, `plot_wavelength_coverage`, `plot_confusion_matrix`, `plot_per_class_metrics`, `plot_accuracy_heatmap`, `plot_roi_overlay`, `plot_summary_dashboard`, `plot_all`, `save_all_to_pdf`. Each saves a PNG and returns its path; the default output directory is `visualizations/` under the **current working directory**.

## Viewers

* `launch_viewer()` opens the tkinter **ME-HSI Viewer**: load a `.pkl` or a raw folder, excitation combo, band slider with animation, false-colour composer, click-to-plot spectra with CSV export, histogram and statistics panels, zoom. Masking tools are stubs ("coming soon").
* `ROIWidget(data, excitation=None, figsize=(8, 6), tool="rectangle"|"lasso")` for notebooks (`%matplotlib widget`, `ipywidgets`): draw regions per class, `get_mask()`, `get_bounds()`, `get_roi_code()`, `to_ground_truth()`, `save_mask()` / `load_mask()` (grayscale PNG, ids shifted by +1).
* Pure helpers in `viewer.py` are the ones the walkthrough notebook uses: `create_rgb_image(cube, method="rgb"|"mean"|"max", percentile=98)`, `compose_false_color(cube, r_band, g_band, b_band, percentile=98)`, `extract_spectrum(cube, x, y)`, `extract_multi_excitation_spectrum(data, x, y)`, `compute_image_statistics(image, mask=None)`, `compute_histogram(image, bins=256, mask=None)`, `detect_cube_format(cube)`.

## Tests

`tests/` at the top level: types and pickle round trips, config validation and serialisation, loader errors (real `.im3` tests are skipped), analyzer integration on the synthetic fixtures in `conftest.py` (10 x 10 x 5-band cubes), the `prepare` / `select` split, the training callback, results manager, validator, widgets, notebooks (JSON validity and imports), edge cases, and `test_analyzer_core_equivalence.py` pinning the delegation to `selection_core`.

## Known issues and gotchas

1. **`Config.random_seed` is never read**; nothing seeds torch or numpy, so runs are not reproducible through config. `save_visualizations`, `data_path`, `mask_path`, `classifier`, `clustering`, `wavelength_ranker` are inert too.
2. **The model cache** (`model_output/<sample_name>/model.pth`) silently reuses a stale model when `model_path` is not set.
3. `import spectral_select` needs **tkinter** (module-level import in `viewer.py`) and takes several seconds (torch, matplotlib, seaborn).
4. All default output paths (`results/`, `model_output/`, `visualizations/`) are relative to the working directory; in Jupyter that is wherever the kernel started.
5. `MaskedHyperspectralDataset` ignores its `normalization_method` / `percentile_range` arguments (always global min-max) and downscales with a pure-Python loop.
6. `training.py` passes `verbose=True` to `ReduceLROnPlateau`, which newer PyTorch versions reject; prints epoch lines regardless of `verbose`; runs the encoder twice per batch (once in `forward`, once for the sparsity loss).
7. `Analyzer.transform` drops excitations with zero selected bands, so `n_excitations` can shrink.
8. Docs drift: `docs/CONFIGURATION.md` says `training_epochs=100`, `training_chunk_size=256`, `training_chunk_overlap=64`, early stopping 10 (code: 30, 64, 8, `None`); `docs/guides/VALIDATION.md` calls `validator.get_metrics()` and passes `color_to_class=` / `class_names=` to `load_ground_truth_from_png` (neither exists), treats class 0 as background (it is -1), and calls `.reshape` on the `SpectraData` returned by `transform`; `docs/USER_GUIDE.md` and `tests/test_notebooks.py`'s docstring refer to `notebooks/examples/` (the notebooks are in `examples/`); the `examples/00_data_loading.ipynb` imports from a `scripts/` module that no longer exists and the other two notebooks compute `project_root` as `Path.cwd().parent.parent` (one level too high).
9. `widgets.py` docstrings mention `SpectraData.from_saved()`; it is `from_pickle`.
10. `ResultsManager.get_layer_path` names layers `layer_0001.tiff` while `Analyzer` writes `layer_01_ex..._em..._inf....tiff`; the two never meet.
11. `protocols.py` is aspirational: the real autoencoder takes a dict of tensors, not an array, and nothing enforces the protocols.
