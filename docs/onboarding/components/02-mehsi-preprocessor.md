# MEHSI preprocessor: the Image Reader and the preprocessing wizard

## Purpose and status

`mehsi_preprocessor` is two layers in one package:

1. A **legacy procedural I/O layer** (`io/`): `HyperspectralDataLoader` reads Nuance `.im3` cubes through PyImageJ, applies the Rayleigh and second-order cutoffs, and saves the loader-dialect pickle; `HyperspectralProcessor` adds exposure and lamp-power normalisation and the full "process a raw folder" pipeline. This is the code the capstone project started from, and it is still what everything calls to open raw data.
2. A **modern GUI layer** (`app.py`, `state.py`, `steps/`, `processing/`, `widgets/`, `workers.py`): the 10-step PyQt6 wizard launched by `spectral-select-gui`, built on `spectral_select.SpectraData` and on pure functions in `processing/`. Steps 9 and 10 train the autoencoder and run band selection by calling `spectral_select.Analyzer`.

The two layers meet in one place: `steps/step1_load.py` calls the legacy loader and converts its output to `SpectraData`. The GUI has a Word user guide with screenshots in `docs/gui_user_guide/` and a scripted, offscreen run of all ten steps in `docs/gui_user_guide/run_pipeline_live.py`, which is the best executable reference for driving the pipeline without clicking.

Since September 2026 the package `__init__` imports the Qt application lazily, so the I/O and processing modules work without PyQt6.

## Files

```
src/mehsi_preprocessor/
├── __init__.py               lazy main(); importing the package no longer needs PyQt6
├── __main__.py               python -m mehsi_preprocessor
├── app.py                    PreprocessorWindow (sidebar of 10 steps + QStackedWidget + Previous / Next), main()
├── state.py                  PipelineState, ClassDef, ROIRegion, DEFAULT_CLASS_COLORS, step constants, invalidate_from()
├── workers.py                TrainWorker, SelectWorker (QThreads around Analyzer.prepare / select), run_selection_job
├── io/
│   ├── hyperspectral_loader.py     HyperspectralDataLoader (.im3 via ImageJ, cutoffs, visualisations, pickle I/O)
│   ├── hyperspectral_processor.py  HyperspectralProcessor (exposure / power normalisation, process_full_pipeline)
│   └── hyperspectral_utils.py      pickle helpers, cube -> pandas DataFrame flatteners
├── processing/               pure SpectraData -> SpectraData functions
│   ├── spectral_filter.py    apply_rayleigh_cutoff, apply_manual_emission_crop
│   ├── normalization.py      normalize_spectra
│   ├── cropping.py           spatial_crop
│   ├── mask_ops.py           roi_regions_to_mask, merge_masks   (currently unused by the GUI, see issues)
│   └── export.py             export_masked_pkl, export_unmasked_pkl, export_mask_png, export_roi_json
├── steps/                    base.py (AbstractStepWidget) + step1_load ... step10_select
└── widgets/                  image_canvas, rect_selector, brush_canvas, band_navigator, spectral_bar_chart, metadata_table
tests/mehsi_preprocessor/     smoke tests for every step, offscreen
docs/gui_user_guide/          content.json + build_docx.py -> spectral-select_GUI_User_Guide.docx, screenshots, run_pipeline_live.py
docs/superpowers/specs/2026-06-16-gui-train-select-design.md, plans/2026-06-16-gui-train-select.md
```

## Reading `.im3` files

`HyperspectralDataLoader(data_path=None, metadata_path=None, cutoff_offset=20, use_fiji=True, verbose=True)`.

* `__init__` calls `_init_imagej()` when `use_fiji` and a `data_path` are given: `import imagej; self._ij = imagej.init("sc.fiji:fiji")`. That is a **Maven coordinate**, not a path: PyImageJ downloads Fiji into `~/.jgo` through Maven and boots a JVM. No environment variable, no path parameter, no auto-detection of a desktop Fiji. If the import or the JVM fails, the loader only warns and sets `use_fiji = False`.
* `load_data(apply_cutoff=True, pattern="*.im3", sheet_name=0)` globs the **top level** of the folder, reads `metadata.xlsx` into `{excitation: exposure}` (columns must be literally `Excitation` and `Exposure`), and for each file:
  * parses the excitation as `float(name.split(".")[0])` (bare numbers only; this line is outside the per-file `try`, so a non-numeric name aborts the whole load);
  * reads the cube with `img = self._ij.io().open(path); cube = np.array(self._ij.py.from_java(img), dtype=float)`, which yields `(H, W, bands)` for our data;
  * reconstructs the emission axis: `em_start = 420 if ex <= 400 else ex + 20`, step 10, `num_bands = cube.shape[2]` (`em_end = 720` is stored but never used);
  * stores a record in `self.raw_data[str(ex)]` with keys `ex, em_start, em_end, step, num_rows, num_cols, num_bands, expos_val, notes, data, em_arr`.
  * With `apply_cutoff=True` it then fills `self.data[str(ex)] = {"cube", "wavelengths", "excitation", "raw"}` via `apply_spectral_cutoff`. With `apply_cutoff=False`, `self.data` stays **empty** (only `raw_data` is filled), which is why GUI step 1 reads `raw_data` directly.
* The non-Fiji fallback `_load_im3_directly` raises `NotImplementedError`. Without a working ImageJ, `load_data` returns an empty dict and a pile of warnings.
* Other methods: `get_cube(ex, processed=True) -> (cube, wavelengths)`, `get_emission_spectrum(ex, row, col)`, `get_mean_spectrum(ex=None, region=None)`, `visualize_spectrum`, `visualize_cutoff(ex)`, `visualize_image(ex, emission_wavelength)`, `visualize_false_color(ex, method="rgb"|"max"|"mean"|"pca")`, `get_features_for_ml`, `apply_dimensionality_reduction`, `save_to_pkl(path)`, `load_from_pkl(path)`, `get_summary()`, `print_summary()`.

**One JVM per process.** Calling `imagej.init` twice returns a broken second gateway. In a notebook, start ImageJ once and inject it: `loader = HyperspectralDataLoader(..., use_fiji=False); loader._ij, loader.use_fiji = ij, True` (walkthrough notebook, section 4a).

### The cutoffs

`apply_spectral_cutoff(data, wavelengths, excitation)` keeps bands with `λ >= ex + offset` (Rayleigh) and `λ < 2 ex - offset or λ > 2 ex + offset` (second order). `processing/spectral_filter.apply_rayleigh_cutoff` re-implements the same rule for `SpectraData`. Default offsets differ by entry point: 20 (`HyperspectralDataLoader`), 30 (`HyperspectralProcessor`, `LoadingOptions`, GUI), 40 (the Lichens paper processing and the file names under `Data/processed/Lichens_2`).

### Normalisation: two conventions

* Legacy (`HyperspectralProcessor.normalize_by_exposure / normalize_by_laser_power`, `reference_type` in `min | max | mean | <float>`): `cube * reference / actual`, and the factor is stored next to the cube (`exposure_normalization_factor`). Output files `data_cutoff_{o}nm_exposure_{ref}.pkl`, `..._power_{ref}.pkl`, plus PNG diagnostics.
* GUI (`processing/normalization.normalize_spectra(spectra, by_exposure=True, by_laser_power=True)`): `cube / exposure_time` and `cube / laser_power`, no reference, skipping excitations whose value is `None` or zero.

Same shape, different absolute scale. Do not compare absolute intensities across the two.

### `HyperspectralProcessor.process_full_pipeline`

`(output_dir=None, exposure_reference="max", power_reference="max", create_parquet=True, sample_size=None, preserve_full_data=True) -> {name: path}` loads, cuts, normalises by exposure then power, and writes three loader-dialect pickles. The parquet branch is broken (see issues) and silently skipped.

## The pure processing functions

All take a `SpectraData`, return a **new** one, never mutate the input, and propagate `mask`, `sample_name`, `loading_options` and a copied `metadata`.

| Function | Signature | Effect |
|---|---|---|
| `apply_rayleigh_cutoff` | `(spectra, cutoff_offset=30, apply_second_order=True)` | drops the Rayleigh and second-order bands per excitation |
| `apply_manual_emission_crop` | `(spectra, per_excitation_ranges: {ex: (em_min, em_max)})` | keeps a wavelength window per excitation; excitations not listed are copied |
| `normalize_spectra` | `(spectra, by_exposure=True, by_laser_power=True)` | divides cubes by the attached metadata |
| `spatial_crop` | `(spectra, roi=(row_min, row_max, col_min, col_max))` | slices every cube and the mask; `row_max` / `col_max` exclusive |
| `export_masked_pkl` | `(spectra, mask, output_path)` | sets `NaN` where `mask == 0`, stores `mask > 0` as `uint8`, writes the export dialect |
| `export_unmasked_pkl` | `(spectra, output_path)` | `spectra.to_pickle` |
| `export_mask_png` | `(mask, class_defs, output_path)` | RGB PNG, one colour per `ClassDef`, black background |
| `export_roi_json` | `(roi_regions, class_defs, output_path)` | `{"classes": [...], "regions": [{"class_id", "class_name", "rect": {...}}]}` |
| `roi_regions_to_mask` | `(regions, shape) -> int32 mask` | rasterises rectangles; later regions overwrite earlier |
| `merge_masks` | `(brush_mask, roi_mask, priority="brush")` | combines two class-id masks |

## The wizard, step by step

Launch: `spectral-select-gui` or `python -m mehsi_preprocessor`. All steps subclass `AbstractStepWidget` (`step_index`, `title`, `on_enter()`, optional `on_leave() -> bool`). Heavy work runs in `QThread`s so the window stays responsive. `PipelineState.invalidate_from(step)` clears every attribute owned by later steps whenever an earlier one changes, and `state.current_spectra` walks `filtered -> cropped -> normalized -> raw` to give the most processed version available.

| Step | Widget | Reads | Writes | Notes |
|---|---|---|---|---|
| 1 Load | `Step1Load` | a folder | `raw_spectra`, `data_folder`, `exposure_times`, `laser_powers` | `_LoaderThread` starts ImageJ and calls `load_data(apply_cutoff=False)`; auto-detects `metadata.xlsx` (three spellings) and `TLS Scans/average_power.xlsx` (four locations); fuzzy column matching (`Excitation`/`Wavelength`/`Laser` and `Exposure`/`Time`/`Integration`); builds `SpectraData(sample_name=folder.name)`; band navigator + canvas preview |
| 2 Verify metadata | `Step2Metadata` | `data_folder`, `raw_spectra` | patches `exposure_time` / `laser_power` in place | shows both spreadsheets and the values applied; lets you browse to other files |
| 3 Normalise | `Step3Normalize` | `raw_spectra` | `normalized_spectra` | two checkboxes; warns about missing values; before / after canvases |
| 4 Spatial crop | `Step4SpatialCrop` | `normalized_spectra or raw_spectra` | `crop_roi`, `cropped_spectra` | `RectSelector` drag, live preview, Apply |
| 5 Spectral crop | `Step5SpectralCrop` | `cropped or normalized or raw` | `filtered_spectra` | Rayleigh offset (0 to 200, default 30) + second-order toggle, per-excitation min / max spin boxes, `SpectralBarChart` preview (green kept, red removed) |
| 6 Draw classes | `Step6DrawClasses` | `current_spectra` | `class_mask` (int32), `class_definitions` | class list with Add / Rename / Delete, brush / eraser radius 1 to 50 on `BrushCanvas`; ids start at 1; colours from `DEFAULT_CLASS_COLORS` |
| 7 ROI regions | `Step7ROIRegions` | `current_spectra`, `class_definitions` | `roi_regions` | rectangles per class in a table; Duplicate Last offsets by 10 px; overlay on `RectSelector` |
| 8 Export | `Step8Export` | `current_spectra`, `class_mask`, `class_definitions`, `roi_regions` | files only | `spectra_masked.pkl`, `spectra_unmasked.pkl`, `class_mask.png`, `roi_regions.json`; masked export refuses to run without a mask |
| 9 Train | `Step9Train` | `current_spectra` | `analyzer`, `training_losses`, `model_source` | train (epochs 1 to 2000, default 50; lr default 1e-3) or load a `.pth`; `Config(device="cpu")` is **hard-coded**; live epoch / loss progress via `TrainWorker` -> `Analyzer.prepare(progress_callback=...)` |
| 10 Select | `Step10Select` | `analyzer` | `selection_config`, `selection_result` | `n_bands_to_select` (default 30), dimension method, normalisation, diversity; `SelectWorker` -> `Analyzer.select(config)`; results table; Export runs `analyzer.save_results(out)` (JSON, TIFF layers, text summary) |

## Widgets

All rendering is matplotlib embedded in Qt (`FigureCanvasQTAgg`, `NavigationToolbar2QT`): `ImageCanvas` (base image widget with zoom / pan toolbar), `RectSelector(ImageCanvas)` (matplotlib `RectangleSelector`, emits `rect_selected(r0, r1, c0, c1)`), `BrushCanvas` (grayscale base + RGBA overlay; disc stamping into an int32 mask; painting disabled while the toolbar is in zoom / pan mode; emits `mask_updated` per stroke), `BandNavigator` (pure Qt: excitation combo + synced slider / spin box, emits `band_changed(ex, band)`), `SpectralBarChart` (kept / removed bars), `MetadataTable` (read-only `QTableWidget` from a DataFrame).

## Outputs

* Step 8: `spectra_masked.pkl` (export dialect with NaN outside the mask), `spectra_unmasked.pkl`, `class_mask.png`, `roi_regions.json`. Formats are specified in `../DATA_GUIDE.md`, sections 5 and 6.
* Step 10: `Analyzer.save_results(out)` writes `wavelength_result.json`, `layers/*.tiff` + `layer_metadata.json`, `analysis_config.json`, `selected_bands.txt`.
* Legacy processor: `data_dual_cutoff_{o}nm.pkl`, `data_cutoff_{o}nm_exposure_{ref}.pkl`, `data_cutoff_{o}nm_exposure_{ref}_power_{ref}.pkl` plus `.exposure_normalization.png` / `.power_normalization.png`.

## Running headless

```bash
QT_QPA_PLATFORM=offscreen python docs/gui_user_guide/run_pipeline_live.py     # drives all ten steps and screenshots them
QT_QPA_PLATFORM=offscreen pytest tests/mehsi_preprocessor -q --no-cov
```

Linux needs the Qt system libraries listed in `../SETUP.md`; without a display always set `QT_QPA_PLATFORM=offscreen`.

## Tests

`tests/mehsi_preprocessor/test_steps_smoke.py` and friends build every step offscreen on synthetic `SpectraData`, exercise the private handlers (`_apply`, `_apply_crop`, `_export`, ...) and the processing functions. There are no tests that open a real `.im3` (those are skipped in `tests/test_loader.py` for lack of ImageJ in CI).

## Known issues and gotchas

1. **Drop Data cannot be opened by the loader** (`310 1500 SPF.im3`, `Background.im3`): the filename parse raises before the per-file `try`. Read those cubes with PyImageJ directly or use the cached `.npy` files; the regex and role logic live in `experiments/_archive/2026_paper_runs/drop_data_inspect.py`.
2. **Silent failure when ImageJ is missing**: `load_data` warns per file and returns nothing; the GUI's tifffile fallback in step 1 is unreachable because the `NotImplementedError` never escapes the loader.
3. **The emission axis is fabricated, not read** (`420 + 10 i`, or `ex + 20` for 415 / 430 nm). A different acquisition protocol produces silently wrong wavelengths. The cube axis order `(H, W, bands)` is assumed without a check.
4. **`load_data_and_create_df` always raises `TypeError`** (three positional arguments to a two-parameter function), so `create_parquet=True` in `process_full_pipeline` never produces parquet; the exception is swallowed.
5. **Painting in step 6 wipes step 7**: every brush release calls `invalidate_from(STEP_DRAW_CLASSES)`, which resets `roi_regions` (and the trained analyzer). Finish painting before drawing rectangles.
6. **ROI rectangles never enter the mask**: `roi_regions_to_mask` / `merge_masks` have no callers in `src/`. Rectangles are exported to JSON and drawn, but `spectra_masked.pkl` and `class_mask.png` come from the brush mask only. A rectangle-only annotation yields an all-zero mask.
7. **Class identity is binarised** in `spectra_masked.pkl` (`mask > 0`); classes survive only in `class_mask.png` (colours) and `roi_regions.json`.
8. **`row_max` / `col_max` in `roi_regions.json` are exclusive** slice bounds.
9. Metadata column matching in the GUI is greedy substring matching (`Time` matches `Timestamp`); the legacy loader needs the exact `Excitation` / `Exposure` names.
10. Units are not normalised: `exposure_time` is in ms on this path though `ExcitationData` documents seconds; `Average Power (W)` differs between datasets (W versus mW).
11. Legacy pickles store each uncut cube twice (about 2x size).
12. `device="cpu"` is hard-coded in step 9; use `Analyzer` directly for GPU training.
13. `state.py`'s docstring says invalidation runs "through 8"; it runs through 10. The design spec's Cancel button, Advanced group, loss curve and influence heatmap in the GUI were never implemented.
14. `docs/INSTALLATION.md` (old) claims Fiji is auto-detected from `/Applications/Fiji.app`; it is not. `docs/guides/DATA_PROCESSING.md` and `examples/00_data_loading.ipynb` import from a `scripts/data_processing` module that no longer exists; the correct imports are `from mehsi_preprocessor.io.hyperspectral_loader import HyperspectralDataLoader` and `from mehsi_preprocessor.io.hyperspectral_processor import HyperspectralProcessor`.
