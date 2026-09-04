# Data guide: datasets, layout, formats, metadata

This is the reference for everything under `Data/`: how raw acquisitions are laid out, what the metadata spreadsheets contain, what an `.im3` file really is, the two pickle dialects used for processed datasets, the annotation formats, and a catalogue of every dataset we keep. The companion notebook `examples/03_hsi_data_walkthrough.ipynb` demonstrates each item with real files.

## 1. Principles

* `Data/` lives at the repository root and is **git-ignored** (so are `*.pkl`, `*.npy`, `*.im3`, `*.png`, `*.tiff`, `*.h5` anywhere in the tree). Datasets travel through the lab's shared storage or Zenodo, never through git.
* `Data/Raw/` is **read-only** in spirit: never write into a raw folder except to add the two metadata spreadsheets. Everything derived goes under `Data/processed/<Sample>/`.
* Folder names are part of the contract. Scripts reference them literally (`Data/processed/Lichens Dataset 1/`, `Data/Raw/PAMAP2_MONSTER/`), spaces included. Do not rename existing folders.
* A "4D dataset" is a set of 3D cubes keyed by excitation wavelength. In code it is a `spectral_select.SpectraData`; on disk it is a pickle in one of the two dialects described in section 5.

## 2. Raw folder contract

```
Data/Raw/<Sample>/
├── 310.im3                        # one cube per excitation wavelength, REQUIRED
├── 325.im3
├── ...
├── metadata.xlsx                  # exposure time per excitation, strongly recommended
├── TLS Scans/
│   ├── average_power.xlsx         # lamp power per excitation, recommended
│   └── *.TRQ                      # raw power-meter scans (optional, git-ignored)
├── Reflectance/White.im3          # optional white reference (Lichens_2)
├── Background.im3, Whitelight.im3 # optional dark / white-light frames (Drop Data)
└── Description.docx               # optional acquisition notes (Collagen)
```

What the loader (`mehsi_preprocessor.io.hyperspectral_loader.HyperspectralDataLoader`) actually does with this:

* It globs `*.im3` in the **top level only**, so `Reflectance/White.im3` is ignored automatically.
* The excitation is parsed from the filename as `float(name.split(".")[0])`. Only bare numbers work: `310.im3`, `365.im3`, `400.5.im3`. Anything else (`Background.im3`, `310 1500 SPF.im3`) raises `ValueError` and aborts the whole load, because that line sits outside the per-file `try`.
* `metadata.xlsx` is read with pandas; the columns must be literally `Excitation` and `Exposure` (whitespace is stripped). The GUI wizard is more lenient and matches column names by substring.
* `TLS Scans/average_power.xlsx` is read by `HyperspectralProcessor.read_laser_power_excel` and by GUI step 1; it wants `Excitation Wavelength (nm)` and `Average Power (W)`, falling back to the first two columns.

### Two filename conventions

| Convention | Example | Used by | Loader support |
|---|---|---|---|
| bare excitation | `365.im3` | Lichens, Collagen, Sponges | yes |
| exposure bracket | `365 10 SPF.im3` (365 nm, 10 ms), plus `Background.im3`, `Whitelight.im3` | Drop Data | **no**: read with PyImageJ directly; the regex `^(\d+)\s+(\d+)\s+SPF$` and the role assignment live in `experiments/_archive/2026_paper_runs/drop_data_inspect.py` and in section 3 of the walkthrough notebook |

## 3. Metadata spreadsheets

### `metadata.xlsx`

| Excitation | Exposure |
|---|---|
| 310 | 5432.9 |
| 325 | 924.52 |
| ... | ... |

One row per excitation. `Exposure` is the camera exposure in **milliseconds** as exported by the Nuance software (long exposures at 310 nm where the lamp is weakest, a few ms at 385 to 400 nm). The value is attached to `ExcitationData.exposure_time`.

### `TLS Scans/average_power.xlsx`

| Excitation Wavelength (nm) | Average Power (W) |
|---|---|
| 310 | 0.000416 |
| 325 | 0.000305 |
| ... | ... |

Average lamp power per excitation, computed from the tunable light source's power-meter scans (`.TRQ` files, one per excitation). The original conversion script (`process_trq_files`) was part of the capstone code base and is no longer in the repository; if you acquire a new dataset, compute the average power per excitation with your own script and write this sheet. Lichens_2 carries a stray third column (the same value in mW); it is ignored.

### Unit caveats

The library treats both values as scale factors, so units only matter when you compare datasets or read the numbers as physics:

| Dataset | `exposure_time` unit in the pickle | `laser_power` unit |
|---|---|---|
| Lichens_2, Collagen (loader / GUI path) | ms (as in the spreadsheet) | W (Lichens) |
| Drop Data Radiometric (`rad_merged`) | **seconds** (converted by the script) | mW (`260414 lamp scan.xlsx`) |
| SpectraForge synthetic | dimensionless | dimensionless |

`ExcitationData.exposure_time` is documented as seconds; the loader path never converts. Normalisation divides by the value, so the result is correct up to a global constant either way. Do not mix datasets normalised with different units in one model.

## 4. What is inside an `.im3`

* Format: **PerkinElmer/CRi Nuance IM3** (Bio-Formats name `Perkin-Elmer Nuance IM3`). The header starts with a `FileVersion` record and names the program `Nuance`.
* Pixels: `uint16`, 348 x 256 (X x Y) for the current camera, 31 channels, dimension order `XYCZT`. PyImageJ delivers it as an `xarray.DataArray` with dims `(row, col, ch)`, which is exactly our `(H, W, bands)` convention; the loader casts to float.
* **No wavelength table is exposed**: Bio-Formats reports zero global-metadata keys and no channel wavelengths. The emission axis is therefore reconstructed in code: `420 + 10 * i` nm for excitations up to 400 nm, `ex + 20 + 10 * i` for 415 and 430 nm. If the acquisition protocol changes, update `HyperspectralDataLoader.load_data` (the `em_start` line) and the `nuance_emission_axis` helper in the notebook.
* Saturation: raw counts clip at **3886**. `drop_data_preprocess.py` uses `SAT_CEILING = 3886`; the notebook's statistics section counts saturated pixels with the same constant.
* Reading it requires Fiji's Bio-Formats through PyImageJ (JDK + Maven, see `SETUP.md`). The "direct" fallback in the loader is a stub that raises `NotImplementedError`.

## 5. Processed datasets: the two pickle dialects

Both are Python pickles produced by our own code. Pickles execute code on load, so only open files produced by this pipeline or by a colleague. `SpectraData.from_pickle` accepts both dialects and dispatches on the keys.

### 5.1 Export dialect (canonical)

Written by `SpectraData.to_pickle`, by the GUI (step 8), by SpectraForge and by the Drop Data scripts. Files: `spectra_masked.pkl`, `spectra_unmasked.pkl`, `spectra_data.pkl`.

```python
{
  "data": {
      "310.0": {"cube": np.ndarray,            # (H, W, n_bands), float64 or float32
                "wavelengths": [420.0, 430.0, ...]},
      "325.0": {...},
  },
  "excitation_wavelengths": [310.0, 325.0, ...],  # sorted floats
  "mask": np.ndarray,                             # optional, (H, W) uint8, 1 = analysed
  "exposure_times": {"310.0": 5432.9, ...},       # optional
  "laser_powers":   {"310.0": 0.000416, ...},     # optional
}
```

* Excitation keys are **strings** in the file and **floats** on the object.
* `spectra_masked.pkl` additionally has `NaN` in every cube outside the mask (that is how the autoencoder's masked loss knows what to ignore). `spectra_unmasked.pkl` is the same data without the NaNs.
* `sample_name` and `metadata` are **not** stored. `from_pickle` names the sample after the file stem.

### 5.2 Loader dialect (legacy, still common)

Written by `HyperspectralDataLoader.save_to_pkl` and `HyperspectralProcessor.process_full_pipeline`. Files: `data_dual_cutoff_40nm.pkl`, `data_cutoff_40nm_exposure_max.pkl`, `data_cutoff_40nm_exposure_max_power_min.pkl`.

```python
{
  "data": {
      "310.0": {"cube": ..., "wavelengths": [...], "excitation": 310.0,
                "raw": {"ex", "em_start", "em_end", "step", "num_rows", "num_cols", "num_bands",
                        "expos_val", "notes", "data": <uncut cube>, "em_arr": [...]},
                "exposure_normalization_factor": float,       # only in the normalised files
                "laser_power_normalization_factor": float},
  },
  "raw_data": {"310.0": {<same record as data[ex]["raw"]>}, ...},
  "metadata": {"processed_date": "...", "cutoff_offset": 40, "excitation_wavelengths": [...],
               "exposure_normalization": {...}, "laser_power_normalization": {...}},
  "excitation_wavelengths": [...],
  "cutoff_offset": 40,
}
```

* The uncut cubes are stored twice, so the file is about twice the size of the data. Loading through `from_pickle` keeps all of it in `SpectraData.metadata` (`raw_data`, `metadata`, `cutoff_offset`); delete `data.metadata["raw_data"]` if memory matters.
* No `mask`; masks live beside the file as `.npy` (`lichens_2_mask.npy`).
* Normalisation in this dialect is **multiplicative to a reference** (`cube * reference / actual`, reference = max exposure, min power); the GUI's `normalize_spectra` **divides by the raw value**. Same shape, different absolute scale.

### 5.3 Reading either dialect

```python
from spectral_select import SpectraData
data = SpectraData.from_pickle("Data/processed/Lichens_2/data_dual_cutoff_40nm.pkl")
data.excitation_wavelengths        # [310.0, 325.0, ...]
e = data.get_excitation(365.0)     # ExcitationData
e.cube.shape, e.emission_wavelengths, e.exposure_time, e.laser_power
data.mask, data.spatial_shape, data.metadata
```

## 6. Masks and annotations

| File | Produced by | Contents |
|---|---|---|
| `*_mask.npy` (e.g. `lichens_2_mask.npy`, `drop_mask.npy`) | scripts | `(H, W)` bool, True = analyse. Load with `np.load`. |
| `mask` key in a pickle | `SpectraData.to_pickle`, GUI step 8 | `(H, W)` uint8 0/1. |
| `*_class_mask.npy` (`lichens_2_class_mask.npy`) | legacy annotation tool | `(H, W)` int16, **-1 = unlabelled**, 0..K-1 = class ids. |
| `*_class_info.json` | legacy annotation tool | `{"0": {"name": "Class 1 (Red)", "color": [255,0,0], "pixel_count": 3568}, ...}` — **0-indexed**, matches `*_class_mask.npy`. |
| `class_mask.png` | GUI step 8 (`export_mask_png`) | RGB image; each class painted in its colour, background black `(0,0,0)`. Load with `load_ground_truth_from_png(path, background_colors=[(0,0,0,255)])`. |
| `roi_regions.json` (GUI form) | GUI step 8 (`export_roi_json`) | `{"classes": [{"id": 1, "name", "color": [r,g,b]}], "regions": [{"class_id", "class_name", "rect": {"row_min","row_max","col_min","col_max"}}]}` — **1-indexed**, `row_max`/`col_max` are **exclusive** (slice bounds). |
| `lichens_2_roi_regions.json` (legacy form) | older tooling | a plain list: `[{"name", "coords": [r0, r1, c0, c1], "color": "#FF0000", "class_id", "class_name"}]` |
| `drop_labels.npy` | `drop_data_preprocess.py` | `(H, W)` int32 instance labels of individual drops (0 = background). |
| `labeled_lichens*.png` | manual | colour-coded ground truth for `examples/02_validation.ipynb`. |

Two ROI JSON variants exist because the GUI export format replaced an older one; readers in `experiments/` handle the one they were written for. When adding annotations, use the GUI format.

`GroundTruth` objects in `spectral_select` use **-1 for background** and non-negative integers for classes; class 0 is a real class. Keep that in mind when converting the 1-indexed GUI JSON.

## 7. Other on-disk formats

| Format | Where | Notes |
|---|---|---|
| `.npy` cube caches | `Data/processed/Drop Data/raw/<name>.npy` | `(256, 348, 31)` float32 raw counts, one per `.im3`, produced by `drop_data_inspect.py` so the JVM is not needed again. |
| `groundtruth.npz` + `groundtruth.json` | SpectraForge output folders | `conc__<fluorophore>` maps, `clean__<excitation>` cubes, `emission_grid`; JSON lists fluorophores, excitations, materials, seed. No loader; unpack with `np.load`. |
| multi-page `.tif` | `data/exported_tiffs/`, `results/**/layers/` | `(bands, H, W)`, for Fiji. `Analyzer.save_results` writes 16-bit, min-max scaled layers. |
| `results/**/wavelength_result.json` | `Analyzer.save_results` | selected bands with rank, excitation, emission, band index, influence score. |
| `.h5` | none today | `h5py` is a dependency; the notebook shows a chunked-per-band layout if you ever need partial reads. |

## 8. Dataset catalogue

Sizes are for the copies on the maintainer's machine (September 2026). Shapes are `(H, W)` pixels; excitations in nm.

| Dataset | Path(s) | Shape / excitations | What it is | Status and consumers |
|---|---|---|---|---|
| **Lichens Dataset 1** | `processed/Lichens Dataset 1/` (2.8 GB: `spectra_masked.pkl` 1.5 GB, `spectra_unmasked.pkl`, `class_mask.png`, `roi_regions.json`) | 1040 x 925; 310, 325, 340, 365, 385, 400, 415, 430 | The full-resolution lichen palette used for the TPAMI submission (four labelled classes). | **Canonical reference.** `experiments/run_master_experiment.py`, `rerun_knn.py`, `export_tiffs.py`. |
| **Lichens_2** | `Raw/Lichens_2/` (50 MB, 8 `.im3` + metadata + `Reflectance/White.im3`), `processed/Lichens_2/` (1.1 GB: three loader-dialect pickles, masks, class mask, ROI JSON, labelled PNGs) | 256 x 348; same 8 excitations | Same specimens at camera resolution. The **onboarding dataset**: small, complete, with raw files. | `tests/test_config.py`, all example notebooks. |
| **Collagen (Acetic Acid)** | `Raw/Collagen_Acetic_Acid/` (34 MB, 6 `.im3`, `Description.docx`), `processed/Collagen_Acetic_Acid/` (304 MB: `spectra_masked.pkl`, `spectra_masked_raw.pkl`, `spectra_masked_exposure_only.pkl`, `class_mask.png`, `roi_regions*.json`) | 256 x 348; 310 to 400 | Collagen digestion series, three ROI classes. | `collagen_*` archived experiments; the CommsAI paper (149 bands after cutoff). |
| **Collagen (Pepsin)** | `Raw/Collagen_Pepsin/`, `processed/Collagen Pepsin/` (215 MB) | 256 x 348; 310 to 400 | Pepsin digestion series. | `pepsin_*` archived experiments; the IASIM 2026 abstract. |
| **Drop Data** | `Raw/Drop Data/` (116 MB: 19 bracketed `.im3`, `Background.im3`, `Whitelight.im3`), `processed/Drop Data/` (587 MB: `raw/*.npy`, variants `raw`, `dark`, `dark_norm`, `dark_norm_mask`, `full`, each with `spectra_data.pkl`; `drop_mask.npy`, `drop_labels.npy`), `processed/Drop Data Cropped/`, `processed/Drop Data Radiometric/` (`rad_merged`, `rad_merged_detrend`, `rad_best`, `rad_best_detrend`) | 256 x 348 (cropped to 175 x 348 in Radiometric); 310 to 415, 2 to 3 exposure brackets each | Liquid drops on a substrate with a ruler; first post-TPAMI blind-validation dataset. | **`Drop Data Radiometric/rad_merged`** is the canonical variant (exposure x lamp-power corrected, HDR-merged brackets). Archived `drop_data_*` scripts. |
| **Sponges (Acid Group 1)** | `processed/Sponges Acid Group 1/` (203 MB) | camera resolution | Sponge specimens. | `poster_*` archived experiments. |
| **PAMAP2 (MONSTER)** | `Raw/PAMAP2_MONSTER/` (1.5 GB: `PAMAP2_X.npy` (N, 52, 100), `PAMAP2_y.csv`, `PAMAP2_subject_id.csv`) | not imaging | Wearable-sensor benchmark for the domain-agnostic `channel_select` work. Download from `huggingface.co/datasets/monster-monash/PAMAP2`. | `experiments/pamap2/`, `saturation_diagnostic.py`. |
| **BCI IV-2a (EEG)** | downloaded on demand into MNE's cache by MOABB | not imaging | Motor-imagery EEG, 22 electrodes, 9 subjects. | `experiments/eeg/`. |
| **Synthetic (SpectraForge)** | generated: `spectraforge-demo -o <dir>` | any | Rendered from painted fluorophores with known ground truth. | `reports/*.py`, `tests/spectraforge/`. |

Archived (moved out of `Data/` on 2026-06-15, recoverable from `archive/redundant_data/`): three redundant Lichens processings, `Raw/Lime` (a 19-excitation early acquisition), `Sample Processed`, `Sample Export 2`. See `docs/ARCHIVE_MANIFEST.md`.

### Public release

The Lichens dataset is published on Zenodo as *250707_4DHSI_Lichens_Palette* (Chilingaryan and Sarvazyan, CC BY 4.0): <https://doi.org/10.5281/zenodo.18640119>. Files: `HSI.zip` (1.8 GB of cubes), `Metadata_HSI.csv`, `Metadata_Classes.csv`, `Mask_Manual.png`, `RGBRendering_WhiteLight.png`, `Photo_SamplePrep.jpg`. Cite the record when you use it outside the lab.

## 9. Adding a new dataset

1. Create `Data/Raw/<Sample>/` and copy the `.im3` files in, renamed to `<excitation>.im3`. Keep reference frames (`Background`, `White`) in a subfolder or with non-numeric names, but note that non-numeric names at the top level break the loader.
2. Write `metadata.xlsx` (`Excitation`, `Exposure` in ms) and `TLS Scans/average_power.xlsx` (`Excitation Wavelength (nm)`, `Average Power (W)`).
3. Open the folder in `spectral-select-gui` (step 1), verify the metadata table (step 2), normalise, crop, apply the cutoffs (steps 3 to 5), annotate (steps 6 and 7), export (step 8) into `Data/processed/<Sample>/`. Or script it with the functions shown in the walkthrough notebook, section 10.
4. Add a row to the catalogue above and to `docs/DATA.md`, with the shape, excitations, size and the scripts that consume it.
5. If the emission grid differs from 420 to 720 nm at 10 nm, fix the reconstruction rule (section 4) **before** processing anything.

## 10. Known pitfalls

* The loader silently returns nothing if ImageJ failed to start; watch for `Failed to initialize ImageJ` warnings.
* `Config(sample_name=...)` drives where models and results are written (`model_output/<sample>/`, `results/<sample>/`), and `SpectraData.sample_name` comes from the file stem. They rarely agree; the `Config` name wins.
* The class mask exported by the GUI is binarised when it goes into `spectra_masked.pkl`: class identity survives only in `class_mask.png` and `roi_regions.json`.
* ROI rectangles drawn in GUI step 7 are exported to JSON but do **not** enter the class mask; paint classes with the brush in step 6 if you need them in the mask.
* Going back to GUI step 6 and painting clears the step 7 rectangles.
* `experiments/export_tiffs.py` refers to `data/processed` in lowercase; it works on macOS and fails on Linux.
