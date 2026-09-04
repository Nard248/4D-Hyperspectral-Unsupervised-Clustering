# Experiments, results and publications

## Purpose

This guide explains how the numbers in the papers are produced: which scripts are the maintained "reproduce the paper" drivers, what they read and write, what the one-off archived scripts are, where run outputs live, and what each publication folder contains. Read it before re-running anything or before adding a new experiment.

## Three tiers of scripts

`experiments/README.md` defines the tiers:

1. **Top level of `experiments/`**: maintained drivers for the canonical Lichens pipeline plus the newer generalisation and screening scripts.
2. **`experiments/pamap2/`, `experiments/eeg/`, `experiments/synthetic/`**: the `channel_select` generalisation domains (guide 05).
3. **`experiments/_archive/2026_paper_runs/`**: 33 one-off scripts for the 2026 submissions (`collagen_*`, `drop_data_*` (16), `pepsin_*`, `poster_*`, `reprocess_with_metadata.py`). Runnable, tied to specific data paths and dates, not maintained.

All drivers assume `pip install -e ".[dev]"`, the datasets under `Data/` as catalogued in `../DATA_GUIDE.md`, and the repository root as the working directory. The hyperspectral drivers anchor paths with `Path(__file__).resolve().parent.parent`; the `channel_select` scripts use relative `Data/...` paths.

## The canonical Lichens pipeline (TPAMI numbers)

Run in this order; step 1 wants a GPU (4 to 8 hours), the rest are minutes on CPU. Everything targets `results/Lichens_Dataset_1_MasterRun/`.

| Step | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 1 | `run_master_experiment.py [--retrain \| --use-existing \| --model PATH] [--data-dir DIR] [--output DIR] [--n-bands 5,10,15] [--n-dims 1,3] [--epochs N]` | grid over every selection configuration through `spectral_select.Analyzer`, KNN evaluation on the labelled ROIs | `Data/processed/Lichens Dataset 1/` (`spectra_masked.pkl`, `class_mask.png`, `roi_regions.json`) | `results.csv`, `experiments/<config>/wavelengths.json`, `model/` |
| 2 | `analyze_results.py` | Excel workbook with charts and per-config sheets; regroups runs | `results.csv` | `comprehensive_analysis.xlsx`, `grouped_by_config/` |
| 3 | `extract_wavelengths.py` | top-10 per configuration group, consensus wavelengths, PCA-versus-variance comparison | `results.csv`, `wavelengths.json` | `wavelength_analysis/*.xlsx, *.csv` |
| 4 | `export_wavelength_combinations.py` | **Pepsin, not Lichens**: exports all 432 Pepsin configurations' selections | `results/Collagen_Pepsin_Normalized/` | `results/Pepsin_Wavelength_Exports/` |
| 5 | `generate_figures.py` | about 35 figure pairs (PNG + PDF): accuracy curves, parameter effects, wavelength heatmaps, Pareto plots, correlations, executive summary | `results.csv`, `wavelengths.json` | `visualizations/` |
| 6 | `rerun_knn.py` | publication classification maps for three fixed configurations (baseline 192 bands, 80 bands, 9 bands) | `Data/processed/Lichens Dataset 1/` + those `wavelengths.json` | `paper_figures/`, `paper_metrics.json` |
| 7 | `export_tiffs.py` | the paper's 3-band and 9-band selections as ImageJ multi-page TIFF stacks | `data/processed/Lichens Dataset 1/spectra_unmasked.pkl` (lowercase path, macOS only) | `data/exported_tiffs/` |

The evaluation protocol shared by all hyperspectral results: train the autoencoder without labels, select K bands, train a k-nearest-neighbour classifier (KNN-5) on labelled ROI pixels using only those bands, report accuracy / macro-F1 against baselines (random, variance, PCA loading, supervised mutual information, SOTA band selectors). The Collagen and Drop Data analyses reuse the same protocol.

## Drop Data (the first post-TPAMI dataset)

The pipeline lives in `_archive/2026_paper_runs/`: `drop_data_inspect.py` (reads the bracketed `.im3` files through PyImageJ, caches `Data/processed/Drop Data/raw/*.npy`, recommends one exposure per excitation) -> `drop_data_preprocess.py` (five cumulative variants: `raw`, `dark`, `dark_norm`, `dark_norm_mask`, `full`; builds `drop_mask.npy` and `drop_labels.npy` by watershed) -> `drop_data_selection_sweep.py`, `drop_data_post_analysis.py`. The first results were wrong because the exposure brackets (250x range) and lamp power (77x range) were ignored; `drop_data_radiometric_rerun.py` applies `(raw - dark) / (exposure x lamp_power)` with a single global scale, HDR-merges the brackets, and writes `Data/processed/Drop Data Radiometric/{rad_merged, rad_merged_detrend, rad_best, rad_best_detrend}/spectra_data.pkl`; `drop_data_radiometric_knn.py` scores them. **`rad_merged` is canonical** (KNN-5 accuracy 0.984 with the autoencoder selection, above all baselines). The lamp-power sheet it needs (`260414 lamp scan.xlsx`) is referenced by an absolute path in the maintainer's Downloads folder; ask for it.

## Where outputs go

| Directory | Contents | Canonical? |
|---|---|---|
| `results/` (git-ignored, 53 entries) | run outputs. `Lichens_Dataset_1_MasterRun/` (the single directory all seven drivers target), `Collagen_*`, `Pepsin_*`, `Drop_Data_*` (11 dirs + a run log), `Sponges_Acid_*`, `Lichens_2_*`, plus timestamped scratch runs (`Lichens_pipeline_2026*`, `Lichens_Dataset_1_2026*`), `debug_output`, `diagnostic_output`, `integration_test_results`, `test`, `models`, `sample_pipeline_*` | only the named dataset folders; the timestamped and debug folders are scratch |
| `model_output/` (git-ignored) | autoencoder checkpoints per `Config.sample_name`: `best_hyperspectral_model.pth`, `final_hyperspectral_model.pth`, `model.pth`, training curves. `Collagen_Acetic_Acid/`, `synth/` are meaningful; `t/`, `v/`, `x/` are throwaway | delete `t/ v/ x/` freely |
| `reports/` | **not reports**: SpectraForge validation *scripts* (`spectraforge_validation_report.py`, `fpbase_validation.py`, `classification_experiment.py`, `cae_vs_spectral_ae.py`) and their artifacts (`fpbase_spectra/`, `model_output/`) | scripts are the runnable record of the SpectraForge investigation |
| `publications/generalization/reports/` | the actual text reports for the generalisation work: `synthetic_recovery.txt`, `eeg_bci_loso.txt`, `pamap2_slice.txt`, `pamap2_loso.txt`, `pamap2_baseline_diag.txt`, `pamap2_richfeat_diag.txt`, `pamap2_pca_diag.txt` | yes |
| `publications/codassca2026/poster/data/` | screen logs (`indian_pines_screen.log`, `opportunity_screen.log`), poster data dumps | yes |
| `htmlcov/`, `.coverage` | local coverage artifacts | scratch |

## Publications

| Folder | Venue and status | Built from |
|---|---|---|
| `publications/tpami/` | IEEE TPAMI, Lichens; submitted, **under revision** (a Phase 0 audit found five paper-versus-data discrepancies; SOTA baselines re-run, method still leads by 1 to 7 points); `paper/`, `revision/` with `MASTER_PLAN.md`, `DROP_DATA_REPORT.md`, `reports/build_reports.py` | the canonical Lichens pipeline |
| `publications/iasim_poster/` | IASIM 2026 (Collagen / Pepsin), poster v2 blocks, design, wireframe, results table | `pepsin_*`, `poster_*` archived scripts |
| `publications/master_thesis/` | master's thesis (LaTeX, `Makefile`) and defence deck (`build_defense_pptx.py`) | Lichens + Collagen figures |
| `publications/commsai_computing/` | Nature-style manuscript on Lichens + Collagen (`overleaf/`, `submission/`); note Collagen has 149 usable bands after cutoff, not 158 | Lichens pipeline + `collagen_*` |
| `publications/codassca2026/` | CODASSCA 2026 short paper on generalisation, **accepted as a poster**; `build_codassca_shortpaper.py`, architecture figure, `poster/` (A0 PowerPoint built with python-pptx, `fig_pipeline.py`) | `channel_select` experiments, screens |
| `publications/generalization/` | the longer generalisation manuscript: `paper/main.tex`, `EXPLAINER.md`, `RESEARCH_LOG.md` (the honest running log of what worked and what did not), `MASTER_PLAN.md`, `CHANGELOG.md`, `figures/`, `reports/`, `baselines/` | `experiments/pamap2`, `eeg`, `synthetic` |
| WHISPERS 2026 SpectraForge papers | on the `whispers2026/papers` branch in a separate worktree, not on `main` | `spectraforge` |

Everything under `publications/` is tracked in git except the large binaries blocked by `.gitignore` (new `.docx`, `.pptx`, `.zip`, `.npz`; existing tracked ones stay).

## Adding an experiment

1. Put a maintained driver at the top level of `experiments/` (or in the matching domain subfolder); put one-off analyses in `_archive/2026_paper_runs/` with a header saying which submission they served.
2. Anchor paths to the repository root with `PROJECT_ROOT = Path(__file__).resolve().parent.parent`; never lowercase `data/`.
3. Write to `results/<Descriptive_Name>/` and `model_output/<sample_name>/`; copy only what a paper needs into `publications/<venue>/`.
4. Set `Config.model_path` explicitly, and record the config snapshot (`WavelengthResult.config_snapshot`, or `ResultsManager.save_run_metadata`).
5. Always report the random baseline next to the method, and for SpectraForge also `mask_coverage`.
6. Update `experiments/README.md` and, if a dataset is involved, `../DATA_GUIDE.md`.

## Known issues

* Four PAMAP2 scripts write to a non-existent `generalization/reports/` at the repository root (see guide 05).
* `export_tiffs.py` uses lowercase `data/` paths and fails on Linux.
* The `channel_select` scripts never use CUDA; the HSI master run does.
* Figure scripts in `pamap2/`, `eeg/`, `synthetic/` hard-code numbers transcribed from the text reports.
* `experiments/README.md` still describes only the top-level drivers and `pamap2/`; `eeg/`, `synthetic/` and the screens are newer and untracked.
* `results/` contains many timestamped scratch runs that nothing references; they can be deleted, but check `publications/tpami/revision/` first, which cites some by name.
