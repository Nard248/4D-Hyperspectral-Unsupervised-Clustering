# Onboarding: start here

Welcome to `spectral-select`, the lab's code base for multi-excitation hyperspectral imaging (ME-HSI): reading the instrument's data, preprocessing and annotating it, selecting informative wavelength bands with an unsupervised deep-learning method, validating the selections, and generating synthetic data to test all of that. This folder is the map. Read it in the order below and you will be productive within a day.

## Reading order

| Step | Document | What you get | Time |
|---|---|---|---|
| 1 | this page | the big picture, the repository map, where the project stands | 15 min |
| 2 | [SETUP.md](SETUP.md) | a working environment on macOS, Windows or Linux, including the Java stack for `.im3` files, GUIs, GPU, data | 30 min + downloads |
| 3 | [DATA_GUIDE.md](DATA_GUIDE.md) | how datasets are organised, the metadata files, the pickle schemas, the annotation formats, the dataset catalogue | 20 min |
| 4 | [`examples/03_hsi_data_walkthrough.ipynb`](../../examples/03_hsi_data_walkthrough.ipynb) | hands-on: open 3D cubes four ways, open the 4D dataset, store it, view slices and false colour, plot spectra and statistics, run the preprocessing functions, a tiny selection run | 45 min |
| 5 | [`examples/01_quickstart.ipynb`](../../examples/01_quickstart.ipynb), [`02_validation.ipynb`](../../examples/02_validation.ipynb) | a full selection run and ground-truth validation | 30 min |
| 6 | [components/](components/README.md) | one technical guide per component: SpectraForge, the preprocessing wizard, `spectral_select`, `selection_core`, `channel_select`, the experiments | as needed |

Older reference material still applies for the core library: [`docs/USER_GUIDE.md`](../USER_GUIDE.md), [`docs/CONFIGURATION.md`](../CONFIGURATION.md), [`docs/guides/`](../guides/) and the GUI user guide in [`docs/gui_user_guide/`](../gui_user_guide/). Where those disagree with the component guides, the component guides are newer and were checked against the code in September 2026.

## What the project does, in one paragraph

A tunable light source excites a sample at 6 to 8 ultraviolet-to-blue wavelengths; at each excitation a Nuance multispectral camera records 31 emission bands from 420 to 720 nm. The result is a 4D dataset (excitation x emission x height x width) with roughly 200 usable bands per pixel. Most of those bands are redundant. `spectral_select` trains a masked convolutional autoencoder on the cube, perturbs the latent space and scores every band by how much the reconstruction reacts; the top-ranked bands are the ones a cheaper instrument, or a downstream classifier, should keep. The method is unsupervised (no labels during selection) and, on our biomedical datasets, its selections match or beat supervised selection when scored by a KNN classifier on labelled regions. The same perturbation engine has been generalised to other multi-channel data (wearable sensors, EEG) in `channel_select`, and a synthetic generator, SpectraForge, exists to test selectors against known ground truth.

## Repository map

```
spectral-select/
├── src/
│   ├── spectral_select/      the HSI library: SpectraData types, .im3 DataLoader wrapper, Analyzer (train + select),
│   │                         autoencoder + training, Visualizer, Validator, Tk viewer, Jupyter ROI widget
│   ├── selection_core/       the shared perturbation-influence algorithm (pure functions on torch tensors)
│   ├── channel_select/       domain-agnostic selection: grouped-channel datasets, temporal / spatial bottleneck AEs,
│   │                         adapters (PAMAP2, EEG), the synthetic factor-recovery benchmark
│   ├── mehsi_preprocessor/   the "Image Reader": .im3 loader (PyImageJ), pure preprocessing functions, 10-step PyQt6 wizard
│   └── spectraforge/         synthetic ME-HSI generator: fluorophores -> materials -> painted scene -> rendered cube + ground truth,
│                             validation harness, PyQt6 painter GUI
├── examples/                 tutorial notebooks (00 data loading [stale], 01 quickstart, 02 validation, 03 walkthrough)
├── experiments/              reproducible drivers: the Lichens paper pipeline, pamap2/, eeg/, synthetic/, benchmark screens
│   └── _archive/2026_paper_runs/   one-off scripts for Collagen, Drop Data, Pepsin, posters (not maintained)
├── tests/                    pytest suite (~490 tests; GUI tests run offscreen)
├── docs/                     this documentation, older guides, design specs and plans under superpowers/
├── publications/             manuscripts, posters, reports per venue (tpami, commsai_computing, codassca2026, generalization, ...)
├── reports/                  SpectraForge validation scripts and their artifacts (despite the name)
├── Data/                     datasets (git-ignored)        results/, model_output/   run outputs (git-ignored)
└── pyproject.toml            the package: extras [dev], [gui], [im3], [all]; console scripts below
```

Console scripts installed with the package:

| Command | Launches |
|---|---|
| `spectral-select-gui` (or `python -m mehsi_preprocessor`) | the 10-step preprocessing and selection wizard |
| `spectraforge-gui` | the SpectraForge painter ("the Forge") |
| `spectraforge-demo -o DIR` | writes a synthetic dataset plus ground truth |
| `python -c "from spectral_select import launch_viewer; launch_viewer()"` | the desktop ME-HSI viewer |

## How the pieces connect

```
 instrument            Image Reader / preprocessing                selection                     validation
 ----------            ----------------------------                ---------                     ----------
 .im3 cubes  --PyImageJ-->  HyperspectralDataLoader  --> SpectraData --> Analyzer.prepare()  --> WavelengthResult
 metadata.xlsx              (cutoffs, normalisation,     (.pkl)          (train masked CAE)       (ranked bands)
 average_power.xlsx          crop, masks, ROI)                            Analyzer.select()             |
                             spectral-select-gui                          (selection_core:              v
                                                                          perturb latent, score)   Validator / KNN
                                                                                                    on labelled ROIs
 SpectraForge -----------------------------------------> SpectraData + GroundTruth  --> validate_selection
 (painted fluorophores, forward model)                    (synthetic)                    (peak recovery, chance baseline)

 channel_select: GroupedChannelDataset -> Temporal/Spatial bottleneck AE -> run_selection (same selection_core) -> KNN / CSP-LDA
```

## Where the project stands (September 2026)

* **Published / submitted.** TPAMI submission on the Lichens dataset (under revision: a data audit found five paper-versus-data discrepancies that are being fixed; SOTA baselines were re-run and the method still leads by 1 to 7 points); IASIM 2026 abstract (Collagen/Pepsin); a Nature Communications-style manuscript on Lichens + Collagen (`publications/commsai_computing/`); CODASSCA 2026 short paper on generalisation beyond HSI (accepted as a poster; the poster is in `publications/codassca2026/poster/`); WHISPERS 2026 papers on SpectraForge (on a separate branch); a master's thesis and defence deck.
* **Generalisation work (`channel_select`).** The perturbation method transfers to heterogeneous-channel data: it recovers planted factors on the synthetic benchmark (coverage 1.0 at K = 8, beating random and variance ranking) and matches supervised mutual-information selection on EEG motor imagery. It does **not** beat random selection on correlated, redundant data (PAMAP2 wearables, Indian Pines, Opportunity); those are saturated benchmarks where any K channels work. Treat "when does channel selection matter at all" as an open question the screening scripts probe.
* **SpectraForge.** The engine and GUI are complete and tested (~70 tests). The research claim that the selector recovers realistic spectra was **withdrawn** in July 2026 after adversarial review: the broad ground-truth mask was saturated and a random selector matched the autoencoder. A tighter peak-recovery metric and a chance baseline are now built in; the verdict is "inconclusive", and the next planned step (a discriminability-grounded ground truth and a per-pixel spectral autoencoder ladder) is designed but not built. Read `components/01-spectraforge.md` before quoting any SpectraForge number.
* **Drop Data.** The first post-TPAMI dataset. Early results were wrong because exposure brackets and lamp power were ignored; the radiometric rerun (`Drop Data Radiometric/rad_merged`) fixed that and the autoencoder selection now beats the baselines (KNN accuracy 0.984). The raw loader still cannot read the bracketed filenames; scripts read the cached `.npy` cubes.
* **Engine unification.** `selection_core` is the single implementation of the perturbation algorithm used by both `spectral_select.Analyzer` and `channel_select.run_selection`; equivalence is pinned by `tests/test_analyzer_core_equivalence.py`. The diversity step (MMR, minimum distance) is deliberately implemented per package.
* **Repository hygiene.** The tree was reorganised in June 2026 (`src/` layout, `publications/` per venue, history rewritten to 120 MB). Several of the newest pieces (EEG adapter, bottleneck autoencoders, synthetic benchmark, screening scripts, the CODASSCA poster) are **untracked in git** at the time of writing; ask before relying on them being on `main`.

Useful people: Narek Meloyan is the sole author of the code and the person to ask about data, results and history.

## Working conventions

* Python 3.11, `src/` layout, editable install. Import packages, never add `src` to `sys.path`.
* Tests: `QT_QPA_PLATFORM=offscreen pytest -q --no-cov` (about one minute). Slow and notebook tests are marked; GUI tests skip without PyQt6. CI runs the suite on Python 3.11 and 3.12 (`.github/workflows/test.yml`).
* Run experiment scripts from the repository root; several use relative `Data/...` paths.
* Long runs go to `results/<name>/` and `model_output/<sample>/`; both are git-ignored. Anything worth keeping is copied into `publications/<venue>/`.
* Design specs and implementation plans for every larger change live in `docs/superpowers/specs/` and `docs/superpowers/plans/`; read the spec before touching a component.
* Keep `docs/DATA.md` and the catalogue in `DATA_GUIDE.md` in sync when datasets change.

## Glossary

| Term | Meaning |
|---|---|
| ME-HSI | multi-excitation hyperspectral imaging: several excitation wavelengths, each with a full emission spectrum per pixel |
| cube | one excitation's `(H, W, bands)` array |
| band | one emission wavelength plane of a cube; "wavelength band" means the (excitation, emission) pair |
| excitation / emission | illumination wavelength / detected wavelength |
| EEM | excitation-emission matrix |
| Rayleigh cutoff, second-order cutoff | removal of bands near the excitation line and near twice the excitation wavelength |
| mask, class mask, ROI | analysed-pixel mask; per-pixel class ids; rectangular labelled regions |
| CAE, AE | (convolutional) autoencoder |
| perturbation influence | the score of a band: how much the reconstruction changes when latent dimensions are perturbed |
| MMR | maximal marginal relevance, the diversity-aware selection step |
| KNN protocol | the evaluation: train a k-nearest-neighbour classifier on labelled ROI pixels using only the selected bands |
| SpectraForge | the synthetic ME-HSI generator; "the Forge" is its GUI |
| MEHSI preprocessor | the `mehsi_preprocessor` package: the `.im3` reader plus the wizard |
| PAMAP2, BCI IV-2a, Indian Pines, Opportunity, TEP | external benchmark datasets used by `channel_select` experiments |
