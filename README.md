# spectral-select

Autoencoder-based wavelength selection for 4D hyperspectral imaging via latent space perturbation analysis.

Data is available at [Zenodo] https://zenodo.org/records/18640119 DOI - https://doi.org/10.5281/zenodo.18640119

<!-- Paper link: TODO -->

## Overview

`spectral-select` identifies the most informative wavelength combinations in multi-excitation hyperspectral datasets by:

1. Training a convolutional autoencoder on the hyperspectral data
2. Analyzing the latent space to identify important dimensions
3. Perturbing latent dimensions and measuring reconstruction sensitivity
4. Ranking and selecting wavelength bands based on influence scores

This enables dimensionality reduction while preserving the most discriminative spectral information for downstream clustering and classification.

## Installation

```bash
pip install -e "."
```

For development (testing, notebooks):

```bash
pip install -e ".[dev]"
```

Extras: `[gui]` (PyQt6 for the two GUIs), `[im3]` (PyImageJ for reading raw Nuance `.im3` cubes; needs a JDK
and Maven), `[all]` (everything). Requires Python >= 3.11. Full platform-by-platform instructions, including the
Java stack, GPU setup and where to get the data: [`docs/onboarding/SETUP.md`](docs/onboarding/SETUP.md).

## Quick Start

```python
from spectral_select import Analyzer, Config, SpectraData

# Configure the analysis
config = Config(
    sample_name="my_sample",
    n_bands_to_select=30,
    model_path="models/autoencoder.pth",
    output_dir="results/",
)

# Load preprocessed hyperspectral data
data = SpectraData.from_raw_dict(your_data_dict)

# Run wavelength selection
analyzer = Analyzer(config)
results = analyzer.fit(data)

# Access selected wavelengths
print(results.selected_wavelengths)
```

## Run the GUI (no code)

```bash
pip install -e ".[gui]"
spectral-select-gui          # or: python -m mehsi_preprocessor
```

The wizard walks you through the whole workflow without writing code: **load → metadata →
normalize → spatial/spectral crop → draw classes → ROI → export → train autoencoder →
select bands**. Step 9 trains the autoencoder (or loads a saved `.pth`) with a live progress
bar; Step 10 runs Perturbation-Based AE selection and exports the chosen bands as CSV / JSON /
TIFF.

## SpectraForge — synthetic ME-HSI data

Generate chemically-grounded synthetic multi-excitation hyperspectral datasets (with perfect
ground truth) to validate the band-selection methods. Define fluorophores → mix them into
materials → paint them onto a scene → render a `SpectraData` cube plus per-fluorophore
concentration maps and "which bands carry signal" ground truth.

```bash
spectraforge-gui                     # the Forge: paint materials on a canvas, render datasets
spectraforge-demo -o my_dataset      # or headless: writes spectra_unmasked.pkl + groundtruth.npz/json
```

The Forge GUI is a workbench: define fluorophores and view their spectra (left), compose
materials (brushes), paint them onto layered canvases (center), manage layers (right), then
configure excitations and **Render** + **Export** (bottom) — output loads in `spectral-select-gui`.

```python
from spectraforge import Fluorophore, Material, Scene, AcquisitionConfig, ArtifactConfig, render, load_builtin_library
lib = load_builtin_library()                         # collagen, NADH, FAD, EGFP, fluorescein, ...
scene = Scene(64, 64)
scene.paint_rect(Material("tissue", {"collagen": 1.0, "NADH": 0.3}), 5, 50, 5, 50)
acq = AcquisitionConfig(excitations=[340, 450, 488], em_min=360, em_max=700, em_step=5)
spectra, gt = render(scene, lib, acq, artifacts=ArtifactConfig(rayleigh_strength=0.15, photon_scale=400), seed=42)
spectra.to_pickle("synthetic.pkl")                   # loads in the Analyzer / GUI
```

The output drops straight into `spectral-select-gui` (Step 1) or `Analyzer`, and
`GroundTruth.informative_bands()` tells you which wavelengths *should* be selected.

## Repository Structure

Uses a `src/` layout (install with `pip install -e .`, then `import spectral_select`).

```
src/
  spectral_select/         Core HSI wavelength-selection library + interactive viewer
    models/                Autoencoder architecture, dataset, training
    analyzer.py            Main analysis engine    config.py  types.py  loader.py
    validation.py          results.py  visualizer.py  viewer.py  widgets.py
  channel_select/          Domain-agnostic selection engine (e.g. PAMAP2 wearables)
  mehsi_preprocessor/      PyQt6 preprocessing GUI — launch: `python -m mehsi_preprocessor`
experiments/               Reusable pipeline drivers (see experiments/README.md)
  pamap2/                  Generalization-domain experiments (channel_select)
  _archive/2026_paper_runs/  One-off paper/figure scripts, kept for provenance
publications/              Manuscripts & posters, one subdir per venue
  tpami/  commsai_computing/  codassca2026/  master_thesis/  iasim_poster/  generalization/
examples/                  Tutorial notebooks
tests/                     Test suite (pytest)
docs/                      Documentation — USER_GUIDE, DATA.md (dataset contract),
                           ARCHIVE_MANIFEST.md, design specs & plans under superpowers/
Data/                      (gitignored) datasets — see docs/DATA.md
archive/                   (gitignored) legacy + archived material — see docs/ARCHIVE_MANIFEST.md
```

A shared `src/selection_core/` (unifying the perturbation algorithm behind both
`spectral_select` and `channel_select`) is planned — see
`docs/superpowers/specs/2026-06-15-repo-cleanup-reorg-design.md`.

## Reproducing Paper Results

See [`experiments/README.md`](experiments/README.md) for instructions to reproduce all paper figures and tables.

## Documentation

**New team members: start at [`docs/onboarding/`](docs/onboarding/README.md)** (setup, data guide, the
`examples/03_hsi_data_walkthrough.ipynb` notebook, and a technical guide per component). See the [`docs/`](docs/)
directory and [`examples/`](examples/) notebooks for the rest.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{meloyan2025spectralselect,
  title={Autoencoder-Based Wavelength Selection for 4D Hyperspectral Imaging},
  author={Meloyan, Narek},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
