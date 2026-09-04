# Spectral-Select Documentation

Welcome to the Spectral-Select documentation. This guide will help you understand and use the hyperspectral wavelength selection toolkit.

> **New to the team?** Start with [`onboarding/README.md`](onboarding/README.md): environment setup for macOS / Windows / Linux (including the Java stack for `.im3` files), the data guide, the hands-on walkthrough notebook, and one technical guide per component.

## Quick Links

| I want to... | Read this |
|--------------|-----------|
| Onboard as a new team member | [Onboarding](onboarding/README.md), [Setup](onboarding/SETUP.md), [Data guide](onboarding/DATA_GUIDE.md), [Component guides](onboarding/components/README.md) |
| Get started quickly | [Quick Reference](QUICKREF.md) |
| Install the software | [Installation Guide](INSTALLATION.md) |
| Understand the concepts | [User Guide](USER_GUIDE.md) |
| Process my data | [Data Processing Guide](guides/DATA_PROCESSING.md) |
| Run wavelength selection | [Wavelength Selection Guide](guides/WAVELENGTH_SELECTION.md) |
| Validate my results | [Validation Guide](guides/VALIDATION.md) |
| Configure parameters | [Configuration Reference](CONFIGURATION.md) |
| Fix a problem | [Troubleshooting](TROUBLESHOOTING.md) |

## Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── QUICKREF.md            # Quick reference card for common tasks
├── USER_GUIDE.md          # Comprehensive user guide
├── INSTALLATION.md        # Installation and setup instructions
├── CONFIGURATION.md       # All configuration parameters
├── TROUBLESHOOTING.md     # Common problems and solutions
├── onboarding/            # New-member path: README (start here), SETUP, DATA_GUIDE, components/*.md
├── spectraforge/          # SpectraForge system overview, investigation findings, architecture plan, runbook
└── guides/
    ├── DATA_PROCESSING.md      # How to prepare your data
    ├── WAVELENGTH_SELECTION.md # How to run analysis
    └── VALIDATION.md           # How to validate results
```

## Getting Started

### New Users

1. **Read the [User Guide](USER_GUIDE.md)** to understand what the software does
2. **Follow [Installation](INSTALLATION.md)** to set up your environment
3. **Process your data** using [Data Processing Guide](guides/DATA_PROCESSING.md)
4. **Run analysis** following [Wavelength Selection Guide](guides/WAVELENGTH_SELECTION.md)

### Returning Users

- **[Quick Reference](QUICKREF.md)**: Copy-paste code snippets
- **[Configuration](CONFIGURATION.md)**: Look up parameter options
- **[Troubleshooting](TROUBLESHOOTING.md)**: Fix common issues

## Interactive Tutorials

Jupyter notebooks provide hands-on tutorials:

| Notebook | Description |
|----------|-------------|
| `examples/03_hsi_data_walkthrough.ipynb` | Data layer end to end: open `.im3` cubes and 4D pickles, store, view, spectra, statistics, preprocessing |
| `examples/01_quickstart.ipynb` | Basic wavelength selection workflow |
| `examples/02_validation.ipynb` | Validating results against ground truth |

To run:
```bash
jupyter lab examples/
```

## Key Concepts

### What is Wavelength Selection?

Hyperspectral imaging captures data at many wavelengths, but not all wavelengths are equally useful. Wavelength selection identifies the most informative bands for distinguishing materials in your sample.

### How Does It Work?

1. **Train an autoencoder** on your hyperspectral data
2. **Perturb the latent space** to measure each wavelength's influence
3. **Rank wavelengths** by their importance
4. **Select top bands** for downstream analysis

### Why Use This?

- **Reduce data dimensionality** from hundreds of bands to tens
- **Focus on informative features** rather than noise
- **Enable faster clustering** and classification
- **Identify physically meaningful bands** for interpretation

## Workflow Overview

```
Raw Data (.im3, TIFF)
        │
        ▼
┌─────────────────┐
│ Data Processing │ ← guides/DATA_PROCESSING.md
│  - Load files   │
│  - Apply cutoffs│
│  - Normalize    │
└────────┬────────┘
         │
         ▼
   Processed Data (.pkl)
         │
         ▼
┌─────────────────────┐
│ Wavelength Selection│ ← guides/WAVELENGTH_SELECTION.md
│  - Configure        │
│  - Train autoencoder│
│  - Rank bands       │
└─────────┬───────────┘
          │
          ▼
    Selected Wavelengths
          │
          ▼
┌─────────────────┐
│   Clustering    │
│  - K-means, etc │
└────────┬────────┘
         │
         ▼
    Cluster Labels
         │
         ▼
┌─────────────────┐
│   Validation    │ ← guides/VALIDATION.md
│  - Compare to GT│
│  - Compute ARI  │
└─────────────────┘
```

## Configuration Quick Reference

### Minimal Configuration

```python
from spectral_select import Config

config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
)
```

### Standard Configuration

```python
config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
    training_epochs=100,
    device="cpu",  # or "cuda" for GPU
)
```

### Publication Quality

```python
config = Config(
    sample_name="MySample",
    n_bands_to_select=30,
    training_epochs=500,
    use_diversity_constraint=True,
    diversity_method="mmr",
    lambda_diversity=0.3,
)
```

See [CONFIGURATION.md](CONFIGURATION.md) for all options.

## Support

### Getting Help

1. Check [Troubleshooting](TROUBLESHOOTING.md) for common issues
2. Review the [User Guide](USER_GUIDE.md) FAQ section
3. Open an issue on GitHub with:
   - Operating system and Python version
   - Full error message
   - Minimal code to reproduce

### Reporting Bugs

Please include:
- Steps to reproduce the issue
- Expected vs actual behavior
- System information (`python --version`, `pip show spectral-select`)

## Version History

See the main [README.md](../README.md) for version information and changelog.
