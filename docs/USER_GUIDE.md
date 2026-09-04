# Spectral-Select User Guide

## What is Spectral-Select?

Spectral-Select is a software tool for analyzing **hyperspectral fluorescence microscopy data**. If you have samples (like lichens, biological tissues, or materials) imaged with multiple excitation wavelengths and many emission bands, this tool helps you:

1. **Find the most important wavelengths** - Identify which combinations of excitation and emission wavelengths carry the most useful information
2. **Reduce data complexity** - Go from hundreds of wavelength channels to just 10-30 that capture the essential variations
3. **Improve clustering** - Better separate different materials/regions in your images
4. **Save time** - Automate the wavelength selection process that would otherwise require expert manual analysis

## Who Should Use This?

- Researchers working with **multispectral or hyperspectral fluorescence imaging**
- Scientists analyzing **lichen, plant, or biological samples** with autofluorescence
- Anyone who needs to **reduce dimensionality** of spectral imaging data while preserving important features

## Key Concepts

### What is Hyperspectral Data?

Traditional cameras capture 3 color channels (Red, Green, Blue). Hyperspectral imaging captures dozens or hundreds of wavelength channels. In fluorescence imaging:

- **Excitation wavelength**: The wavelength of light used to illuminate the sample (e.g., 365nm UV)
- **Emission wavelength**: The wavelengths of fluorescence light emitted by the sample (e.g., 400-700nm range)
- **Data cube**: A 3D array where each pixel has a full spectrum instead of just RGB values

### What is Wavelength Selection?

Not all wavelengths are equally useful. Some wavelengths:
- Show strong contrast between different materials
- Contain unique information not found in other channels
- Are redundant with nearby wavelengths

Wavelength selection finds the **most informative wavelengths** so you can:
- Focus your analysis on what matters
- Reduce storage and computation requirements
- Improve machine learning results by removing redundant features

### How Does Spectral-Select Work?

The software uses a **deep learning approach**:

1. **Autoencoder Training**: A neural network learns to compress and reconstruct your spectral data
2. **Latent Space Analysis**: The compressed representation reveals what patterns the network found important
3. **Perturbation Testing**: The software tests which wavelengths most affect the learned representation
4. **Ranking**: Wavelengths are ranked by their "influence score" - how important they are to the data structure
5. **Selection**: The top-ranked wavelengths are selected for your downstream analysis

## Getting Started

### Prerequisites

Before you begin, you'll need:

1. **Python 3.11 or higher** installed on your computer
2. **Your hyperspectral data** in one of the supported formats
3. Basic familiarity with running Python scripts (or Jupyter notebooks)

### Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

Quick version:
```bash
pip install -e ".[dev]"
```

### Your First Analysis

The easiest way to get started is with the example notebooks:

1. **Open the notebooks folder**: `examples/`
2. **Start with data loading**: `00_data_loading.ipynb` - Learn how to load your data
3. **Run wavelength selection**: `01_quickstart.ipynb` - Perform your first analysis
4. **Validate results**: `02_validation.ipynb` - Compare to ground truth (if available)

## Typical Workflow

```
Raw Data (.im3)  -->  Process Data  -->  Load SpectraData
                      (cutoffs,
                       normalize)
                                               |
                                               v
Selected Bands   <--  Run Analysis  <--  Configure
(10-30 bands)         (Analyzer)         (Config)
      |
      v
Downstream Analysis --> Results (clusters, maps)
```

## Step-by-Step Tutorials

### Step 1: Prepare Your Data

Your raw hyperspectral data needs to be processed before analysis. The processing includes:

- **Spectral cutoffs**: Remove artifacts from Rayleigh scattering and second-order diffraction
- **Normalization**: Correct for different exposure times and laser powers at each excitation

See [DATA_PROCESSING.md](guides/DATA_PROCESSING.md) for detailed instructions.

### Step 2: Configure Your Analysis

Create a configuration that specifies:

- How many wavelengths to select
- Which methods to use for analysis
- Where to save results

See [CONFIGURATION.md](CONFIGURATION.md) for all options.

**Simple example:**
```python
from spectral_select import Config

config = Config(
    sample_name="MySample",
    n_bands_to_select=20,    # Select 20 wavelength combinations
    training_epochs=100,      # Train the autoencoder for 100 epochs
    device="cpu",             # Use CPU (or "cuda" for GPU)
)
```

### Step 3: Run the Analysis

```python
from spectral_select import Analyzer, SpectraData

# Load your processed data
data = SpectraData.from_pickle("path/to/your/processed_data.pkl")

# Create analyzer and run
analyzer = Analyzer(config)
analyzer.fit(data)

# Get the selected wavelengths
wavelengths = analyzer.get_wavelengths()

# See what was selected
for band in wavelengths[:10]:
    print(f"Rank {band.rank}: Excitation={band.excitation_nm}nm, Emission={band.emission_nm}nm")
```

### Step 4: Review and Save Results

```python
# Save all results
analyzer.save_results("results/my_analysis/")

# This creates:
# - wavelength_result.json  (machine-readable results)
# - selected_bands.txt      (human-readable summary)
# - analysis_config.json    (your configuration for reproducibility)
```

### Step 5: Validate (Optional)

If you have ground truth labels (manually annotated regions), you can measure how well the selected wavelengths perform for clustering:

```python
from spectral_select import Validator, load_ground_truth_from_png

# Load ground truth from annotated image
ground_truth = load_ground_truth_from_png("ground_truth.png")

# After running clustering on selected wavelengths...
validator = Validator()
validator.fit(cluster_labels, ground_truth)

print(f"Adjusted Rand Index: {validator.score():.3f}")
```

## Understanding the Results

### Selected Wavelengths File

The `selected_bands.txt` file shows your results in human-readable format:

```
Wavelength Analysis Results: MySample
Analysis Date: 2024-01-15 14:30:22
Method: activation + percentile
Total bands selected: 20 out of 187
Compression ratio: 9.35x

Rank  Excitation(nm)  Emission(nm)    Influence
------------------------------------------------------------
1     310.0           420.0           0.362400
2     310.0           430.0           0.143700
3     340.0           420.0           0.066600
...
```

**How to interpret:**
- **Rank**: Higher rank (lower number) = more important
- **Excitation**: The laser wavelength used
- **Emission**: The detected fluorescence wavelength
- **Influence**: How important this wavelength is (higher = more informative)

### Visualizations

The software generates several plots:

1. **Influence Heatmap**: Shows which excitation/emission combinations are most important
2. **Wavelength Scatter**: Displays selected wavelengths in 2D space
3. **Influence Ranking**: Bar chart showing how importance decreases with rank

### What Makes a Good Result?

- **High compression ratio**: Going from 200+ to 20-30 wavelengths while maintaining information
- **Diverse selections**: Wavelengths spread across different excitations (not all from one laser)
- **High validation scores**: If you have ground truth, ARI > 0.5 indicates good clustering agreement

## Common Use Cases

### Case 1: Lichen Species Identification

```python
config = Config(
    sample_name="Lichens",
    n_bands_to_select=25,
    use_diversity_constraint=True,  # Ensure wavelengths from different excitations
    diversity_method="mmr",
    lambda_diversity=0.3,
)
```

### Case 2: Fast Exploration (Quick Results)

```python
config = Config(
    sample_name="QuickTest",
    n_bands_to_select=10,
    training_epochs=20,     # Fewer epochs = faster
    n_baseline_patches=20,  # Fewer patches = faster
)
```

### Case 3: High-Quality Publication Results

```python
config = Config(
    sample_name="Publication",
    n_bands_to_select=30,
    training_epochs=500,          # More epochs = better model
    n_baseline_patches=100,       # More patches = more robust
    save_tiff_layers=True,        # Export TIFF images
    save_visualizations=True,     # Generate all plots
)
```

## Frequently Asked Questions

### Q: How many wavelengths should I select?

**A:** This depends on your application:
- For quick visualization: 5-10 wavelengths
- For clustering: 15-30 wavelengths
- For detailed analysis: 30-50 wavelengths

Start with 20-30 and adjust based on your results.

### Q: How long does analysis take?

**A:** Depends on:
- Data size (larger images = longer)
- Number of training epochs
- CPU vs GPU (GPU is 10-50x faster)

Typical times:
- Small dataset (256x348 pixels), 20 epochs, CPU: ~2-5 minutes
- Large dataset (1024x1024 pixels), 100 epochs, GPU: ~5-10 minutes

### Q: What if I don't have ground truth?

**A:** That's fine! The wavelength selection works without ground truth. Ground truth is only needed if you want to validate clustering results afterward.

### Q: Can I use this with my own clustering algorithm?

**A:** Yes! The selected wavelengths can be used with any clustering method:
- K-means
- DBSCAN
- Hierarchical clustering
- Gaussian Mixture Models
- Your custom algorithm

### Q: My data format isn't supported. What do I do?

**A:** You need to convert your data to the expected format. See [DATA_PROCESSING.md](guides/DATA_PROCESSING.md) for how to create `SpectraData` objects from custom formats.

## Getting Help

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems
- Review the example notebooks in `examples/`
- Open an issue on GitHub for bugs or feature requests

## Next Steps

1. Read [INSTALLATION.md](INSTALLATION.md) if you haven't installed yet
2. Work through [DATA_PROCESSING.md](guides/DATA_PROCESSING.md) to prepare your data
3. Follow [WAVELENGTH_SELECTION.md](guides/WAVELENGTH_SELECTION.md) for detailed analysis guide
4. Review [CONFIGURATION.md](CONFIGURATION.md) for all available options
