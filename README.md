# Hyperspectral Data Analysis Framework

This repository provides a comprehensive framework for the analysis of hyperspectral data, including data processing, normalization, autoencoder-based dimensionality reduction, and clustering analysis. It's designed to handle 4D hyperspectral datasets (spatial x, y + emission wavelength + excitation wavelength) with efficient processing and visualization tools.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [Examples](#examples)

## Installation

### Requirements

Clone the repository and install the required packages:

```bash
git clone https://github.com/Nard248/4D-Hyperspectral-Unsupervised-Clustering.git
cd hyperspectral-analysis
pip install -r requirements.txt
```

### Special Dependencies for Fiji/ImageJ Integration

To enable reading .im3 hyperspectral files, you need to install:

1. **Apache Maven 3.9.9**:
   - Download from [Apache Maven website](https://maven.apache.org/download.cgi)
   - Install to a location like `C:\Program Files\apache-maven-3.9.9`
   - Add to your PATH environment variable: `C:\Program Files\apache-maven-3.9.9\bin`
   - Create MAVEN_HOME environment variable: `C:\Program Files\apache-maven-3.9.9`

2. **Java JDK**:
   - Download from [Oracle JDK](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://jdk.java.net/)
   - Install to a location like `C:\Program Files\Java\jdk-24`
   - Add to your PATH environment variable: `C:\Program Files\Java\jdk-24\bin`

These dependencies are required for the ImageJ/Fiji integration that allows reading specialized hyperspectral file formats.

**Torch**: Additionally use command 
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
to install PyTorch with GPU support as sometimes installing requirements does not ensure the download of the Torch automatically. 

## Project Structure

```
├── data/
│   ├── processed/       # Processed data files
│   └── raw/             # Raw data files
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks for examples
├── results/             # Results from analysis
│   ├── clustering/      # Clustering results
│   ├── evaluation/      # Model evaluation results
│   ├── model/           # Saved models
│   └── visualizations/  # Generated visualizations
└── scripts/
    ├── data_processing/ # Data processing utilities
    ├── models/          # ML models and training
    └── utils/           # Utility functions
```

## Core Components

### Data Processing (`scripts/data_processing/`)

- **`hyperspectral_loader.py`**: Core class for loading hyperspectral data from various formats
- **`hyperspectral_processor.py`**: Preprocessing pipeline including normalization and cutoff filtering
- **`hyperspectral_utils.py`**: Utility functions for data manipulation

#### Key features:
- Dual cutoff for Rayleigh and second-order scattering artifacts
- Exposure time normalization
- Laser power normalization
- Data structure conversion for analysis

### Models (`scripts/models/`)

- **`autoencoder.py`**: Convolutional autoencoder for dimensionality reduction
- **`dataset.py`**: Custom PyTorch dataset for hyperspectral data with masking support
- **`training.py`**: Efficient training with spatial chunking and masked loss
- **`clustering.py`**: Algorithms for pixel-wise clustering in 3D and 4D data
- **`visualization.py`**: Comprehensive visualization utilities
- **`workflow.py`**: End-to-end analysis pipeline

#### Key features:
- Support for 4D data (spatial dimensions + emission wavelength + excitation wavelength)
- Masked processing that handles missing or invalid data
- Memory-efficient spatial chunking for processing large data

### Utilities (`scripts/utils/`)

- **`masking_tool.py`**: Interactive tool for creating masks
- **`masking_utils.py`**: Functions for applying masks to data
- **`process_trq_files.py`**: Processing tool for .trq files (TLS scans)
- **`viewer.py`**: Interactive visualization tool for hyperspectral data

## Usage Guide

### 1. Raw Data Processing

Process raw hyperspectral files (.im3, .trq) and normalize them:

```python
from scripts.utils.process_trq_files import process_and_save
from scripts.data_processing.hyperspectral_processor import HyperspectralProcessor

# Process power files for normalization
trq_folder_path = "data/raw/Lime/TLS Scans"
output_path = "data/raw/Lime/TLS Scans/average_power.xlsx"
power_data = process_and_save(trq_folder_path, output_path)

# Process hyperspectral data
processor = HyperspectralProcessor(
    data_path="data/raw/Lime",
    metadata_path="data/raw/Lime/metadata.xlsx",
    laser_power_excel="data/raw/Lime/TLS Scans/average_power.xlsx",
    cutoff_offset=40,
    verbose=True
)

output_files = processor.process_full_pipeline(
    output_dir="data/processed/Lime",
    exposure_reference="max",
    power_reference="min",
    create_parquet=False,
    preserve_full_data=True
)
```

The processor performs several important preprocessing steps:
1. Loads raw .im3 hyperspectral files
2. Applies spectral cutoffs to remove Rayleigh scattering and second-order artifacts
3. Normalizes for exposure time differences between excitations
4. Normalizes for laser power variations
5. Saves processed data in a structured format

### 2. Creating a Mask

Create masks to define regions of interest or remove invalid pixels:

```python
from scripts.utils.masking_utils import create_masked_pickle

# Apply an existing mask
masked_file = create_masked_pickle(
    data_path="data/processed/Lime/lime_data.pkl",
    mask_path="data/processed/Lime/lime_mask.npy",
    output_path="data/processed/Lime/lime_data_masked.pkl"
)
```

You can create masks using:
1. The interactive `masking_tool.py` (run as a script)
2. Creating numpy binary masks manually
3. Automated segmentation methods

### 3. Dataset Creation

Create a dataset for model training and analysis:

```python
from scripts.models.dataset import MaskedHyperspectralDataset, load_hyperspectral_data, load_mask

# Load processed data and mask
data_dict = load_hyperspectral_data("data/processed/Lime/lime_data.pkl")
mask = load_mask("data/processed/Lime/lime_mask.npy")

# Create dataset
dataset = MaskedHyperspectralDataset(
    data_dict=data_dict,
    mask=mask,
    normalize=True,
    downscale_factor=1  # Set higher for large datasets
)
```

The dataset handles:
- Proper masking of invalid regions
- Normalization of intensity values
- Optional spatial downscaling for large datasets
- Tracking emission wavelengths for each excitation

### 4. Model Creation and Training

Train an autoencoder for dimensionality reduction:

```python
from scripts.models.autoencoder import HyperspectralCAEWithMasking
from scripts.models.training import train_with_masking

# Get data
all_data = dataset.get_all_data()

# Create model
model = HyperspectralCAEWithMasking(
    excitations_data={ex: data.numpy() for ex, data in all_data.items()},
    k1=20,  # Number of first-layer filters
    k3=20,  # Number of third-layer filters
    filter_size=5,
    sparsity_target=0.1,
    sparsity_weight=1.0,
    dropout_rate=0.5
)

# Train model
model, losses = train_with_masking(
    model=model,
    dataset=dataset,
    num_epochs=50,
    learning_rate=0.001,
    chunk_size=256,  # Size of spatial chunks for processing
    chunk_overlap=64,  # Overlap between chunks
    batch_size=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience=5,
    mask=dataset.processed_mask,
    output_dir="results/lime_analysis/model"
)
```

Key features:
- Memory-efficient chunking for large datasets
- Masked loss computation (ignores invalid/masked pixels)
- Early stopping for optimal results
- Sparsity regularization for better features

### 5. Evaluation and Visualization

Evaluate model performance and visualize results:

```python
from scripts.models.training import evaluate_model_with_masking
from scripts.models.visualization import create_rgb_visualization, visualize_reconstruction_comparison

# Evaluate model
evaluation_results = evaluate_model_with_masking(
    model=model,
    dataset=dataset,
    chunk_size=256,
    chunk_overlap=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="results/lime_analysis/evaluation"
)

# Get reconstructed data
reconstructions = evaluation_results['reconstructions']

# Create visualizations
original_rgb = create_rgb_visualization(
    data_dict=all_data,
    emission_wavelengths=dataset.emission_wavelengths,
    mask=dataset.processed_mask,
    output_dir="results/lime_analysis/visualizations"
)

recon_rgb = create_rgb_visualization(
    data_dict=reconstructions,
    emission_wavelengths=dataset.emission_wavelengths,
    mask=dataset.processed_mask,
    output_dir="results/lime_analysis/visualizations/reconstructed"
)

# Create detailed comparisons
for ex in all_data:
    if ex in reconstructions:
        metrics = visualize_reconstruction_comparison(
            original_data=all_data[ex],
            reconstructed_data=reconstructions[ex],
            excitation=ex,
            emission_wavelengths=dataset.emission_wavelengths.get(ex, None),
            mask=dataset.processed_mask,
            output_dir="results/lime_analysis/visualizations"
        )
```

Visualization options:
- RGB false-color representations
- Side-by-side original vs. reconstruction comparisons
- Spectral profile plots
- Error maps and metrics

### 6. Clustering and Analysis

Perform pixel-wise clustering for segmentation:

```python
from scripts.models.clustering import run_4d_pixel_wise_clustering, visualize_4d_cluster_profiles

# Run 4D clustering (across all excitations)
cluster_results = run_4d_pixel_wise_clustering(
    model=model,
    dataset=dataset,
    n_clusters=7,
    chunk_size=256,
    chunk_overlap=64,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="results/lime_analysis/clustering",
    calculate_metrics=True,
    use_pca=False
)

# Analyze cluster profiles
cluster_stats = visualize_4d_cluster_profiles(
    cluster_results=cluster_results,
    dataset=dataset,
    original_data=all_data,
    output_dir="results/lime_analysis/clustering"
)

# Create overlay visualization
from scripts.models.visualization import overlay_clusters_on_rgb

visualization_excitation = cluster_results['excitations_used'][0]
overlay = overlay_clusters_on_rgb(
    cluster_labels=cluster_results['cluster_labels'],
    rgb_image=original_rgb[visualization_excitation],
    mask=dataset.processed_mask,
    output_path="results/lime_analysis/clustering/cluster_overlay.png"
)
```

Clustering features:
- 4D clustering across all excitation and emission wavelengths
- Optimized K-means with PCA preprocessing option
- Evaluation metrics for cluster quality
- Spectral profile visualization for each cluster
- Colored overlays for interpretation

### 7. Comparative Clustering Analysis

Compare different clustering parameters:

```python
# Run clustering with different numbers of clusters
cluster_numbers = [4, 5, 6, 7, 8, 9, 10]
cluster_results_by_k = {}

for k in cluster_numbers:
    results_k = run_4d_pixel_wise_clustering(
        model=model,
        dataset=dataset,
        n_clusters=k,
        chunk_size=256,
        chunk_overlap=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=f"results/lime_analysis/clustering/4d_k_{k}",
        calculate_metrics=True,
        use_pca=False
    )
    
    cluster_stats = visualize_4d_cluster_profiles(
        cluster_results=results_k,
        dataset=dataset,
        original_data=all_data,
        output_dir=f"results/lime_analysis/clustering/4d_k_{k}"
    )
    
    cluster_results_by_k[k] = results_k
```

Analysis options:
- Comparing cluster quality metrics across different K values
- Visualizing cluster size distributions
- Consistent color mapping for better comparison

## Examples

Complete examples are available in the `notebooks/` directory:
- `complete_lime_analysis_notebook.ipynb`: End-to-end analysis of lime data
- `complete_kiwi_analysis_notebook.ipynb`: End-to-end analysis of kiwi data

These notebooks walk through the entire workflow from raw data processing to final clustering analysis.