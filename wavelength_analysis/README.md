# Wavelength Analysis for Hyperspectral Imaging

A comprehensive analysis framework for identifying the most informative wavelength combinations in 4D hyperspectral data using latent space perturbation techniques.

## 🔬 Overview

This project implements an intelligent wavelength selection algorithm that:

- **Reduces data acquisition time** from hours to minutes
- **Maintains discrimination capability** while achieving 10-50x compression
- **Uses latent space analysis** to identify informative wavelength combinations  
- **Provides interpretable visualizations** for domain experts
- **Supports multiple sample types** (Lime, Kiwi, Lichens)

## 🏗️ Project Structure

```
wavelength_analysis/
├── core/                          # Core analysis modules
│   ├── __init__.py               # Package initialization
│   ├── analyzer.py               # Main analysis engine  
│   ├── config.py                 # Configuration management
│   └── visualization.py          # Visualization tools
├── results/                       # Organized results by sample
│   ├── Lime/                     # Lime sample results
│   ├── Kiwi/                     # Kiwi sample results  
│   └── Lichens/                  # Lichens sample results
├── run_analysis.py               # Main runner script
└── README.md                     # This documentation
```

## 🚀 Quick Start

### Basic Usage

```bash
# Analyze Lime sample with default settings
python run_analysis.py --sample Lime

# Analyze all samples 
python run_analysis.py --all-samples

# Compare different configurations
python run_analysis.py --comparison
```

### Advanced Configurations

```bash
# Use aggressive perturbations for higher sensitivity
python run_analysis.py --sample Lime --config aggressive_std

# High-resolution analysis with fine-grained perturbations
python run_analysis.py --sample Kiwi --config high_resolution

# PCA-based dimension selection
python run_analysis.py --sample Lichens --config pca_based
```

## 📊 Output Structure

Each analysis generates a comprehensive set of outputs:

### Results Directory (`results/{Sample}/`)

```
results/Lime/
├── layers/                        # Wavelength layer extractions
│   ├── layer_01_ex370nm_em420nm_inf53.345337.tiff
│   ├── layer_02_ex360nm_em420nm_inf41.120421.tiff
│   ├── ...
│   └── layer_metadata.json       # Layer metadata
├── visualizations/               # Analysis visualizations  
│   ├── influence_heatmap.png     # Heatmap of all influences
│   ├── wavelength_scatter.png    # Scatter plot of selections
│   ├── excitation_distribution.png # Distribution chart
│   └── summary_dashboard.png     # Comprehensive dashboard
├── selected_bands.json          # Machine-readable results
├── selected_bands.txt           # Human-readable band list
└── analysis_config.json         # Configuration used
```

### Key Output Files

1. **TIFF Layers** (`layers/`)
   - 16-bit grayscale images (256×348 pixels)
   - Top 10 most informative wavelength combinations
   - Normalized and ready for analysis

2. **Visualizations** (`visualizations/`)
   - Influence heatmaps across all wavelengths
   - Scatter plots of selected combinations
   - Distribution charts and summary dashboards

3. **Results Files**
   - JSON format for programmatic access
   - TXT format for human readability
   - Complete metadata and configuration tracking

## 🔧 Configuration Options

### Analysis Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `default` | Activation-based + percentile perturbations | General use, balanced performance |
| `aggressive_std` | Variance-based + large std perturbations | Maximum sensitivity detection |
| `high_resolution` | Fine-grained percentile analysis | Detailed influence mapping |
| `pca_based` | PCA selection + absolute range | Dimensionality reduction focus |

### Configuration Parameters

```python
# Example custom configuration
config = AnalysisConfig(
    sample_name="Lime",
    dimension_selection_method="activation",  # "variance", "activation", "pca"
    perturbation_method="percentile",        # "percentile", "standard_deviation", "absolute_range"
    perturbation_magnitudes=[10, 20, 30],    # Perturbation strengths
    n_important_dimensions=15,               # Number of latent dims to analyze
    n_bands_to_select=30,                    # Number of bands to select
    n_layers_to_extract=10                   # Number of TIFF layers to save
)
```

## 📈 Results Interpretation

### Influence Scores

- **High scores** (>10): Highly informative wavelength combinations
- **Medium scores** (1-10): Moderately informative
- **Low scores** (<1): Less informative but may still be useful

### Wavelength Patterns

**Typical findings for biological samples:**
- **420nm emission** often dominant (blue fluorescence)
- **350-390nm excitation** range commonly important (UV-A region)
- **Compression ratios** of 10-50x achievable with minimal information loss

### Performance Metrics

- **Compression Ratio**: Original bands / Selected bands
- **Max Influence Score**: Highest individual band influence
- **Coverage**: Distribution across excitation wavelengths

## 🧪 Scientific Background

### Method Overview

1. **Latent Space Extraction**: Train 3D CNN autoencoder on 4D hyperspectral data
2. **Dimension Selection**: Identify most important latent dimensions using activation/variance/PCA
3. **Perturbation Analysis**: Systematically perturb important dimensions  
4. **Influence Measurement**: Track reconstruction changes across wavelength bands
5. **Band Selection**: Rank and select top wavelength combinations

### Key Innovations

- **Percentile-based perturbations**: Data-driven scaling instead of fixed epsilon values
- **Multi-scale analysis**: Test multiple perturbation magnitudes simultaneously  
- **Activation-based selection**: Mean absolute activation outperforms variance for dimension selection
- **Comprehensive validation**: Statistical normalization and extensive visualization

## 🛠️ Technical Requirements

### Dependencies

```python
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tifffile>=2021.11.2
Pillow>=8.3.0
pandas>=1.3.0
```

### Hardware Requirements

- **GPU**: CUDA-capable GPU recommended (analysis uses ~2-4GB VRAM)
- **RAM**: 8GB+ recommended for large datasets  
- **Storage**: ~500MB-1GB per sample analysis

### Data Requirements

- **Hyperspectral data**: 4D format [height, width, emission, excitation]
- **Trained model**: Pre-trained autoencoder weights (.pth file)
- **Optional mask**: Binary mask for region-of-interest analysis

## 📚 API Reference

### Core Classes

#### `WavelengthAnalyzer`

Main analysis engine that orchestrates the complete wavelength selection pipeline.

```python
from core.analyzer import WavelengthAnalyzer
from core.config import AnalysisConfig

# Create configuration
config = AnalysisConfig(sample_name="Lime")

# Run analysis
analyzer = WavelengthAnalyzer(config)
results = analyzer.run_complete_analysis()
```

#### `AnalysisConfig`

Configuration management with predefined templates and validation.

```python
from core.config import LIME_CONFIG, EXPERIMENTAL_CONFIGS

# Use predefined config
config = LIME_CONFIG

# Use experimental config
config = EXPERIMENTAL_CONFIGS["aggressive_std"]
```

#### `WavelengthVisualizer`

Comprehensive visualization tools for results analysis.

```python
from core.visualization import WavelengthVisualizer

visualizer = WavelengthVisualizer(selected_bands, influence_matrix, output_dir)
visualizer.create_all_visualizations()
```

## 🔬 Example Results

### Lime Sample Analysis

**Configuration**: Activation-based selection with percentile perturbations

**Key Findings**:
- **Top combination**: 370nm excitation → 420nm emission (influence: 53.35)
- **Compression achieved**: 17.1x (512 bands → 30 bands)
- **Dominant pattern**: Blue emission (420-430nm) with UV-A excitation

**Selected Wavelengths** (Top 5):
1. Ex 370nm / Em 420nm: 53.345
2. Ex 360nm / Em 420nm: 41.120  
3. Ex 380nm / Em 420nm: 37.699
4. Ex 350nm / Em 420nm: 31.289
5. Ex 370nm / Em 430nm: 28.514

## 🚨 Troubleshooting

### Common Issues

**"Data file not found"**
- Verify file paths in configuration
- Ensure data files exist in `data/processed/{Sample}/` directory

**"CUDA out of memory"**  
- Reduce `n_baseline_patches` in configuration
- Use CPU instead: set `device="cpu"`

**"Low influence scores"**
- Try `aggressive_std` configuration
- Increase `perturbation_magnitudes` values
- Verify model is properly trained

**"No valid patches found"**
- Check mask file validity
- Reduce `patch_size` or increase `patch_stride`
- Verify spatial dimensions match data

### Performance Optimization

- **GPU Memory**: Reduce batch sizes and patch counts
- **Speed**: Use fewer perturbation magnitudes for faster analysis  
- **Quality**: Increase `n_important_dimensions` for more thorough analysis

## 📄 Citation

If you use this wavelength selection framework in your research, please cite:

```bibtex
@article{wavelength_selection_2024,
  title={Intelligent Wavelength Selection for 4D Hyperspectral Imaging via Latent Space Analysis},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our contribution guidelines for details on:

- Code style and formatting
- Testing requirements  
- Documentation standards
- Pull request process

## 📧 Support

For questions, issues, or feature requests:

1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed description and logs

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy analyzing! 🔬✨**