# Perturbation Dimension Analysis

This directory contains analysis and visualizations comparing the impact of different perturbation dimension levels (1-7) on wavelength selection quality and classification accuracy.

## Directory Contents

```
perturbation_dimension_analysis/
├── README.md                              # This file (overview)
├── DATA_DESCRIPTION.md                    # Comprehensive data structure documentation
├── OBJECT_LEVEL_ANALYSIS_README.md        # Object-level analysis documentation
├── perturbation_dimension_analysis.ipynb  # Main analysis: dimensions 1-7 comparison
├── object_level_analysis_1dim.ipynb       # Object-level analysis: 1D deep dive
├── figures/                               # Generated plots from main analysis
└── object_figures/                        # Generated plots from object analysis
    ├── individual_objects/                # Individual plots for each object (16 plots)
    ├── by_class/                          # Class-level visualizations
    └── summary/                           # Summary plots
```

## Quick Start

### For Perturbation Dimension Comparison (1D-7D)
1. Open the main analysis notebook:
   ```bash
   jupyter notebook perturbation_dimension_analysis.ipynb
   ```

2. Run all cells to generate the complete analysis and visualizations

3. All plots will be saved to the `figures/` directory

### For Object-Level Deep Dive (1D only)
1. Open the object-level analysis notebook:
   ```bash
   jupyter notebook object_level_analysis_1dim.ipynb
   ```

2. Run all cells to generate object-specific visualizations

3. All plots will be saved to the `object_figures/` directory (including 16 individual object plots)

## Analysis Overview

The notebook provides:

### Core Visualizations (As Requested)

1. **Plot 1: Best Accuracy vs Perturbation Dimensions**
   - Shows the best achievable accuracy for each perturbation dimension level
   - Compares against baseline (204 bands)
   - Identifies optimal perturbation dimension level

2. **Plot 2: Tolerance Region Analysis**
   - Defines multiple confidence levels (95%, 97%, 99% of baseline)
   - Finds minimum bands needed to reach each tolerance level
   - Shows data reduction potential at different quality thresholds

3. **Plot 3: Difference from Baseline**
   - Visualizes accuracy delta (absolute and percentage)
   - Highlights which dimensions beat or fall short of baseline
   - Easy comparison across all dimension levels

### Additional Analysis & Visualizations

4. **Accuracy Distribution Analysis**
   - Box plots and violin plots showing accuracy distributions
   - Understand variability within each dimension level

5. **Accuracy vs Bands (Pareto Front)**
   - Trade-off visualization between feature reduction and accuracy
   - All configurations plotted with tolerance lines

6. **Efficiency Analysis**
   - Accuracy per band metric
   - Identifies most efficient dimension levels

7. **Accuracy Heatmap**
   - 2D heatmap of Dimensions × Band bins
   - Quick overview of performance landscape

8. **Stability Analysis**
   - Coefficient of variation
   - Standard deviation and range analysis
   - Identifies most consistent dimension levels

## Data Source

The notebook analyzes data from:
```
wavelength_analysis/validation_results_v2/
├── 1Dimensions/wavelength_selection_results_v2.xlsx
├── 2Dimensions/wavelength_selection_results_v2.xlsx
├── 3Dimensions/wavelength_selection_results_v2.xlsx
├── 4Dimensions/wavelength_selection_results_v2.xlsx
├── 5Dimensions/wavelength_selection_results_v2.xlsx
├── 6Dimensions/wavelength_selection_results_v2.xlsx
└── 7Dimensions/wavelength_selection_results_v2.xlsx
```

For detailed information about the data structure, see [DATA_DESCRIPTION.md](DATA_DESCRIPTION.md).

## Key Metrics Analyzed

- **Accuracy**: Primary classification metric
- **Number of Bands**: Feature count (3-204)
- **Data Reduction**: Percentage of features eliminated
- **Perturbation Dimensions**: Algorithm parameter (1-7)

## Brainstorming: Understanding the Correlation

### Questions Explored

1. **Does increasing perturbation dimensions improve selection quality?**
   - Measured through peak accuracy, stability, and efficiency

2. **What is the optimal perturbation dimension level?**
   - Depends on goal: maximum accuracy, efficiency, or balance

3. **How does dimensionality affect the intelligence of wavelength selection?**
   - Higher dimensions may explore search space better
   - But too many dimensions could lead to overfitting or instability

### Additional Visualization Ideas

The notebook includes discussion of additional analyses:
- 3D surface plots
- Time/performance trade-offs
- Clustering quality metrics (ARI, NMI)
- Per-class accuracy breakdown
- ROI-specific trends

## Two Complementary Notebooks

This analysis package includes two Jupyter notebooks that work together:

### 1. Main Analysis: `perturbation_dimension_analysis.ipynb`
**Focus**: Compare perturbation dimensions 1-7

**Questions Answered**:
- Which perturbation dimension level achieves best accuracy?
- How does dimensionality affect data reduction potential?
- What's the stability across different dimensions?
- Where's the sweet spot for accuracy vs efficiency?

**Key Outputs**:
- 8 summary plots comparing all dimension levels
- Tolerance region analysis
- Efficiency metrics
- Stability analysis

**Use This When**: You want to understand the macro-level impact of the perturbation dimension parameter

---

### 2. Object-Level Analysis: `object_level_analysis_1dim.ipynb`
**Focus**: Deep dive into individual object performance at 1 perturbation dimension

**Questions Answered**:
- How does each of the 16 objects respond to band selection?
- Which objects are stable vs sensitive to wavelength choice?
- What's the optimal band count for specific objects?
- Are there class-level patterns in object behavior?
- Can we group objects by performance patterns?

**Key Outputs**:
- 16 individual object plots (perfect for presentations!)
- Object sensitivity analysis
- Class-level comparisons
- Object clustering
- Per-object optimal configurations

**Use This When**: You want to understand micro-level behavior and find interesting individual cases to highlight

---

### How They Complement Each Other

**Main Analysis** tells you: "3 perturbation dimensions gives best overall accuracy"

**Object Analysis** tells you: "But Object 11 actually performs best with 5 dimensions at 30 bands, while Object 6 is stable regardless of band count"

Together, they provide:
- **Strategic insight**: Which dimension level to use overall
- **Tactical insight**: How to handle specific problematic objects
- **Presentation material**: Both high-level trends and specific examples

## Usage Tips

### For Main Analysis
- **Interactive Analysis**: The notebook is designed for exploration. Adjust parameters like tolerance levels, bin sizes, or dimension ranges as needed.

- **Custom Visualizations**: All plotting functions are modular and can be easily modified or extended.

- **Data Loading**: The data loading utilities handle all dimension levels automatically. You can filter or subset data as needed.

### For Object-Level Analysis
- **Cherry-Pick for Presentations**: Browse the 16 individual object plots and select the most interesting cases

- **Focus on Outliers**: Objects with high CV or large improvement potential are good storytelling examples

- **Class Comparison**: Use class-level plots to show differential impact across lichen types

- **Identify Patterns**: Use clustering results to group similar behavioral patterns

## Dependencies

Required Python packages:
```python
pandas
numpy
matplotlib
seaborn
openpyxl      # For reading Excel files
scikit-learn  # For clustering in object-level analysis
scipy         # For statistical analysis
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy
```

## Output

All figures are saved as high-resolution PNG files (300 DPI) in the `figures/` directory:
- `plot1_best_accuracy_vs_dimensions.png`
- `plot2_tolerance_region_analysis.png`
- `plot3_difference_from_baseline.png`
- `plot4_accuracy_distributions.png`
- `plot5_accuracy_vs_bands_pareto.png`
- `plot6_efficiency_analysis.png`
- `plot7_accuracy_heatmap.png`
- `plot8_stability_analysis.png`

## Next Steps

After reviewing the analysis, consider:

1. **Identifying optimal configuration** based on your priorities (accuracy vs efficiency)
2. **Running additional experiments** at promising dimension levels
3. **Investigating specific configurations** that show interesting behavior
4. **Exploring other metrics** like clustering quality (ARI, NMI)
5. **Analyzing per-class or ROI-specific trends**

## Questions or Issues?

Refer to the inline documentation in the notebook or the comprehensive data description for detailed information about metrics, formulas, and data structure.
