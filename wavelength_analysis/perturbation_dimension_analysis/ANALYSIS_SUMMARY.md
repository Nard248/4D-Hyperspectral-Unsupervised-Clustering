# Analysis Package Summary

## What Was Delivered

A comprehensive analysis package for understanding wavelength selection performance across perturbation dimensions, with special focus on object-level behavior.

## Files Created

### Documentation (3 files)
1. **README.md** - Main overview and quick start guide
2. **DATA_DESCRIPTION.md** - Comprehensive data structure reference
3. **OBJECT_LEVEL_ANALYSIS_README.md** - Object-level analysis guide
4. **ANALYSIS_SUMMARY.md** - This file

### Notebooks (2 files)

#### 1. perturbation_dimension_analysis.ipynb
**Purpose**: Compare perturbation dimensions 1-7

**Visualizations** (8 plots):
1. Best accuracy vs perturbation dimensions
2. Tolerance region analysis (2 subplots)
3. Difference from baseline (2 subplots)
4. Accuracy distributions (box & violin plots)
5. Accuracy vs bands Pareto front
6. Efficiency analysis (2 subplots)
7. Accuracy heatmap (dimensions × bands)
8. Stability analysis (4 subplots)

**Data Source**:
- Excel files from all 7 dimension folders
- Combines 43+ configurations per dimension level

---

#### 2. object_level_analysis_1dim.ipynb
**Purpose**: Deep dive into individual object performance for 1D

**Visualizations**:
- **16 individual object plots** (saved separately for presentations!)
- All objects overlay plot
- Objects by class (4 subplots)
- Object accuracy heatmap
- Object sensitivity analysis (4 subplots)
- Class-level analysis (4 subplots)
- Optimal bands analysis (4 subplots)
- Object clustering (4 clusters)

**Data Source**:
- Object-level CSV with 16 objects × 43 configurations = 688 data points

**Exports**:
- `object_statistics_summary.csv`
- `optimal_configurations_per_object.csv`
- `object_clusters.csv`

## Directory Structure

```
perturbation_dimension_analysis/
├── README.md
├── DATA_DESCRIPTION.md
├── OBJECT_LEVEL_ANALYSIS_README.md
├── ANALYSIS_SUMMARY.md                    (this file)
├── perturbation_dimension_analysis.ipynb
├── object_level_analysis_1dim.ipynb
├── figures/                               (8 main analysis plots)
│   ├── plot1_best_accuracy_vs_dimensions.png
│   ├── plot2_tolerance_region_analysis.png
│   ├── plot3_difference_from_baseline.png
│   ├── plot4_accuracy_distributions.png
│   ├── plot5_accuracy_vs_bands_pareto.png
│   ├── plot6_efficiency_analysis.png
│   ├── plot7_accuracy_heatmap.png
│   └── plot8_stability_analysis.png
└── object_figures/                        (object-level plots)
    ├── individual_objects/                (16 object plots)
    │   ├── object_01_*.png
    │   ├── object_02_*.png
    │   └── ...
    ├── by_class/                          (2 plots)
    │   ├── objects_by_class.png
    │   └── class_level_analysis.png
    └── summary/                           (5 plots)
        ├── all_objects_accuracy_vs_bands.png
        ├── object_accuracy_heatmap.png
        ├── object_sensitivity_analysis.png
        ├── optimal_bands_analysis.png
        └── object_clusters.png
```

## Key Features

### Your 3 Requested Plots ✓
1. **Best accuracy vs n-perturbation dimensions** - Shows optimal dimension level
2. **Tolerance region analysis** - Finds minimum bands for 95%, 97%, 99% of baseline
3. **Difference from baseline** - Visualizes improvements/degradations

### Additional Value-Added Features

#### Main Analysis (Dimensions 1-7)
- Distribution analysis (box & violin plots)
- Pareto efficiency visualization
- Accuracy per band metrics
- Comprehensive heatmaps
- Stability metrics (CV, std dev, range)
- Data reduction potential analysis

#### Object-Level Analysis (1D Deep Dive)
- **16 presentation-ready individual plots**
- Object stability/sensitivity metrics
- Class-level behavioral patterns
- K-means clustering (4 groups)
- Per-object optimal band selection
- Achievable data reduction per object
- Within-class variability analysis

## How to Use

### Quick Start
```bash
cd wavelength_analysis/perturbation_dimension_analysis

# For main analysis
jupyter notebook perturbation_dimension_analysis.ipynb

# For object-level analysis
jupyter notebook object_level_analysis_1dim.ipynb
```

### For Presentations

**Scenario: High-level overview of perturbation dimension impact**
→ Use plots from `figures/` directory

**Scenario: Show specific interesting cases**
→ Browse `object_figures/individual_objects/` and pick compelling examples

**Scenario: Class-specific behavior**
→ Use plots from `object_figures/by_class/`

**Scenario: Demonstrate data reduction potential**
→ Use tolerance region analysis or efficiency plots

## Key Insights You Can Discover

### From Main Analysis:
1. Which perturbation dimension level is optimal?
2. How stable are results across dimensions?
3. What's the accuracy vs efficiency trade-off?
4. Can we beat baseline with fewer bands?
5. Which dimension level is most consistent?

### From Object-Level Analysis:
1. Which objects are easy/hard to classify?
2. Which objects benefit most from band selection?
3. Are certain classes more sensitive than others?
4. What's the minimum bands needed per object?
5. Do objects cluster into behavioral groups?
6. Which objects should we focus improvement efforts on?

## Additional Analysis Ideas (Brainstormed)

The notebooks include infrastructure for:
- 3D surface plots (dimensions × bands × accuracy)
- Time/performance trade-offs
- Clustering quality metrics (ARI, NMI) analysis
- ROI-specific accuracy trends
- Correlation analysis across all metrics
- Feature importance by object
- Spatial pattern analysis

## Data Sources

### Main Analysis
```
validation_results_v2/
├── 1Dimensions/wavelength_selection_results_v2.xlsx
├── 2Dimensions/wavelength_selection_results_v2.xlsx
├── 3Dimensions/wavelength_selection_results_v2.xlsx
├── 4Dimensions/wavelength_selection_results_v2.xlsx
├── 5Dimensions/wavelength_selection_results_v2.xlsx
├── 6Dimensions/wavelength_selection_results_v2.xlsx
└── 7Dimensions/wavelength_selection_results_v2.xlsx
```

### Object-Level Analysis
```
validation_results_v2/1Dimensions/analysis_summary/all_object_metrics_across_configs.csv
```

## Statistics

- **Total configurations analyzed**: 301 (43 per dimension × 7 dimensions)
- **Individual objects tracked**: 16 (4 per class)
- **Total visualizations generated**: 31+ plots
- **Classes analyzed**: 4 (Lichen types 0, 1, 2, 5)
- **Band selection range**: 3-204 bands

## Requirements

```bash
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn scipy
```

## Next Steps

After running the notebooks, consider:

1. **Identify optimal dimension**: Based on your priority (accuracy, efficiency, stability)
2. **Select presentation examples**: Cherry-pick interesting objects for storytelling
3. **Investigate outliers**: Why do certain objects/dimensions behave differently?
4. **Cross-dimensional comparison**: Run object-level analysis for other dimensions (2D, 3D, etc.)
5. **Wavelength investigation**: Which specific wavelengths are selected in best configs?
6. **Ensemble strategies**: Can we use different band selections for different object types?

## Questions Answered

✓ Does n-perturbation dimensions correlate with selection quality?
✓ What's the optimal dimension level?
✓ Which objects are sensitive to band selection?
✓ How much data reduction is achievable?
✓ Are there class-specific patterns?
✓ What's the stability across configurations?
✓ Can we beat baseline with fewer bands?

## Support

- **Data structure questions**: See DATA_DESCRIPTION.md
- **Object analysis questions**: See OBJECT_LEVEL_ANALYSIS_README.md
- **General questions**: See main README.md
- **Code questions**: All notebooks have inline documentation

---

**Created**: 2025-10-29
**For**: Capstone Project - Wavelength Selection Analysis
**Data**: validation_results_v2 (1-7 Dimensions)
