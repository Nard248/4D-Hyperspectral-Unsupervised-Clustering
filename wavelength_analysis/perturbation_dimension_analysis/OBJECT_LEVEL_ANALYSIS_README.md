# Object-Level Analysis for 1 Perturbation Dimension

This document describes the object-level analysis notebook that tracks how individual classified objects perform across different wavelength band selections.

## Overview

While the main perturbation dimension analysis looks at overall accuracy trends, this notebook dives deeper into **individual object performance**. Each of the 16 classified objects is tracked across all 43 different band selection configurations to understand:

1. Which objects are robust to band selection changes
2. Which objects are sensitive and need careful wavelength selection
3. Per-object optimal band counts
4. Class-level patterns in object behavior
5. Object clustering by performance patterns

## Data Source

The notebook uses:
```
validation_results_v2/1Dimensions/analysis_summary/all_object_metrics_across_configs.csv
```

This file contains object-level metrics for:
- **16 distinct objects** (enumerated in the classification)
- **4 objects per class** (Lichen_Type_0, Lichen_Type_1, Lichen_Type_2, Lichen_Type_5)
- **43 configurations** (baseline + various band selections from 3 to 170 bands)

## Key Metrics

For each object across all configurations:
- **object_id**: Unique identifier (1-16)
- **true_class**: Ground truth class label
- **num_pixels**: Number of pixels in the object
- **accuracy**: Object-level classification accuracy
- **n_bands**: Number of bands used in the configuration
- **baseline_accuracy**: Accuracy with full 204 bands
- **accuracy_diff_from_baseline**: Change from baseline

## Notebook Structure

### 1. Data Loading and Preparation
- Loads object-level metrics
- Calculates baseline comparisons
- Adds derived metrics

### 2. Overall Object Performance Summary
- Statistics for each object across all configurations
- Identification of:
  - Most/least stable objects
  - Best/worst baseline performers
  - Objects with highest improvement potential

### 3. Main Visualizations

#### 3.1 All Objects Accuracy vs Bands
- Single plot with all 16 objects
- Color-coded by class
- Shows overall landscape

#### 3.2 Objects Grouped by Class
- Four subplots (one per class)
- Easier to compare objects within same class

#### 3.3 Individual Object Plots (FOR PRESENTATIONS)
- **16 separate detailed plots** saved to `individual_objects/`
- Each shows:
  - Accuracy trajectory across band counts
  - Baseline reference line
  - Best configuration marked with star
  - Difference from baseline plot
  - Statistics box
- **Perfect for cherry-picking interesting cases for presentations!**

### 4. Object Performance Heatmaps
- 2D heatmap: Objects × Band counts
- Color intensity shows accuracy
- Sorted by baseline accuracy
- Quick visual overview of patterns

### 5. Object Sensitivity Analysis
Four subplots analyzing:
- **Coefficient of Variation (CV)**: Stability measure
- **Accuracy Range**: Max - Min accuracy
- **Best Improvement Potential**: How much can each object improve?
- **Baseline vs Best Achievable**: Comparison bars

### 6. Class-Level Analysis
- Average performance trends by class
- Within-class variability
- Box plots at key band counts
- Identifies if certain classes benefit more from band selection

### 7. Optimal Band Selection
- Finds best configuration for each object
- Finds efficient configuration (99% of best with fewer bands)
- Calculates achievable data reduction per object
- Summary statistics

### 8. Object Clustering
- K-means clustering (4 clusters) based on performance patterns
- Groups objects with similar behavior
- Visualizes each cluster separately
- Useful for understanding different "types" of object responses

### 9. Summary and Insights
- Comprehensive summary of findings
- Key statistics and recommendations
- Identifies objects needing special attention

### 10. Export Summary Statistics
Exports three CSV files:
- `object_statistics_summary.csv`: Per-object statistics
- `optimal_configurations_per_object.csv`: Best configs per object
- `object_clusters.csv`: Cluster assignments

## Output Structure

```
object_figures/
├── individual_objects/          # 16 individual object plots
│   ├── object_01_*.png
│   ├── object_02_*.png
│   └── ...
├── by_class/                    # Class-level visualizations
│   ├── objects_by_class.png
│   └── class_level_analysis.png
└── summary/                     # Overall summary plots
    ├── all_objects_accuracy_vs_bands.png
    ├── object_accuracy_heatmap.png
    ├── object_sensitivity_analysis.png
    ├── optimal_bands_analysis.png
    └── object_clusters.png
```

## Key Insights You Can Discover

### Object Stability
**Question**: Which objects maintain consistent accuracy regardless of band selection?

**Use Case**: Stable objects indicate features that are well-represented across many wavelength combinations. Unstable objects may require specific wavelengths.

### Class-Specific Patterns
**Question**: Do certain lichen classes benefit more from careful wavelength selection?

**Use Case**: If Class 2 objects show high variability but Class 0 objects are stable, it suggests Class 2 requires more careful feature selection.

### Improvement Opportunities
**Question**: Which objects have the most room for improvement over baseline?

**Use Case**: Objects with low baseline but high peak accuracy are good candidates for targeted wavelength selection strategies.

### Efficiency Analysis
**Question**: Can we achieve near-optimal accuracy with far fewer bands?

**Use Case**: If most objects reach 99% of best accuracy with only 30 bands (vs 204), this demonstrates strong dimensionality reduction potential.

### Object Grouping
**Question**: Do objects cluster into distinct behavioral groups?

**Use Case**:
- **Cluster 1**: "Stable performers" - always high accuracy
- **Cluster 2**: "Improvers" - benefit greatly from selection
- **Cluster 3**: "Sensitive" - highly dependent on band choice
- **Cluster 4**: "Difficult" - consistently low accuracy

## Usage for Presentations

### Scenario 1: Show Overall Trends
Use: `summary/all_objects_accuracy_vs_bands.png`

### Scenario 2: Highlight Specific Classes
Use: `by_class/objects_by_class.png`

### Scenario 3: Deep Dive on Interesting Cases
Use individual plots from `individual_objects/`:
- Pick Object 11 if you want to show low baseline with big improvement
- Pick Object 6 if you want to show consistently high performance
- Pick most variable object to show sensitivity
- Pick least variable object to show robustness

### Scenario 4: Show Diversity of Behaviors
Use: `summary/object_clusters.png` - Shows 4 different patterns

### Scenario 5: Demonstrate Efficiency
Use: `summary/optimal_bands_analysis.png` - Shows data reduction potential

## Interesting Questions to Explore

1. **Do larger objects (more pixels) perform differently than smaller ones?**
   - Check correlation between `num_pixels` and stability metrics

2. **Are edge objects more sensitive than central objects?**
   - Would need spatial information, but could infer from object IDs

3. **Do certain band ranges work better for specific objects?**
   - Would require analyzing which wavelengths are selected for best configs

4. **Is there a "sweet spot" band count that works well for most objects?**
   - Look for vertical patterns in the heatmap

5. **Do improvements correlate with baseline performance?**
   - Check if low baseline objects improve more (regression to mean?)

## Tips for Analysis

1. **Start with the heatmap** - Get quick overview of all patterns
2. **Check stability metrics** - Understand which objects need attention
3. **Review class-level analysis** - See if issues are class-specific
4. **Examine individual plots** - Deep dive on outliers and interesting cases
5. **Look at clusters** - Understand different behavioral groups
6. **Check optimal configs** - See if there's consensus on good band counts

## Connection to Main Analysis

This object-level analysis complements the main perturbation dimension analysis:

**Main Analysis**: How does varying perturbation dimensions (1-7) affect overall accuracy?

**Object-Level Analysis**: For a specific perturbation dimension (1D), how do individual objects respond to different band selections?

Together, they provide:
- **Macro view**: Overall trends across perturbation dimensions
- **Micro view**: Individual object sensitivity and patterns

## Next Steps

After running this analysis, you might want to:

1. **Investigate specific wavelengths** - Which bands are selected in best configs for problematic objects?
2. **Spatial analysis** - Map object locations and see if spatial patterns emerge
3. **Cross-perturbation comparison** - Do the same objects remain stable/variable across 2D, 3D, etc.?
4. **Feature importance** - Which wavelength bands are most critical for each object?
5. **Ensemble strategies** - Can we combine different band selections for different objects?

## Running the Notebook

```bash
cd wavelength_analysis/perturbation_dimension_analysis
jupyter notebook object_level_analysis_1dim.ipynb
```

Run all cells to generate complete analysis. The notebook takes ~2-5 minutes to complete all visualizations.
