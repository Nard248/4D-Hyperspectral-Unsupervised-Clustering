# Validation Results V2 - Data Description

## Overview
This document describes the data structure and contents of the wavelength_analysis/validation_results_v2 directory, which contains results from WavelengthSelectionV2-2 experiments with varying perturbation dimensions (1-7 dimensions).

## Directory Structure

```
validation_results_v2/
├── 1Dimensions/
├── 2Dimensions/
├── 3Dimensions/
├── 4Dimensions/
├── 5Dimensions/
├── 6Dimensions/
└── 7Dimensions/
```

Each dimension folder represents experiments with a specific number of perturbation dimensions and follows an identical structure.

## Files Within Each Dimension Folder

### 1. Primary Results File

**File**: `wavelength_selection_results_v2.xlsx`
**Location**: Root of each dimension folder (e.g., `1Dimensions/wavelength_selection_results_v2.xlsx`)

**Sheet**: `Results_V2`

**Columns**:
- `config_name` (str): Configuration identifier (e.g., "mmr_lambda050_variance_1dim_140bands")
- `n_combinations_selected` (int): Number of wavelength combinations selected
- `n_features` (int): Number of features/bands selected
- `data_reduction_pct` (float): Percentage of data reduction achieved
- `purity` (float): Cluster purity metric
- `ari` (float): Adjusted Rand Index
- `nmi` (float): Normalized Mutual Information
- `accuracy` (float): Classification accuracy (PRIMARY METRIC)
- `precision_weighted` (float): Weighted average precision
- `recall_weighted` (float): Weighted average recall
- `f1_weighted` (float): Weighted F1-score
- `cohen_kappa` (float): Cohen's Kappa coefficient
- `selection_time` (float): Time taken for wavelength selection
- `clustering_time` (float): Time taken for clustering
- `speedup_factor` (float): Computational speedup achieved

**Special Entries**:
- `BASELINE_FULL_DATA`: Results using all 204 wavelength bands (no selection)

### 2. Supervised Metrics (JSON)

**Location**:
- Baseline: `supervised_metrics/baseline_supervised_metrics.json`
- Other configs: `experiments/{config_name}/{config_name}_supervised_metrics.json`

**Structure**:
```json
{
  "accuracy": float,
  "balanced_accuracy": float,
  "precision_micro": float,
  "precision_macro": float,
  "precision_weighted": float,
  "recall_micro": float,
  "recall_macro": float,
  "recall_weighted": float,
  "f1_micro": float,
  "f1_macro": float,
  "f1_weighted": float,
  "cohen_kappa": float,
  "matthews_corrcoef": float,
  "confusion_matrix": [[int]],
  "per_class": {
    "class_id": {
      "true_positives": int,
      "false_positives": int,
      "false_negatives": int,
      "true_negatives": int,
      "precision": float,
      "recall": float,
      "f1": float,
      "specificity": float,
      "support": int,
      "class_name": str
    }
  },
  "total_pixels": int,
  "correct_pixels": float,
  "error_rate": float,
  "roi_metrics": {
    "Region N": {
      "ground_truth_class": int,
      "class_name": str,
      "coordinates": [y1, y2, x1, x2],
      "accuracy": float,
      "pixel_count": int,
      "correct_pixels": float,
      "dominant_prediction": int,
      "unique_predictions": [int],
      "class_match": bool
    }
  },
  "classification_report": str
}
```

### 3. Object-Level Metrics (CSV)

**Location**:
- Per config: `experiments/{config_name}/{config_name}_object_metrics.csv`
- Summary: `analysis_summary/all_object_metrics_across_configs.csv`

**Columns**:
- Object identification and metrics
- Per-object accuracy
- Class predictions
- Spatial information

### 4. Concatenated Data (CSV)

**Location**: `concat-data/{config_name}_concatenated_data.csv`

Contains the full dataset for each configuration with selected wavelengths.

### 5. Visualizations

**Location**:
- Main results: `paper-results/`
- Detailed: `experiments/{config_name}/supervised_visualizations/`

**Available Plots**:
- `confusion_matrix.png`: Confusion matrix heatmap
- `per_class_metrics.png`: Per-class precision, recall, F1 scores
- `accuracy_heatmap.png`: Spatial accuracy distribution
- `misclassification_patterns.png`: Error analysis
- `roi_performance.png`: ROI-specific metrics
- `roi_overlay_accuracy.png`: Spatial ROI overlay with accuracy
- `metrics_comparison.png`: Comparison of multiple metrics
- `class_distribution.png`: Class distribution visualization
- `{config_name}_classification.png`: Classification results
- `{config_name}_roi_overlay_main.png`: Main ROI overlay visualization

### 6. Analysis Summary Files

**Location**: `analysis_summary/`

**Files**:
- `all_object_metrics_across_configs.csv`: Object-level metrics for all configurations
- `per_config_object_metrics_summary.csv`: Summary statistics per configuration
- `full_dataset_object_metrics_summary.csv`: Dataset-wide summary

## Configuration Naming Convention

Format: `mmr_lambda{lambda}_variance_{n}dim_{k}bands`

**Components**:
- `mmr`: Maximal Marginal Relevance method
- `lambda{lambda}`: Lambda parameter value (e.g., lambda050 = 0.50)
- `variance`: Feature selection criterion
- `{n}dim`: Number of perturbation dimensions (e.g., 1dim, 2dim, ..., 7dim)
- `{k}bands`: Number of bands/features selected (e.g., 3bands, 4bands, ..., 170bands)

**Example**: `mmr_lambda050_variance_1dim_140bands`
- Uses MMR method with lambda=0.50
- Variance-based selection
- 1 perturbation dimension
- 140 bands selected

## Baseline Configuration

**Name**: `BASELINE_FULL_DATA`
- Uses all 204 wavelength bands
- No feature selection applied
- Serves as reference for comparison
- Typical accuracy: ~0.855 (85.5%)

## Classes/Lichen Types

The dataset includes 4 lichen types:
- **Class 0**: Lichen_Type_0
- **Class 1**: Lichen_Type_1
- **Class 2**: Lichen_Type_2
- **Class 5**: Lichen_Type_5

**Note**: Classes are not consecutive (0, 1, 2, 5), which is intentional based on the original dataset labeling.

## Key Metrics Definitions

### Accuracy
- Primary evaluation metric
- Percentage of correctly classified pixels
- Range: [0, 1], where 1 = 100% correct

### Balanced Accuracy
- Average of recall for each class
- Better for imbalanced datasets
- Accounts for different class sizes

### Cohen's Kappa
- Inter-rater reliability metric
- Adjusts accuracy for chance agreement
- Range: [-1, 1], where 1 = perfect agreement

### Confusion Matrix
- 4×4 matrix for 4 classes
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions

## Regions of Interest (ROI)

Four spatial regions defined for detailed analysis:

| Region | Class | Coordinates [y1, y2, x1, x2] | Pixels |
|--------|-------|------------------------------|--------|
| Region 1 | Lichen_Type_0 | [175, 225, 100, 150] | 2,500 |
| Region 2 | Lichen_Type_1 | [175, 225, 250, 300] | 2,500 |
| Region 3 | Lichen_Type_2 | [175, 225, 425, 475] | 2,500 |
| Region 4 | Lichen_Type_5 | [185, 225, 675, 700] | 1,000 |

## Data Usage for Comparative Analysis

### For Perturbation Dimension Analysis:

1. **Primary Data Source**: `wavelength_selection_results_v2.xlsx` from each dimension folder
   - Compare accuracy across different perturbation dimensions
   - Track feature reduction vs accuracy trade-off

2. **Baseline Reference**: Use `BASELINE_FULL_DATA` entry
   - Baseline accuracy: ~0.855
   - All experiments should be compared against this

3. **Key Comparisons**:
   - Best accuracy for each perturbation dimension level
   - Minimum features needed to reach within X% of baseline
   - Trend analysis: accuracy vs perturbation dimensions

### Recommended Analysis Approaches:

1. **Best Choice Analysis**: For each perturbation dimension (1-7), find the configuration with highest accuracy (excluding baseline)

2. **Tolerance Region Analysis**: Define tolerance levels (e.g., 95%, 97%, 99% of baseline) and find minimum features needed

3. **Efficiency Analysis**: Plot accuracy vs n_features for each perturbation dimension to understand efficiency

4. **Stability Analysis**: Examine variance in results across different feature counts within each perturbation dimension

## Notes

- Total dataset size: 223,597 pixels
- Original wavelength bands: 204
- Feature selection range: 3-170 bands typically
- All experiments use MMR with lambda=0.50 and variance criterion
- Experiments are organized by perturbation dimensions (1-7)
