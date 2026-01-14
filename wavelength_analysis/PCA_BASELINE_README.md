# PCA Baseline Comparison

## Overview

This script (`wavelength_selection_pca_baseline.py`) provides a direct comparison baseline to the autoencoder-based wavelength selection approach (V2-2). It answers the fundamental question:

**"Is the autoencoder + perturbation complexity necessary, or can we achieve similar results with simple PCA directly on the normalized hyperspectral data?"**

## Architecture Comparison

### Original Pipeline (WavelengthSelectionV2-2)
```
Raw 4D Hyperspectral Data
    ↓
Normalize/Preprocess
    ↓
Train Autoencoder (unsupervised feature learning)
    ↓
Perturbation-based Selection (MMR with n dimensions)
    ↓
Selected Wavelength Combinations
    ↓
Supervised Classification (KNN)
    ↓
Metrics + Visualizations
```

### PCA Baseline Pipeline (This Script)
```
Raw 4D Hyperspectral Data
    ↓
Normalize/Preprocess (SAME)
    ↓
PCA-based Selection (direct, no autoencoder)
    ↓
Selected Wavelength Combinations
    ↓
Supervised Classification (KNN) (SAME)
    ↓
Metrics + Visualizations (SAME)
```

**Key Difference**: Only the wavelength selection step is different. Everything else (data loading, preprocessing, clustering, metrics, visualizations) is identical.

## Selection Methods Available

The script supports 3 PCA-based selection methods:

### 1. **pca_loadings** (Recommended for comparison)
- Uses PCA component loadings to select wavelengths
- Selects wavelengths with highest absolute loadings on principal components
- Captures maximum variance directions
- Most directly comparable to PCA dimensionality reduction

### 2. **variance** (Simplest baseline)
- Selects wavelengths with highest variance
- Dead simple: `sorted(variances)[-n:]`
- Good "lower bound" baseline
- Shows if any complexity is needed at all

### 3. **greedy_diversity** (Mimics MMR without autoencoder)
- Balances high variance with low correlation
- Greedy selection: pick bands iteratively
- Similar to MMR logic but operates directly on raw data
- Shows if diversity matters without autoencoder

## Usage

### Basic Usage (PCA Loadings method)
```bash
python wavelength_selection_pca_baseline.py
```

### Specify Selection Method
```bash
python wavelength_selection_pca_baseline.py pca_loadings

python wavelength_selection_pca_baseline.py variance

python wavelength_selection_pca_baseline.py greedy_diversity
```

### Limit Number of Configurations (for testing)
```bash
python wavelength_selection_pca_baseline.py pca_loadings 10
```

## Output Structure

Generates same directory structure as V2-2:

```
pca_baseline_results/
└── YYYYMMDD_HHMMSS/
    ├── wavelength_selection_results_pca_baseline.xlsx  # Main results
    ├── supervised_metrics/
    │   ├── baseline_supervised_metrics.json
    │   └── ground_truth_tracker_state.pkl
    ├── experiments/
    │   ├── BASELINE_FULL_DATA/
    │   │   ├── BASELINE_object_metrics.csv
    │   │   ├── supervised_visualizations/
    │   │   └── *.png
    │   ├── pca_loadings_10bands/
    │   ├── pca_loadings_20bands/
    │   └── ...
    ├── paper-results/
    │   ├── BASELINE_roi_overlay.png
    │   ├── BASELINE_classification.png
    │   ├── pca_loadings_10bands_roi_overlay.png
    │   └── ...
    ├── concat-data/
    │   ├── BASELINE_FULL_DATA_concatenated_data.csv
    │   └── ...
    └── analysis_summary/
        ├── all_object_metrics_across_configs.csv
        └── per_config_object_metrics_summary.csv
```

## Excel Output

The main results Excel file contains:

| Column | Description |
|--------|-------------|
| config_name | e.g., "pca_loadings_20bands" |
| n_combinations_selected | Number of wavelength pairs selected |
| n_features | Total features after selection |
| data_reduction_pct | % reduction from baseline |
| accuracy | Supervised accuracy |
| f1_weighted | Weighted F1 score |
| precision_weighted | Weighted precision |
| recall_weighted | Weighted recall |
| cohen_kappa | Cohen's Kappa coefficient |
| purity | Clustering purity |
| ari | Adjusted Rand Index |
| nmi | Normalized Mutual Information |
| selection_time | Time for wavelength selection (s) |
| clustering_time | Time for clustering (s) |
| speedup_factor | Speedup vs baseline |

## Comparison with V2-2

### What's Identical:
- Data loading and preprocessing
- Cropping and masking
- Ground truth tracking
- KNN clustering algorithm
- Supervised metrics calculation
- ROI analysis
- Object-level tracking
- All visualizations
- Excel output format

### What's Different:
- **Wavelength Selection Method**:
  - V2-2: Autoencoder + Perturbation + MMR
  - PCA Baseline: PCA component loadings / Variance / Greedy

- **Computational Cost**:
  - V2-2: Requires autoencoder training (slower)
  - PCA Baseline: Direct PCA (faster)

- **Configuration Naming**:
  - V2-2: `mmr_lambda050_variance_1dim_20bands`
  - PCA Baseline: `pca_loadings_20bands`

## Expected Results

### Scenario A: V2-2 Autoencoder Approach Wins
If autoencoder approach achieves significantly higher accuracy:
- **Interpretation**: Complex feature learning is necessary
- **Conference Message**: "PCA achieves only X% accuracy, while our autoencoder achieves Y% - justifying the added complexity"

### Scenario B: PCA Competitive
If PCA achieves similar accuracy:
- **Interpretation**: Raw data structure is sufficient
- **Conference Message**: Emphasize OTHER advantages (stability, object-level performance, interpretability)

### Scenario C: Trade-offs
If different methods excel at different band counts:
- **Interpretation**: Method choice depends on application constraints
- **Conference Message**: "For extreme compression (<20 bands), autoencoder excels; for moderate reduction, simpler methods sufficient"

## Band Counts Tested

Same as V2-2 for direct comparison:
```python
[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 50, 60, 70, 80, 90,
 100, 110, 120, 130, 140, 150, 160, 170]
```

## Analysis with Existing Notebooks

The output from this script can be analyzed using your existing perturbation dimension analysis notebooks!

### Step 1: Copy Results to Comparison Location
```bash
cp -r pca_baseline_results/TIMESTAMP validation_results_v2/PCA_Baseline
```

### Step 2: Modify Analysis Notebooks
In `perturbation_dimension_analysis.ipynb`, add PCA baseline to comparison:

```python
# Load V2-2 results
v2_1d = load_results('validation_results_v2/1Dimensions')

# Load PCA baseline results
pca_baseline = load_results('validation_results_v2/PCA_Baseline')

# Compare
plt.plot(v2_1d['n_bands'], v2_1d['accuracy'], label='Autoencoder (1D)')
plt.plot(pca_baseline['n_bands'], pca_baseline['accuracy'], label='PCA Baseline')
plt.legend()
```

## Key Comparison Metrics

### 1. Accuracy at Same Band Count
**Question**: For N bands, which method achieves higher accuracy?

**Analysis**:
```python
n_bands = 20
v2_acc = v2_results[v2_results['n_bands'] == n_bands]['accuracy'].values[0]
pca_acc = pca_results[pca_results['n_bands'] == n_bands]['accuracy'].values[0]
print(f"V2 Autoencoder: {v2_acc:.4f}")
print(f"PCA Baseline: {pca_acc:.4f}")
print(f"Difference: {v2_acc - pca_acc:.4f}")
```

### 2. Bands Needed for Target Accuracy
**Question**: To reach 95% of baseline accuracy, how many bands does each method need?

### 3. Stability
**Question**: Which method has lower variance across configurations?

### 4. Object-Level Performance
**Question**: Do difficult objects benefit more from autoencoder?

## Integration with Presentation

### Slide: "Baseline Comparison"

**Setup**: "We compared our autoencoder approach against direct PCA selection"

**Show**: Plot of accuracy vs bands for both methods

**Key Messages**:
- Quantify accuracy difference
- Highlight efficiency gains/losses
- Show object-level improvements
- Discuss when each method is appropriate

## Troubleshooting

### Issue: Import Errors
**Solution**: Ensure you're running from the same environment as V2-2:
```bash
cd wavelength_analysis
python wavelength_selection_pca_baseline.py
```

### Issue: Memory Errors
**Solution**: Limit configurations:
```bash
python wavelength_selection_pca_baseline.py pca_loadings 20
```

### Issue: Results Directory Not Found
**Solution**: Check that base paths match your setup in the script header

## Time Estimates

For full run (42 configurations):
- **PCA Loadings**: ~2-3 hours (faster than V2-2)
- **Variance**: ~1-2 hours (fastest)
- **Greedy Diversity**: ~3-4 hours

For limited run (10 configurations):
- ~30-60 minutes

## Next Steps After Running

1. **Compare Results**: Open both Excel files side-by-side
2. **Visualize Comparison**: Use plotting notebook to create comparison figures
3. **Statistical Testing**: Run paired t-test on accuracies
4. **Object Analysis**: Compare object-level metrics
5. **Presentation Integration**: Add comparison slide to conference presentation

## Scientific Rigor

This comparison demonstrates:
- ✅ Thoroughness: Tested simpler alternatives
- ✅ Transparency: Same evaluation framework
- ✅ Reproducibility: Same data, same metrics
- ✅ Objectivity: Let results speak for themselves

**Either way, you strengthen your paper!**

## Questions This Answers

1. **Is the autoencoder necessary?** → See accuracy comparison
2. **What's the computational trade-off?** → See selection_time column
3. **Does complexity buy us performance?** → See accuracy vs bands plot
4. **Are there diminishing returns?** → See where curves converge
5. **Which objects benefit most?** → See object-level analysis

## Files Generated

Per Configuration:
- Object metrics CSV
- Supervised metrics JSON
- 8+ visualization PNGs
- Classification maps
- ROI overlays

Total:
- 1 Excel summary file
- ~43 experiment folders
- ~500+ visualization files
- Object-level summaries

## Contact

For issues or questions about this comparison script:
- Check that V2-2 runs successfully first
- Verify all import paths are correct
- Ensure data files are accessible

---

**Created**: 2025-10-29
**Purpose**: Baseline comparison for wavelength selection methods
**Compatible with**: WavelengthSelectionV2-2.py outputs
