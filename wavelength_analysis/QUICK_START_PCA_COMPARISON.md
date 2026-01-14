# Quick Start: PCA Baseline Comparison

## What You Have Now

✅ **wavelength_selection_pca_baseline.py** - Complete PCA comparison script
✅ **PCA_BASELINE_README.md** - Comprehensive documentation
✅ Same output format as your V2-2 results

## Run Instructions

### 1. Quick Test Run (10 configurations, ~30 min)

```bash
cd C:\Users\meloy\PycharmProjects\Capstone\wavelength_analysis
python wavelength_selection_pca_baseline.py pca_loadings 10
```

### 2. Full Run - PCA Loadings Method (~2-3 hours)

```bash
python wavelength_selection_pca_baseline.py pca_loadings
```

### 3. Full Run - Variance Method (~1-2 hours, simplest)

```bash
python wavelength_selection_pca_baseline.py variance
```

### 4. Full Run - Greedy Diversity (~3-4 hours)

```bash
python wavelength_selection_pca_baseline.py greedy_diversity
```

## What It Does

1. **Loads same data** as V2-2 (Lichens hyperspectral)
2. **Uses PCA** to select wavelength combinations (NO autoencoder)
3. **Runs KNN clustering** (same as V2-2)
4. **Generates same outputs**:
   - Excel file with results
   - Supervised metrics (JSON)
   - Object-level metrics (CSV)
   - All visualizations (PNG)
   - ROI overlays
   - Classification maps

## Where to Find Results

```
wavelength_analysis/pca_baseline_results/TIMESTAMP/
├── wavelength_selection_results_pca_baseline.xlsx  ← Main results here!
├── experiments/
│   ├── BASELINE_FULL_DATA/
│   ├── pca_loadings_10bands/
│   ├── pca_loadings_20bands/
│   └── ...
├── paper-results/  ← Presentation-ready figures
└── analysis_summary/  ← Object-level data
```

## Comparison Steps

### Step 1: Run PCA Baseline
```bash
python wavelength_selection_pca_baseline.py pca_loadings
```

### Step 2: Compare Excel Files

Open side-by-side:
- `validation_results_v2/1Dimensions/wavelength_selection_results_v2.xlsx`
- `pca_baseline_results/TIMESTAMP/wavelength_selection_results_pca_baseline.xlsx`

Look at accuracy column for same n_bands values.

### Step 3: Create Comparison Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
v2 = pd.read_excel('validation_results_v2/1Dimensions/wavelength_selection_results_v2.xlsx')
pca = pd.read_excel('pca_baseline_results/TIMESTAMP/wavelength_selection_results_pca_baseline.xlsx')

# Extract n_bands from config names
v2['n_bands'] = v2['n_features']
pca['n_bands'] = pca['n_features']

# Plot
plt.figure(figsize=(12, 7))
plt.plot(v2['n_bands'], v2['accuracy'], 'o-', label='Autoencoder (1D)', linewidth=2.5)
plt.plot(pca['n_bands'], pca['accuracy'], 's-', label='PCA Baseline', linewidth=2.5)
plt.axhline(y=v2[v2['config_name']=='BASELINE_FULL_DATA']['accuracy'].values[0],
            color='red', linestyle='--', label='Baseline (204 bands)')
plt.xlabel('Number of Bands Selected', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
plt.title('Autoencoder vs PCA Baseline Comparison', fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison_autoencoder_vs_pca.png', dpi=300)
plt.show()
```

## Expected Output

### Console Output Example
```
================================================================================
WAVELENGTH SELECTION - PCA BASELINE COMPARISON
================================================================================
  Working directory: C:\Users\meloy\PycharmProjects\Capstone
  Results directory: pca_baseline_results/20251029_143022
  Method: PCA-based selection (no autoencoder)

...

Configuration 1/42: pca_loadings_3bands
================================================================================
Running PCA-based wavelength selection (method: pca_loadings)
  Target bands: 3
  Data matrix shape: (241600, 192)
  Total wavelength combinations: 192
  Selected 3 unique wavelength combinations

[RESULTS - pca_loadings_3bands]
  Accuracy: 0.7845
  F1 (weighted): 0.7621
  Data reduction: 98.4%
  Speedup: 1.23x

...

Top configurations by accuracy:
                         config_name  accuracy  f1_weighted  n_features
0              BASELINE_FULL_DATA      0.8554       0.8586         192
1      pca_loadings_140bands         0.8521       0.8548         140
2      pca_loadings_130bands         0.8515       0.8541         130
...
```

## Key Files to Check

### 1. Excel Summary
`pca_baseline_results/TIMESTAMP/wavelength_selection_results_pca_baseline.xlsx`
- Compare accuracy values directly

### 2. Object Metrics
`pca_baseline_results/TIMESTAMP/analysis_summary/all_object_metrics_across_configs.csv`
- Same format as V2-2 object analysis

### 3. Visualizations
`pca_baseline_results/TIMESTAMP/paper-results/`
- ROI overlays for each configuration
- Classification maps
- Ready for presentation

## Troubleshooting

### Error: Module not found
**Fix**: Make sure you're in the correct directory
```bash
cd C:\Users\meloy\PycharmProjects\Capstone\wavelength_analysis
```

### Error: CUDA out of memory
**Fix**: This script doesn't use CUDA/autoencoder, so this shouldn't happen
If it does, it's from clustering - reduce max_configs

### Script runs but no output
**Check**: Look in `pca_baseline_results/` for timestamped folder
The script prints the directory at the start

## Interpretation Guide

### If PCA Accuracy is Close to V2-2 (within 1%)
**Meaning**: Simple PCA is competitive
**Presentation Angle**:
- Emphasize other V2-2 advantages (stability, object-level, interpretability)
- Show V2-2 excels at extreme compression (<10 bands)
- Discuss computational efficiency trade-offs

### If V2-2 Significantly Better (>2% accuracy gain)
**Meaning**: Autoencoder complexity is justified
**Presentation Angle**:
- "PCA achieves X%, our method achieves Y% - Z% improvement"
- Show the value of learned representations
- Emphasize consistent superiority across band counts

### If Mixed Results
**Meaning**: Method choice depends on constraints
**Presentation Angle**:
- Show which method wins at which band counts
- Discuss application-specific recommendations
- Demonstrate thorough evaluation

## Next Steps

1. ✅ Run PCA baseline script
2. ✅ Check Excel output
3. ✅ Compare with V2-2 results
4. ✅ Create comparison visualizations
5. ✅ Analyze object-level differences
6. ✅ Add comparison slide to presentation
7. ✅ Prepare talking points for Q&A

## For Your Presentation

### New Slide: "Baseline Comparison"

**Title**: "Comparison with Simple PCA Selection"

**Content**:
- Plot showing both methods
- Table with key comparisons
- Discussion of results

**Talking Points**:
- "We validated our approach against direct PCA selection"
- "Our method achieves [X] vs PCA's [Y]"
- "This demonstrates [why complexity is/isn't needed]"

## Time Budget

- Test run (10 configs): 30 minutes
- Full run: 2-3 hours
- Analysis: 1 hour
- Presentation integration: 1 hour

**Total**: Half day of work for major presentation boost!

## Questions This Answers for Reviewers

1. ✅ "Why not just use PCA?"
2. ✅ "Is the autoencoder necessary?"
3. ✅ "What about simpler methods?"
4. ✅ "How does this compare to standard approaches?"

**Shows scientific rigor and thoroughness!**

---

**Ready to run?**
```bash
python wavelength_selection_pca_baseline.py pca_loadings 10
```
