# Configuration Parameters - Quick Reference Guide

## Quick Configuration Template

```python
config = {
    'name': 'experiment_name',
    'dimension_selection_method': 'variance',        # ✓ BEST: variance
    'perturbation_method': 'standard_deviation',     # ✓ BEST: standard_deviation
    'perturbation_magnitudes': [15, 30, 45],        # ✓ BEST: [15,30,45]
    'n_important_dimensions': 7,                     # ✓ BEST: 6-8
    'n_bands_to_select': 10,                        # ✓ BEST: 7-11
    'normalization_method': 'max_per_excitation',   # ✓ BEST: max_per_excitation
    'use_diversity_constraint': True,                # ✓ BEST: True
    'diversity_method': 'mmr',                       # ✓ BEST: mmr
    'lambda_diversity': 0.5                          # ✓ BEST: 0.5
}
```

---

## Parameter Reference Table

| Parameter | Type | Options | Best Value | Impact |
|-----------|------|---------|------------|--------|
| `dimension_selection_method` | str | variance, activation, pca | **variance** | HIGH - Most important parameter |
| `perturbation_method` | str | percentile, standard_deviation, absolute_range | **standard_deviation** | MEDIUM |
| `perturbation_magnitudes` | list | [small, medium, large] | **[15, 30, 45]** | MEDIUM |
| `n_important_dimensions` | int | 5-20 | **6-8** | MEDIUM |
| `n_bands_to_select` | int | 5-30 | **7-11** | HIGH |
| `normalization_method` | str | variance, max_per_excitation, none | **max_per_excitation** | MEDIUM |
| `use_diversity_constraint` | bool | True, False | **True** | HIGH |
| `diversity_method` | str | mmr, min_distance, none | **mmr** | HIGH |
| `lambda_diversity` | float | 0.0-1.0 | **0.5** | MEDIUM |

---

## Quick Decision Tree

### 1. Choose Dimension Selection Method

```
Do you want discriminative features for clustering?
  YES → use 'variance' ✓ (Purity: 0.866+)
  NO  → Consider 'activation' (Purity: 0.814)

Do you have highly correlated features?
  YES → Try 'pca' (Purity: 0.784)
  NO  → Stick with 'variance' ✓
```

### 2. Choose Number of Bands

```
What is your priority?

  Maximum efficiency (fastest):
    n_bands_to_select = 7
    Data reduction: 93.7%
    Purity: 0.860
    Speedup: 3.1×

  Balanced (recommended):
    n_bands_to_select = 10 ✓
    Data reduction: 91.0%
    Purity: 0.868
    Speedup: 2.24×

  Maximum quality:
    n_bands_to_select = 11
    Data reduction: 90.1%
    Purity: 0.868
    Speedup: 2.1×
```

### 3. Choose Perturbation Settings

```
What type of latent distribution do you have?

  Approximately Gaussian:
    perturbation_method = 'standard_deviation' ✓
    perturbation_magnitudes = [15, 30, 45]

  Non-Gaussian / Unknown:
    perturbation_method = 'percentile'
    perturbation_magnitudes = [10, 20, 35]

  Want aggressive exploration:
    perturbation_method = 'absolute_range'
    perturbation_magnitudes = [20, 40, 60]
```

### 4. Choose Diversity Settings

```
Do you want to avoid redundant wavelengths?
  YES → use_diversity_constraint = True ✓

Which diversity method?

  Best results (recommended):
    diversity_method = 'mmr'
    lambda_diversity = 0.5 ✓

  Simple distance constraint:
    diversity_method = 'min_distance'
    min_distance_nm = 15.0

  Focus on influence:
    diversity_method = 'mmr'
    lambda_diversity = 0.3

  Focus on diversity:
    diversity_method = 'mmr'
    lambda_diversity = 0.7
```

---

## Pre-configured Profiles

### Profile 1: Best Overall (Recommended)

**Use for:** Highest clustering quality with good efficiency

```python
{
    'name': 'best_overall',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [15, 30, 45],
    'n_important_dimensions': 7,
    'n_bands_to_select': 10,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.5
}
```
**Expected:** Purity 0.868, 91% reduction, 2.24× speedup

---

### Profile 2: Maximum Efficiency

**Use for:** When speed is critical, acceptable quality

```python
{
    'name': 'max_efficiency',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [20, 40, 60],
    'n_important_dimensions': 6,
    'n_bands_to_select': 7,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.5
}
```
**Expected:** Purity 0.860, 93.7% reduction, 3.1× speedup

---

### Profile 3: Conservative

**Use for:** When you want to be safe, minimize risk

```python
{
    'name': 'conservative',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [12, 25, 40],
    'n_important_dimensions': 8,
    'n_bands_to_select': 11,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.5
}
```
**Expected:** Purity 0.868, 90% reduction, 2.1× speedup

---

### Profile 4: Influence-Focused

**Use for:** When you want highest-influence wavelengths

```python
{
    'name': 'influence_focused',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [15, 30, 45],
    'n_important_dimensions': 7,
    'n_bands_to_select': 10,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.3  # Lower = more influence weight
}
```
**Expected:** Purity 0.868, 91% reduction, high influence scores

---

### Profile 5: Diversity-Focused

**Use for:** When you want maximum spectral coverage

```python
{
    'name': 'diversity_focused',
    'dimension_selection_method': 'variance',
    'perturbation_method': 'standard_deviation',
    'perturbation_magnitudes': [15, 30, 45],
    'n_important_dimensions': 7,
    'n_bands_to_select': 10,
    'normalization_method': 'max_per_excitation',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.7  # Higher = more diversity weight
}
```
**Expected:** Purity 0.867, 91% reduction, wide spectral spread

---

### Profile 6: Experimental PCA

**Use for:** Testing PCA-based selection (typically underperforms)

```python
{
    'name': 'experimental_pca',
    'dimension_selection_method': 'pca',
    'perturbation_method': 'absolute_range',
    'perturbation_magnitudes': [20, 40, 60],
    'n_important_dimensions': 8,
    'n_bands_to_select': 10,
    'normalization_method': 'variance',
    'use_diversity_constraint': True,
    'diversity_method': 'mmr',
    'lambda_diversity': 0.5
}
```
**Expected:** Purity 0.784 (significantly lower)

---

## Parameter Tuning Guide

### If purity is too low (<0.85):

1. **Check dimension selection method**
   - Switch from 'activation' or 'pca' → 'variance'

2. **Enable diversity constraint**
   - Set use_diversity_constraint = True
   - Use diversity_method = 'mmr'

3. **Increase number of bands**
   - Try n_bands_to_select = 10-11

4. **Check normalization**
   - Use 'max_per_excitation' instead of 'variance' or 'none'

### If execution is too slow:

1. **Reduce important dimensions**
   - Decrease n_important_dimensions to 6-7

2. **Reduce perturbation magnitudes**
   - Use only 3 values: [15, 30, 45]

3. **Simplify diversity method**
   - Use 'min_distance' instead of 'mmr'

### If wavelengths are too clustered:

1. **Increase diversity weight**
   - Increase lambda_diversity from 0.5 → 0.7

2. **Use larger min distance**
   - Increase min_distance_nm from 15 → 25

3. **Select more bands**
   - Increase n_bands_to_select to spread selection

### If wavelengths miss important features:

1. **Decrease diversity weight**
   - Decrease lambda_diversity from 0.5 → 0.3

2. **Increase important dimensions**
   - Increase n_important_dimensions to 8-10

3. **Check perturbation magnitudes**
   - Ensure adequate range: [15, 30, 45]

---

## Common Mistakes to Avoid

❌ **Don't use 'activation' or 'pca' for clustering**
   → These underperform significantly (0.78-0.81 vs 0.87)

❌ **Don't disable diversity constraints**
   → Will select redundant wavelengths

❌ **Don't use lambda_diversity = 1.0**
   → Same as no diversity, defeats the purpose

❌ **Don't select too few bands (<6)**
   → Quality drops significantly

❌ **Don't select too many bands (>15)**
   → Diminishing returns, wastes computation

❌ **Don't use 'none' normalization**
   → Biases toward noisy or high-magnitude bands

---

## Validation Checklist

After running your configuration, verify:

✓ **Purity ≥ 0.85**
   → If lower, check dimension method and diversity settings

✓ **Data reduction ≥ 70%**
   → If lower, reduce n_bands_to_select

✓ **Selected wavelengths are diverse**
   → Check that they span different spectral regions

✓ **Execution time is reasonable**
   → Should complete in <20 minutes for typical data

✓ **Wavelength combinations make sense**
   → Check excitation-emission pairs are physically reasonable

---

## Advanced Tuning

### Fine-tuning lambda_diversity:

```
λ = 0.1-0.2: Extreme influence focus (may cluster)
λ = 0.3:     Influence-focused (good for high-importance bands)
λ = 0.5:     Balanced ✓ (recommended starting point)
λ = 0.7:     Diversity-focused (good for coverage)
λ = 0.8-0.9: Extreme diversity (may miss important bands)
```

### Fine-tuning n_important_dimensions:

```
n_dims = 5:   Fast but may miss features
n_dims = 6-7: Optimal efficiency ✓
n_dims = 8-9: Balanced ✓
n_dims = 10+: Comprehensive but slower
```

### Fine-tuning perturbation_magnitudes:

```
Conservative:  [10, 20, 30]
Balanced:      [15, 30, 45] ✓
Aggressive:    [20, 40, 60]
Very aggressive: [25, 50, 75]
```

---

## Example Usage

```python
from wavelength_analysis.core.config import AnalysisConfig
from wavelength_analysis.core.analyzer import WavelengthAnalyzer

# Create configuration
config = AnalysisConfig(
    sample_name="Lichens",
    data_path="path/to/data.pkl",
    mask_path="path/to/mask.npy",
    model_path="path/to/model.pth",

    # Use best overall profile
    dimension_selection_method='variance',
    perturbation_method='standard_deviation',
    perturbation_magnitudes=[15, 30, 45],
    n_important_dimensions=7,
    n_bands_to_select=10,
    normalization_method='max_per_excitation',
    use_diversity_constraint=True,
    diversity_method='mmr',
    lambda_diversity=0.5,

    output_dir="results/lichens_wavelength_selection"
)

# Run analysis
analyzer = WavelengthAnalyzer(config)
results = analyzer.run_complete_analysis()

# Check results
print(f"Selected {len(results['selected_bands'])} wavelengths")
print(f"Compression ratio: {results['performance_metrics']['compression_ratio']:.1f}x")
```

---

## Results Interpretation

### High-Quality Results:
- Purity: 0.865-0.870
- ARI: 0.78-0.82
- NMI: 0.80-0.85
- Data reduction: 90-93%

### Acceptable Results:
- Purity: 0.850-0.865
- ARI: 0.75-0.78
- NMI: 0.77-0.80
- Data reduction: 85-90%

### Poor Results (needs tuning):
- Purity: <0.850
- ARI: <0.75
- NMI: <0.77

If you get poor results, revisit:
1. dimension_selection_method (should be 'variance')
2. use_diversity_constraint (should be True)
3. n_bands_to_select (try 8-11)
4. normalization_method (try 'max_per_excitation')
