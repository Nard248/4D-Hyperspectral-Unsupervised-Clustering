# Robustness Analysis for Wavelength Selection

## Overview

The robustness analysis script evaluates how optimal your autoencoder+perturbation wavelength selection is by testing many (or all) possible combinations of n bands and comparing their performance.

**Purpose**: Prove that the autoencoder method selects near-optimal wavelength combinations.

## How It Works

The script:
1. Takes a number `n` (e.g., n=13 bands to select)
2. Tests combinations of n bands from the 192 available bands
3. For each combination:
   - Extracts those specific wavelength bands from the hyperspectral data
   - Runs KMeans clustering (4 clusters, same as V2-2)
   - Computes supervised metrics (accuracy, F1, precision, recall, Cohen's kappa)
   - Stores the results
4. Generates statistics and visualizations showing:
   - Distribution of accuracies across all tested combinations
   - Where the autoencoder-selected combination ranks
   - Top-performing combinations

## Computational Complexity

The number of possible combinations grows exponentially:

| n bands | Total combinations C(192,n) | Strategy |
|---------|------------------------------|----------|
| 3       | 1,161,280                    | Random sampling |
| 4       | 74,364,290                   | Random sampling |
| 5       | 2,536,766,080                | Random sampling |
| 10      | 8.28 × 10^14                 | Random sampling |
| 13      | 1.5 × 10^18                  | Random sampling |

**Solution**:
- For small n (≤6): Can test all combinations (takes hours/days)
- For larger n (>6): Must use random sampling (e.g., 10,000-50,000 combinations)

## Usage

### Basic Command

```bash
python robustness_analysis.py <n_bands> [max_combinations]
```

### Examples

```bash
# Test 1000 random combinations of 3 bands (quick test)
python robustness_analysis.py 3 1000

# Test 5000 random combinations of 13 bands (recommended for paper)
python robustness_analysis.py 13 5000

# Test 10000 random combinations of 13 bands (more thorough)
python robustness_analysis.py 13 10000

# Test ALL possible combinations of 3 bands (will take ~15 hours)
python robustness_analysis.py 3

# Custom output directory
python robustness_analysis.py 13 10000 --output my_results/robustness
```

### Recommended Test Configurations

For paper/thesis:

| Goal | Command | Time Estimate |
|------|---------|---------------|
| Quick proof of concept | `python robustness_analysis.py 3 5000` | ~1.5 hours |
| Medium robustness test | `python robustness_analysis.py 13 5000` | ~2 hours |
| **Recommended for paper** | `python robustness_analysis.py 13 10000` | ~3-4 hours |
| Thorough analysis | `python robustness_analysis.py 13 50000` | ~18 hours |

## Output Files

All results are saved to `validation_results_v2/robustness/`:

### Main Files

1. **`robustness_{n}bands_results.csv`**
   - Complete results for all tested combinations
   - Columns: combo_id, accuracy, f1_weighted, precision_weighted, recall_weighted, cohen_kappa, n_bands, band_indices

2. **`robustness_{n}bands_summary.csv`**
   - Summary statistics
   - Includes: mean, std, min, max, median accuracies
   - Autoencoder performance and percentile rank

3. **`robustness_{n}bands_analysis.png`**
   - 4-panel visualization:
     - Histogram of accuracies
     - Cumulative distribution function
     - Box plot
     - Top 20 combinations

4. **`robustness_{n}bands_temp.csv`**
   - Intermediate results (saved every 1000 combinations)
   - Allows recovery if script crashes

## Interpreting Results

### Statistical Output

When the script completes, you'll see output like:

```
================================================================================
STATISTICAL SUMMARY
================================================================================

Accuracy Statistics (10,000 combinations tested):
  Mean:   0.7823
  Std:    0.0542
  Min:    0.6102
  Max:    0.8789
  Median: 0.7856

Percentiles:
  1th:  0.6543
  5th:  0.6891
  10th: 0.7123
  25th: 0.7512
  50th: 0.7856
  75th: 0.8234
  90th: 0.8567
  95th: 0.8689
  99th: 0.8756

Autoencoder Selection Accuracy: 0.8789
Percentile Rank: 99.85th percentile
Combinations better than autoencoder: 15 (0.15%)
```

### What This Means

- **Percentile Rank**: If autoencoder is at 99.85th percentile, it's better than 99.85% of all random combinations
- **Combinations Better**: Only 15 out of 10,000 random combinations performed better
- **Conclusion**: The autoencoder method is highly effective at selecting informative wavelength combinations

### For Your Paper/Thesis

You can write:

> "To validate the robustness of our wavelength selection method, we tested 10,000 random combinations of 13 wavelength bands from the 192 available bands. Our autoencoder+perturbation method achieved an accuracy of 0.8789, ranking in the **99.85th percentile** of all tested combinations. Only 15 combinations (0.15%) achieved higher accuracy, demonstrating that our method effectively identifies near-optimal wavelength subsets."

## Visualization Plots

### Plot 1: Histogram
- Shows the distribution of accuracies across all tested combinations
- Red dashed line: mean accuracy
- Green dashed line: median accuracy
- Purple solid line: autoencoder selection (if available)

### Plot 2: Cumulative Distribution
- Shows what percentage of combinations achieve each accuracy level
- Purple line: marks where the autoencoder selection falls

### Plot 3: Box Plot
- Shows quartiles, outliers, and overall distribution
- Purple line: autoencoder selection

### Plot 4: Top 20 Combinations
- Bar chart of the best-performing combinations
- Purple bar: highlights autoencoder selection if it's in top 20

## Performance Tips

1. **Start small**: Test with 1000 combinations first to verify everything works

2. **Use background execution** for long runs:
   ```bash
   nohup python robustness_analysis.py 13 10000 > robustness.log 2>&1 &
   ```

3. **Monitor progress**: The script saves temp results every 1000 combinations

4. **Time estimation**: Each combination takes ~1.1-1.3 seconds
   - 1000 combinations: ~20 minutes
   - 5000 combinations: ~2 hours
   - 10000 combinations: ~3-4 hours
   - 50000 combinations: ~18 hours

## Troubleshooting

### Error: FileNotFoundError
- Check that data files exist:
  - `data/processed/Lichens/lichens_data_masked.pkl`
  - `C:\Users\meloy\Downloads\Mask_Manual.png`

### Error: Memory Error
- Reduce `max_combinations` to a smaller number
- Close other applications to free up RAM

### Script Crashes Mid-Run
- Check the temp file: `robustness_{n}bands_temp.csv`
- You can see how many combinations were completed before the crash

## Advanced Usage

### Compare Multiple n Values

To test robustness across different band counts:

```bash
# Test n=3, 5, 7, 10, 13 bands
python robustness_analysis.py 3 5000
python robustness_analysis.py 5 5000
python robustness_analysis.py 7 5000
python robustness_analysis.py 10 5000
python robustness_analysis.py 13 5000
```

Then create a plot showing how the autoencoder's percentile rank varies with n.

### Extract Best Combinations

To see which wavelength bands appear most frequently in top combinations:

```python
import pandas as pd

# Load results
df = pd.read_csv('validation_results_v2/robustness/robustness_13bands_results.csv')

# Get top 100 combinations
top_100 = df.nlargest(100, 'accuracy')

# Extract band indices (will need custom analysis)
# band_indices column contains tuples of selected band indices
```

## Integration with Existing Results

The script automatically looks for autoencoder results in:
- `validation_results_v2/1Dimensions/wavelength_selection_results_v2.xlsx`

If found, it will:
- Display the autoencoder's accuracy for that n value
- Calculate its percentile rank
- Count how many combinations performed better
- Highlight it in visualizations

## Notes

- **Random seed**: Set to 42 for reproducibility
- **Clustering**: Uses same KMeans parameters as V2-2 (n_clusters=4, random_state=42)
- **Metrics**: Computes same supervised metrics as V2-2
- **Ground truth**: Uses same ROI mapping (0→0, 1→1, 2→2, 5→3)

## Citation

If you use this robustness analysis in your paper, consider including:

> We validated our wavelength selection method through robustness analysis, testing [N] random combinations of [n] bands and demonstrating that our method achieves performance in the top [X]% of all tested combinations.

---

**Script**: `robustness_analysis.py`
**Version**: 1.0
**Last Updated**: 2025-11-07
