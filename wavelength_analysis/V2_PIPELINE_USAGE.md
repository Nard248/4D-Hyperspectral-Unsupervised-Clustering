# V2 Pipeline Usage Guide

## Configuration Limit Fix

### The Issue
The V2 pipeline was originally limited to running only the first 3 configurations for testing purposes:
```python
# BEFORE:
for i, config in enumerate(tqdm(configurations[:3], desc="Testing configurations")):  # Test first 3
```

### The Fix
Now runs ALL configurations by default, with optional control:
```python
# AFTER:
for i, config in enumerate(tqdm(configs_to_run, desc="Running configurations")):  # Run selected configurations
```

## Usage Options

### 1. Run ALL Configurations (43 total)
```bash
python wavelengthselectionV2.py
```
- Runs all 43 configurations
- Complete analysis for your paper
- Takes longer but provides comprehensive results

### 2. Run Limited Configurations (for testing)
```bash
# Run first 5 configurations
python wavelengthselectionV2.py 5

# Run first 10 configurations
python wavelengthselectionV2.py 10
```

### 3. Quick Test Run
```bash
# Run just 3 for quick testing
python wavelengthselectionV2.py 3
```

## Configuration Details

- **Total Available**: 43 configurations
- **Range**: 3 to 170 wavelength bands
- **Each configuration tests**: Different wavelength selection parameters

## Time Estimates

Based on typical performance:
- **3 configurations**: ~5-10 minutes (testing)
- **10 configurations**: ~20-30 minutes (quick analysis)
- **43 configurations**: ~1.5-2 hours (full analysis)

## What Happens in Full Run

When running all 43 configurations, the pipeline:

1. **Tests each configuration**:
   - Runs wavelength selection
   - Extracts data subset
   - Performs clustering
   - Calculates supervised metrics
   - Creates ROI overlay visualizations

2. **Generates summary visualizations**:
   - Combinations vs all metrics (6-panel)
   - Metrics progression plot
   - Pareto frontier analysis
   - Individual metric plots
   - Correlation matrix

3. **Saves comprehensive results**:
   - Excel file with all metrics
   - Individual visualizations per config
   - Summary visualizations
   - Paper-ready figures

## Output Structure

```
validation_results_v2/[timestamp]/
├── supervised_visualizations/      # Baseline visualizations
├── paper-results/                 # ROI overlays for paper
│   ├── BASELINE_roi_overlay.png
│   ├── Config_1_roi_overlay.png
│   └── ... (43 total)
├── experiments/                   # Per-configuration results
│   ├── Config_1/
│   ├── Config_2/
│   └── ... (43 folders)
├── summary_visualizations/        # Overall analysis
│   ├── combinations_vs_all_metrics.png
│   ├── metrics_progression.png
│   ├── pareto_frontier_accuracy.png
│   └── ... (12+ summary plots)
├── supervised_metrics/             # Metric files
└── wavelength_selection_results_v2.xlsx  # All results
```

## Recommendations

### For Paper Results
Run the full pipeline:
```bash
python wavelengthselectionV2.py
```
This ensures you have complete data for:
- Comprehensive performance analysis
- Pareto frontier identification
- Statistical validation
- All publication figures

### For Development/Testing
Use limited runs:
```bash
python wavelengthselectionV2.py 5
```
This allows you to:
- Test code changes quickly
- Verify pipeline functionality
- Debug issues

### For Quick Analysis
Medium run:
```bash
python wavelengthselectionV2.py 15
```
Provides:
- Reasonable sample size
- Good trend visualization
- Faster turnaround

## Memory Considerations

- Each configuration processes ~1GB of hyperspectral data
- Multiple visualizations are generated
- Recommended: 16GB+ RAM for full run
- Close other applications during full run

## Monitoring Progress

The pipeline provides detailed progress information:
- Progress bar for configurations
- Individual timing for each step
- Running accuracy metrics
- Memory usage (if issues arise)

## Troubleshooting

If the pipeline crashes:
1. Check available memory
2. Try running fewer configurations first
3. Review error messages in console
4. Check intermediate results in experiments/ folder

## Key Metrics to Watch

During execution, monitor:
- **Accuracy**: Should generally increase with more wavelengths
- **Data Reduction %**: Should decrease with more wavelengths
- **Processing Time**: Should decrease with fewer wavelengths
- **F1 Score**: Balanced performance metric

## After Completion

Once finished:
1. Check `summary_visualizations/` for overall trends
2. Review `wavelength_selection_results_v2.xlsx` for detailed metrics
3. Use `paper-results/` for publication figures
4. Identify Pareto optimal configurations for final selection