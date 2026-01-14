# Wavelength Combinations vs Metrics Visualizations

## Feature Summary
Enhanced the V2 pipeline with comprehensive visualizations showing the relationship between the number of wavelength combinations selected and all supervised learning metrics (accuracy, precision, recall, F1, Cohen's Kappa, purity). This replaces and extends the original single "Combinations vs Purity" plot.

## What Was Added

### 1. **Multi-Metric Comparison Plot** (`plot_combinations_vs_metrics`)
- 6-panel visualization showing all key metrics
- Each panel shows:
  - Scatter plot of metric vs combinations
  - Polynomial trend line
  - Best performing point highlighted with red star
  - Correlation coefficient displayed
- Metrics included: Accuracy, Precision, Recall, F1, Cohen's Kappa, Purity

### 2. **Metrics Progression Plot** (`plot_metrics_progression`)
- Single plot tracking multiple metrics simultaneously
- Features:
  - Primary metric on left y-axis
  - Data reduction percentage on right y-axis
  - Multiple metrics with different markers/colors
  - Best configuration annotated
  - Shows performance vs efficiency trade-off

### 3. **Pareto Frontier Analysis** (`plot_pareto_frontier`)
- Identifies optimal trade-offs between performance and complexity
- Features:
  - Pareto optimal points highlighted in red
  - Pareto frontier line connecting optimal solutions
  - Best performance and minimal complexity points annotated
  - Color-coded by performance level
- Generated for both Accuracy and F1 score

### 4. **Individual Metric Plots**
- Separate, publication-ready plots for each metric
- Clean design for paper inclusion
- Features:
  - Large, clear scatter points
  - Trend line visualization
  - Best and baseline points highlighted
  - Statistical information (correlation, sample size)

### 5. **Metrics Correlation Matrix**
- Shows relationships between all metrics
- Diagonal: Histogram of each metric's distribution
- Off-diagonal: Scatter plots showing metric correlations
- Correlation coefficients displayed

## Output Structure

```
validation_results_v2/[timestamp]/
└── summary_visualizations/
    ├── combinations_vs_all_metrics.png       # 6-panel comprehensive view
    ├── metrics_progression.png               # Multi-metric line plot
    ├── pareto_frontier_accuracy.png          # Pareto analysis for accuracy
    ├── pareto_frontier_f1.png               # Pareto analysis for F1
    ├── combinations_vs_accuracy.png          # Individual plot
    ├── combinations_vs_precision_weighted.png
    ├── combinations_vs_recall_weighted.png
    ├── combinations_vs_f1_weighted.png
    ├── combinations_vs_cohen_kappa.png
    ├── combinations_vs_purity.png
    └── metrics_correlation_matrix.png        # Metric relationships
```

## Usage

### Automatic Generation in V2 Pipeline
```python
# Automatically generated when running:
python wavelengthselectionV2.py

# After all experiments complete, generates:
# 1. Multi-metric comparison
# 2. Metrics progression
# 3. Pareto frontier analysis
# 4. Individual metric plots
# 5. Correlation matrix
```

### Standalone Usage
```python
from supervised_visualizations import SupervisedVisualizations
import pandas as pd

# Load results DataFrame
df_results = pd.read_csv("results.csv")

# Create visualizer
viz = SupervisedVisualizations(output_dir=output_path, dpi=300)

# Generate all combination vs metrics plots
viz.plot_combinations_vs_metrics(
    df_results,
    metrics_to_plot=['accuracy', 'precision_weighted', 'recall_weighted',
                    'f1_weighted', 'cohen_kappa', 'purity']
)

# Generate metrics progression
viz.plot_metrics_progression(
    df_results,
    primary_metric='accuracy',
    secondary_metrics=['f1_weighted', 'precision_weighted']
)

# Generate Pareto frontier
viz.plot_pareto_frontier(
    df_results,
    performance_metric='accuracy',
    complexity_metric='n_combinations_selected'
)
```

## Key Features

### Visual Elements
- **Trend Lines**: Polynomial fit showing general relationship
- **Best Points**: Red stars marking optimal configurations
- **Baseline Comparison**: Blue squares showing full data performance
- **Correlation Values**: Statistical correlation displayed
- **Color Coding**: Performance levels shown through color gradients

### Statistical Information
- **Correlation Coefficients**: Linear correlation with combinations
- **Sample Size**: Number of experiments
- **Best Performance**: Highlighted and annotated
- **Pareto Optimality**: Non-dominated solutions identified

## Benefits for Research

1. **Comprehensive Analysis**: See all metrics at once, not just purity
2. **Trade-off Visualization**: Understand performance vs complexity
3. **Statistical Validation**: Correlation analysis included
4. **Publication Ready**: Individual plots for papers
5. **Decision Support**: Pareto frontier helps choose optimal configuration

## Interpretation Guide

### Multi-Metric Plot
- **Positive Correlation**: Metric improves with more combinations
- **Plateau Effect**: Performance saturates at certain point
- **Optimal Range**: Where performance/complexity trade-off is best

### Metrics Progression
- **Convergence**: Where metrics stabilize
- **Divergence**: Where metrics behave differently
- **Sweet Spot**: Best balance of all metrics

### Pareto Frontier
- **Red Line**: Non-dominated solutions (no better trade-offs exist)
- **Red Diamonds**: Pareto optimal configurations
- **Off-frontier**: Suboptimal configurations

## Paper Usage

For publications, recommended figures:
1. **Individual metric plots** for specific metrics of interest
2. **Pareto frontier** to show optimization approach
3. **Metrics progression** for comprehensive performance view
4. **Multi-metric panel** for supplementary materials

## Testing

Demo scripts provided:
```bash
# Run comprehensive demo
python demo_combinations_metrics.py

# Creates sample visualizations showing all features
```

## Comparison with Original Pipeline

| Feature | Original (V1) | Enhanced (V2) |
|---------|--------------|---------------|
| Metrics Plotted | Purity only | All supervised metrics |
| Plot Types | Single scatter | Multiple visualization types |
| Trade-off Analysis | None | Pareto frontier |
| Correlation Analysis | None | Full correlation matrix |
| Trend Visualization | None | Polynomial trend lines |
| Best Point Marking | None | Highlighted with annotation |
| Publication Plots | Limited | Individual plots for each metric |

## Key Insights Enabled

1. **Performance Saturation**: Identify where adding more wavelengths stops helping
2. **Metric Agreement**: See which metrics move together
3. **Optimal Configuration**: Find best trade-off using Pareto analysis
4. **Statistical Significance**: Correlation analysis validates trends
5. **Comprehensive View**: All metrics visible simultaneously

## Future Enhancements

Potential additions:
- Confidence intervals on trend lines
- Interactive HTML plots with Plotly
- 3D visualization for three metrics
- Animation showing progression
- Bootstrap analysis for uncertainty