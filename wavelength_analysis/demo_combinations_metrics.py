"""
Demo: Wavelength Combinations vs Metrics Visualizations
========================================================
Demonstrates the new visualization capabilities for analyzing the relationship
between number of wavelength combinations and various performance metrics.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

from supervised_visualizations import SupervisedVisualizations


def create_sample_results():
    """Create sample results DataFrame for demonstration."""

    # Simulate results for different numbers of wavelength combinations
    np.random.seed(42)

    n_configs = 20
    combinations = np.linspace(10, 180, n_configs).astype(int)

    # Simulate metrics that generally improve with more combinations but with diminishing returns
    results = []

    for i, n_comb in enumerate(combinations):
        # Simulate realistic metric values
        base_performance = 0.6 + 0.3 * (1 - np.exp(-n_comb / 50))
        noise = np.random.normal(0, 0.02)

        result = {
            'config_name': f'Config_{n_comb}',
            'n_combinations_selected': n_comb,
            'accuracy': np.clip(base_performance + noise, 0, 1),
            'precision_weighted': np.clip(base_performance + noise * 0.8, 0, 1),
            'recall_weighted': np.clip(base_performance + noise * 1.2, 0, 1),
            'f1_weighted': np.clip(base_performance + noise * 0.9, 0, 1),
            'cohen_kappa': np.clip(base_performance * 0.95 + noise, 0, 1),
            'purity': np.clip(base_performance * 1.02 + noise * 0.7, 0, 1),
            'data_reduction_pct': 100 * (1 - n_comb / 180)
        }
        results.append(result)

    # Add baseline with all combinations
    results.append({
        'config_name': 'BASELINE_FULL_DATA',
        'n_combinations_selected': 180,
        'accuracy': 0.89,
        'precision_weighted': 0.88,
        'recall_weighted': 0.89,
        'f1_weighted': 0.88,
        'cohen_kappa': 0.86,
        'purity': 0.90,
        'data_reduction_pct': 0
    })

    return pd.DataFrame(results)


def demonstrate_visualizations():
    """Demonstrate all visualization types."""

    print("=" * 80)
    print("WAVELENGTH COMBINATIONS VS METRICS VISUALIZATION DEMO")
    print("=" * 80)

    # Create sample data
    print("\n1. Creating sample results data...")
    df_results = create_sample_results()
    print(f"   Generated {len(df_results)} configurations")
    print(f"   Combination range: {df_results['n_combinations_selected'].min()} - {df_results['n_combinations_selected'].max()}")

    # Create output directory
    output_dir = Path("demo_combinations_output")
    output_dir.mkdir(exist_ok=True)

    # Initialize visualizer
    viz = SupervisedVisualizations(output_dir=output_dir, dpi=150)

    # 1. Multi-metric plot (6 panels)
    print("\n2. Creating multi-metric visualization (6 panels)...")
    viz.plot_combinations_vs_metrics(
        df_results,
        metrics_to_plot=['accuracy', 'precision_weighted', 'recall_weighted',
                        'f1_weighted', 'cohen_kappa', 'purity'],
        save_name="demo_combinations_vs_all_metrics.png"
    )
    print("   Shows: Scatter plots with trend lines for each metric")
    print("   Features: Best point marked, correlation coefficient, trend line")

    # 2. Metrics progression plot
    print("\n3. Creating metrics progression plot...")
    viz.plot_metrics_progression(
        df_results,
        primary_metric='accuracy',
        secondary_metrics=['f1_weighted', 'precision_weighted', 'recall_weighted'],
        save_name="demo_metrics_progression.png"
    )
    print("   Shows: Line plot of multiple metrics vs combinations")
    print("   Features: Dual y-axis (metrics + data reduction), best point annotation")

    # 3. Pareto frontier
    print("\n4. Creating Pareto frontier analysis...")
    viz.plot_pareto_frontier(
        df_results,
        performance_metric='accuracy',
        complexity_metric='n_combinations_selected',
        save_name="demo_pareto_frontier.png"
    )
    print("   Shows: Performance vs complexity trade-off")
    print("   Features: Pareto optimal points highlighted, best performance marked")

    # 4. Individual metric plots
    print("\n5. Creating individual metric plots...")
    metrics = ['accuracy', 'f1_weighted', 'purity']

    for metric in metrics:
        viz.plot_combinations_vs_metrics(
            df_results,
            metrics_to_plot=[metric],
            save_name=f"demo_combinations_vs_{metric}.png"
        )
        print(f"   Created individual plot for {metric}")

    print(f"\n✅ All visualizations saved to: {output_dir}")

    # Print statistics
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS SUMMARY")
    print("=" * 80)

    print("\nTop 5 configurations by accuracy:")
    top_5 = df_results.nlargest(5, 'accuracy')[['config_name', 'n_combinations_selected', 'accuracy', 'f1_weighted']]
    print(top_5.to_string(index=False))

    # Calculate correlations
    print("\nCorrelations with n_combinations_selected:")
    metrics_cols = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'cohen_kappa', 'purity']
    for metric in metrics_cols:
        corr = df_results['n_combinations_selected'].corr(df_results[metric])
        print(f"  {metric:20s}: {corr:+.3f}")

    return df_results


def explain_visualizations():
    """Explain what each visualization shows."""

    print("\n" + "=" * 80)
    print("VISUALIZATION EXPLANATIONS")
    print("=" * 80)

    print("""
1. COMBINATIONS VS ALL METRICS (6-panel plot)
   - Purpose: Compare how different metrics change with wavelength selection
   - Each panel: One metric vs number of combinations
   - Features:
     • Scatter points: Individual experiments
     • Red trend line: Polynomial fit showing general trend
     • Red star: Best performing configuration
     • Correlation value: Linear correlation coefficient

2. METRICS PROGRESSION
   - Purpose: Track multiple metrics simultaneously
   - Left y-axis: Performance metrics (accuracy, F1, etc.)
   - Right y-axis: Data reduction percentage
   - Features:
     • Multiple line plots with different markers
     • Best configuration marked with annotation
     • Shows trade-off between performance and data reduction

3. PARETO FRONTIER
   - Purpose: Identify optimal trade-offs
   - X-axis: Complexity (number of combinations)
   - Y-axis: Performance (accuracy or other metric)
   - Features:
     • Red line: Pareto frontier (non-dominated solutions)
     • Red diamonds: Pareto optimal configurations
     • Annotations: Best performance and minimal complexity points

4. INDIVIDUAL METRIC PLOTS
   - Purpose: Detailed analysis of single metrics
   - Clean plots for paper inclusion
   - Features:
     • Larger markers for better visibility
     • Clear trend visualization
     • Statistical information included

USE CASES:
- Paper figures: Use individual metric plots
- Analysis: Use multi-panel for comprehensive view
- Decision making: Use Pareto frontier to choose optimal configuration
- Presentation: Use progression plot to show improvements
    """)


if __name__ == "__main__":
    # Run demonstration
    df_results = demonstrate_visualizations()
    explain_visualizations()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("""
The V2 pipeline automatically generates all these visualizations after
running all experiments. They will be saved to:

validation_results_v2/[timestamp]/summary_visualizations/
├── combinations_vs_all_metrics.png     # 6-panel overview
├── metrics_progression.png             # Multi-metric progression
├── pareto_frontier_accuracy.png        # Pareto analysis for accuracy
├── pareto_frontier_f1.png             # Pareto analysis for F1
├── combinations_vs_accuracy.png        # Individual plot
├── combinations_vs_precision_weighted.png
├── combinations_vs_recall_weighted.png
├── combinations_vs_f1_weighted.png
├── combinations_vs_cohen_kappa.png
├── combinations_vs_purity.png
└── metrics_correlation_matrix.png      # Correlation analysis

These provide comprehensive analysis of how wavelength selection impacts
all supervised learning metrics!
    """)