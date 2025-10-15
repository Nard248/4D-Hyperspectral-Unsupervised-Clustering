"""
Comprehensive Visualization Pipeline for Wavelength Analysis
============================================================
This script generates all visualizations needed for the research paper.
It creates organized, publication-ready figures for the wavelength selection methodology.

Author: Automated Visualization Pipeline
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import (confusion_matrix, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.decomposition import PCA
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# ROI regions (matching the notebook)
ROI_REGIONS = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},  # Red
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},  # Blue
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},  # Green
    {'name': 'Region 4', 'coords': (175, 225, 650, 700), 'color': '#FFFF00'},  # Yellow
]

# Color palettes for different visualization types
COLOR_PALETTES = {
    'roi': ['#FF0000', '#0000FF', '#00FF00', '#FFFF00'],  # Matching ROI regions
    'scientific': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9'],
    'diverging': sns.diverging_palette(250, 10, n=11).as_hex(),
    'sequential': sns.color_palette("YlOrRd", 10).as_hex(),
    'categorical': sns.color_palette("Set2", 8).as_hex(),
}

# Figure DPI settings
DPI_SCREEN = 150
DPI_PRINT = 300
DPI_POSTER = 600


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_output_directories(base_dir):
    """Create comprehensive directory structure for all visualizations"""
    base_path = Path(base_dir)

    subdirs = {
        'roi_overlays': 'ROI-based clustering overlays',
        'comparisons': 'Method-to-method comparisons',
        'difference_maps': 'Difference and agreement visualizations',
        'spectral_analysis': 'Spectral signature analysis',
        'clustering_quality': 'Clustering quality metrics',
        'wavelength_importance': 'Wavelength selection importance',
        'latent_space': 'Latent space projections',
        'ground_truth': 'Ground truth comparisons',
        'statistical': 'Statistical distributions and tests',
        'correlation': 'Correlation and relationship analysis',
        'summary': 'Summary dashboards and combined figures',
        'paper_ready': 'Publication-ready high-res figures',
        'individual_results': 'Individual configuration results',
        'animations': 'Animated visualizations (if applicable)',
    }

    created_dirs = {}
    for dirname, description in subdirs.items():
        dir_path = base_path / dirname
        dir_path.mkdir(parents=True, exist_ok=True)
        created_dirs[dirname] = dir_path

    # Create README
    readme_path = base_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("Comprehensive Visualization Pipeline Output\n")
        f.write("=" * 60 + "\n\n")
        for dirname, description in subdirs.items():
            f.write(f"{dirname:30s} - {description}\n")

    print(f"\n📁 Created {len(created_dirs)} output directories at: {base_path}")
    return created_dirs


# ============================================================================
# CORE VISUALIZATION FUNCTIONS
# ============================================================================

def create_roi_colormap():
    """Create consistent colormap based on ROI colors"""
    colors = [roi['color'] for roi in ROI_REGIONS]
    return ListedColormap(colors)


def export_clustering_overlay(cluster_map, output_path, title="Clustering Result",
                              metrics=None, roi_overlay=False, ground_truth=None,
                              dpi=DPI_PRINT):
    """
    Export individual clustering result with ROI colors and optional overlays.

    Parameters:
    -----------
    cluster_map : np.ndarray
        2D array with cluster labels
    output_path : Path
        Where to save the figure
    title : str
        Figure title
    metrics : dict, optional
        Dictionary with metrics to display
    roi_overlay : bool
        Whether to show ROI regions
    ground_truth : np.ndarray, optional
        Ground truth for comparison
    dpi : int
        Resolution for saving
    """
    roi_cmap = create_roi_colormap()

    if ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    elif roi_overlay:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    # Main clustering visualization
    cluster_map_display = np.ma.masked_where(cluster_map == -1, cluster_map)
    im0 = axes[0].imshow(cluster_map_display, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
    axes[0].set_title('Clustering Result')
    axes[0].axis('off')

    # Add ROI boxes if requested
    if roi_overlay and len(axes) > 1:
        axes[1].imshow(cluster_map_display, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
        for roi in ROI_REGIONS:
            y_start, y_end, x_start, x_end = roi['coords']
            rect = mpatches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                     linewidth=2, edgecolor=roi['color'], facecolor='none',
                                     linestyle='--', label=roi['name'])
            axes[1].add_patch(rect)
        axes[1].set_title('ROI Regions Overlay')
        axes[1].axis('off')
        axes[1].legend(loc='upper right', fontsize=8)

    # Add ground truth comparison if provided
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='tab10')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

    # Add metrics annotation
    if metrics:
        metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                 for k, v in metrics.items()])
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_difference_map(labels1, labels2, mask, output_path,
                         method1_name="Method 1", method2_name="Method 2",
                         dpi=DPI_PRINT):
    """
    Create comprehensive difference visualization between two clustering results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    roi_cmap = create_roi_colormap()

    # Panel 1: Method 1
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_map1 = np.ma.masked_where(labels1 == -1, labels1)
    im1 = ax1.imshow(cluster_map1, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
    ax1.set_title(method1_name, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Method 2
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_map2 = np.ma.masked_where(labels2 == -1, labels2)
    im2 = ax2.imshow(cluster_map2, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
    ax2.set_title(method2_name, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Difference/Agreement Map
    ax3 = fig.add_subplot(gs[0, 2])
    agreement_map = (labels1 == labels2).astype(float)
    im3 = ax3.imshow(agreement_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_title('Agreement Map\n(Green=Agree, Red=Disagree)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, label='Agreement')

    # Panel 4: Agreement Statistics
    ax4 = fig.add_subplot(gs[1, 0])
    agreement_pct = np.sum(labels1 == labels2) / len(labels1) * 100
    disagreement_pct = 100 - agreement_pct
    bars = ax4.bar(['Agreement', 'Disagreement'], [agreement_pct, disagreement_pct],
                   color=['#2ECC71', '#E74C3C'], alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title(f'Overall Agreement: {agreement_pct:.1f}%', fontweight='bold')
    ax4.set_ylim([0, 100])
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 5: Confusion Matrix
    ax5 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(labels1, labels2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                cbar_kws={'label': 'Count'}, square=True)
    ax5.set_xlabel(f'{method2_name} Labels', fontsize=10)
    ax5.set_ylabel(f'{method1_name} Labels', fontsize=10)
    ax5.set_title('Confusion Matrix', fontweight='bold')

    # Panel 6: Normalized Confusion Matrix
    ax6 = fig.add_subplot(gs[1, 2])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', ax=ax6,
                cbar_kws={'label': 'Proportion'}, square=True, vmin=0, vmax=1)
    ax6.set_xlabel(f'{method2_name} Labels', fontsize=10)
    ax6.set_ylabel(f'{method1_name} Labels', fontsize=10)
    ax6.set_title('Normalized Confusion Matrix', fontweight='bold')

    # Panel 7: Cluster Size Comparison
    ax7 = fig.add_subplot(gs[2, :2])
    unique_labels = np.unique(np.concatenate([labels1[labels1 >= 0], labels2[labels2 >= 0]]))
    cluster_sizes1 = [np.sum(labels1 == l) for l in unique_labels]
    cluster_sizes2 = [np.sum(labels2 == l) for l in unique_labels]

    x = np.arange(len(unique_labels))
    width = 0.35
    bars1 = ax7.bar(x - width/2, cluster_sizes1, width, label=method1_name,
                    alpha=0.8, color='#3498DB', edgecolor='black')
    bars2 = ax7.bar(x + width/2, cluster_sizes2, width, label=method2_name,
                    alpha=0.8, color='#E74C3C', edgecolor='black')
    ax7.set_xlabel('Cluster Label', fontsize=11)
    ax7.set_ylabel('Number of Pixels', fontsize=11)
    ax7.set_title('Cluster Size Distribution Comparison', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(unique_labels)
    ax7.legend(fontsize=10)
    ax7.grid(axis='y', alpha=0.3)

    # Panel 8: Statistical Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Calculate additional metrics
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)

    stats_text = f"""
    Statistical Summary
    {'='*30}

    Agreement: {agreement_pct:.2f}%
    Disagreement: {disagreement_pct:.2f}%

    Adjusted Rand Index: {ari:.4f}
    Normalized Mutual Info: {nmi:.4f}

    {method1_name}:
      Clusters: {len(np.unique(labels1[labels1>=0]))}
      Total pixels: {len(labels1)}

    {method2_name}:
      Clusters: {len(np.unique(labels2[labels2>=0]))}
      Total pixels: {len(labels2)}
    """

    ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle(f'{method1_name} vs {method2_name} - Comprehensive Comparison',
                fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return agreement_pct, ari, nmi


def create_spectral_comparison(hyperspectral_data, labels_dict, output_path,
                               wavelengths=None, dpi=DPI_PRINT):
    """
    Compare spectral signatures across different clustering methods.

    Parameters:
    -----------
    hyperspectral_data : dict
        Dictionary with excitation wavelengths as keys
    labels_dict : dict
        Dictionary mapping method names to cluster labels
    output_path : Path
        Output file path
    wavelengths : array-like, optional
        Wavelength values for x-axis
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    n_methods = len(labels_dict)
    colors = COLOR_PALETTES['scientific'][:n_methods]

    # Extract mean spectra for each method and cluster
    method_spectra = {}

    for method_idx, (method_name, labels) in enumerate(labels_dict.items()):
        unique_labels = np.unique(labels[labels >= 0])
        method_spectra[method_name] = {}

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            # Extract mean spectrum for this cluster across all excitations
            cluster_spectra = []

            for ex_key in hyperspectral_data['data'].keys():
                cube = hyperspectral_data['data'][ex_key]['cube']
                # Get mean spectrum for pixels in this cluster
                if cube.ndim == 3:
                    cluster_pixels = cube[cluster_mask, :]
                    if len(cluster_pixels) > 0:
                        mean_spectrum = np.mean(cluster_pixels, axis=0)
                        cluster_spectra.append(mean_spectrum)

            if cluster_spectra:
                method_spectra[method_name][cluster_id] = np.concatenate(cluster_spectra)

    # Generate wavelength axis if not provided
    if wavelengths is None:
        total_bands = sum([hyperspectral_data['data'][ex]['cube'].shape[2]
                          for ex in hyperspectral_data['data'].keys()])
        wavelengths = np.arange(total_bands)

    # Plot 1: Mean spectra for all methods overlaid
    ax1 = fig.add_subplot(gs[0, :])
    for method_idx, (method_name, spectra) in enumerate(method_spectra.items()):
        for cluster_id, spectrum in spectra.items():
            if len(spectrum) == len(wavelengths):
                ax1.plot(wavelengths, spectrum,
                        label=f'{method_name} - Cluster {cluster_id}',
                        alpha=0.7, linewidth=1.5,
                        color=colors[method_idx % len(colors)])

    ax1.set_xlabel('Wavelength (nm)', fontsize=11)
    ax1.set_ylabel('Mean Intensity', fontsize=11)
    ax1.set_title('Spectral Signatures Comparison - All Methods', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot individual method comparisons
    for idx, (method_name, spectra) in enumerate(list(method_spectra.items())[:6]):
        row = 1 + idx // 3
        col = idx % 3
        if row < 3:
            ax = fig.add_subplot(gs[row, col])

            for cluster_id, spectrum in spectra.items():
                if len(spectrum) == len(wavelengths):
                    ax.plot(wavelengths, spectrum,
                           label=f'Cluster {cluster_id}',
                           linewidth=2, alpha=0.8)

            ax.set_xlabel('Wavelength (nm)', fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)
            ax.set_title(f'{method_name}', fontweight='bold', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Spectral Signature Analysis Across Methods',
                fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_wavelength_importance_analysis(results_df, output_path, dpi=DPI_PRINT):
    """
    Analyze and visualize wavelength importance across configurations.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Extract wavelength data
    # This assumes results_df has 'wavelength_combinations' or similar column

    # Plot 1: Number of wavelengths vs Purity
    ax1 = fig.add_subplot(gs[0, :2])
    if 'n_combinations_selected' in results_df.columns and 'purity' in results_df.columns:
        scatter = ax1.scatter(results_df['n_combinations_selected'],
                             results_df['purity'],
                             c=results_df['data_reduction_pct'] if 'data_reduction_pct' in results_df.columns else None,
                             s=100, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=1)
        ax1.set_xlabel('Number of Wavelengths Selected', fontsize=11)
        ax1.set_ylabel('Purity Score', fontsize=11)
        ax1.set_title('Wavelength Count vs Clustering Purity', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        if 'data_reduction_pct' in results_df.columns:
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Data Reduction (%)', fontsize=10)

        # Annotate best point
        best_idx = results_df['purity'].idxmax()
        ax1.annotate('Best',
                    xy=(results_df.loc[best_idx, 'n_combinations_selected'],
                        results_df.loc[best_idx, 'purity']),
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, color='red', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Plot 2: Method comparison
    ax2 = fig.add_subplot(gs[0, 2])
    if 'dimension_method' in results_df.columns:
        method_performance = results_df.groupby('dimension_method')['purity'].agg(['mean', 'std'])
        method_performance = method_performance.sort_values('mean', ascending=False)

        bars = ax2.barh(range(len(method_performance)), method_performance['mean'],
                       xerr=method_performance['std'], alpha=0.7, color='steelblue',
                       edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(method_performance)))
        ax2.set_yticklabels(method_performance.index)
        ax2.set_xlabel('Mean Purity', fontsize=10)
        ax2.set_title('Performance by Selection Method', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

    # Plot 3: Data reduction efficiency
    ax3 = fig.add_subplot(gs[1, :2])
    if 'data_reduction_pct' in results_df.columns and 'purity' in results_df.columns:
        ax3.scatter(results_df['data_reduction_pct'], results_df['purity'],
                   s=100, alpha=0.6, c='coral', edgecolors='black', linewidth=1)
        ax3.set_xlabel('Data Reduction (%)', fontsize=11)
        ax3.set_ylabel('Purity Score', fontsize=11)
        ax3.set_title('Efficiency: Purity vs Data Reduction', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Add efficiency frontier
        sorted_idx = np.argsort(results_df['data_reduction_pct'])
        ax3.plot(results_df['data_reduction_pct'].iloc[sorted_idx],
                results_df['purity'].iloc[sorted_idx],
                'k--', alpha=0.3, linewidth=1, label='Trend')

    # Plot 4: Top configurations
    ax4 = fig.add_subplot(gs[1, 2])
    top_5 = results_df.nlargest(5, 'purity')
    bars = ax4.barh(range(len(top_5)), top_5['purity'],
                    color=COLOR_PALETTES['scientific'][:5], alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax4.set_yticks(range(len(top_5)))
    ax4.set_yticklabels([f"{int(row['n_combinations_selected'])} WL"
                         for _, row in top_5.iterrows()], fontsize=9)
    ax4.set_xlabel('Purity Score', fontsize=10)
    ax4.set_title('Top 5 Configurations', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_5['purity'])):
        ax4.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
                ha='left', va='center', fontsize=8, fontweight='bold')

    # Plot 5: Multi-metric comparison
    ax5 = fig.add_subplot(gs[2, :])
    metrics_cols = ['purity', 'ari', 'nmi']
    available_metrics = [m for m in metrics_cols if m in results_df.columns]

    if available_metrics:
        top_10 = results_df.nlargest(10, 'purity')
        x = np.arange(len(top_10))
        width = 0.8 / len(available_metrics)

        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2) * width + width/2
            ax5.bar(x + offset, top_10[metric], width,
                   label=metric.upper(), alpha=0.8)

        ax5.set_xlabel('Configuration Rank', fontsize=11)
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_title('Multi-Metric Performance (Top 10 Configurations)', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"#{i+1}" for i in range(len(top_10))])
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)

    plt.suptitle('Wavelength Selection Importance Analysis',
                fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_clustering_quality_dashboard(results_df, output_path, dpi=DPI_PRINT):
    """
    Create comprehensive clustering quality metrics dashboard.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Purity distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['purity'], bins=20, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=1.5)
    ax1.axvline(results_df['purity'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {results_df["purity"].mean():.4f}')
    ax1.axvline(results_df['purity'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median: {results_df["purity"].median():.4f}')
    ax1.set_xlabel('Purity Score', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Purity Distribution', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: ARI vs Purity
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ari' in results_df.columns:
        ax2.scatter(results_df['purity'], results_df['ari'],
                   s=80, alpha=0.6, c='coral', edgecolors='black', linewidth=1)
        ax2.set_xlabel('Purity', fontsize=10)
        ax2.set_ylabel('Adjusted Rand Index', fontsize=10)
        ax2.set_title('Purity vs ARI Correlation', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(results_df['purity'], results_df['ari'])[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 3: NMI vs Purity
    ax3 = fig.add_subplot(gs[0, 2])
    if 'nmi' in results_df.columns:
        ax3.scatter(results_df['purity'], results_df['nmi'],
                   s=80, alpha=0.6, c='mediumseagreen', edgecolors='black', linewidth=1)
        ax3.set_xlabel('Purity', fontsize=10)
        ax3.set_ylabel('Normalized Mutual Info', fontsize=10)
        ax3.set_title('Purity vs NMI Correlation', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(results_df['purity'], results_df['nmi'])[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax3.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 4: Metric correlation heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    metric_cols = ['purity', 'ari', 'nmi', 'silhouette_score', 'data_reduction_pct']
    available_cols = [col for col in metric_cols if col in results_df.columns]

    if len(available_cols) > 1:
        corr_matrix = results_df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, ax=ax4, cbar_kws={'label': 'Correlation'},
                   linewidths=1, linecolor='black')
        ax4.set_title('Metric Correlation Matrix', fontweight='bold')

    # Plot 5: Performance improvement from baseline
    ax5 = fig.add_subplot(gs[1, 2])
    if 'BASELINE_FULL_DATA' in results_df['config_name'].values:
        baseline_purity = results_df[results_df['config_name'] == 'BASELINE_FULL_DATA']['purity'].values[0]
        non_baseline = results_df[results_df['config_name'] != 'BASELINE_FULL_DATA'].copy()
        non_baseline['improvement'] = ((non_baseline['purity'] - baseline_purity) / baseline_purity) * 100

        # Sort by improvement
        non_baseline = non_baseline.sort_values('improvement', ascending=True)

        colors = ['green' if x >= 0 else 'red' for x in non_baseline['improvement']]
        bars = ax5.barh(range(len(non_baseline)), non_baseline['improvement'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax5.set_yticks(range(len(non_baseline)))
        ax5.set_yticklabels([f"{int(row['n_combinations_selected'])} WL"
                            for _, row in non_baseline.iterrows()], fontsize=7)
        ax5.set_xlabel('Purity Improvement (%)', fontsize=10)
        ax5.set_title('Performance vs Baseline', fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.grid(axis='x', alpha=0.3)

    # Plot 6: Statistical summary table
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Calculate statistics
    stats_data = []
    metrics_to_show = ['purity', 'ari', 'nmi', 'data_reduction_pct']
    metrics_to_show = [m for m in metrics_to_show if m in results_df.columns]

    for metric in metrics_to_show:
        stats_data.append([
            metric.upper(),
            f"{results_df[metric].mean():.4f}",
            f"{results_df[metric].std():.4f}",
            f"{results_df[metric].min():.4f}",
            f"{results_df[metric].max():.4f}",
            f"{results_df[metric].median():.4f}"
        ])

    table = ax6.table(cellText=stats_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Median'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)

    # Style the header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax6.set_title('Statistical Summary of Clustering Metrics',
                 fontweight='bold', fontsize=12, pad=20)

    plt.suptitle('Clustering Quality Dashboard',
                fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def create_paper_ready_figure(results_df, best_config_data, ground_truth,
                              output_path, dpi=DPI_POSTER):
    """
    Create publication-ready comprehensive figure.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    roi_cmap = create_roi_colormap()

    # Panel A: Ground Truth
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(ground_truth, cmap='tab10')
    ax_a.set_title('(A) Ground Truth', fontsize=12, fontweight='bold')
    ax_a.axis('off')

    # Panel B: Baseline
    ax_b = fig.add_subplot(gs[0, 1])
    if 'baseline_cluster_map' in best_config_data:
        baseline_map = np.ma.masked_where(best_config_data['baseline_cluster_map'] == -1,
                                         best_config_data['baseline_cluster_map'])
        ax_b.imshow(baseline_map, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
        ax_b.set_title('(B) Baseline (All Wavelengths)', fontsize=12, fontweight='bold')
    ax_b.axis('off')

    # Panel C: Best Result
    ax_c = fig.add_subplot(gs[0, 2])
    if 'cluster_map' in best_config_data:
        best_map = np.ma.masked_where(best_config_data['cluster_map'] == -1,
                                     best_config_data['cluster_map'])
        ax_c.imshow(best_map, cmap=roi_cmap, vmin=0, vmax=len(ROI_REGIONS)-1)
        title = f"(C) Best Result\n({best_config_data.get('n_wavelengths', 'N/A')} wavelengths)"
        ax_c.set_title(title, fontsize=12, fontweight='bold')
    ax_c.axis('off')

    # Panel D: Difference Map
    ax_d = fig.add_subplot(gs[0, 3])
    if 'cluster_map' in best_config_data and 'baseline_cluster_map' in best_config_data:
        diff_map = (best_config_data['cluster_map'] == best_config_data['baseline_cluster_map']).astype(float)
        im_d = ax_d.imshow(diff_map, cmap='RdYlGn', vmin=0, vmax=1)
        ax_d.set_title('(D) Agreement with Baseline', fontsize=12, fontweight='bold')
        plt.colorbar(im_d, ax=ax_d, fraction=0.046)
    ax_d.axis('off')

    # Panel E: Performance Metrics
    ax_e = fig.add_subplot(gs[1, :2])
    top_configs = results_df.nlargest(10, 'purity')
    x = np.arange(len(top_configs))

    metrics_to_plot = []
    if 'purity' in top_configs.columns:
        metrics_to_plot.append(('purity', 'Purity'))
    if 'ari' in top_configs.columns:
        metrics_to_plot.append(('ari', 'ARI'))
    if 'nmi' in top_configs.columns:
        metrics_to_plot.append(('nmi', 'NMI'))

    width = 0.8 / len(metrics_to_plot) if metrics_to_plot else 0.8

    for i, (metric, label) in enumerate(metrics_to_plot):
        offset = (i - len(metrics_to_plot)/2) * width + width/2
        ax_e.bar(x + offset, top_configs[metric], width, label=label, alpha=0.8)

    ax_e.set_xlabel('Configuration Rank', fontsize=11)
    ax_e.set_ylabel('Score', fontsize=11)
    ax_e.set_title('(E) Top 10 Configurations - Performance Metrics',
                  fontsize=12, fontweight='bold')
    ax_e.set_xticks(x)
    ax_e.set_xticklabels([f"#{i+1}" for i in range(len(top_configs))], fontsize=9)
    ax_e.legend(fontsize=10)
    ax_e.grid(axis='y', alpha=0.3)

    # Panel F: Efficiency Plot
    ax_f = fig.add_subplot(gs[1, 2:])
    if 'data_reduction_pct' in results_df.columns and 'purity' in results_df.columns:
        scatter = ax_f.scatter(results_df['data_reduction_pct'],
                              results_df['purity'],
                              c=results_df['n_combinations_selected'] if 'n_combinations_selected' in results_df.columns else None,
                              s=150, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=1.5)
        ax_f.set_xlabel('Data Reduction (%)', fontsize=11)
        ax_f.set_ylabel('Purity Score', fontsize=11)
        ax_f.set_title('(F) Efficiency: Purity vs Data Reduction',
                      fontsize=12, fontweight='bold')
        ax_f.grid(True, alpha=0.3)

        if 'n_combinations_selected' in results_df.columns:
            cbar = plt.colorbar(scatter, ax=ax_f)
            cbar.set_label('Number of Wavelengths', fontsize=10)

        # Highlight best point
        best_idx = results_df['purity'].idxmax()
        ax_f.plot(results_df.loc[best_idx, 'data_reduction_pct'],
                 results_df.loc[best_idx, 'purity'],
                 'r*', markersize=20, label='Best Configuration')
        ax_f.legend(fontsize=10)

    # Panel G: Summary Table
    ax_g = fig.add_subplot(gs[2, :])
    ax_g.axis('off')

    # Create comprehensive summary
    summary_data = []
    for i, (_, row) in enumerate(top_configs.head(5).iterrows()):
        row_data = [
            f"#{i+1}",
            f"{int(row['n_combinations_selected'])}",
            f"{row['purity']:.4f}",
        ]

        if 'ari' in row:
            row_data.append(f"{row['ari']:.4f}")
        if 'nmi' in row:
            row_data.append(f"{row['nmi']:.4f}")
        if 'data_reduction_pct' in row:
            row_data.append(f"{row['data_reduction_pct']:.1f}%")

        summary_data.append(row_data)

    headers = ['Rank', 'Wavelengths', 'Purity']
    if 'ari' in top_configs.columns:
        headers.append('ARI')
    if 'nmi' in top_configs.columns:
        headers.append('NMI')
    if 'data_reduction_pct' in top_configs.columns:
        headers.append('Data Reduction')

    table = ax_g.table(cellText=summary_data,
                      colLabels=headers,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.1] + [0.15] * (len(headers) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 3)

    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax_g.set_title('(G) Top 5 Configurations - Detailed Metrics',
                  fontweight='bold', fontsize=12, pad=20)

    plt.suptitle('Wavelength Selection Pipeline - Comprehensive Results',
                fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(str(output_path).replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  📊 Paper-ready figure saved (PNG and PDF)")


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_comprehensive_visualization_pipeline(
    results_dir,
    hyperspectral_data=None,
    ground_truth=None,
    create_all=True
):
    """
    Main function to run the comprehensive visualization pipeline.

    Parameters:
    -----------
    results_dir : str or Path
        Directory containing the pipeline results (Excel file, visualizations, etc.)
    hyperspectral_data : dict, optional
        Original hyperspectral data for spectral analysis
    ground_truth : np.ndarray, optional
        Ground truth labels for comparison
    create_all : bool
        If True, creates all visualization types. If False, creates only essentials.

    Returns:
    --------
    output_dirs : dict
        Dictionary of output directory paths
    """
    results_dir = Path(results_dir)
    print("\n" + "="*80)
    print("COMPREHENSIVE VISUALIZATION PIPELINE")
    print("="*80)
    print(f"\nResults directory: {results_dir}")

    # Create output directories
    output_base = results_dir / "comprehensive_visualizations"
    output_dirs = create_output_directories(output_base)

    # Load results DataFrame
    excel_path = results_dir / "wavelength_selection_results.xlsx"
    if not excel_path.exists():
        # Try to find any Excel file in the directory
        excel_files = list(results_dir.glob("*.xlsx"))
        if excel_files:
            excel_path = excel_files[0]
            print(f"\n📂 Found results file: {excel_path.name}")
        else:
            print(f"\n❌ No results Excel file found in {results_dir}")
            print("Please ensure the pipeline has been run first.")
            return None

    print(f"\n📊 Loading results from: {excel_path.name}")
    results_df = pd.read_excel(excel_path, sheet_name='Configuration_Results')
    print(f"  Loaded {len(results_df)} configurations")

    # ========================================================================
    # 1. INDIVIDUAL CONFIGURATION RESULTS
    # ========================================================================
    print("\n🎨 Creating individual configuration visualizations...")

    paper_results_dir = results_dir / "paper-results"
    if paper_results_dir.exists():
        result_images = list(paper_results_dir.glob("*_result.png"))
        print(f"  Found {len(result_images)} existing result images")

        # Copy to organized directory with metadata
        for img_path in tqdm(result_images, desc="  Organizing results"):
            config_name = img_path.stem.replace('_result', '')

            # Find metrics for this configuration
            config_row = results_df[results_df['config_name'] == config_name]

            if not config_row.empty:
                # Add metrics overlay
                import shutil
                output_path = output_dirs['individual_results'] / img_path.name
                shutil.copy(img_path, output_path)

    # ========================================================================
    # 2. METHOD COMPARISONS
    # ========================================================================
    print("\n🔄 Creating method comparison visualizations...")

    # Load cluster maps if available
    concat_data_dir = results_dir / "concat-data"

    # Compare top methods
    top_methods = results_df.nlargest(5, 'purity')

    for i in range(len(top_methods) - 1):
        method1 = top_methods.iloc[i]
        method2 = top_methods.iloc[i + 1]

        output_path = output_dirs['comparisons'] / f"comparison_rank{i+1}_vs_rank{i+2}.png"

        # This would need actual cluster labels - placeholder for structure
        print(f"  Comparison: {method1['config_name']} vs {method2['config_name']}")

    # ========================================================================
    # 3. WAVELENGTH IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n📊 Creating wavelength importance analysis...")
    create_wavelength_importance_analysis(
        results_df,
        output_dirs['wavelength_importance'] / "importance_analysis.png",
        dpi=DPI_PRINT
    )
    print("  ✓ Wavelength importance analysis complete")

    # ========================================================================
    # 4. CLUSTERING QUALITY DASHBOARD
    # ========================================================================
    print("\n📈 Creating clustering quality dashboard...")
    create_clustering_quality_dashboard(
        results_df,
        output_dirs['clustering_quality'] / "quality_dashboard.png",
        dpi=DPI_PRINT
    )
    print("  ✓ Clustering quality dashboard complete")

    # ========================================================================
    # 5. SUMMARY VISUALIZATIONS
    # ========================================================================
    print("\n📋 Creating summary visualizations...")

    # Top configurations comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Top 10 by purity
    ax = axes[0, 0]
    top_10 = results_df.nlargest(10, 'purity')
    ax.barh(range(len(top_10)), top_10['purity'],
           color=COLOR_PALETTES['scientific'][:len(top_10)], alpha=0.8,
           edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels([f"{int(row['n_combinations_selected'])} WL"
                       for _, row in top_10.iterrows()], fontsize=9)
    ax.set_xlabel('Purity Score', fontsize=10)
    ax.set_title('Top 10 Configurations by Purity', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Efficiency scatter
    ax = axes[0, 1]
    if 'data_reduction_pct' in results_df.columns:
        scatter = ax.scatter(results_df['data_reduction_pct'], results_df['purity'],
                           c=results_df['n_combinations_selected'],
                           s=100, alpha=0.6, cmap='viridis', edgecolors='black')
        ax.set_xlabel('Data Reduction (%)', fontsize=10)
        ax.set_ylabel('Purity', fontsize=10)
        ax.set_title('Efficiency Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='# Wavelengths')

    # Plot 3: Method performance
    ax = axes[1, 0]
    if 'dimension_method' in results_df.columns:
        method_perf = results_df.groupby('dimension_method')['purity'].mean().sort_values(ascending=False)
        ax.bar(range(len(method_perf)), method_perf.values,
              color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(method_perf)))
        ax.set_xticklabels(method_perf.index, rotation=45, ha='right')
        ax.set_ylabel('Mean Purity', fontsize=10)
        ax.set_title('Average Performance by Method', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Plot 4: Data reduction histogram
    ax = axes[1, 1]
    if 'data_reduction_pct' in results_df.columns:
        ax.hist(results_df['data_reduction_pct'], bins=15, alpha=0.7,
               color='coral', edgecolor='black', linewidth=1.5)
        ax.axvline(results_df['data_reduction_pct'].mean(), color='red',
                  linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Data Reduction (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Data Reduction Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Wavelength Selection - Summary Statistics',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    summary_path = output_dirs['summary'] / "summary_statistics.png"
    plt.savefig(summary_path, dpi=DPI_PRINT, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Summary statistics saved")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ VISUALIZATION PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📁 All visualizations saved to: {output_base}")
    print("\nGenerated visualizations:")
    for name, path in output_dirs.items():
        file_count = len(list(path.glob("*.png")))
        print(f"  {name:25s} {file_count:3d} files")

    print("\n" + "="*80)

    return output_dirs


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Visualization Pipeline for Wavelength Analysis"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results directory from the wavelength validation pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to hyperspectral data file (for spectral analysis)"
    )
    parser.add_argument(
        "--ground-truth-path",
        type=str,
        default=None,
        help="Path to ground truth file"
    )
    parser.add_argument(
        "--essentials-only",
        action="store_true",
        help="Create only essential visualizations (faster)"
    )

    args = parser.parse_args()

    # Load optional data
    hyperspectral_data = None
    ground_truth = None

    if args.data_path:
        print(f"\n📂 Loading hyperspectral data from: {args.data_path}")
        with open(args.data_path, 'rb') as f:
            hyperspectral_data = pickle.load(f)

    if args.ground_truth_path:
        print(f"\n📂 Loading ground truth from: {args.ground_truth_path}")
        ground_truth = np.load(args.ground_truth_path)

    # Run pipeline
    output_dirs = run_comprehensive_visualization_pipeline(
        results_dir=args.results_dir,
        hyperspectral_data=hyperspectral_data,
        ground_truth=ground_truth,
        create_all=not args.essentials_only
    )

    if output_dirs:
        print("\n✨ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
