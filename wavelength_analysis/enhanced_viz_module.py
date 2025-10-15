"""
Enhanced Visualization Module for Wavelength Analysis
This module provides comprehensive visualization capabilities for paper preparation
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import (confusion_matrix, silhouette_score,
                           davies_bouldin_score, calinski_harabasz_score)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots (optional)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Interactive 3D plots will be static only.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Enhanced color palettes for consistent visualization
COLOR_PALETTES = {
    'default': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51'],
    'scientific': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9'],
    'diverging': ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                  '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#67001F'],
    'categorical': sns.color_palette("Set2", 8).as_hex(),
    'roi': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#74B9FF'],
    'paper': ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
}

# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def create_visualization_dirs(base_dir="results/enhanced_visualizations"):
    """Create organized directory structure for all visualizations"""
    base_path = Path(base_dir)

    dirs = [
        "overlays",
        "comparisons",
        "spectral_analysis",
        "clustering_metrics",
        "latent_space",
        "wavelength_importance",
        "reconstruction_quality",
        "statistical_distributions",
        "combined_figures",
        "paper_ready"
    ]

    for dir_name in dirs:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

    print(f"📁 Visualization directories created at: {base_path}")
    return base_path

# ============================================================================
# ROI AND OVERLAY VISUALIZATIONS
# ============================================================================

def export_roi_overlay(mask, labels, save_path, title="Clustering Result",
                       figsize=(12, 8), dpi=300, color_palette='paper',
                       roi_regions=None):
    """Export clustering results with ROI overlay"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get colors
    colors = COLOR_PALETTES.get(color_palette, COLOR_PALETTES['default'])

    # Create cluster visualization
    cluster_img = np.zeros(mask.shape + (3,))
    unique_labels = np.unique(labels[labels >= 0])

    for i, label in enumerate(unique_labels):
        color_rgb = mcolors.hex2color(colors[i % len(colors)])
        cluster_img[labels == label] = color_rgb

    # Panel 1: Original clustering
    axes[0].imshow(cluster_img)
    axes[0].set_title("Clustering Result")
    axes[0].axis('off')

    # Panel 2: With ROI overlay if provided
    if roi_regions:
        roi_overlay = cluster_img.copy()
        for roi_name, roi_mask in roi_regions.items():
            roi_overlay[roi_mask] = roi_overlay[roi_mask] * 0.7 + 0.3
        axes[1].imshow(roi_overlay)
        axes[1].set_title("ROI Highlighted")
    else:
        axes[1].imshow(cluster_img)
        axes[1].set_title("Clustering Result")
    axes[1].axis('off')

    # Panel 3: Legend
    legend_elements = []
    for i, label in enumerate(unique_labels):
        color = colors[i % len(colors)]
        legend_elements.append(mpatches.Patch(color=color, label=f'Cluster {label}'))

    axes[2].legend(handles=legend_elements, loc='center', frameon=False, fontsize=10)
    axes[2].axis('off')
    axes[2].set_title("Legend")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return cluster_img

# ============================================================================
# METHOD COMPARISON VISUALIZATIONS
# ============================================================================

def create_method_comparison(labels1, labels2, mask, save_path,
                            method1_name="Method 1", method2_name="Method 2",
                            title=None, dpi=300):
    """Create comprehensive comparison between two methods"""
    if title is None:
        title = f"{method1_name} vs {method2_name} Comparison"

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Visualization 1: Method 1 clustering
    ax1 = fig.add_subplot(gs[0, 0])
    viz1 = np.zeros(mask.shape)
    viz1[mask] = labels1
    im1 = ax1.imshow(viz1, cmap='tab10')
    ax1.set_title(method1_name)
    ax1.axis('off')

    # Visualization 2: Method 2 clustering
    ax2 = fig.add_subplot(gs[0, 1])
    viz2 = np.zeros(mask.shape)
    viz2[mask] = labels2
    im2 = ax2.imshow(viz2, cmap='tab10')
    ax2.set_title(method2_name)
    ax2.axis('off')

    # Visualization 3: Difference map
    ax3 = fig.add_subplot(gs[0, 2])
    diff_map = np.zeros(mask.shape)
    diff_map[mask] = (labels1 != labels2).astype(int)
    im3 = ax3.imshow(diff_map, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_title("Difference Map")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Agreement percentage
    ax4 = fig.add_subplot(gs[1, 0])
    agreement = np.sum(labels1 == labels2) / len(labels1) * 100
    colors = ['green' if agreement > 80 else 'orange' if agreement > 60 else 'red']
    ax4.bar(['Agreement'], [agreement], color=colors, alpha=0.7)
    ax4.bar(['Disagreement'], [100-agreement], bottom=0, color='gray', alpha=0.3)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_ylim([0, 100])
    ax4.set_title(f"Agreement: {agreement:.1f}%")

    # Confusion matrix
    ax5 = fig.add_subplot(gs[1, 1:])
    cm = confusion_matrix(labels1, labels2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5, cbar=True)
    ax5.set_xlabel(f'{method2_name} Labels')
    ax5.set_ylabel(f'{method1_name} Labels')
    ax5.set_title('Label Confusion Matrix')

    # Cluster size comparison
    ax6 = fig.add_subplot(gs[2, :])
    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    cluster_sizes1 = [np.sum(labels1 == l) for l in unique_labels]
    cluster_sizes2 = [np.sum(labels2 == l) for l in unique_labels]

    x = np.arange(len(unique_labels))
    width = 0.35
    ax6.bar(x - width/2, cluster_sizes1, width, label=method1_name, alpha=0.7, color='#3498DB')
    ax6.bar(x + width/2, cluster_sizes2, width, label=method2_name, alpha=0.7, color='#E74C3C')
    ax6.set_xlabel('Cluster Label')
    ax6.set_ylabel('Number of Pixels')
    ax6.set_title('Cluster Size Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(unique_labels)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return agreement

# ============================================================================
# SPECTRAL ANALYSIS
# ============================================================================

def plot_spectral_signatures(spectra_dict, wavelengths, save_path,
                            title="Spectral Signatures Analysis", dpi=300):
    """Compare spectral signatures from different methods/clusters"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: All spectra
    ax = axes[0, 0]
    for name, spectrum in spectra_dict.items():
        ax.plot(wavelengths, spectrum, label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Spectral Signatures')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean spectrum with std bands
    ax = axes[0, 1]
    all_spectra = np.array(list(spectra_dict.values()))
    mean_spectrum = np.mean(all_spectra, axis=0)
    std_spectrum = np.std(all_spectra, axis=0)

    ax.plot(wavelengths, mean_spectrum, 'k-', linewidth=2, label='Mean')
    ax.fill_between(wavelengths, mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum, alpha=0.3, color='gray', label='±1 STD')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('Mean Spectrum ± Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Correlation heatmap
    ax = axes[1, 0]
    corr_matrix = np.corrcoef(all_spectra)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Spectral Correlation Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Plot 4: Variability analysis
    ax = axes[1, 1]
    cv = std_spectrum / (mean_spectrum + 1e-10) * 100  # Coefficient of variation
    ax.plot(wavelengths, cv, linewidth=2, color='purple')
    ax.fill_between(wavelengths, 0, cv, alpha=0.3, color='purple')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('Spectral Variability')
    ax.grid(True, alpha=0.3)

    # Highlight high variability regions
    threshold = np.percentile(cv, 75)
    high_var = cv > threshold
    ax.fill_between(wavelengths, 0, cv * high_var, alpha=0.5, color='red')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# ============================================================================
# WAVELENGTH IMPORTANCE
# ============================================================================

def create_wavelength_importance_plot(importance_scores, wavelengths, save_path,
                                     title="Wavelength Importance Analysis", dpi=300,
                                     top_n=20):
    """Visualize wavelength importance scores"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Importance curve
    ax = axes[0, 0]
    ax.plot(wavelengths, importance_scores, linewidth=2, color='darkred')
    ax.fill_between(wavelengths, 0, importance_scores, alpha=0.3, color='red')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Importance Score')
    ax.set_title('Wavelength Importance Profile')
    ax.grid(True, alpha=0.3)

    # Plot 2: Top N wavelengths
    ax = axes[0, 1]
    top_indices = np.argsort(importance_scores)[-top_n:]
    top_wavelengths = wavelengths[top_indices]
    top_scores = importance_scores[top_indices]

    y_pos = np.arange(len(top_wavelengths))
    ax.barh(y_pos, top_scores, color='darkred', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{w:.1f} nm' for w in top_wavelengths], fontsize=8)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Important Wavelengths')

    # Plot 3: Cumulative importance
    ax = axes[1, 0]
    sorted_scores = np.sort(importance_scores)[::-1]
    cumulative = np.cumsum(sorted_scores) / np.sum(sorted_scores) * 100
    ax.plot(range(len(cumulative)), cumulative, linewidth=2, color='navy')
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Number of Wavelengths')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Cumulative Importance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Statistics
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    Wavelength Importance Statistics
    --------------------------------
    Total wavelengths: {len(wavelengths)}
    Mean importance: {np.mean(importance_scores):.4f}
    Std importance: {np.std(importance_scores):.4f}
    Max importance: {np.max(importance_scores):.4f}

    Top wavelength: {wavelengths[np.argmax(importance_scores)]:.1f} nm

    Wavelengths for 80% importance: {np.sum(cumulative <= 80)}
    Wavelengths for 90% importance: {np.sum(cumulative <= 90)}
    Wavelengths for 95% importance: {np.sum(cumulative <= 95)}
    """

    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# ============================================================================
# ADVANCED CLUSTERING METRICS
# ============================================================================

def calculate_advanced_metrics(data, labels):
    """Calculate advanced clustering metrics"""
    metrics = {}

    # Filter valid points
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 2:
        return {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0}

    valid_data = data[valid_mask]
    valid_labels = labels[valid_mask]

    # Calculate metrics only if we have multiple clusters
    if len(np.unique(valid_labels)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(valid_data, valid_labels)
        except:
            metrics['silhouette'] = 0

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(valid_data, valid_labels)
        except:
            metrics['davies_bouldin'] = 0

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_labels)
        except:
            metrics['calinski_harabasz'] = 0
    else:
        metrics['silhouette'] = 0
        metrics['davies_bouldin'] = 0
        metrics['calinski_harabasz'] = 0

    return metrics

# ============================================================================
# 3D LATENT SPACE VISUALIZATION
# ============================================================================

def create_latent_space_plot(features, labels, save_path,
                            title="Latent Space Visualization", dpi=300):
    """Create 3D latent space visualization"""

    # Reduce to 3D if needed
    if features.shape[1] > 3:
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
        variance_explained = pca.explained_variance_ratio_
    else:
        features_3d = features[:, :3] if features.shape[1] >= 3 else features
        variance_explained = [1.0, 0.0, 0.0]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    unique_labels = np.unique(labels[labels >= 0])
    colors = COLOR_PALETTES['paper']

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=colors[i % len(colors)], label=f'Cluster {label}', s=10, alpha=0.6)

    ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax1.set_zlabel(f'PC3 ({variance_explained[2]*100:.1f}%)')
    ax1.set_title('3D Latent Space')
    ax1.legend(fontsize=8)

    # 2D projections
    projections = [(0, 1, '(PC1 vs PC2)'), (0, 2, '(PC1 vs PC3)'), (1, 2, '(PC2 vs PC3)')]

    for idx, (dim1, dim2, proj_name) in enumerate(projections):
        ax = fig.add_subplot(2, 2, idx+2)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_3d[mask, dim1], features_3d[mask, dim2],
                      c=colors[i % len(colors)], label=f'Cluster {label}',
                      s=10, alpha=0.6)

        ax.set_xlabel(f'PC{dim1+1} ({variance_explained[dim1]*100:.1f}%)')
        ax.set_ylabel(f'PC{dim2+1} ({variance_explained[dim2]*100:.1f}%)')
        ax.set_title(f'2D Projection {proj_name}')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Create interactive version if plotly is available
    if PLOTLY_AVAILABLE:
        fig_plotly = go.Figure()

        for i, label in enumerate(unique_labels):
            mask = labels == label
            fig_plotly.add_trace(go.Scatter3d(
                x=features_3d[mask, 0],
                y=features_3d[mask, 1],
                z=features_3d[mask, 2],
                mode='markers',
                marker=dict(size=3, color=colors[i % len(colors)], opacity=0.7),
                name=f'Cluster {label}'
            ))

        fig_plotly.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'PC1 ({variance_explained[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({variance_explained[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({variance_explained[2]*100:.1f}%)'
            ),
            width=800,
            height=600
        )

        # Save interactive HTML
        html_path = str(save_path).replace('.png', '.html')
        fig_plotly.write_html(html_path)
        print(f"  💾 Interactive plot saved: {html_path}")

# ============================================================================
# STATISTICAL DISTRIBUTIONS
# ============================================================================

def create_distribution_plots(data_dict, save_path,
                            title="Statistical Distribution Analysis", dpi=300):
    """Create comprehensive distribution visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Prepare data
    data_list = []
    labels = []
    for name, data in data_dict.items():
        data_flat = data.flatten() if hasattr(data, 'flatten') else data
        data_list.append(data_flat)
        labels.append(name)

    # Box plot
    ax = axes[0, 0]
    bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLOR_PALETTES['paper']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Value')
    ax.set_title('Distribution Box Plots')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Violin plot
    ax = axes[0, 1]
    parts = ax.violinplot(data_list, positions=range(len(data_list)),
                          showmeans=True, showmedians=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Distribution Violin Plots')
    ax.grid(True, alpha=0.3)

    # Histogram comparison
    ax = axes[0, 2]
    for data, label, color in zip(data_list, labels, COLOR_PALETTES['paper']):
        ax.hist(data, bins=30, alpha=0.5, label=label, color=color, density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Histograms')
    ax.legend(fontsize=8)

    # KDE plots
    ax = axes[1, 0]
    for data, label, color in zip(data_list, labels, COLOR_PALETTES['paper']):
        try:
            density = stats.gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 100)
            ax.plot(x, density(x), label=label, linewidth=2, color=color)
        except:
            pass
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1, 1]
    for data, label in zip(data_list[:3], labels[:3]):  # Limit to 3 for clarity
        stats.probplot(data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plots (Normal Distribution)')
    ax.grid(True, alpha=0.3)

    # Statistics table
    ax = axes[1, 2]
    ax.axis('off')

    stats_data = []
    for data, label in zip(data_list, labels):
        stats_data.append([
            label[:15],
            f"{np.mean(data):.3f}",
            f"{np.std(data):.3f}",
            f"{np.median(data):.3f}",
            f"{stats.skew(data):.3f}"
        ])

    table = ax.table(cellText=stats_data,
                    colLabels=['Method', 'Mean', 'Std', 'Median', 'Skew'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Statistical Summary')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# ============================================================================
# PAPER-READY COMBINED FIGURE
# ============================================================================

def create_paper_figure(results_df, best_result, mask, wavelengths, save_path, dpi=300):
    """Create publication-ready combined figure (handles missing data gracefully)"""
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Purity vs Data Reduction
    ax_a = fig.add_subplot(gs[0, :])
    scatter = ax_a.scatter(results_df['data_reduction_pct'],
                          results_df['purity'],
                          c=results_df['num_wavelengths'],
                          s=100, cmap='viridis', alpha=0.6, edgecolors='black')
    ax_a.set_xlabel('Data Reduction (%)', fontsize=11)
    ax_a.set_ylabel('Purity Score', fontsize=11)
    ax_a.set_title('(A) Purity vs Data Reduction Trade-off', fontsize=12, fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax_a)
    cbar.set_label('Number of Wavelengths', fontsize=10)

    # Annotate best point
    best_idx = results_df['purity'].idxmax()
    ax_a.annotate('Best', xy=(results_df.loc[best_idx, 'data_reduction_pct'],
                              results_df.loc[best_idx, 'purity']),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red', fontweight='bold')

    # Panel B: Top Methods Comparison
    ax_b = fig.add_subplot(gs[1, :])
    top_5 = results_df.nlargest(5, 'purity')
    x = range(len(top_5))
    bars = ax_b.bar(x, top_5['purity'].values, color=COLOR_PALETTES['paper'][:5], alpha=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{row['num_wavelengths']} WLs" for _, row in top_5.iterrows()])
    ax_b.set_ylabel('Purity Score', fontsize=11)
    ax_b.set_title('(B) Top 5 Methods by Purity', fontsize=12, fontweight='bold')
    ax_b.set_ylim([top_5['purity'].min() * 0.95, top_5['purity'].max() * 1.02])

    # Add value labels
    for bar, val in zip(bars, top_5['purity'].values):
        height = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Panel C: Cluster visualization (if available)
    ax_c = fig.add_subplot(gs[2, 0])
    if 'cluster_labels' in best_result:
        viz = np.zeros(mask.shape)
        viz[mask] = best_result['cluster_labels']
        im = ax_c.imshow(viz, cmap='tab10')
        ax_c.set_title(f"(C) Best Method Clustering\n({best_result['num_wavelengths']} wavelengths)",
                      fontsize=11, fontweight='bold')
    else:
        ax_c.text(0.5, 0.5, 'Cluster labels\nnot available',
                 ha='center', va='center', transform=ax_c.transAxes,
                 fontsize=12, color='gray')
        ax_c.set_title("(C) Clustering Result", fontsize=11, fontweight='bold')
    ax_c.axis('off')

    # Panel D: Cluster sizes (if available)
    ax_d = fig.add_subplot(gs[2, 1])
    if 'cluster_labels' in best_result:
        unique_labels, counts = np.unique(best_result['cluster_labels'][best_result['cluster_labels'] >= 0],
                                         return_counts=True)
        ax_d.bar(unique_labels, counts, color=COLOR_PALETTES['paper'][:len(unique_labels)], alpha=0.8)
        ax_d.set_xlabel('Cluster ID', fontsize=10)
        ax_d.set_ylabel('Number of Pixels', fontsize=10)
        ax_d.set_title('(D) Cluster Size Distribution', fontsize=11, fontweight='bold')
        ax_d.grid(True, alpha=0.3)
    else:
        ax_d.text(0.5, 0.5, 'Cluster data\nnot available',
                 ha='center', va='center', transform=ax_d.transAxes,
                 fontsize=12, color='gray')
        ax_d.set_title('(D) Cluster Sizes', fontsize=11, fontweight='bold')
        ax_d.axis('off')

    # Panel E: Advanced metrics comparison
    ax_e = fig.add_subplot(gs[2, 2])
    if 'silhouette_score' in results_df.columns:
        metrics = ['purity', 'silhouette_score']
        top_3 = results_df.nlargest(3, 'purity')

        x = np.arange(len(top_3))
        width = 0.35

        ax_e.bar(x - width/2, top_3['purity'], width, label='Purity', color='#3498DB', alpha=0.8)
        ax_e.bar(x + width/2, top_3['silhouette_score'], width, label='Silhouette', color='#E74C3C', alpha=0.8)

        ax_e.set_xlabel('Method Rank', fontsize=10)
        ax_e.set_ylabel('Score', fontsize=10)
        ax_e.set_title('(E) Multi-Metric Comparison', fontsize=11, fontweight='bold')
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(['Top 1', 'Top 2', 'Top 3'])
        ax_e.legend(fontsize=9)
        ax_e.grid(True, alpha=0.3)

    # Panel F: Performance summary
    ax_f = fig.add_subplot(gs[3:, :])
    ax_f.axis('off')

    # Create summary table
    summary_data = []
    for i, (_, row) in enumerate(top_5.iterrows()):
        summary_data.append([
            f"Rank {i+1}",
            f"{row['num_wavelengths']}",
            f"{row['purity']:.4f}",
            f"{row['data_reduction_pct']:.1f}%"
        ])

    if 'silhouette_score' in top_5.columns:
        headers = ['Rank', 'Wavelengths', 'Purity', 'Data Reduction', 'Silhouette']
        for i, row_data in enumerate(summary_data):
            row_data.append(f"{top_5.iloc[i]['silhouette_score']:.4f}")
    else:
        headers = ['Rank', 'Wavelengths', 'Purity', 'Data Reduction']

    table = ax_f.table(cellText=summary_data,
                      colLabels=headers,
                      cellLoc='center',
                      loc='upper center',
                      colWidths=[0.15, 0.2, 0.2, 0.25, 0.2] if len(headers) == 5 else [0.2, 0.25, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Add text summary
    summary_text = f"""
    BEST METHOD SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Wavelengths Used: {best_result['num_wavelengths']}
    Purity Score: {best_result['purity']:.4f}
    Data Reduction: {best_result['data_reduction_pct']:.1f}%
    """

    # Add cluster info if available
    if 'cluster_labels' in best_result:
        n_clusters = len(np.unique(best_result['cluster_labels'][best_result['cluster_labels'] >= 0]))
        summary_text += f"    Number of Clusters: {n_clusters}\n"

    if 'silhouette_score' in best_result:
        summary_text += f"""    Silhouette Score: {best_result['silhouette_score']:.4f}
    Davies-Bouldin: {best_result.get('davies_bouldin', 'N/A')}
    """

    ax_f.text(0.5, 0.3, summary_text, transform=ax_f.transAxes,
             fontsize=11, ha='center', va='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Main title
    plt.suptitle('Wavelength Selection Analysis - Comprehensive Results',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(str(save_path).replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  📊 Paper-ready figure saved: {save_path}")
    print(f"  📊 PDF version saved: {str(save_path).replace('.png', '.pdf')}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_all_visualizations(results, mask, wavelengths, hypercube=None, base_dir="results/enhanced_visualizations"):
    """Generate all visualizations from results (handles missing data gracefully)"""
    vis_dir = Path(base_dir)

    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)

    # Convert results to DataFrame if needed
    if isinstance(results, list):
        df_results = pd.DataFrame(results)
    else:
        df_results = results

    # Find best result
    best_idx = df_results['purity'].idxmax()
    best_result = df_results.iloc[best_idx].to_dict()

    # Try to get cluster_labels if available
    if 'cluster_labels' not in best_result and isinstance(results, list):
        if best_idx < len(results) and 'cluster_labels' in results[best_idx]:
            best_result['cluster_labels'] = results[best_idx]['cluster_labels']

    # 1. Export top method overlays (only if cluster_labels exist)
    has_overlays = False
    top_5 = df_results.nlargest(5, 'purity')

    for i, (idx, row) in enumerate(top_5.iterrows()):
        labels = None

        # Try to find cluster labels
        if isinstance(results, list) and idx < len(results):
            if 'cluster_labels' in results[idx]:
                labels = results[idx]['cluster_labels']
        elif 'cluster_labels' in row:
            labels = row['cluster_labels']

        if labels is not None:
            if not has_overlays:
                print("\n📸 Exporting clustering overlays...")
                has_overlays = True

            export_roi_overlay(
                mask=mask,
                labels=labels,
                save_path=vis_dir / "overlays" / f"top_{i+1}_clustering.png",
                title=f"Top {i+1}: {row['num_wavelengths']} wavelengths (Purity: {row['purity']:.4f})"
            )

    if has_overlays:
        print(f"  ✓ Saved overlay visualizations")
    else:
        print("\n  ⚠️ No cluster labels found - skipping overlay visualizations")

    # 2. Create paper-ready figure
    print("\n📊 Creating paper-ready combined figure...")
    create_paper_figure(
        results_df=df_results,
        best_result=best_result,
        mask=mask,
        wavelengths=wavelengths,
        save_path=vis_dir / "paper_ready" / "main_figure.png"
    )

    # 3. Wavelength importance
    if 'wavelength_indices' in best_result or 'selected_wavelengths' in best_result:
        print("\n📈 Creating wavelength importance plot...")
        # Create importance scores based on selection frequency
        importance = np.zeros(len(wavelengths))
        for _, row in df_results.iterrows():
            if 'wavelength_indices' in row:
                for idx in row['wavelength_indices']:
                    if idx < len(wavelengths):
                        importance[idx] += row['purity']

        create_wavelength_importance_plot(
            importance_scores=importance,
            wavelengths=wavelengths,
            save_path=vis_dir / "wavelength_importance" / "importance_analysis.png"
        )

    print("\n" + "="*60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Location: {vis_dir}")
    print("="*60)

    return vis_dir

# Print module info when imported
print("="*60)
print("Enhanced Visualization Module Loaded")
print("="*60)
print("Available functions:")
print("  • create_visualization_dirs()")
print("  • export_roi_overlay()")
print("  • create_method_comparison()")
print("  • plot_spectral_signatures()")
print("  • create_wavelength_importance_plot()")
print("  • calculate_advanced_metrics()")
print("  • create_latent_space_plot()")
print("  • create_distribution_plots()")
print("  • create_paper_figure()")
print("  • generate_all_visualizations()")
print("="*60)