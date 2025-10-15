# ============================================================================
# COMPREHENSIVE VISUALIZATION MODULE FOR WAVELENGTH ANALYSIS
# ============================================================================
# This module provides enhanced visualization capabilities for the paper
# All visualizations are saved as individual files for flexible paper design

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Create enhanced visualization directory structure
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
        "perturbation_analysis",
        "reconstruction_quality",
        "statistical_distributions",
        "animations",
        "combined_figures",
        "paper_ready"
    ]

    for dir_name in dirs:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

    return base_path

# Enhanced color palettes for consistent visualization
COLOR_PALETTES = {
    'default': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51'],
    'scientific': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#ECE133', '#56B4E9'],
    'diverging': ['#053061', '#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
                  '#FDDBC7', '#F4A582', '#D6604D', '#B2182B', '#67001F'],
    'categorical': sns.color_palette("Set2", 8).as_hex(),
    'roi': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#74B9FF']
}

def export_roi_overlay_with_colors(mask, labels, roi_regions, save_path, title="ROI Overlay",
                                   figsize=(12, 10), dpi=300, color_palette='roi'):
    """Export ROI overlay with consistent colors for all clustering cases"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Get color palette
    colors = COLOR_PALETTES.get(color_palette, COLOR_PALETTES['default'])

    # Create cluster visualization
    cluster_img = np.zeros(mask.shape + (3,))
    for i, label in enumerate(np.unique(labels[labels >= 0])):
        color = mcolors.hex2color(colors[i % len(colors)])
        cluster_img[labels == label] = color

    # Original clustering
    axes[0].imshow(cluster_img)
    axes[0].set_title(f"{title} - Clustering Result")
    axes[0].axis('off')

    # ROI overlay
    roi_overlay = cluster_img.copy()
    if roi_regions:
        for roi_name, roi_mask in roi_regions.items():
            roi_overlay[roi_mask] = roi_overlay[roi_mask] * 0.5 + 0.5  # Highlight ROI

    axes[1].imshow(roi_overlay)
    axes[1].set_title(f"{title} - ROI Highlighted")
    axes[1].axis('off')

    # Create legend
    legend_elements = []
    for i, label in enumerate(np.unique(labels[labels >= 0])):
        legend_elements.append(mpatches.Patch(color=colors[i % len(colors)],
                                             label=f'Cluster {label}'))

    axes[2].legend(handles=legend_elements, loc='center', frameon=False)
    axes[2].axis('off')
    axes[2].set_title("Legend")

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return cluster_img

def create_difference_visualization(img1, img2, labels1, labels2, mask, save_path,
                                   title="Method Comparison", dpi=300):
    """Create comprehensive difference visualization between two methods"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Method 1 clustering
    ax1 = fig.add_subplot(gs[0, 0])
    viz1 = np.zeros(mask.shape)
    viz1[mask] = labels1
    im1 = ax1.imshow(viz1, cmap='tab10')
    ax1.set_title("Method 1 Clustering")
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Method 2 clustering
    ax2 = fig.add_subplot(gs[0, 1])
    viz2 = np.zeros(mask.shape)
    viz2[mask] = labels2
    im2 = ax2.imshow(viz2, cmap='tab10')
    ax2.set_title("Method 2 Clustering")
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Difference map
    ax3 = fig.add_subplot(gs[0, 2])
    diff_map = np.zeros(mask.shape)
    diff_map[mask] = (labels1 != labels2).astype(int)
    im3 = ax3.imshow(diff_map, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax3.set_title("Difference Map")
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Agreement percentage
    ax4 = fig.add_subplot(gs[0, 3])
    agreement = np.sum(labels1 == labels2) / len(labels1) * 100
    ax4.bar(['Agreement', 'Disagreement'], [agreement, 100-agreement],
            color=['green', 'red'], alpha=0.7)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title(f"Agreement: {agreement:.1f}%")

    # Confusion matrix
    ax5 = fig.add_subplot(gs[1, :2])
    cm = confusion_matrix(labels1, labels2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_xlabel('Method 2 Labels')
    ax5.set_ylabel('Method 1 Labels')
    ax5.set_title('Confusion Matrix')

    # Cluster size comparison
    ax6 = fig.add_subplot(gs[1, 2:])
    unique_labels = np.unique(np.concatenate([labels1, labels2]))
    cluster_sizes1 = [np.sum(labels1 == l) for l in unique_labels]
    cluster_sizes2 = [np.sum(labels2 == l) for l in unique_labels]

    x = np.arange(len(unique_labels))
    width = 0.35
    ax6.bar(x - width/2, cluster_sizes1, width, label='Method 1', alpha=0.7)
    ax6.bar(x + width/2, cluster_sizes2, width, label='Method 2', alpha=0.7)
    ax6.set_xlabel('Cluster Label')
    ax6.set_ylabel('Number of Pixels')
    ax6.set_title('Cluster Size Comparison')
    ax6.set_xticks(x)
    ax6.set_xticklabels(unique_labels)
    ax6.legend()

    # Pixel-wise difference histogram
    ax7 = fig.add_subplot(gs[2, :2])
    if img1 is not None and img2 is not None:
        pixel_diff = np.abs(img1 - img2).flatten()
        ax7.hist(pixel_diff, bins=50, edgecolor='black', alpha=0.7)
        ax7.axvline(np.mean(pixel_diff), color='red', linestyle='--',
                   label=f'Mean: {np.mean(pixel_diff):.3f}')
        ax7.set_xlabel('Absolute Difference')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Pixel-wise Difference Distribution')
        ax7.legend()

    # Statistical summary
    ax8 = fig.add_subplot(gs[2, 2:])
    ax8.axis('off')

    stats_text = f"""Statistical Summary:

Agreement: {agreement:.2f}%
Disagreement: {100-agreement:.2f}%

Method 1 - Unique clusters: {len(np.unique(labels1))}
Method 2 - Unique clusters: {len(np.unique(labels2))}

Matched pixels: {np.sum(labels1 == labels2)}
Mismatched pixels: {np.sum(labels1 != labels2)}"""

    if img1 is not None and img2 is not None:
        stats_text += f"""

Mean Absolute Error: {np.mean(np.abs(img1 - img2)):.4f}
RMSE: {np.sqrt(np.mean((img1 - img2)**2)):.4f}
Max Difference: {np.max(np.abs(img1 - img2)):.4f}"""

    ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_spectral_signatures_comparison(spectra_dict, wavelengths, save_path,
                                       title="Spectral Signatures Comparison", dpi=300):
    """Compare spectral signatures from different methods/clusters"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot all spectra together
    ax = axes[0, 0]
    for name, spectrum in spectra_dict.items():
        ax.plot(wavelengths, spectrum, label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title('All Spectral Signatures')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot difference from mean
    ax = axes[0, 1]
    mean_spectrum = np.mean(list(spectra_dict.values()), axis=0)
    for name, spectrum in spectra_dict.items():
        diff = spectrum - mean_spectrum
        ax.plot(wavelengths, diff, label=f'{name} - Mean', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Difference from Mean')
    ax.set_title('Deviation from Mean Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot correlation matrix
    ax = axes[1, 0]
    spectra_array = np.array(list(spectra_dict.values()))
    corr_matrix = np.corrcoef(spectra_array)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(spectra_dict)))
    ax.set_yticks(range(len(spectra_dict)))
    ax.set_xticklabels(list(spectra_dict.keys()), rotation=45, ha='right')
    ax.set_yticklabels(list(spectra_dict.keys()))
    ax.set_title('Spectral Correlation Matrix')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Add correlation values
    for i in range(len(spectra_dict)):
        for j in range(len(spectra_dict)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                         ha="center", va="center",
                         color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")

    # Plot standard deviation across methods
    ax = axes[1, 1]
    std_spectrum = np.std(list(spectra_dict.values()), axis=0)
    ax.plot(wavelengths, std_spectrum, linewidth=2, color='purple')
    ax.fill_between(wavelengths, 0, std_spectrum, alpha=0.3, color='purple')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Spectral Variability Across Methods')
    ax.grid(True, alpha=0.3)

    # Highlight regions of high variability
    threshold = np.percentile(std_spectrum, 75)
    high_var_mask = std_spectrum > threshold
    ax.fill_between(wavelengths, 0, std_spectrum * high_var_mask,
                    alpha=0.5, color='red', label='High variability')
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_3d_latent_space_visualization(latent_features, labels, save_path,
                                        title="3D Latent Space Visualization", dpi=300):
    """Create interactive 3D visualization of latent space"""
    # Reduce to 3D if necessary
    if latent_features.shape[1] > 3:
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(latent_features)
        variance_explained = pca.explained_variance_ratio_
    else:
        features_3d = latent_features
        variance_explained = [1.0, 0.0, 0.0]

    # Create interactive plotly figure
    fig = go.Figure()

    unique_labels = np.unique(labels[labels >= 0])
    colors = px.colors.qualitative.Set1[:len(unique_labels)]

    for i, label in enumerate(unique_labels):
        mask = labels == label
        fig.add_trace(go.Scatter3d(
            x=features_3d[mask, 0],
            y=features_3d[mask, 1],
            z=features_3d[mask, 2],
            mode='markers',
            marker=dict(size=3, color=colors[i % len(colors)], opacity=0.7),
            name=f'Cluster {label}'
        ))

    fig.update_layout(
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
    fig.write_html(html_path)

    # Also create static matplotlib version
    fig_static = plt.figure(figsize=(14, 10))

    # 3D scatter plot
    ax1 = fig_static.add_subplot(221, projection='3d')
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                   c=[colors[i % len(colors)]], label=f'Cluster {label}', s=5, alpha=0.6)
    ax1.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}%)')
    ax1.set_zlabel(f'PC3 ({variance_explained[2]*100:.1f}%)')
    ax1.legend()
    ax1.set_title('3D Latent Space')

    # 2D projections
    projections = [(0, 1, 222), (0, 2, 223), (1, 2, 224)]
    proj_names = [('PC1', 'PC2'), ('PC1', 'PC3'), ('PC2', 'PC3')]

    for (dim1, dim2, subplot), (name1, name2) in zip(projections, proj_names):
        ax = fig_static.add_subplot(subplot)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_3d[mask, dim1], features_3d[mask, dim2],
                      c=[colors[i % len(colors)]], label=f'Cluster {label}',
                      s=5, alpha=0.6)
        ax.set_xlabel(f'{name1} ({variance_explained[dim1]*100:.1f}%)')
        ax.set_ylabel(f'{name2} ({variance_explained[dim2]*100:.1f}%)')
        ax.set_title(f'{name1} vs {name2}')
        if subplot == 222:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_wavelength_importance_heatmap(importance_matrix, wavelengths, save_path,
                                        title="Wavelength Importance Heatmap", dpi=300):
    """Create heatmap showing importance of different wavelengths"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Main heatmap
    ax = axes[0, 0]
    im = ax.imshow(importance_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Wavelength Index')
    ax.set_ylabel('Method/Iteration')
    ax.set_title('Wavelength Importance Across Methods')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Average importance
    ax = axes[0, 1]
    avg_importance = np.mean(importance_matrix, axis=0)
    ax.plot(wavelengths, avg_importance, linewidth=2, color='darkred')
    ax.fill_between(wavelengths, 0, avg_importance, alpha=0.3, color='red')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Average Importance')
    ax.set_title('Average Wavelength Importance')
    ax.grid(True, alpha=0.3)

    # Top N important wavelengths
    ax = axes[1, 0]
    top_n = 20
    top_indices = np.argsort(avg_importance)[-top_n:]
    ax.barh(range(top_n), avg_importance[top_indices], color='darkred', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'{wavelengths[i]:.1f} nm' for i in top_indices], fontsize=8)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Important Wavelengths')

    # Importance distribution
    ax = axes[1, 1]
    ax.hist(importance_matrix.flatten(), bins=50, edgecolor='black', alpha=0.7, color='darkred')
    ax.axvline(np.mean(importance_matrix), color='blue', linestyle='--',
              label=f'Mean: {np.mean(importance_matrix):.3f}')
    ax.axvline(np.median(importance_matrix), color='green', linestyle='--',
              label=f'Median: {np.median(importance_matrix):.3f}')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Importance Scores')
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def calculate_ssim(img1, img2, window_size=11):
    """Calculate Structural Similarity Index (SSIM) between two images"""
    # Simplified SSIM calculation
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))

    return ssim

def calculate_advanced_clustering_metrics(data, labels):
    """Calculate advanced clustering metrics"""
    metrics = {}

    # Filter out noise points (-1 labels)
    valid_mask = labels >= 0
    if np.sum(valid_mask) < 2:
        return {'silhouette': 0, 'davies_bouldin': 0, 'calinski_harabasz': 0}

    valid_data = data[valid_mask]
    valid_labels = labels[valid_mask]

    # Calculate metrics
    if len(np.unique(valid_labels)) > 1:
        metrics['silhouette'] = silhouette_score(valid_data, valid_labels)
        metrics['davies_bouldin'] = davies_bouldin_score(valid_data, valid_labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(valid_data, valid_labels)
    else:
        metrics['silhouette'] = 0
        metrics['davies_bouldin'] = 0
        metrics['calinski_harabasz'] = 0

    return metrics

def create_reconstruction_quality_plots(original, reconstructions_dict, save_path,
                                       title="Reconstruction Quality Analysis", dpi=300):
    """Create comprehensive reconstruction quality visualizations"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Calculate quality metrics
    metrics = {}
    for name, recon in reconstructions_dict.items():
        metrics[name] = {
            'psnr': 20 * np.log10(np.max(original) / (np.sqrt(np.mean((original - recon)**2)) + 1e-10)),
            'ssim': calculate_ssim(original, recon),
            'mae': np.mean(np.abs(original - recon)),
            'correlation': np.corrcoef(original.flatten(), recon.flatten())[0, 1]
        }

    # Plot quality metrics comparison
    ax = axes[0, 0]
    metric_names = list(metrics[list(metrics.keys())[0]].keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics)

    for i, (name, metric_vals) in enumerate(metrics.items()):
        values = [metric_vals[m] for m in metric_names]
        normalized_values = values / (np.max(values) + 1e-10)
        ax.bar(x + i * width, normalized_values, width, label=name, alpha=0.7)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Reconstruction Quality Metrics (Normalized)')
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter plot: Original vs Reconstructed
    ax = axes[0, 1]
    for name, recon in list(reconstructions_dict.items())[:3]:  # Limit for clarity
        sample_size = min(1000, original.size)
        idx = np.random.choice(original.size, sample_size, replace=False)
        ax.scatter(original.flatten()[idx], recon.flatten()[idx],
                  alpha=0.3, s=1, label=name)
    ax.plot([original.min(), original.max()], [original.min(), original.max()],
           'r--', label='Perfect reconstruction')
    ax.set_xlabel('Original Values')
    ax.set_ylabel('Reconstructed Values')
    ax.set_title('Original vs Reconstructed')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual distributions
    ax = axes[0, 2]
    for name, recon in reconstructions_dict.items():
        residuals = (original - recon).flatten()
        ax.hist(residuals, bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spatial error maps (if 2D data)
    if len(original.shape) >= 2:
        for idx, (name, recon) in enumerate(list(reconstructions_dict.items())[:6]):
            if idx < 6:
                ax = axes[1 + idx // 3, idx % 3]
                error_map = np.abs(original - recon)
                if len(error_map.shape) > 2:
                    error_map = np.mean(error_map, axis=-1)
                im = ax.imshow(error_map[:, :] if len(error_map.shape) >= 2 else error_map.reshape(-1, 1),
                              cmap='hot', interpolation='nearest')
                ax.set_title(f'{name} - Error Map')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_perturbation_impact_visualization(original, perturbed_dict, save_path,
                                            title="Perturbation Impact Analysis", dpi=300):
    """Visualize the impact of perturbations on reconstruction"""
    n_perturbations = len(perturbed_dict)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Calculate impacts
    impacts = {}
    for name, perturbed in perturbed_dict.items():
        impacts[name] = {
            'mae': np.mean(np.abs(original - perturbed)),
            'rmse': np.sqrt(np.mean((original - perturbed)**2)),
            'max_error': np.max(np.abs(original - perturbed)),
            'correlation': np.corrcoef(original.flatten(), perturbed.flatten())[0, 1]
        }

    # Plot impact metrics
    metrics = ['mae', 'rmse', 'max_error', 'correlation']
    metric_names = ['Mean Absolute Error', 'RMSE', 'Max Error', 'Correlation']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        values = [impacts[name][metric] for name in perturbed_dict.keys()]
        colors = ['red' if metric != 'correlation' else 'green' for _ in values]

        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(list(perturbed_dict.keys()), rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Perturbation')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Difference distributions
    ax = axes[4]
    for name, perturbed in list(perturbed_dict.items())[:5]:  # Limit to 5 for clarity
        diff = np.abs(original - perturbed).flatten()
        ax.hist(diff, bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Absolute Difference')
    ax.set_ylabel('Density')
    ax.set_title('Difference Distributions')
    ax.legend()

    # Summary statistics table
    ax = axes[5]
    ax.axis('off')

    # Create summary table
    summary_data = []
    for name in perturbed_dict.keys():
        summary_data.append([
            name,
            f"{impacts[name]['mae']:.4f}",
            f"{impacts[name]['rmse']:.4f}",
            f"{impacts[name]['correlation']:.4f}"
        ])

    table = ax.table(cellText=summary_data,
                    colLabels=['Perturbation', 'MAE', 'RMSE', 'Correlation'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def create_statistical_distribution_plots(data_dict, save_path,
                                         title="Statistical Distributions", dpi=300):
    """Create statistical distribution visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Box plots
    ax = axes[0, 0]
    box_data = [data.flatten() for data in data_dict.values()]
    bp = ax.boxplot(box_data, labels=list(data_dict.keys()), patch_artist=True)
    for patch, color in zip(bp['boxes'], COLOR_PALETTES['default']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Value')
    ax.set_title('Distribution Box Plots')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Violin plots
    ax = axes[0, 1]
    parts = ax.violinplot(box_data, positions=range(len(data_dict)),
                          showmeans=True, showmedians=True)
    ax.set_xticks(range(len(data_dict)))
    ax.set_xticklabels(list(data_dict.keys()), rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Distribution Violin Plots')
    ax.grid(True, alpha=0.3)

    # KDE plots
    ax = axes[0, 2]
    for name, data in data_dict.items():
        values = data.flatten()
        density = stats.gaussian_kde(values)
        x = np.linspace(values.min(), values.max(), 100)
        ax.plot(x, density(x), label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plots
    ax = axes[1, 0]
    for i, (name, data) in enumerate(list(data_dict.items())[:3]):  # Limit for clarity
        values = data.flatten()
        stats.probplot(values, dist="norm", plot=ax)
    ax.set_title('Q-Q Plots (Normal Distribution)')
    ax.grid(True, alpha=0.3)

    # Cumulative distributions
    ax = axes[1, 1]
    for name, data in data_dict.items():
        values = np.sort(data.flatten())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values[::100], cumulative[::100], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Functions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistical summary table
    ax = axes[1, 2]
    ax.axis('off')

    summary_stats = []
    for name, data in data_dict.items():
        values = data.flatten()
        summary_stats.append([
            name[:10],  # Truncate long names
            f"{np.mean(values):.3f}",
            f"{np.std(values):.3f}",
            f"{np.median(values):.3f}",
            f"{stats.skew(values):.3f}",
            f"{stats.kurtosis(values):.3f}"
        ])

    table = ax.table(cellText=summary_stats,
                    colLabels=['Method', 'Mean', 'Std', 'Median', 'Skew', 'Kurtosis'],
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax.set_title('Statistical Summary')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# Initialize visualization directory structure
vis_base_dir = create_visualization_dirs()
print(f"Visualization directories created at: {vis_base_dir}")
print("Enhanced visualization module loaded successfully!")
print("Available functions:")
print("  - export_roi_overlay_with_colors()")
print("  - create_difference_visualization()")
print("  - plot_spectral_signatures_comparison()")
print("  - create_3d_latent_space_visualization()")
print("  - create_wavelength_importance_heatmap()")
print("  - create_perturbation_impact_visualization()")
print("  - create_reconstruction_quality_plots()")
print("  - calculate_advanced_clustering_metrics()")
print("  - create_statistical_distribution_plots()")