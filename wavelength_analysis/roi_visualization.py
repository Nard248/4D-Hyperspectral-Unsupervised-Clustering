"""
ROI Overlay Visualization Module
Creates visualizations showing ROI regions overlaid on clustering results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path


def create_roi_overlay_visualization(cluster_map, roi_regions, output_path, title="Clustering with ROI Overlay", dpi=300):
    """
    Create visualization showing clustering result with ROI regions overlaid.

    Args:
        cluster_map: 2D array of cluster labels
        roi_regions: List of ROI dictionaries with 'coords' and 'color' keys
        output_path: Path to save the figure
        title: Title for the plot
        dpi: Resolution for saving
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Clustering result without overlay
    ax1 = axes[0]
    im1 = ax1.imshow(cluster_map, cmap='tab10', interpolation='nearest')
    ax1.set_title('Clustering Result', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='Cluster ID', fraction=0.046)

    # Panel 2: Clustering with ROI boxes
    ax2 = axes[1]
    im2 = ax2.imshow(cluster_map, cmap='tab10', interpolation='nearest', alpha=0.8)

    # Draw ROI rectangles
    for roi in roi_regions:
        y_start, y_end, x_start, x_end = roi['coords']
        width = x_end - x_start
        height = y_end - y_start

        # Draw rectangle
        rect = Rectangle((x_start, y_start), width, height,
                        linewidth=3, edgecolor=roi['color'],
                        facecolor='none', linestyle='-')
        ax2.add_patch(rect)

        # Add label
        ax2.text(x_start + width/2, y_start - 5, roi['name'],
                color=roi['color'], fontsize=10, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.set_title('Clustering with ROI Overlay', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Cluster ID', fraction=0.046)

    # Panel 3: ROI regions highlighted
    ax3 = axes[2]

    # Create highlight mask
    highlight_img = np.ones((*cluster_map.shape, 3)) * 0.3  # Dark background

    for idx, roi in enumerate(roi_regions):
        y_start, y_end, x_start, x_end = roi['coords']

        # Get color as RGB
        color_hex = roi['color'].lstrip('#')
        color_rgb = tuple(int(color_hex[i:i+2], 16)/255.0 for i in (0, 2, 4))

        # Highlight this ROI
        highlight_img[y_start:y_end, x_start:x_end] = color_rgb

    ax3.imshow(highlight_img)
    ax3.set_title('ROI Regions', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Add legend
    legend_elements = [mpatches.Patch(facecolor=roi['color'], label=roi['name'])
                      for roi in roi_regions]
    ax3.legend(handles=legend_elements, loc='center', fontsize=10,
              bbox_to_anchor=(0.5, -0.05), ncol=len(roi_regions))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROI overlay: {output_path.name}")


def create_roi_analysis_report(cluster_map, roi_regions, output_path):
    """
    Create a detailed analysis report of how each ROI is clustered.

    Args:
        cluster_map: 2D array of cluster labels
        roi_regions: List of ROI dictionaries
        output_path: Path to save the report
    """
    fig, axes = plt.subplots(len(roi_regions), 2, figsize=(12, 4*len(roi_regions)))

    if len(roi_regions) == 1:
        axes = axes.reshape(1, -1)

    for idx, roi in enumerate(roi_regions):
        y_start, y_end, x_start, x_end = roi['coords']

        # Extract ROI data
        roi_clusters = cluster_map[y_start:y_end, x_start:x_end]

        # Panel 1: ROI clustering
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(roi_clusters, cmap='tab10', interpolation='nearest')
        ax1.set_title(f"{roi['name']} - Clustering", fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Panel 2: Cluster distribution
        ax2 = axes[idx, 1]
        unique, counts = np.unique(roi_clusters[roi_clusters >= 0], return_counts=True)

        if len(unique) > 0:
            colors = plt.cm.tab10(unique / 10.0)
            bars = ax2.bar(unique, counts, color=colors, alpha=0.7)
            ax2.set_xlabel('Cluster ID', fontsize=10)
            ax2.set_ylabel('Pixel Count', fontsize=10)
            ax2.set_title(f"{roi['name']} - Distribution", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Add percentage labels
            total = counts.sum()
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                pct = 100 * count / total
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{pct:.1f}%',
                        ha='center', va='bottom', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No valid pixels', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title(f"{roi['name']} - Distribution", fontsize=12, fontweight='bold')

    plt.suptitle('ROI Analysis Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved ROI analysis: {output_path.name}")
