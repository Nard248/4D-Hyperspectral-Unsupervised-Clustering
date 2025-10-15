"""
Enhanced Difference Map Visualization
=====================================
Shows the difference between baseline and optimized clustering to demonstrate noise reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
import seaborn as sns


def create_enhanced_difference_map(baseline_labels, optimized_labels, mask=None,
                                   output_path=None, title="Noise Reduction Visualization",
                                   dpi=300, save_individual_panels=True, show_title=True,
                                   background_color='white'):
    """
    Create an enhanced difference map showing noise reduction.

    Parameters:
    -----------
    baseline_labels : np.ndarray
        Cluster labels from baseline (full wavelength) method
    optimized_labels : np.ndarray
        Cluster labels from optimized (subset wavelength) method
    mask : np.ndarray, optional
        Boolean mask for valid pixels
    output_path : str or Path, optional
        Where to save the figure
    title : str
        Figure title
    dpi : int
        Resolution for saving
    save_individual_panels : bool
        If True, saves each panel as a separate image
    show_title : bool
        If True, shows titles on panels and main figure
    background_color : str
        Background color for masked regions ('white', 'black', 'grey', etc.)

    Returns:
    --------
    difference_stats : dict
        Statistics about the differences
    """

    # Handle mask
    if mask is None:
        mask = np.ones_like(baseline_labels, dtype=bool)

    # Calculate difference map
    difference_map = (baseline_labels != optimized_labels).astype(float)

    # Apply mask
    difference_map_masked = np.ma.masked_where(~mask, difference_map)

    # Calculate statistics
    total_pixels = np.sum(mask)
    different_pixels = np.sum(difference_map[mask])
    agreement_pixels = total_pixels - different_pixels
    agreement_pct = (agreement_pixels / total_pixels) * 100
    noise_reduction_pct = (different_pixels / total_pixels) * 100

    # Create figure with specified background color
    fig = plt.figure(figsize=(18, 12), facecolor=background_color)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Setup for individual panel saving
    if save_individual_panels and output_path:
        individual_dir = Path(output_path).parent / f"{Path(output_path).stem}_panels"
        individual_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to save individual panel
    def save_panel(fig_panel, panel_name, panel_title=""):
        if save_individual_panels and output_path:
            panel_path = individual_dir / f"{panel_name}.png"
            fig_panel.savefig(panel_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
            return panel_path
        return None

    # ========================================================================
    # Panel 1: Baseline Clustering
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0], facecolor=background_color)
    baseline_display = np.ma.masked_where(~mask, baseline_labels)
    # Set masked values color to background
    cmap1 = plt.cm.tab10
    cmap1.set_bad(background_color)
    im1 = ax1.imshow(baseline_display, cmap=cmap1, interpolation='nearest')
    if show_title:
        ax1.set_title('Baseline (All Wavelengths)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, label='Cluster ID')

    # Save individual panel
    if save_individual_panels and output_path:
        fig_panel = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_temp = fig_panel.add_subplot(111, facecolor=background_color)
        ax_temp.imshow(baseline_display, cmap=cmap1, interpolation='nearest')
        if show_title:
            ax_temp.set_title('Baseline (All Wavelengths)', fontsize=14, fontweight='bold')
        ax_temp.axis('off')
        save_panel(fig_panel, "01_baseline_clustering")
        plt.close(fig_panel)

    # ========================================================================
    # Panel 2: Optimized Clustering
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1], facecolor=background_color)
    optimized_display = np.ma.masked_where(~mask, optimized_labels)
    cmap2 = plt.cm.tab10
    cmap2.set_bad(background_color)
    im2 = ax2.imshow(optimized_display, cmap=cmap2, interpolation='nearest')
    if show_title:
        ax2.set_title('Optimized (Selected Wavelengths)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, label='Cluster ID')

    # ========================================================================
    # Panel 3: Enhanced Difference Map (RED for differences)
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2], facecolor=background_color)

    # Create custom colormap: White (agreement) to Red (difference)
    # Enhanced exposure by making red brighter
    colors_diff = ['#00FF00', '#FF0000']  # Green = agreement, Red = difference
    n_bins = 100
    cmap_diff = mcolors.LinearSegmentedColormap.from_list('agreement', colors_diff, N=n_bins)
    cmap_diff.set_bad(background_color)

    # Apply contrast enhancement
    enhanced_diff = difference_map_masked.copy()
    # Enhance the red (difference) areas
    enhanced_diff = np.power(enhanced_diff, 0.5)  # Gamma correction for visibility

    im3 = ax3.imshow(enhanced_diff, cmap=cmap_diff, interpolation='nearest',
                     vmin=0, vmax=1, alpha=0.9)
    if show_title:
        ax3.set_title('Difference Map (Green=Agreement, Red=Changed)',
                     fontsize=12, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
    cbar3.set_label('Difference Intensity', fontsize=10)

    # ========================================================================
    # Panel 4: Red-Only Difference Overlay
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 0], facecolor=background_color)

    # Create RGB image with red highlighting differences
    rgb_diff = np.zeros((*difference_map.shape, 3))
    # Green channel for agreement areas (subtle)
    rgb_diff[:, :, 1] = 0.3 * (1 - difference_map) * mask
    # Red channel for differences (bright)
    rgb_diff[:, :, 0] = 1.0 * difference_map * mask
    # Enhance red visibility
    rgb_diff[:, :, 0] = np.clip(rgb_diff[:, :, 0] * 2.5, 0, 1)

    ax4.imshow(rgb_diff, interpolation='nearest')
    if show_title:
        ax4.set_title('Red-Enhanced Differences\n(Noise Reduction Areas)',
                     fontsize=12, fontweight='bold')
    ax4.axis('off')

    # ========================================================================
    # Panel 5: Difference Intensity Heatmap
    # ========================================================================
    ax5 = fig.add_subplot(gs[1, 1], facecolor=background_color)

    # Create intensity map based on local difference density
    from scipy.ndimage import gaussian_filter
    diff_intensity = gaussian_filter(difference_map.astype(float), sigma=3)
    diff_intensity_masked = np.ma.masked_where(~mask, diff_intensity)

    cmap5 = plt.cm.hot
    cmap5.set_bad(background_color)
    im5 = ax5.imshow(diff_intensity_masked, cmap=cmap5, interpolation='bilinear')
    if show_title:
        ax5.set_title('Noise Density Map\n(Smoothed Differences)',
                     fontsize=12, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, label='Noise Density')

    # ========================================================================
    # Panel 6: Agreement Statistics
    # ========================================================================
    ax6 = fig.add_subplot(gs[1, 2], facecolor=background_color)

    # Create pie chart
    sizes = [agreement_pct, noise_reduction_pct]
    colors = ['#2ECC71', '#E74C3C']  # Green for agreement, Red for differences
    explode = (0, 0.1)  # Explode the noise slice

    wedges, texts, autotexts = ax6.pie(sizes, explode=explode, labels=['Agreement', 'Changed'],
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})

    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    if show_title:
        ax6.set_title('Pixel Agreement Analysis', fontsize=12, fontweight='bold')

    # ========================================================================
    # Panel 7: Spatial Distribution of Differences
    # ========================================================================
    ax7 = fig.add_subplot(gs[2, :2], facecolor=background_color)

    # Calculate row-wise and column-wise difference distributions
    row_diff = np.sum(difference_map * mask, axis=1)
    col_diff = np.sum(difference_map * mask, axis=0)

    ax7_twin = ax7.twinx()

    ax7.fill_between(range(len(row_diff)), row_diff, alpha=0.6, color='#E74C3C',
                     label='Row-wise differences')
    ax7_twin.fill_between(range(len(col_diff)), col_diff, alpha=0.4, color='#3498DB',
                         label='Column-wise differences')

    ax7.set_xlabel('Spatial Position', fontsize=11)
    ax7.set_ylabel('Row Differences', fontsize=11, color='#E74C3C')
    ax7_twin.set_ylabel('Column Differences', fontsize=11, color='#3498DB')
    if show_title:
        ax7.set_title('Spatial Distribution of Noise/Changes', fontsize=12, fontweight='bold')
    ax7.tick_params(axis='y', labelcolor='#E74C3C')
    ax7_twin.tick_params(axis='y', labelcolor='#3498DB')
    ax7.grid(True, alpha=0.3)

    # ========================================================================
    # Panel 8: Statistics Summary
    # ========================================================================
    ax8 = fig.add_subplot(gs[2, 2], facecolor=background_color)
    ax8.axis('off')

    # Calculate cluster-wise differences
    unique_baseline = np.unique(baseline_labels[mask])
    unique_optimized = np.unique(optimized_labels[mask])

    stats_text = f"""
    NOISE REDUCTION ANALYSIS
    {'='*35}

    Total Pixels Analyzed: {total_pixels:,}

    Agreement: {agreement_pixels:,} ({agreement_pct:.2f}%)
    Changed: {different_pixels:,} ({noise_reduction_pct:.2f}%)

    Baseline Clusters: {len(unique_baseline)}
    Optimized Clusters: {len(unique_optimized)}

    Noise Reduction Rate: {noise_reduction_pct:.2f}%

    Interpretation:
    • Green areas: Consistent clustering
    • Red areas: Refined/corrected pixels
    • Higher red intensity = More correction

    The optimized method corrected
    {different_pixels:,} pixels by using
    selected informative wavelengths.
    """

    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Main title
    if show_title:
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    # Save combined figure
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color)
        plt.savefig(str(output_path).replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight', facecolor=background_color)
        print(f"✅ Saved difference map to: {output_path}")

        if save_individual_panels:
            print(f"✅ Saved individual panels to: {individual_dir}")
            print(f"   Total panels saved: {len(list(individual_dir.glob('*.png')))}")

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Save all remaining individual panels
    # ========================================================================
    if save_individual_panels and output_path:
        # Panel 2: Optimized clustering
        fig_p2 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p2 = fig_p2.add_subplot(111, facecolor=background_color)
        optimized_display = np.ma.masked_where(~mask, optimized_labels)
        cmap2 = plt.cm.tab10
        cmap2.set_bad(background_color)
        ax_p2.imshow(optimized_display, cmap=cmap2, interpolation='nearest')
        if show_title:
            ax_p2.set_title('Optimized (Selected Wavelengths)', fontsize=14, fontweight='bold')
        ax_p2.axis('off')
        save_panel(fig_p2, "02_optimized_clustering")
        plt.close(fig_p2)

        # Panel 3: Enhanced difference map
        fig_p3 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p3 = fig_p3.add_subplot(111, facecolor=background_color)
        colors_diff = ['#00FF00', '#FF0000']
        cmap_diff = mcolors.LinearSegmentedColormap.from_list('agreement', colors_diff, N=100)
        enhanced_diff = difference_map_masked.copy()
        enhanced_diff = np.power(enhanced_diff, 0.5)
        ax_p3.imshow(enhanced_diff, cmap=cmap_diff, interpolation='nearest', vmin=0, vmax=1, alpha=0.9)
        if show_title:
            ax_p3.set_title('Difference Map (Green=Agreement, Red=Changed)', fontsize=14, fontweight='bold')
        ax_p3.axis('off')
        save_panel(fig_p3, "03_enhanced_difference_map")
        plt.close(fig_p3)

        # Panel 4: Red-only overlay
        fig_p4 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p4 = fig_p4.add_subplot(111, facecolor=background_color)
        rgb_diff = np.zeros((*difference_map.shape, 3))
        rgb_diff[:, :, 1] = 0.3 * (1 - difference_map) * mask
        rgb_diff[:, :, 0] = 1.0 * difference_map * mask
        rgb_diff[:, :, 0] = np.clip(rgb_diff[:, :, 0] * 2.5, 0, 1)
        ax_p4.imshow(rgb_diff, interpolation='nearest')
        if show_title:
            ax_p4.set_title('Red-Enhanced Differences', fontsize=14, fontweight='bold')
        ax_p4.axis('off')
        save_panel(fig_p4, "04_red_enhanced_differences")
        plt.close(fig_p4)

        # Panel 5: Difference intensity heatmap
        fig_p5 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p5 = fig_p5.add_subplot(111, facecolor=background_color)
        from scipy.ndimage import gaussian_filter
        diff_intensity = gaussian_filter(difference_map.astype(float), sigma=3)
        diff_intensity_masked = np.ma.masked_where(~mask, diff_intensity)
        ax_p5.imshow(diff_intensity_masked, cmap='hot', interpolation='bilinear')
        if show_title:
            ax_p5.set_title('Noise Density Map', fontsize=14, fontweight='bold')
        ax_p5.axis('off')
        save_panel(fig_p5, "05_noise_density_heatmap")
        plt.close(fig_p5)

        # Panel 6: Agreement pie chart
        fig_p6 = plt.figure(figsize=(6, 6), facecolor=background_color)
        ax_p6 = fig_p6.add_subplot(111, facecolor=background_color)
        sizes = [agreement_pct, noise_reduction_pct]
        colors = ['#2ECC71', '#E74C3C']
        explode = (0, 0.1)
        ax_p6.pie(sizes, explode=explode, labels=['Agreement', 'Changed'],
                  colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,
                  textprops={'fontsize': 12, 'fontweight': 'bold'})
        if show_title:
            ax_p6.set_title('Pixel Agreement Analysis', fontsize=14, fontweight='bold')
        save_panel(fig_p6, "06_agreement_pie_chart")
        plt.close(fig_p6)

        # Panel 7: Spatial distribution
        fig_p7 = plt.figure(figsize=(12, 6), facecolor=background_color)
        ax_p7 = fig_p7.add_subplot(111, facecolor=background_color)
        ax_p7_twin = ax_p7.twinx()
        row_diff = np.sum(difference_map * mask, axis=1)
        col_diff = np.sum(difference_map * mask, axis=0)
        ax_p7.fill_between(range(len(row_diff)), row_diff, alpha=0.6, color='#E74C3C',
                           label='Row-wise differences')
        ax_p7_twin.fill_between(range(len(col_diff)), col_diff, alpha=0.4, color='#3498DB',
                                label='Column-wise differences')
        ax_p7.set_xlabel('Spatial Position', fontsize=12)
        ax_p7.set_ylabel('Row Differences', fontsize=12, color='#E74C3C')
        ax_p7_twin.set_ylabel('Column Differences', fontsize=12, color='#3498DB')
        if show_title:
            ax_p7.set_title('Spatial Distribution of Noise/Changes', fontsize=14, fontweight='bold')
        ax_p7.tick_params(axis='y', labelcolor='#E74C3C')
        ax_p7_twin.tick_params(axis='y', labelcolor='#3498DB')
        ax_p7.grid(True, alpha=0.3)
        save_panel(fig_p7, "07_spatial_distribution")
        plt.close(fig_p7)

        # Panel 8: Statistics summary
        fig_p8 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p8 = fig_p8.add_subplot(111, facecolor=background_color)
        ax_p8.axis('off')
        unique_baseline = np.unique(baseline_labels[mask])
        unique_optimized = np.unique(optimized_labels[mask])
        stats_text = f"""
    NOISE REDUCTION ANALYSIS
    {'='*35}

    Total Pixels Analyzed: {total_pixels:,}

    Agreement: {agreement_pixels:,} ({agreement_pct:.2f}%)
    Changed: {different_pixels:,} ({noise_reduction_pct:.2f}%)

    Baseline Clusters: {len(unique_baseline)}
    Optimized Clusters: {len(unique_optimized)}

    Noise Reduction Rate: {noise_reduction_pct:.2f}%

    Interpretation:
    • Green areas: Consistent clustering
    • Red areas: Refined/corrected pixels
    • Higher red intensity = More correction

    The optimized method corrected
    {different_pixels:,} pixels by using
    selected informative wavelengths.
    """
        ax_p8.text(0.05, 0.95, stats_text, transform=ax_p8.transAxes,
                   fontsize=11, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        save_panel(fig_p8, "08_statistics_summary")
        plt.close(fig_p8)

    # Return statistics
    difference_stats = {
        'total_pixels': total_pixels,
        'agreement_pixels': agreement_pixels,
        'different_pixels': different_pixels,
        'agreement_pct': agreement_pct,
        'noise_reduction_pct': noise_reduction_pct,
        'baseline_clusters': len(unique_baseline),
        'optimized_clusters': len(unique_optimized)
    }

    return difference_stats


def create_simple_difference_overlay(baseline_labels, optimized_labels, mask=None,
                                     output_path=None, alpha=0.7,
                                     save_individual_panels=True, show_title=True,
                                     background_color='white'):
    """
    Create a simple red overlay showing differences (for quick visualization).

    Parameters:
    -----------
    baseline_labels : np.ndarray
        Baseline cluster labels
    optimized_labels : np.ndarray
        Optimized cluster labels
    mask : np.ndarray, optional
        Boolean mask
    output_path : str or Path, optional
        Save path
    alpha : float
        Transparency of overlay
    save_individual_panels : bool
        If True, saves each panel as a separate image
    show_title : bool
        If True, shows titles on panels and main figure
    background_color : str
        Background color for masked regions ('white', 'black', 'grey', etc.)
    """

    if mask is None:
        mask = np.ones_like(baseline_labels, dtype=bool)

    # Calculate difference
    diff = (baseline_labels != optimized_labels).astype(float)

    # Create figure with background color
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=background_color)

    # Setup for individual panel saving
    if save_individual_panels and output_path:
        individual_dir = Path(output_path).parent / f"{Path(output_path).stem}_panels"
        individual_dir.mkdir(parents=True, exist_ok=True)

    # Helper function to save individual panel
    def save_panel(fig_panel, panel_name):
        if save_individual_panels and output_path:
            panel_path = individual_dir / f"{panel_name}.png"
            fig_panel.savefig(panel_path, dpi=300, bbox_inches='tight', facecolor=background_color)
            plt.close(fig_panel)
            return panel_path
        return None

    # Baseline
    baseline_masked = np.ma.masked_where(~mask, baseline_labels)
    cmap_base = plt.cm.tab10
    cmap_base.set_bad(background_color)
    axes[0].set_facecolor(background_color)
    axes[0].imshow(baseline_masked, cmap=cmap_base)
    if show_title:
        axes[0].set_title('Baseline', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Optimized with red overlay
    optimized_masked = np.ma.masked_where(~mask, optimized_labels)
    cmap_opt = plt.cm.tab10
    cmap_opt.set_bad(background_color)
    axes[1].set_facecolor(background_color)
    axes[1].imshow(optimized_masked, cmap=cmap_opt)
    # Add red overlay for differences
    diff_overlay = np.zeros((*diff.shape, 4))
    diff_overlay[:, :, 0] = diff  # Red channel
    diff_overlay[:, :, 3] = diff * alpha  # Alpha channel
    axes[1].imshow(diff_overlay, interpolation='nearest')
    if show_title:
        axes[1].set_title('Optimized (Red = Changed Pixels)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Pure difference map
    diff_masked = np.ma.masked_where(~mask, diff)
    cmap_red = plt.cm.Reds
    cmap_red.set_bad(background_color)
    axes[2].set_facecolor(background_color)
    axes[2].imshow(diff_masked, cmap=cmap_red, interpolation='nearest')
    if show_title:
        axes[2].set_title('Pure Difference Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    agreement_pct = (1 - np.sum(diff[mask]) / np.sum(mask)) * 100
    if show_title:
        plt.suptitle(f'Difference Visualization (Agreement: {agreement_pct:.1f}%)',
                    fontsize=16, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=background_color)
        print(f"✅ Saved simple difference map to: {output_path}")

    # Save individual panels
    if save_individual_panels and output_path:
        # Panel 1: Baseline
        fig_p1 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p1 = fig_p1.add_subplot(111, facecolor=background_color)
        ax_p1.imshow(baseline_masked, cmap=cmap_base)
        if show_title:
            ax_p1.set_title('Baseline', fontsize=14, fontweight='bold')
        ax_p1.axis('off')
        save_panel(fig_p1, "01_baseline")

        # Panel 2: Optimized with red overlay
        fig_p2 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p2 = fig_p2.add_subplot(111, facecolor=background_color)
        ax_p2.imshow(optimized_masked, cmap=cmap_opt)
        ax_p2.imshow(diff_overlay, interpolation='nearest')
        if show_title:
            ax_p2.set_title('Optimized (Red = Changed Pixels)', fontsize=14, fontweight='bold')
        ax_p2.axis('off')
        save_panel(fig_p2, "02_optimized_with_overlay")

        # Panel 3: Pure difference
        fig_p3 = plt.figure(figsize=(8, 8), facecolor=background_color)
        ax_p3 = fig_p3.add_subplot(111, facecolor=background_color)
        ax_p3.imshow(diff_masked, cmap=cmap_red, interpolation='nearest')
        if show_title:
            ax_p3.set_title('Pure Difference Map', fontsize=14, fontweight='bold')
        ax_p3.axis('off')
        save_panel(fig_p3, "03_pure_difference")

        print(f"✅ Saved individual panels to: {individual_dir}")

    plt.show()


if __name__ == "__main__":
    print("Enhanced Difference Visualization Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from enhanced_difference_visualization import create_enhanced_difference_map")
    print("  stats = create_enhanced_difference_map(baseline_labels, optimized_labels)")
