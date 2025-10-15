"""
Enhanced visualization functions for paper publication
Generates comparison difference maps between ground truth and classifications
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Tuple, Dict


def create_ground_truth_difference_maps(
        ground_truth: np.ndarray,
        baseline_labels: np.ndarray,
        optimized_labels: np.ndarray,
        mask: np.ndarray,
        config_name: str,
        output_dir: Path,
        baseline_purity: float,
        optimized_purity: float
) -> Dict[str, any]:
    """
    Create two difference maps comparing against ground truth:
    1. Ground Truth vs Full Spectral (baseline)
    2. Ground Truth vs Sub-selection (our method)

    This visually demonstrates noise reduction in our approach.

    Args:
        ground_truth: Ground truth labels
        baseline_labels: Baseline (full spectral) classification
        optimized_labels: Optimized (sub-selection) classification
        mask: Valid pixel mask
        config_name: Configuration name
        output_dir: Output directory for images
        baseline_purity: Baseline purity score
        optimized_purity: Optimized purity score

    Returns:
        Dictionary with statistics about both difference maps
    """
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Map cluster labels to ground truth using Hungarian algorithm
    # ========================================================================
    # Cluster labels are arbitrary, so we need to find the best mapping
    # between predicted labels and ground truth labels before comparison

    def map_labels_to_ground_truth(predicted_labels, ground_truth_labels, mask):
        """
        Find optimal mapping from predicted labels to ground truth labels.
        Uses the SAME method as purity calculation: assign each cluster to its majority ground truth class.
        """
        # Flatten and filter valid pixels
        pred_flat = predicted_labels[mask].flatten()
        gt_flat = ground_truth_labels[mask].flatten()

        # Get unique labels
        unique_clusters = np.unique(pred_flat[pred_flat >= 0])
        unique_gt = np.unique(gt_flat[gt_flat >= 0])

        # Build contingency table: rows=clusters, cols=ground truth classes
        contingency = np.zeros((len(unique_clusters), len(unique_gt)))

        for i, cluster in enumerate(unique_clusters):
            for j, gt_class in enumerate(unique_gt):
                contingency[i, j] = np.sum((pred_flat == cluster) & (gt_flat == gt_class))

        # For each cluster, assign it to the ground truth class with maximum overlap
        label_mapping = {}
        for i, cluster in enumerate(unique_clusters):
            # Find the ground truth class with maximum overlap
            best_gt_idx = np.argmax(contingency[i, :])
            best_gt = unique_gt[best_gt_idx]
            label_mapping[int(cluster)] = int(best_gt)

        # Map predicted labels to ground truth space
        mapped_labels = np.copy(predicted_labels)
        for cluster_id, gt_id in label_mapping.items():
            mapped_labels[predicted_labels == cluster_id] = gt_id

        return mapped_labels, label_mapping

    # Map baseline labels to ground truth
    baseline_mapped, baseline_mapping = map_labels_to_ground_truth(baseline_labels, ground_truth, mask)

    # Map optimized labels to ground truth
    optimized_mapped, optimized_mapping = map_labels_to_ground_truth(optimized_labels, ground_truth, mask)

    print(f"  Label mapping - Baseline: {baseline_mapping}")
    print(f"  Label mapping - Optimized: {optimized_mapping}")

    # Calculate misclassifications using MAPPED labels
    baseline_correct = (baseline_mapped == ground_truth) & mask
    optimized_correct = (optimized_mapped == ground_truth) & mask

    baseline_misclassified = (~baseline_correct) & mask
    optimized_misclassified = (~optimized_correct) & mask

    # Count pixels
    n_valid = np.sum(mask)
    n_baseline_correct = np.sum(baseline_correct)
    n_optimized_correct = np.sum(optimized_correct)
    n_baseline_wrong = np.sum(baseline_misclassified)
    n_optimized_wrong = np.sum(optimized_misclassified)

    # Calculate noise reduction
    noise_reduction = n_baseline_wrong - n_optimized_wrong
    noise_reduction_pct = (noise_reduction / n_baseline_wrong * 100) if n_baseline_wrong > 0 else 0

    # Create figure with both difference maps
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Define colors
    correct_color = '#2ecc71'  # Green for correct
    wrong_color = '#e74c3c'    # Red for misclassified
    background_color = '#ecf0f1'  # Light gray for background

    # Color map: 0=background, 1=correct, 2=misclassified
    colors = [background_color, correct_color, wrong_color]
    cmap = mcolors.ListedColormap(colors)

    # --- Plot 1: Ground Truth ---
    axes[0, 0].imshow(ground_truth, cmap='tab10')
    axes[0, 0].set_title('Ground Truth\n(Manual Annotation)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # --- Plot 2: Baseline Difference Map (GT vs Full Spectral) ---
    baseline_diff_map = np.zeros_like(ground_truth, dtype=int)
    baseline_diff_map[baseline_correct] = 1  # Correct
    baseline_diff_map[baseline_misclassified] = 2  # Misclassified

    im2 = axes[0, 1].imshow(baseline_diff_map, cmap=cmap, vmin=0, vmax=2)
    axes[0, 1].set_title(
        f'Ground Truth vs Full Spectral\n'
        f'Purity: {baseline_purity:.3f} | Errors: {n_baseline_wrong:,} ({n_baseline_wrong/n_valid*100 if n_valid > 0 else 0:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    axes[0, 1].axis('off')

    # --- Plot 3: Optimized Difference Map (GT vs Sub-selection) ---
    optimized_diff_map = np.zeros_like(ground_truth, dtype=int)
    optimized_diff_map[optimized_correct] = 1  # Correct
    optimized_diff_map[optimized_misclassified] = 2  # Misclassified

    im3 = axes[1, 0].imshow(optimized_diff_map, cmap=cmap, vmin=0, vmax=2)
    axes[1, 0].set_title(
        f'Ground Truth vs Sub-selection ({config_name})\n'
        f'Purity: {optimized_purity:.3f} | Errors: {n_optimized_wrong:,} ({n_optimized_wrong/n_valid*100 if n_valid > 0 else 0:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    axes[1, 0].axis('off')

    # --- Plot 4: Noise Reduction Visualization ---
    # Show where we improved (baseline wrong, optimized correct)
    # and where we got worse (baseline correct, optimized wrong)
    improvement_map = np.zeros_like(ground_truth, dtype=int)

    improved = baseline_misclassified & optimized_correct
    degraded = baseline_correct & optimized_misclassified
    still_wrong = baseline_misclassified & optimized_misclassified

    improvement_map[improved] = 1  # Improved (green)
    improvement_map[degraded] = 2  # Degraded (red)
    improvement_map[still_wrong] = 3  # Still wrong (orange)

    improve_colors = [background_color, '#27ae60', '#c0392b', '#e67e22']  # background, green, red, orange
    improve_cmap = mcolors.ListedColormap(improve_colors)

    im4 = axes[1, 1].imshow(improvement_map, cmap=improve_cmap, vmin=0, vmax=3)
    axes[1, 1].set_title(
        f'Noise Reduction Analysis\n'
        f'Improved: {np.sum(improved):,} | Degraded: {np.sum(degraded):,} | '
        f'Net Reduction: {noise_reduction:,} ({noise_reduction_pct:+.1f}%)',
        fontsize=14, fontweight='bold'
    )
    axes[1, 1].axis('off')

    # Add legends
    # Legend for difference maps
    from matplotlib.patches import Patch
    legend_elements_diff = [
        Patch(facecolor=correct_color, label=f'Correct Classification'),
        Patch(facecolor=wrong_color, label=f'Misclassified')
    ]
    axes[0, 1].legend(handles=legend_elements_diff, loc='upper right', fontsize=10)
    axes[1, 0].legend(handles=legend_elements_diff, loc='upper right', fontsize=10)

    # Legend for improvement map
    legend_elements_improve = [
        Patch(facecolor='#27ae60', label=f'Improved ({np.sum(improved):,} pixels)'),
        Patch(facecolor='#c0392b', label=f'Degraded ({np.sum(degraded):,} pixels)'),
        Patch(facecolor='#e67e22', label=f'Still Wrong ({np.sum(still_wrong):,} pixels)')
    ]
    axes[1, 1].legend(handles=legend_elements_improve, loc='upper right', fontsize=10)

    # Overall title
    plt.suptitle(
        f'Ground Truth Comparison: {config_name}\n'
        f'Baseline Errors: {n_baseline_wrong:,} → Our Method Errors: {n_optimized_wrong:,} '
        f'(Noise Reduction: {noise_reduction:,} pixels, {noise_reduction_pct:+.1f}%)',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = output_dir / f"{config_name}_ground_truth_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also save individual maps for flexibility
    _save_individual_difference_maps(
        baseline_diff_map, optimized_diff_map, improvement_map,
        config_name, output_dir, cmap, improve_cmap,
        baseline_purity, optimized_purity,
        n_baseline_wrong, n_optimized_wrong, n_valid
    )

    # Return statistics
    stats = {
        'baseline_correct': int(n_baseline_correct),
        'baseline_wrong': int(n_baseline_wrong),
        'optimized_correct': int(n_optimized_correct),
        'optimized_wrong': int(n_optimized_wrong),
        'noise_reduction': int(noise_reduction),
        'noise_reduction_pct': float(noise_reduction_pct),
        'improved_pixels': int(np.sum(improved)),
        'degraded_pixels': int(np.sum(degraded)),
        'still_wrong_pixels': int(np.sum(still_wrong)),
        'baseline_purity': float(baseline_purity),
        'optimized_purity': float(optimized_purity)
    }

    return stats


def _save_individual_difference_maps(
        baseline_diff_map, optimized_diff_map, improvement_map,
        config_name, output_dir, cmap, improve_cmap,
        baseline_purity, optimized_purity,
        n_baseline_wrong, n_optimized_wrong, n_valid
):
    """Save individual difference maps as separate high-resolution images"""

    # Baseline difference map
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(baseline_diff_map, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(
        f'Ground Truth vs Full Spectral\n'
        f'Purity: {baseline_purity:.3f} | Errors: {n_baseline_wrong:,}/{n_valid:,} ({n_baseline_wrong/n_valid*100 if n_valid > 0 else 0:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{config_name}_GT_vs_Baseline.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Optimized difference map
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(optimized_diff_map, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(
        f'Ground Truth vs Sub-selection\n'
        f'Purity: {optimized_purity:.3f} | Errors: {n_optimized_wrong:,}/{n_valid:,} ({n_optimized_wrong/n_valid*100 if n_valid > 0 else 0:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{config_name}_GT_vs_Optimized.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Improvement map
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(improvement_map, cmap=improve_cmap, vmin=0, vmax=3)
    noise_reduction = n_baseline_wrong - n_optimized_wrong
    noise_reduction_pct = (noise_reduction / n_baseline_wrong * 100) if n_baseline_wrong > 0 else 0
    ax.set_title(
        f'Noise Reduction Analysis\n'
        f'Net Reduction: {noise_reduction:,} pixels ({noise_reduction_pct:+.1f}%)',
        fontsize=14, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{config_name}_noise_reduction.png", dpi=300, bbox_inches='tight')
    plt.close()
