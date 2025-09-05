"""
Ground truth validation module for hyperspectral clustering.
This module extracts ground truth labels from colored PNG annotations
and validates clustering results against them.
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
    confusion_matrix
)
from scipy.optimize import linear_sum_assignment
import warnings


def extract_ground_truth_from_png(
    png_path: Union[str, Path],
    background_colors: List[Tuple[int, int, int, int]] = None,
    target_shape: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Extract ground truth labels from a colored PNG annotation file.
    
    Args:
        png_path: Path to the PNG annotation file
        background_colors: List of RGBA tuples representing background colors to exclude
        target_shape: Optional target shape (height, width) to resize to
        
    Returns:
        Tuple of (ground_truth_labels, color_mapping, unique_colors_array)
        - ground_truth_labels: 2D array with integer labels (-1 for background)
        - color_mapping: Dict mapping label integers to color tuples
        - unique_colors_array: Array of unique colors found
    """
    if background_colors is None:
        background_colors = [
            (24, 24, 24, 255),      # Dark gray background
            (168, 168, 168, 255)    # Light gray background
        ]
    
    # Load the PNG image
    img = Image.open(png_path)
    
    # Convert to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Convert to numpy array
    img_array = np.array(img)
    original_shape = img_array.shape[:2]
    
    print(f"Original PNG shape: {original_shape}")
    
    # Handle resizing if needed
    if target_shape is not None and original_shape != target_shape:
        target_h, target_w = target_shape
        current_h, current_w = original_shape
        
        # Create new array with target shape, filled with first background color
        resized_array = np.full((target_h, target_w, 4), background_colors[0], dtype=np.uint8)
        
        # Calculate padding or cropping
        h_diff = target_h - current_h
        w_diff = target_w - current_w
        
        if h_diff >= 0 and w_diff >= 0:
            # Padding needed
            h_pad = h_diff // 2
            w_pad = w_diff // 2
            resized_array[h_pad:h_pad+current_h, w_pad:w_pad+current_w] = img_array
        elif h_diff >= 0 and w_diff < 0:
            # Pad height, crop width
            h_pad = h_diff // 2
            w_crop = abs(w_diff) // 2
            resized_array[h_pad:h_pad+current_h, :] = img_array[:, w_crop:w_crop+target_w]
        elif h_diff < 0 and w_diff >= 0:
            # Crop height, pad width
            h_crop = abs(h_diff) // 2
            w_pad = w_diff // 2
            resized_array[:, w_pad:w_pad+current_w] = img_array[h_crop:h_crop+target_h, :]
        else:
            # Crop both dimensions
            h_crop = abs(h_diff) // 2
            w_crop = abs(w_diff) // 2
            resized_array = img_array[h_crop:h_crop+target_h, w_crop:w_crop+target_w]
        
        img_array = resized_array
        print(f"Resized to: {img_array.shape[:2]}")
    
    # Find all unique colors in the image
    img_flat = img_array.reshape(-1, 4)
    unique_colors = np.unique(img_flat, axis=0)
    
    print(f"Found {len(unique_colors)} unique colors in the PNG")
    
    # Filter out background colors
    lichen_colors = []
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple not in background_colors:
            lichen_colors.append(color_tuple)
    
    print(f"Found {len(lichen_colors)} lichen type colors (excluding background)")
    
    # Create ground truth label array
    ground_truth = np.full(img_array.shape[:2], -1, dtype=int)
    
    # Create color mapping
    color_mapping = {-1: (0, 0, 0, 0)}  # Background
    
    # Assign labels to each lichen color
    for label, color in enumerate(lichen_colors):
        color_mapping[label] = color
        # Find pixels matching this color
        mask = np.all(img_array == color, axis=2)
        ground_truth[mask] = label
    
    # Print statistics
    print("\nGround Truth Statistics:")
    print(f"  Shape: {ground_truth.shape}")
    print(f"  Number of lichen types: {len(lichen_colors)}")
    
    for label in range(len(lichen_colors)):
        count = np.sum(ground_truth == label)
        percentage = 100 * count / ground_truth.size
        print(f"  Type {label} (color {color_mapping[label]}): {count} pixels ({percentage:.2f}%)")
    
    background_count = np.sum(ground_truth == -1)
    background_percentage = 100 * background_count / ground_truth.size
    print(f"  Background: {background_count} pixels ({background_percentage:.2f}%)")
    
    return ground_truth, color_mapping, np.array(lichen_colors)


def calculate_clustering_accuracy(
    cluster_labels: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate various metrics to evaluate clustering accuracy against ground truth.
    
    Args:
        cluster_labels: Predicted cluster labels (can be 1D or 2D)
        ground_truth: Ground truth labels (same shape as cluster_labels)
        valid_mask: Optional boolean mask for valid pixels
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Flatten arrays if needed
    if cluster_labels.ndim == 2:
        cluster_labels_flat = cluster_labels.flatten()
    else:
        cluster_labels_flat = cluster_labels
    
    if ground_truth.ndim == 2:
        ground_truth_flat = ground_truth.flatten()
    else:
        ground_truth_flat = ground_truth
    
    # Apply mask if provided
    if valid_mask is not None:
        if valid_mask.ndim == 2:
            valid_mask_flat = valid_mask.flatten()
        else:
            valid_mask_flat = valid_mask
        
        # Filter both arrays
        cluster_labels_flat = cluster_labels_flat[valid_mask_flat]
        ground_truth_flat = ground_truth_flat[valid_mask_flat]
    
    # Remove background pixels (-1) from both
    valid_indices = (cluster_labels_flat >= 0) & (ground_truth_flat >= 0)
    cluster_labels_valid = cluster_labels_flat[valid_indices]
    ground_truth_valid = ground_truth_flat[valid_indices]
    
    if len(cluster_labels_valid) == 0:
        warnings.warn("No valid pixels to compare!")
        return {}
    
    metrics = {}
    
    # Basic metrics that don't require same number of clusters
    metrics['adjusted_rand_score'] = adjusted_rand_score(ground_truth_valid, cluster_labels_valid)
    metrics['normalized_mutual_info'] = normalized_mutual_info_score(ground_truth_valid, cluster_labels_valid)
    metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(ground_truth_valid, cluster_labels_valid)
    metrics['fowlkes_mallows_score'] = fowlkes_mallows_score(ground_truth_valid, cluster_labels_valid)
    
    # V-measure and its components
    metrics['v_measure'] = v_measure_score(ground_truth_valid, cluster_labels_valid)
    metrics['homogeneity'] = homogeneity_score(ground_truth_valid, cluster_labels_valid)
    metrics['completeness'] = completeness_score(ground_truth_valid, cluster_labels_valid)
    
    # Calculate purity (best matching accuracy)
    purity, mapping = calculate_purity_and_mapping(cluster_labels_valid, ground_truth_valid)
    metrics['purity'] = purity
    metrics['cluster_to_gt_mapping'] = mapping
    
    # Calculate per-class metrics
    n_gt_classes = len(np.unique(ground_truth_valid))
    n_clusters = len(np.unique(cluster_labels_valid))
    
    metrics['n_ground_truth_classes'] = n_gt_classes
    metrics['n_predicted_clusters'] = n_clusters
    
    # Confusion matrix
    conf_matrix = confusion_matrix(ground_truth_valid, cluster_labels_valid)
    metrics['confusion_matrix'] = conf_matrix
    
    # Per-class precision and recall
    per_class_metrics = calculate_per_class_metrics(conf_matrix, mapping)
    metrics['per_class_precision'] = per_class_metrics['precision']
    metrics['per_class_recall'] = per_class_metrics['recall']
    metrics['per_class_f1'] = per_class_metrics['f1']
    
    return metrics


def calculate_purity_and_mapping(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> Tuple[float, Dict]:
    """
    Calculate purity score and optimal cluster-to-ground-truth mapping.
    
    Args:
        cluster_labels: Predicted cluster labels
        ground_truth: Ground truth labels
        
    Returns:
        Tuple of (purity_score, cluster_to_gt_mapping)
    """
    unique_clusters = np.unique(cluster_labels)
    unique_gt = np.unique(ground_truth)
    
    # Build contingency table
    contingency = np.zeros((len(unique_clusters), len(unique_gt)))
    
    for i, cluster in enumerate(unique_clusters):
        for j, gt_class in enumerate(unique_gt):
            contingency[i, j] = np.sum((cluster_labels == cluster) & (ground_truth == gt_class))
    
    # For purity, assign each cluster to its majority class
    cluster_to_gt = {}
    total_correct = 0
    
    for i, cluster in enumerate(unique_clusters):
        # Find the ground truth class with maximum overlap
        best_gt_idx = np.argmax(contingency[i, :])
        best_gt = unique_gt[best_gt_idx]
        cluster_to_gt[int(cluster)] = int(best_gt)
        total_correct += contingency[i, best_gt_idx]
    
    purity = total_correct / len(cluster_labels)
    
    return purity, cluster_to_gt


def calculate_per_class_metrics(confusion_matrix: np.ndarray, mapping: Dict) -> Dict:
    """
    Calculate per-class precision, recall, and F1 scores.
    
    Args:
        confusion_matrix: Confusion matrix
        mapping: Cluster to ground truth mapping
        
    Returns:
        Dictionary with per-class metrics
    """
    n_classes = confusion_matrix.shape[0]
    
    precision = {}
    recall = {}
    f1 = {}
    
    for i in range(n_classes):
        # True positives for this class
        tp = confusion_matrix[i, i] if i < confusion_matrix.shape[1] else 0
        
        # False positives (predicted as this class but actually other classes)
        fp = np.sum(confusion_matrix[:, i]) - tp if i < confusion_matrix.shape[1] else 0
        
        # False negatives (actually this class but predicted as other classes)
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Calculate metrics
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0
        
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        else:
            recall[i] = 0
        
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def visualize_clustering_vs_ground_truth(
    cluster_map: np.ndarray,
    ground_truth: np.ndarray,
    metrics: Dict,
    color_mapping: Optional[Dict] = None,
    save_path: Optional[Path] = None
):
    """
    Create comprehensive visualization comparing clustering results to ground truth.
    
    Args:
        cluster_map: 2D array of cluster labels
        ground_truth: 2D array of ground truth labels
        metrics: Dictionary of evaluation metrics
        color_mapping: Optional dict mapping ground truth labels to colors
        save_path: Optional path to save the figure
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Ground Truth
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(ground_truth, cmap='tab10')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='True Class')
    
    # 2. Clustering Result
    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(cluster_map, cmap='tab10')
    ax2.set_title('Clustering Result', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Cluster ID')
    
    # 3. Difference map
    ax3 = plt.subplot(2, 4, 3)
    
    # Create difference map based on optimal mapping
    if 'cluster_to_gt_mapping' in metrics:
        mapping = metrics['cluster_to_gt_mapping']
        mapped_clusters = np.copy(cluster_map)
        for cluster_id, gt_id in mapping.items():
            mapped_clusters[cluster_map == cluster_id] = gt_id
        
        # Show where clusters match ground truth
        match_map = np.zeros_like(cluster_map)
        match_map[ground_truth == -1] = -1  # Background
        match_map[(ground_truth >= 0) & (mapped_clusters == ground_truth)] = 1  # Correct
        match_map[(ground_truth >= 0) & (mapped_clusters != ground_truth)] = 2  # Incorrect
        
        colors = ['gray', 'green', 'red']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        im3 = ax3.imshow(match_map, cmap=cmap, vmin=-1, vmax=2)
        ax3.set_title('Agreement Map', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Custom colorbar
        cbar = plt.colorbar(im3, ax=ax3, ticks=[-1, 1, 2])
        cbar.ax.set_yticklabels(['Background', 'Correct', 'Incorrect'])
    
    # 4. Metrics display
    ax4 = plt.subplot(2, 4, 4)
    ax4.axis('off')
    
    metrics_text = "Evaluation Metrics:\n\n"
    metrics_text += f"Clusters: {metrics.get('n_predicted_clusters', 'N/A')}\n"
    metrics_text += f"True Classes: {metrics.get('n_ground_truth_classes', 'N/A')}\n\n"
    
    key_metrics = [
        ('Purity', 'purity'),
        ('Adjusted Rand', 'adjusted_rand_score'),
        ('NMI', 'normalized_mutual_info'),
        ('AMI', 'adjusted_mutual_info'),
        ('V-Measure', 'v_measure'),
        ('Homogeneity', 'homogeneity'),
        ('Completeness', 'completeness'),
        ('Fowlkes-Mallows', 'fowlkes_mallows_score')
    ]
    
    for name, key in key_metrics:
        if key in metrics:
            value = metrics[key]
            metrics_text += f"{name}: {value:.4f}\n"
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, transform=ax4.transAxes,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Confusion Matrix
    ax5 = plt.subplot(2, 4, 5)
    if 'confusion_matrix' in metrics:
        conf_matrix = metrics['confusion_matrix']
        im5 = ax5.imshow(conf_matrix, cmap='Blues')
        ax5.set_xlabel('Predicted Cluster')
        ax5.set_ylabel('True Class')
        ax5.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text = ax5.text(j, i, f'{int(conf_matrix[i, j])}',
                              ha="center", va="center", color="black" if conf_matrix[i, j] < conf_matrix.max() / 2 else "white")
        
        plt.colorbar(im5, ax=ax5)
    
    # 6. Per-class F1 scores
    ax6 = plt.subplot(2, 4, 6)
    if 'per_class_f1' in metrics:
        f1_scores = metrics['per_class_f1']
        classes = list(f1_scores.keys())
        scores = list(f1_scores.values())
        
        bars = ax6.bar(classes, scores)
        ax6.set_xlabel('Ground Truth Class')
        ax6.set_ylabel('F1 Score')
        ax6.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Color bars based on score
        for bar, score in zip(bars, scores):
            if score >= 0.8:
                bar.set_color('green')
            elif score >= 0.6:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
    
    # 7. Class distribution comparison
    ax7 = plt.subplot(2, 4, 7)
    
    # Count pixels per class/cluster
    gt_unique, gt_counts = np.unique(ground_truth[ground_truth >= 0], return_counts=True)
    cluster_unique, cluster_counts = np.unique(cluster_map[cluster_map >= 0], return_counts=True)
    
    x = np.arange(max(len(gt_unique), len(cluster_unique)))
    width = 0.35
    
    # Pad arrays if different lengths
    gt_counts_padded = np.zeros(len(x))
    gt_counts_padded[:len(gt_counts)] = gt_counts
    
    cluster_counts_padded = np.zeros(len(x))
    cluster_counts_padded[:len(cluster_counts)] = cluster_counts
    
    ax7.bar(x - width/2, gt_counts_padded, width, label='Ground Truth', alpha=0.8)
    ax7.bar(x + width/2, cluster_counts_padded, width, label='Clusters', alpha=0.8)
    
    ax7.set_xlabel('Class/Cluster ID')
    ax7.set_ylabel('Number of Pixels')
    ax7.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Optimal mapping visualization
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    if 'cluster_to_gt_mapping' in metrics:
        mapping_text = "Optimal Cluster Mapping:\n\n"
        mapping = metrics['cluster_to_gt_mapping']
        
        for cluster_id, gt_id in sorted(mapping.items()):
            cluster_size = np.sum(cluster_map == cluster_id)
            gt_size = np.sum(ground_truth == gt_id)
            overlap = np.sum((cluster_map == cluster_id) & (ground_truth == gt_id))
            overlap_pct = 100 * overlap / cluster_size if cluster_size > 0 else 0
            
            mapping_text += f"Cluster {cluster_id} → Class {gt_id}\n"
            mapping_text += f"  Overlap: {overlap_pct:.1f}%\n"
        
        ax8.text(0.1, 0.5, mapping_text, fontsize=10, transform=ax8.transAxes,
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Clustering Validation Against Ground Truth', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation visualization saved to {save_path}")
    
    plt.show()


def compare_multiple_clusterings_to_ground_truth(
    clustering_results: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Compare multiple clustering results to ground truth.
    
    Args:
        clustering_results: Dict mapping method names to cluster label arrays
        ground_truth: Ground truth labels
        valid_mask: Optional valid pixel mask
        save_dir: Directory to save results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for method_name, cluster_labels in clustering_results.items():
        print(f"\nEvaluating {method_name}...")
        
        # Calculate metrics
        metrics = calculate_clustering_accuracy(
            cluster_labels,
            ground_truth,
            valid_mask
        )
        
        # Add to comparison data
        row = {
            'Method': method_name,
            'N_Clusters': metrics.get('n_predicted_clusters', np.nan),
            'Purity': metrics.get('purity', np.nan),
            'ARI': metrics.get('adjusted_rand_score', np.nan),
            'NMI': metrics.get('normalized_mutual_info', np.nan),
            'AMI': metrics.get('adjusted_mutual_info', np.nan),
            'V-Measure': metrics.get('v_measure', np.nan),
            'Homogeneity': metrics.get('homogeneity', np.nan),
            'Completeness': metrics.get('completeness', np.nan),
            'FM-Score': metrics.get('fowlkes_mallows_score', np.nan)
        }
        
        comparison_data.append(row)
    
    # Create DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by purity (or another metric)
    df_comparison = df_comparison.sort_values('Purity', ascending=False)
    
    # Display
    print("\n" + "="*80)
    print("Clustering Methods Comparison Against Ground Truth")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    # Create visualization
    if len(clustering_results) > 1:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['Purity', 'ARI', 'NMI', 'V-Measure', 'Homogeneity', 'Completeness']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            bars = ax.bar(df_comparison['Method'], df_comparison[metric])
            ax.set_xlabel('Method')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(clustering_results) > 3:
                ax.set_xticklabels(df_comparison['Method'], rotation=45, ha='right')
            
            # Color bars based on value
            for bar, value in zip(bars, df_comparison[metric]):
                if value >= 0.8:
                    bar.set_color('green')
                elif value >= 0.6:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        plt.suptitle('Clustering Methods Performance Against Ground Truth', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_path = save_dir / 'methods_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plot saved to {save_path}")
        
        plt.show()
    
    # Save DataFrame
    if save_dir:
        csv_path = save_dir / 'methods_comparison.csv'
        df_comparison.to_csv(csv_path, index=False)
        print(f"Comparison table saved to {csv_path}")
    
    return df_comparison


if __name__ == "__main__":
    # Example usage
    print("Ground Truth Validation Module")
    print("This module provides functions to validate clustering against ground truth from colored PNG annotations.")