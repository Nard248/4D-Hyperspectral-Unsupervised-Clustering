"""
Simple Object-Wise Analysis for Wavelength Selection
=====================================================
Straightforward implementation that:
1. Spatially separates objects using connected components
2. Calculates metrics for each object individually

No bullshit, just what you asked for.
"""

import numpy as np
from scipy import ndimage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from typing import Dict, List, Tuple


def segment_objects_spatially(ground_truth: np.ndarray,
                             background_value: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Segment objects based on spatial connectivity.
    Objects of the same class that are spatially separated are different objects.

    Args:
        ground_truth: 2D array with class labels
        background_value: Value for background pixels (default 0)

    Returns:
        Tuple of (list of boolean masks for each object, labeled object map)
    """
    # Create binary mask (non-background pixels)
    foreground_mask = ground_truth != background_value

    # Use connected components to find spatially separated regions
    labeled_objects, num_objects = ndimage.label(foreground_mask)

    # Create mask for each object
    object_masks = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        object_masks.append(obj_mask)

    print(f"Found {num_objects} spatially separated objects")

    return object_masks, labeled_objects


def calculate_metrics_per_object(ground_truth: np.ndarray,
                                predictions: np.ndarray,
                                object_masks: List[np.ndarray]) -> pd.DataFrame:
    """
    Calculate classification metrics for each spatially separated object.

    Args:
        ground_truth: 2D array with true labels
        predictions: 2D array with predicted labels
        object_masks: List of boolean masks for each object

    Returns:
        DataFrame with metrics for each object
    """
    results = []

    for obj_id, mask in enumerate(object_masks, 1):
        # Get pixels for this object
        y_true = ground_truth[mask]
        y_pred = predictions[mask]

        # Skip if empty
        if len(y_true) == 0:
            continue

        # Calculate metrics
        metrics = {
            'object_id': obj_id,
            'num_pixels': len(y_true),
            'true_class': np.unique(y_true)[0] if len(np.unique(y_true)) == 1 else -1,
            'accuracy': accuracy_score(y_true, y_pred)
        }

        # Add precision, recall, F1 if applicable
        try:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            if len(unique_labels) > 1:
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                # Single class - perfect if all predictions match
                metrics['precision'] = 1.0 if np.all(y_pred == y_true) else 0.0
                metrics['recall'] = metrics['precision']
                metrics['f1'] = metrics['precision']
        except:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0

        results.append(metrics)

    return pd.DataFrame(results)


def visualize_object_metrics(ground_truth: np.ndarray,
                            predictions: np.ndarray,
                            object_labels: np.ndarray,
                            metrics_df: pd.DataFrame,
                            save_path: str = None) -> None:
    """
    Simple visualization of objects and their metrics.

    Args:
        ground_truth: 2D ground truth array
        predictions: 2D predictions array
        object_labels: 2D array with object IDs
        metrics_df: DataFrame with object metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Ground truth
    axes[0, 0].imshow(ground_truth, cmap='tab20')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')

    # 2. Predictions
    axes[0, 1].imshow(predictions, cmap='tab20')
    axes[0, 1].set_title('Predictions')
    axes[0, 1].axis('off')

    # 3. Object segmentation
    axes[0, 2].imshow(object_labels, cmap='nipy_spectral')
    axes[0, 2].set_title(f'Object Segmentation ({len(metrics_df)} objects)')
    axes[0, 2].axis('off')

    # 4. Accuracy per object
    axes[1, 0].bar(metrics_df['object_id'], metrics_df['accuracy'])
    axes[1, 0].set_xlabel('Object ID')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy per Object')
    axes[1, 0].set_ylim([0, 1.1])

    # 5. Metrics heatmap
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    heatmap_data = metrics_df[metric_cols].values.T
    im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1, 1].set_yticks(range(len(metric_cols)))
    axes[1, 1].set_yticklabels(metric_cols)
    axes[1, 1].set_xticks(range(len(metrics_df)))
    axes[1, 1].set_xticklabels(metrics_df['object_id'].values)
    axes[1, 1].set_xlabel('Object ID')
    axes[1, 1].set_title('Metrics Heatmap')
    plt.colorbar(im, ax=axes[1, 1])

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"Object-Wise Summary:\n\n"
    summary_text += f"Total Objects: {len(metrics_df)}\n"
    summary_text += f"Mean Accuracy: {metrics_df['accuracy'].mean():.3f}\n"
    summary_text += f"Std Accuracy: {metrics_df['accuracy'].std():.3f}\n"
    summary_text += f"Best Object: #{metrics_df.loc[metrics_df['accuracy'].idxmax(), 'object_id']} "
    summary_text += f"({metrics_df['accuracy'].max():.3f})\n"
    summary_text += f"Worst Object: #{metrics_df.loc[metrics_df['accuracy'].idxmin(), 'object_id']} "
    summary_text += f"({metrics_df['accuracy'].min():.3f})"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Object-Wise Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_object_wise_analysis(ground_truth: np.ndarray,
                            predictions: np.ndarray) -> pd.DataFrame:
    """
    Main function to run object-wise analysis.

    Args:
        ground_truth: 2D ground truth array
        predictions: 2D predictions array

    Returns:
        DataFrame with metrics for each object
    """
    # 1. Segment objects spatially
    object_masks, object_labels = segment_objects_spatially(ground_truth)

    # 2. Calculate metrics for each object
    metrics_df = calculate_metrics_per_object(ground_truth, predictions, object_masks)

    # 3. Print results
    print("\nObject-Wise Metrics:")
    print(metrics_df.to_string())

    # 4. Create visualization
    visualize_object_metrics(ground_truth, predictions, object_labels, metrics_df)

    return metrics_df


# Example usage in your pipeline
def integrate_with_wavelength_selection(data_path: str,
                                       cluster_map: np.ndarray,
                                       ground_truth: np.ndarray) -> Dict:
    """
    Simple integration with your wavelength selection pipeline.

    Args:
        data_path: Path to data file
        cluster_map: Clustering results from your pipeline
        ground_truth: Ground truth labels

    Returns:
        Dictionary with results
    """
    # Run object-wise analysis
    metrics_df = run_object_wise_analysis(ground_truth, cluster_map)

    # Aggregate results
    results = {
        'num_objects': len(metrics_df),
        'mean_accuracy': metrics_df['accuracy'].mean(),
        'std_accuracy': metrics_df['accuracy'].std(),
        'per_object_metrics': metrics_df.to_dict('records'),
        'best_object': {
            'id': metrics_df.loc[metrics_df['accuracy'].idxmax(), 'object_id'],
            'accuracy': metrics_df['accuracy'].max()
        },
        'worst_object': {
            'id': metrics_df.loc[metrics_df['accuracy'].idxmin(), 'object_id'],
            'accuracy': metrics_df['accuracy'].min()
        }
    }

    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing simple object-wise analysis...")

    # Create synthetic ground truth with 4 spatially separated objects
    ground_truth = np.zeros((100, 100), dtype=int)

    # Object 1: Class 1
    ground_truth[10:30, 10:30] = 1

    # Object 2: Class 2
    ground_truth[10:35, 60:85] = 2

    # Object 3: Class 3
    ground_truth[60:90, 15:40] = 3

    # Object 4: Class 1 (same class as object 1, but spatially separated)
    ground_truth[65:85, 65:90] = 1

    # Create predictions with some errors
    predictions = ground_truth.copy()
    # Add some misclassifications
    predictions[15:20, 15:20] = 2  # Error in object 1
    predictions[70:75, 70:75] = 3  # Error in object 4

    # Run analysis
    metrics = run_object_wise_analysis(ground_truth, predictions)

    print("\nSimple object-wise analysis complete!")
    print("To integrate with your pipeline, just call:")
    print("  metrics_df = run_object_wise_analysis(ground_truth, cluster_map)")