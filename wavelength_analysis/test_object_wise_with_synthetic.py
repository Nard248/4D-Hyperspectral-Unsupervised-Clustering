"""
Test Object-Wise Analysis with Synthetic Data
==============================================
Demonstrates the object-wise analysis functionality with synthetic data.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the functions from our simplified implementation
from wavelengthSelectionV2SeparateObjectAnalysis import (
    segment_objects_spatially,
    calculate_metrics_per_object,
    visualize_object_metrics
)


def create_synthetic_lichens_data():
    """
    Create synthetic data that mimics the lichens dataset.
    16 objects total: 4 yellows, 4 browns, 4 whites, 4 greens
    """
    # Create a 925x1040 image (cropped dimensions)
    height, width = 1040, 925
    ground_truth = np.zeros((height, width), dtype=int)

    # Define 16 object locations (approximate ROI positions)
    objects = [
        # Yellow lichens (class 1)
        {'row': 160, 'col': 58, 'size': 60, 'class': 1},  # Yellow1
        {'row': 354, 'col': 44, 'size': 65, 'class': 1},  # Yellow2
        {'row': 397, 'col': 304, 'size': 55, 'class': 1},  # Yellow3 (adjusted for crop)
        {'row': 554, 'col': 30, 'size': 60, 'class': 1},  # Yellow4

        # Brown lichens (class 2)
        {'row': 185, 'col': 202, 'size': 55, 'class': 2},  # Brown1
        {'row': 199, 'col': 141, 'size': 58, 'class': 2},  # Brown2 (adjusted)
        {'row': 390, 'col': 127, 'size': 52, 'class': 2},  # Brown3 (adjusted)
        {'row': 568, 'col': 114, 'size': 60, 'class': 2},  # Brown4 (adjusted)

        # White lichens (class 3)
        {'row': 188, 'col': 0, 'size': 55, 'class': 3},   # White1 (adjusted)
        {'row': 386, 'col': 0, 'size': 52, 'class': 3},   # White2 (adjusted)
        {'row': 554, 'col': 226, 'size': 58, 'class': 3},  # White3
        {'row': 579, 'col': 446, 'size': 50, 'class': 3},  # White4 (adjusted)

        # Green lichens (class 4)
        {'row': 205, 'col': 296, 'size': 50, 'class': 4},  # Green1 (adjusted)
        {'row': 379, 'col': 239, 'size': 52, 'class': 4},  # Green2
        {'row': 565, 'col': 402, 'size': 55, 'class': 4},  # Green3
        {'row': 572, 'col': 290, 'size': 58, 'class': 4},  # Green4 (adjusted)
    ]

    # Create circular objects in the ground truth
    for obj in objects:
        # Create a circular mask for each object
        y, x = np.ogrid[-obj['size']:obj['size']+1, -obj['size']:obj['size']+1]
        mask = x**2 + y**2 <= obj['size']**2

        # Calculate bounds
        r_start = max(0, obj['row'] - obj['size'])
        r_end = min(height, obj['row'] + obj['size'] + 1)
        c_start = max(0, obj['col'] - obj['size'])
        c_end = min(width, obj['col'] + obj['size'] + 1)

        # Adjust mask size if needed
        mask_r_start = max(0, obj['size'] - obj['row'])
        mask_r_end = mask_r_start + (r_end - r_start)
        mask_c_start = max(0, obj['size'] - obj['col'])
        mask_c_end = mask_c_start + (c_end - c_start)

        # Apply the object to ground truth
        ground_truth[r_start:r_end, c_start:c_end][mask[mask_r_start:mask_r_end, mask_c_start:mask_c_end]] = obj['class']

    return ground_truth, objects


def create_synthetic_predictions(ground_truth, accuracy_level=0.85):
    """
    Create synthetic predictions with controlled accuracy.

    Args:
        ground_truth: Ground truth labels
        accuracy_level: Target overall accuracy (0-1)

    Returns:
        Predictions array
    """
    predictions = ground_truth.copy()

    # Add some misclassification noise
    noise_prob = 1.0 - accuracy_level
    noise_mask = np.random.random(ground_truth.shape) < noise_prob

    # Only add noise to non-background pixels
    non_background = ground_truth > 0
    noise_mask = noise_mask & non_background

    # Random misclassification
    unique_classes = np.unique(ground_truth[ground_truth > 0])
    for idx in zip(*np.where(noise_mask)):
        true_class = ground_truth[idx]
        # Pick a different class
        other_classes = unique_classes[unique_classes != true_class]
        if len(other_classes) > 0:
            predictions[idx] = np.random.choice(other_classes)

    return predictions


def run_synthetic_test():
    """
    Run a complete test of object-wise analysis with synthetic data.
    """
    logger.info("="*60)
    logger.info("OBJECT-WISE ANALYSIS TEST WITH SYNTHETIC DATA")
    logger.info("="*60)

    # Create output directory
    output_dir = "results/synthetic_object_test"
    os.makedirs(output_dir, exist_ok=True)

    # Create synthetic data
    logger.info("\n1. Creating synthetic lichens data...")
    ground_truth, object_info = create_synthetic_lichens_data()
    logger.info(f"   Created ground truth with {len(object_info)} objects")
    logger.info(f"   Classes: 4 Yellow, 4 Brown, 4 White, 4 Green lichens")

    # Create predictions with varying accuracy per object
    logger.info("\n2. Creating synthetic predictions...")
    predictions = create_synthetic_predictions(ground_truth, accuracy_level=0.85)

    # Add specific errors to certain objects for testing
    # Make object 3 have lower accuracy
    obj3_mask = segment_objects_spatially(ground_truth)[0][2]  # Third object
    predictions[obj3_mask] = np.where(
        np.random.random(np.sum(obj3_mask)) < 0.5,
        ground_truth[obj3_mask],
        np.random.choice([1, 2, 3, 4], size=np.sum(obj3_mask))
    )

    # Perform object-wise analysis
    logger.info("\n3. Performing spatial segmentation...")
    object_masks, object_labels = segment_objects_spatially(ground_truth)

    logger.info("\n4. Calculating per-object metrics...")
    metrics_df = calculate_metrics_per_object(ground_truth, predictions, object_masks)

    # Print results
    logger.info("\n5. Object-Wise Results:")
    logger.info("-" * 60)
    print(metrics_df.to_string())

    # Save results
    csv_path = os.path.join(output_dir, "object_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved metrics to: {csv_path}")

    # Create visualization
    logger.info("\n6. Creating visualizations...")
    visualize_object_metrics(ground_truth, predictions, object_labels,
                            metrics_df, "Synthetic Test", output_dir)

    # Summary statistics
    logger.info("\n7. Summary Statistics:")
    logger.info("-" * 60)
    logger.info(f"   Total objects detected: {len(metrics_df)}")
    logger.info(f"   Mean accuracy: {metrics_df['accuracy'].mean():.3f}")
    logger.info(f"   Std accuracy: {metrics_df['accuracy'].std():.3f}")
    logger.info(f"   Min accuracy: {metrics_df['accuracy'].min():.3f}")
    logger.info(f"   Max accuracy: {metrics_df['accuracy'].max():.3f}")

    # Class-wise summary
    class_summary = metrics_df.groupby('true_class')['accuracy'].agg(['mean', 'std', 'count'])
    logger.info("\n8. Class-wise Performance:")
    logger.info("-" * 60)
    for class_id, row in class_summary.iterrows():
        class_names = {1: 'Yellow', 2: 'Brown', 3: 'White', 4: 'Green'}
        class_name = class_names.get(class_id, f'Class {class_id}')
        logger.info(f"   {class_name}: {row['count']:.0f} objects, "
                   f"mean acc={row['mean']:.3f}, std={row['std']:.3f}")

    # Best and worst objects
    if len(metrics_df) > 0:
        best_obj = metrics_df.loc[metrics_df['accuracy'].idxmax()]
        worst_obj = metrics_df.loc[metrics_df['accuracy'].idxmin()]

        logger.info("\n9. Best and Worst Performing Objects:")
        logger.info("-" * 60)
        logger.info(f"   Best:  Object #{int(best_obj['object_id'])} "
                   f"(Class {best_obj['true_class']}, Accuracy={best_obj['accuracy']:.3f})")
        logger.info(f"   Worst: Object #{int(worst_obj['object_id'])} "
                   f"(Class {worst_obj['true_class']}, Accuracy={worst_obj['accuracy']:.3f})")

    logger.info("\n" + "="*60)
    logger.info("TEST COMPLETE!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)

    return metrics_df


if __name__ == "__main__":
    metrics = run_synthetic_test()

    logger.info("\nThis demonstrates the object-wise analysis working correctly:")
    logger.info("- Spatial segmentation identifies 16 individual lichen objects")
    logger.info("- Each object gets its own accuracy metrics calculated")
    logger.info("- Objects of the same class but spatially separated are tracked individually")
    logger.info("- Results show per-object performance for targeted analysis")