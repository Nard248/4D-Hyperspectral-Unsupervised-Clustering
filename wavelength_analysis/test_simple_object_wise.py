"""
Quick test of object-wise analysis with minimal dependencies
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import accuracy_score
import os

# Import the simple functions
from wavelengthSelectionV2SeparateObjectAnalysis import segment_objects_spatially, calculate_metrics_per_object

print("Testing object-wise analysis with synthetic data...")

# Create synthetic ground truth with multiple objects
ground_truth = np.zeros((1040, 925), dtype=int)

# Add 16 synthetic objects (4 of each class)
objects = [
    # Class 1 objects
    (100, 100, 50, 1), (100, 300, 50, 1), (100, 500, 50, 1), (100, 700, 50, 1),
    # Class 2 objects
    (300, 100, 50, 2), (300, 300, 50, 2), (300, 500, 50, 2), (300, 700, 50, 2),
    # Class 3 objects
    (500, 100, 50, 3), (500, 300, 50, 3), (500, 500, 50, 3), (500, 700, 50, 3),
    # Class 4 objects
    (700, 100, 50, 4), (700, 300, 50, 4), (700, 500, 50, 4), (700, 700, 50, 4),
]

for row, col, size, cls in objects:
    ground_truth[row:row+size, col:col+size] = cls

print(f"Created ground truth with {len(objects)} objects")

# Create predictions with some errors
predictions = ground_truth.copy()
# Add 10% random noise
noise_mask = np.random.random(ground_truth.shape) < 0.1
predictions[noise_mask & (ground_truth > 0)] = np.random.randint(1, 5, size=np.sum(noise_mask & (ground_truth > 0)))

# Perform object segmentation
print("\nPerforming object segmentation...")
object_masks, object_labels = segment_objects_spatially(ground_truth)
print(f"Found {len(object_masks)} objects")

# Calculate metrics
print("\nCalculating object-wise metrics...")
metrics_df = calculate_metrics_per_object(ground_truth, predictions, object_masks)

print("\nResults:")
print(metrics_df)

# Save results
os.makedirs("results/test_object_wise", exist_ok=True)
metrics_df.to_csv("results/test_object_wise/object_metrics_test.csv", index=False)
print("\nSaved to results/test_object_wise/object_metrics_test.csv")

# Summary
print(f"\nSummary:")
print(f"  Total objects: {len(metrics_df)}")
print(f"  Mean accuracy: {metrics_df['accuracy'].mean():.3f}")
print(f"  Std accuracy: {metrics_df['accuracy'].std():.3f}")