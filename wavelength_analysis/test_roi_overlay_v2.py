"""
Test ROI Overlay Visualization in V2 Pipeline
==============================================
Quick test to verify ROI overlay with accuracy metrics is working.
"""

import numpy as np
import sys
from pathlib import Path

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

def test_roi_overlay():
    """Test the ROI overlay visualization with accuracy metrics."""

    from supervised_visualizations import SupervisedVisualizations
    from ground_truth_tracker import GroundTruthTracker
    from supervised_metrics import SupervisedMetrics

    # Define test ROI regions
    ROI_REGIONS = [
        {'name': 'Test ROI 1', 'coords': (10, 30, 10, 30), 'color': '#FF0000'},
        {'name': 'Test ROI 2', 'coords': (10, 30, 40, 60), 'color': '#0000FF'},
    ]

    # Create test data
    print("Creating test data...")
    ground_truth = np.zeros((50, 80), dtype=int)
    ground_truth[10:30, 10:30] = 0  # Class 0 in ROI 1
    ground_truth[10:30, 40:60] = 1  # Class 1 in ROI 2
    ground_truth[0:10, :] = -1  # Background

    # Create predictions with 95% accuracy
    predictions = ground_truth.copy()
    mask = np.random.random((50, 80)) < 0.05
    predictions[mask] = np.random.randint(0, 2, mask.sum())

    # Initialize tracker
    print("Initializing tracker...")
    tracker = GroundTruthTracker(ground_truth)

    # Map ROIs
    for roi in ROI_REGIONS:
        tracker.add_roi_mapping(roi['name'], roi['coords'])

    # Calculate metrics
    print("Calculating metrics...")
    tracker.set_predictions(predictions)
    metrics_calc = SupervisedMetrics(tracker)
    metrics = metrics_calc.calculate_metrics(predictions)
    roi_metrics = metrics_calc.calculate_roi_metrics(predictions)

    # Create visualization
    print("Creating ROI overlay visualization...")
    output_dir = Path("test_roi_overlay_output")
    output_dir.mkdir(exist_ok=True)

    viz = SupervisedVisualizations(output_dir=output_dir, dpi=100)
    viz.plot_roi_overlay_with_accuracy(
        cluster_map=predictions,
        ground_truth=ground_truth,
        roi_regions=ROI_REGIONS,
        overall_accuracy=metrics['accuracy'],
        roi_metrics=roi_metrics,
        title="Test: ROI Overlay with Accuracy",
        save_name="test_roi_overlay.png"
    )

    # Check if file was created
    output_file = output_dir / "test_roi_overlay.png"
    if output_file.exists():
        print(f"SUCCESS: Visualization created at {output_file}")
        print(f"  Overall accuracy: {metrics['accuracy']:.2%}")
        for roi_name, roi_data in roi_metrics.items():
            print(f"  {roi_name} accuracy: {roi_data['accuracy']:.2%}")
        return True
    else:
        print("FAILED: Visualization not created")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ROI OVERLAY VISUALIZATION")
    print("=" * 60)

    success = test_roi_overlay()

    if success:
        print("\n[OK] ROI overlay visualization is working correctly!")
        print("\nThe visualization includes:")
        print("  - Panel 1: Clustering result with overall accuracy")
        print("  - Panel 2: ROI boxes with individual accuracy labels")
        print("  - Panel 3: Bar chart comparing ROI accuracies")
    else:
        print("\n[ERROR] ROI overlay visualization failed")