"""
Integration test for visualization updates
Tests that the new visualization methods work correctly with sample data
"""

import numpy as np
import sys
from pathlib import Path
from scipy import ndimage

# Add path
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir / "wavelength_analysis"))

from supervised_visualizations import SupervisedVisualizations

def create_sample_data():
    """Create sample data for testing"""
    # Create a simple ground truth with 4 classes
    ground_truth = np.zeros((100, 100), dtype=int)
    ground_truth[:] = -1  # Background

    # Add 4 objects with different classes
    ground_truth[10:30, 10:30] = 0  # Object 1, class 0
    ground_truth[10:30, 70:90] = 1  # Object 2, class 1
    ground_truth[70:90, 10:30] = 2  # Object 3, class 2
    ground_truth[70:90, 70:90] = 3  # Object 4, class 3

    # Create predictions (with some errors)
    predictions = ground_truth.copy()
    # Add some misclassifications
    predictions[15:20, 15:20] = 1  # Some errors in object 1
    predictions[75:80, 75:80] = 2  # Some errors in object 4

    # Label objects using scipy
    foreground_mask = ground_truth > -1
    labeled_objects, num_objects = ndimage.label(foreground_mask)

    # Define ROI regions
    roi_regions = [
        {'name': 'Region 1', 'coords': (10, 30, 10, 30), 'color': '#FF0000'},  # Red
        {'name': 'Region 2', 'coords': (10, 30, 70, 90), 'color': '#0000FF'},  # Blue
        {'name': 'Region 3', 'coords': (70, 90, 10, 30), 'color': '#00FF00'},  # Green
        {'name': 'Region 4', 'coords': (70, 90, 70, 90), 'color': '#FFFF00'},  # Yellow
    ]

    # Calculate object metrics
    object_metrics = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        y_true = ground_truth[obj_mask]
        y_pred = predictions[obj_mask]

        from sklearn.metrics import accuracy_score
        obj_accuracy = accuracy_score(y_true, y_pred)

        object_metrics.append({
            'object_id': obj_id,
            'num_pixels': np.sum(obj_mask),
            'true_class': int(np.unique(y_true)[0]),
            'accuracy': obj_accuracy
        })

    return ground_truth, predictions, labeled_objects, num_objects, roi_regions, object_metrics


def test_visualization_methods():
    """Test all new visualization methods"""
    print("=== INTEGRATION TEST: Visualizations ===")
    print()

    # Create output directory
    test_output_dir = Path("test_visualization_output")
    test_output_dir.mkdir(exist_ok=True)

    print("Step 1: Creating sample data...")
    ground_truth, predictions, labeled_objects, num_objects, roi_regions, object_metrics = create_sample_data()
    print(f"[PASS] Created sample data with {num_objects} objects")

    print()
    print("Step 2: Initializing visualization module...")
    viz = SupervisedVisualizations(output_dir=test_output_dir, dpi=100)
    print("[PASS] Visualization module initialized")

    print()
    print("Step 3: Testing plot_enumerated_objects...")
    try:
        viz.plot_enumerated_objects(
            ground_truth=ground_truth,
            labeled_objects=labeled_objects,
            num_objects=num_objects,
            title="Test: Ground Truth with Enumerated Objects",
            save_name="test_enumerated_objects.png"
        )
        assert (test_output_dir / "test_enumerated_objects.png").exists()
        print("[PASS] Enumerated objects visualization created")
    except Exception as e:
        print(f"[FAIL] Error in plot_enumerated_objects: {e}")
        raise

    print()
    print("Step 4: Testing plot_roi_overlay_with_accuracy (updated version)...")
    try:
        viz.plot_roi_overlay_with_accuracy(
            cluster_map=predictions,
            ground_truth=ground_truth,
            roi_regions=roi_regions,
            overall_accuracy=0.85,
            roi_metrics=None,
            title="Test: ROI Overlay with Per-Pixel Colors",
            save_name="test_roi_overlay_perpixel.png"
        )
        assert (test_output_dir / "test_roi_overlay_perpixel.png").exists()
        print("[PASS] ROI overlay with per-pixel colors created")
    except Exception as e:
        print(f"[FAIL] Error in plot_roi_overlay_with_accuracy: {e}")
        raise

    print()
    print("Step 5: Testing plot_roi_overlay_with_object_accuracy...")
    try:
        viz.plot_roi_overlay_with_object_accuracy(
            cluster_map=predictions,
            ground_truth=ground_truth,
            roi_regions=roi_regions,
            labeled_objects=labeled_objects,
            object_metrics=object_metrics,
            overall_accuracy=0.85,
            title="Test: ROI Overlay with Object Accuracy",
            save_name="test_roi_overlay_object_accuracy.png"
        )
        assert (test_output_dir / "test_roi_overlay_object_accuracy.png").exists()
        print("[PASS] ROI overlay with object accuracy created")
    except Exception as e:
        print(f"[FAIL] Error in plot_roi_overlay_with_object_accuracy: {e}")
        raise

    print()
    print("Step 6: Verifying all output files...")
    expected_files = [
        "test_enumerated_objects.png",
        "test_roi_overlay_perpixel.png",
        "test_roi_overlay_object_accuracy.png"
    ]

    all_exist = True
    for filename in expected_files:
        filepath = test_output_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  [OK] {filename} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {filename}")
            all_exist = False

    if not all_exist:
        raise AssertionError("Not all output files were created")

    print()
    print("=" * 50)
    print("[SUCCESS] All integration tests passed!")
    print(f"Output files saved to: {test_output_dir.absolute()}")
    print("=" * 50)

    return True


if __name__ == "__main__":
    try:
        test_visualization_methods()
        sys.exit(0)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"[FAIL] Integration test failed: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        sys.exit(1)
