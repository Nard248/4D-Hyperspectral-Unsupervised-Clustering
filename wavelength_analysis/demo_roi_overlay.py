"""
Demo Script for ROI Overlay Visualization with Accuracy Metrics
================================================================
Shows how to create ROI overlay visualizations with accuracy metrics displayed on top.
"""

import numpy as np
import sys
from pathlib import Path

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

from supervised_visualizations import SupervisedVisualizations
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics

# Define ROI regions (same as in main pipeline)
ROI_REGIONS = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},  # Red
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},  # Blue
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},  # Green
    {'name': 'Region 4', 'coords': (185, 225, 675, 700), 'color': '#FFFF00'},  # Yellow
]


def create_roi_overlay_demo():
    """
    Demo function to create ROI overlay visualization with accuracy metrics.
    """
    print("=" * 80)
    print("ROI OVERLAY VISUALIZATION DEMO")
    print("=" * 80)

    # Create sample data for demonstration
    print("\n1. Creating sample data...")
    height, width = 350, 800

    # Create sample ground truth (4 classes in different regions)
    ground_truth = np.full((height, width), -1, dtype=int)

    # Fill ROI regions with their respective classes
    for i, roi in enumerate(ROI_REGIONS):
        y_start, y_end, x_start, x_end = roi['coords']
        ground_truth[y_start:y_end, x_start:x_end] = i

    # Create sample predictions (with some errors)
    predictions = ground_truth.copy()

    # Introduce some errors for demonstration
    noise = np.random.random((height, width))
    error_mask = noise < 0.1  # 10% error rate
    predictions[error_mask] = np.random.randint(0, 4, np.sum(error_mask))

    print(f"  Created {height}x{width} sample data")
    print(f"  Ground truth classes: {np.unique(ground_truth[ground_truth >= 0])}")

    # Initialize ground truth tracker
    print("\n2. Initializing ground truth tracker...")
    tracker = GroundTruthTracker(ground_truth,
                                 class_names=["Red Class", "Blue Class", "Green Class", "Yellow Class"])

    # Map ROIs to ground truth
    for roi in ROI_REGIONS:
        tracker.add_roi_mapping(
            roi_id=roi['name'],
            coordinates=roi['coords'],
            verify_single_class=True
        )

    # Calculate metrics
    print("\n3. Calculating supervised metrics...")
    tracker.set_predictions(predictions)
    metrics_calc = SupervisedMetrics(tracker)
    metrics = metrics_calc.calculate_metrics(predictions, use_hungarian_mapping=True)
    roi_metrics = metrics_calc.calculate_roi_metrics(predictions)

    print(f"  Overall accuracy: {metrics['accuracy']:.2%}")
    print("  ROI accuracies:")
    for roi_name, roi_data in roi_metrics.items():
        print(f"    {roi_name}: {roi_data['accuracy']:.2%}")

    # Create visualization
    print("\n4. Creating ROI overlay visualization...")
    output_dir = Path("demo_roi_overlay_output")
    output_dir.mkdir(exist_ok=True)

    viz = SupervisedVisualizations(output_dir=output_dir, dpi=150)

    # Create standalone ROI overlay visualization
    viz.plot_roi_overlay_with_accuracy(
        cluster_map=predictions,
        ground_truth=ground_truth,
        roi_regions=ROI_REGIONS,
        overall_accuracy=metrics['accuracy'],
        roi_metrics=roi_metrics,
        title="Demo: ROI Overlay with Accuracy Metrics",
        save_name="demo_roi_overlay_accuracy.png"
    )

    print(f"\n✅ Visualization saved to: {output_dir / 'demo_roi_overlay_accuracy.png'}")

    # Also create full visualization suite
    print("\n5. Creating full visualization suite...")
    viz.create_all_visualizations(
        metrics=metrics,
        ground_truth=ground_truth,
        predictions=predictions,
        roi_metrics=roi_metrics,
        roi_regions=ROI_REGIONS
    )

    print(f"\n✅ All visualizations saved to: {output_dir}")

    return metrics, roi_metrics


def visualize_existing_results(cluster_map_path, ground_truth_path, output_dir):
    """
    Create ROI overlay visualization from existing cluster map and ground truth.

    Args:
        cluster_map_path: Path to saved cluster map (numpy array)
        ground_truth_path: Path to saved ground truth (numpy array)
        output_dir: Directory to save visualizations
    """
    import pickle

    print("\nLoading existing results...")

    # Load cluster map and ground truth
    if cluster_map_path.suffix == '.pkl':
        with open(cluster_map_path, 'rb') as f:
            cluster_map = pickle.load(f)
    else:
        cluster_map = np.load(cluster_map_path)

    if ground_truth_path.suffix == '.pkl':
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)
    else:
        ground_truth = np.load(ground_truth_path)

    print(f"  Cluster map shape: {cluster_map.shape}")
    print(f"  Ground truth shape: {ground_truth.shape}")

    # Initialize tracker and calculate metrics
    tracker = GroundTruthTracker(ground_truth)

    # Map ROIs
    for roi in ROI_REGIONS:
        tracker.add_roi_mapping(
            roi_id=roi['name'],
            coordinates=roi['coords'],
            verify_single_class=True
        )

    # Calculate metrics
    tracker.set_predictions(cluster_map)
    metrics_calc = SupervisedMetrics(tracker)
    metrics = metrics_calc.calculate_metrics(cluster_map, use_hungarian_mapping=True)
    roi_metrics = metrics_calc.calculate_roi_metrics(cluster_map)

    # Create visualization
    viz = SupervisedVisualizations(output_dir=Path(output_dir), dpi=300)
    viz.plot_roi_overlay_with_accuracy(
        cluster_map=cluster_map,
        ground_truth=ground_truth,
        roi_regions=ROI_REGIONS,
        overall_accuracy=metrics['accuracy'],
        roi_metrics=roi_metrics,
        title="ROI Overlay with Accuracy Metrics",
        save_name="roi_overlay_accuracy.png"
    )

    print(f"\n✅ Visualization saved to: {output_dir / 'roi_overlay_accuracy.png'}")
    print(f"  Overall accuracy: {metrics['accuracy']:.2%}")

    return metrics, roi_metrics


if __name__ == "__main__":
    # Run the demo
    print("\nRunning ROI overlay visualization demo...")
    metrics, roi_metrics = create_roi_overlay_demo()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("""
    The ROI overlay visualization shows:

    Panel 1: Clustering result with overall accuracy displayed
    Panel 2: ROI boxes overlaid on clustering with individual accuracies
    Panel 3: Bar chart comparing ROI accuracies vs overall accuracy

    Features:
    - Accuracy metrics displayed on top of each ROI
    - Color-coded ROI boundaries
    - Overall accuracy comparison line
    - Individual ROI performance metrics

    To use with your own data:
    1. Load your cluster map and ground truth
    2. Initialize GroundTruthTracker
    3. Map ROIs to ground truth classes
    4. Calculate metrics with SupervisedMetrics
    5. Create visualization with SupervisedVisualizations

    The V2 pipeline automatically generates these visualizations for each experiment!
    """)

    # Example of how to use with existing results
    print("\nTo visualize existing results, use:")
    print("  visualize_existing_results(cluster_map_path, ground_truth_path, output_dir)")