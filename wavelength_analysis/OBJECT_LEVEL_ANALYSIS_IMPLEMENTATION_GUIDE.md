# Object-Level Analysis Implementation Guide

## Executive Summary

This guide provides step-by-step instructions for implementing object-level analysis in the wavelength selection pipeline. The goal is to analyze 16 individual lichen objects (4 objects per class) separately, calculating per-object metrics and enabling single-object selection for detailed analysis.

---

## Prerequisites

### Already Implemented (Current State)
- ✅ Ground truth tracking at pixel level (`ground_truth_tracker.py`)
- ✅ Supervised metrics calculation (`supervised_metrics.py`)
- ✅ ROI-to-class mapping
- ✅ V2 pipeline with full integration

### Requirements for Object-Level
- **Spatial Separability**: Objects must be spatially distinct (confirmed by user)
- **Ground Truth Data**: 16 objects within 4 classes
- **Connected Components**: For object segmentation

---

## Implementation Plan

### Step 1: Create object_segmentation.py

```python
"""
Object Segmentation Module
==========================
Segments ground truth into individual objects using connected components.
Enables object-level analysis and metrics.
"""

import numpy as np
from scipy.ndimage import label, find_objects
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class ObjectSegmentation:
    """
    Segments ground truth into individual objects for detailed analysis.
    """

    def __init__(self, ground_truth_tracker):
        """
        Initialize with existing ground truth tracker.

        Args:
            ground_truth_tracker: Instance of GroundTruthTracker
        """
        self.tracker = ground_truth_tracker
        self.ground_truth = self.tracker.ground_truth
        self.objects = {}
        self.object_map = np.full_like(self.ground_truth, -1, dtype=int)
        self.n_objects = 0

    def segment_objects(self, connectivity: int = 8, min_size: int = 100):
        """
        Segment ground truth into individual objects using connected components.

        Args:
            connectivity: 4 or 8 connectivity for connected components
            min_size: Minimum pixels for valid object

        Returns:
            Number of objects found
        """
        from scipy.ndimage import label, find_objects
        import cv2

        object_id = 0

        # Process each class separately
        for class_id in self.tracker.unique_classes:
            print(f"\nSegmenting class {class_id} ({self.tracker.class_names.get(class_id, 'Unknown')})")

            # Get mask for this class
            class_mask = (self.ground_truth == class_id).astype(np.uint8)

            # Find connected components
            if connectivity == 4:
                connectivity_struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
            else:  # 8-connectivity
                connectivity_struct = np.ones((3,3), dtype=np.uint8)

            labeled_array, num_features = label(class_mask, structure=connectivity_struct)
            print(f"  Found {num_features} connected components")

            # Process each component
            component_sizes = []
            for component_id in range(1, num_features + 1):
                component_mask = (labeled_array == component_id)
                component_size = np.sum(component_mask)
                component_sizes.append(component_size)

                # Filter by minimum size
                if component_size < min_size:
                    print(f"    Skipping component {component_id}: too small ({component_size} < {min_size})")
                    continue

                # Get bounding box
                bbox = find_objects(labeled_array == component_id)[0]

                # Calculate centroid
                y_coords, x_coords = np.where(component_mask)
                centroid = (int(np.mean(y_coords)), int(np.mean(x_coords)))

                # Store object information
                self.objects[object_id] = {
                    'object_id': object_id,
                    'class_id': int(class_id),
                    'class_name': self.tracker.class_names.get(class_id, f"Class_{class_id}"),
                    'pixel_mask': component_mask,
                    'pixel_count': int(component_size),
                    'pixel_coords': np.column_stack((y_coords, x_coords)),
                    'bounding_box': (bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop),
                    'centroid': centroid,
                    'area_percentage': float(100 * component_size / np.sum(self.ground_truth >= 0))
                }

                # Update object map
                self.object_map[component_mask] = object_id

                print(f"    Object {object_id}: {component_size} pixels, "
                      f"centroid {centroid}, area {self.objects[object_id]['area_percentage']:.2f}%")

                object_id += 1

            # Report class statistics
            if component_sizes:
                print(f"  Class {class_id} statistics:")
                print(f"    Objects created: {len([s for s in component_sizes if s >= min_size])}")
                print(f"    Avg size: {np.mean(component_sizes):.0f} pixels")
                print(f"    Size range: {min(component_sizes)} - {max(component_sizes)} pixels")

        self.n_objects = object_id
        print(f"\nTotal objects segmented: {self.n_objects}")

        # Verify we got expected 16 objects
        if self.n_objects != 16:
            print(f"WARNING: Expected 16 objects, found {self.n_objects}")
            print("Consider adjusting connectivity or min_size parameters")

        return self.n_objects

    def get_object_by_id(self, object_id: int) -> Optional[Dict]:
        """Get object information by ID."""
        return self.objects.get(object_id)

    def get_objects_by_class(self, class_id: int) -> List[Dict]:
        """Get all objects belonging to a specific class."""
        return [obj for obj in self.objects.values() if obj['class_id'] == class_id]

    def get_object_at_position(self, y: int, x: int) -> Optional[int]:
        """Get object ID at specific pixel position."""
        if 0 <= y < self.object_map.shape[0] and 0 <= x < self.object_map.shape[1]:
            obj_id = self.object_map[y, x]
            return obj_id if obj_id >= 0 else None
        return None

    def create_object_visualization(self, object_id: int) -> np.ndarray:
        """Create binary mask for single object."""
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} not found")

        mask = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        mask[self.objects[object_id]['pixel_mask']] = 255
        return mask

    def create_all_objects_visualization(self) -> np.ndarray:
        """Create color-coded visualization of all objects."""
        import matplotlib.pyplot as plt

        # Create random colors for each object
        np.random.seed(42)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, self.n_objects)))

        # Create RGB image
        vis = np.zeros((*self.ground_truth.shape, 3))

        for obj_id, obj_info in self.objects.items():
            color = colors[obj_id % len(colors)][:3]
            vis[obj_info['pixel_mask']] = color

        return vis

    def export_segmentation(self, filepath: Path):
        """Export segmentation data."""
        export_data = {
            'n_objects': self.n_objects,
            'objects': {}
        }

        for obj_id, obj_info in self.objects.items():
            export_data['objects'][obj_id] = {
                'object_id': obj_info['object_id'],
                'class_id': obj_info['class_id'],
                'class_name': obj_info['class_name'],
                'pixel_count': obj_info['pixel_count'],
                'centroid': obj_info['centroid'],
                'bounding_box': obj_info['bounding_box'],
                'area_percentage': obj_info['area_percentage']
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Segmentation exported to {filepath}")
```

### Step 2: Create object_metrics.py

```python
"""
Object-Level Metrics Module
===========================
Calculates metrics for individual objects.
"""

import numpy as np
from typing import Dict, Optional, List
from supervised_metrics import SupervisedMetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ObjectLevelMetrics(SupervisedMetrics):
    """
    Extends SupervisedMetrics to calculate object-level performance.
    """

    def __init__(self, ground_truth_tracker, object_segmentation):
        """
        Initialize with tracker and segmentation.

        Args:
            ground_truth_tracker: GroundTruthTracker instance
            object_segmentation: ObjectSegmentation instance
        """
        super().__init__(ground_truth_tracker)
        self.segmentation = object_segmentation

    def calculate_object_metrics(self, predictions: np.ndarray) -> Dict:
        """
        Calculate metrics for each individual object.

        Args:
            predictions: 2D array of predicted labels

        Returns:
            Dictionary with per-object metrics
        """
        object_metrics = {}

        for obj_id, obj_info in self.segmentation.objects.items():
            # Extract predictions for this object
            obj_mask = obj_info['pixel_mask']
            obj_true = self.ground_truth[obj_mask]
            obj_pred = predictions[obj_mask]

            # Remove any background pixels
            valid = obj_true >= 0
            obj_true = obj_true[valid]
            obj_pred = obj_pred[valid]

            if len(obj_true) == 0:
                continue

            # Calculate basic metrics
            accuracy = accuracy_score(obj_true, obj_pred)

            # Find dominant prediction
            unique_preds, counts = np.unique(obj_pred, return_counts=True)
            dominant_pred = unique_preds[np.argmax(counts)]
            prediction_purity = np.max(counts) / len(obj_pred)

            # Check if dominant prediction matches true class
            class_match = (dominant_pred == obj_info['class_id'])

            # Calculate per-pixel metrics within object
            tp = np.sum((obj_pred == obj_info['class_id']) & (obj_true == obj_info['class_id']))
            fp = np.sum((obj_pred == obj_info['class_id']) & (obj_true != obj_info['class_id']))
            fn = np.sum((obj_pred != obj_info['class_id']) & (obj_true == obj_info['class_id']))
            tn = np.sum((obj_pred != obj_info['class_id']) & (obj_true != obj_info['class_id']))

            # Calculate derived metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Store metrics
            object_metrics[obj_id] = {
                'object_id': obj_id,
                'class_id': obj_info['class_id'],
                'class_name': obj_info['class_name'],
                'pixel_count': obj_info['pixel_count'],
                'centroid': obj_info['centroid'],
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'dominant_prediction': int(dominant_pred),
                'prediction_purity': float(prediction_purity),
                'class_match': bool(class_match),
                'confusion': {
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn)
                },
                'unique_predictions': unique_preds.tolist(),
                'prediction_distribution': dict(zip(unique_preds.tolist(), counts.tolist()))
            }

        # Calculate aggregate statistics
        aggregate_stats = self._calculate_aggregate_stats(object_metrics)

        return {
            'per_object': object_metrics,
            'aggregate': aggregate_stats
        }

    def _calculate_aggregate_stats(self, object_metrics: Dict) -> Dict:
        """Calculate aggregate statistics across all objects."""

        if not object_metrics:
            return {}

        accuracies = [m['accuracy'] for m in object_metrics.values()]
        f1_scores = [m['f1_score'] for m in object_metrics.values()]
        class_matches = [m['class_match'] for m in object_metrics.values()]

        # Group by class
        class_stats = {}
        for class_id in self.tracker.unique_classes:
            class_objects = [m for m in object_metrics.values() if m['class_id'] == class_id]
            if class_objects:
                class_stats[int(class_id)] = {
                    'n_objects': len(class_objects),
                    'mean_accuracy': np.mean([o['accuracy'] for o in class_objects]),
                    'std_accuracy': np.std([o['accuracy'] for o in class_objects]),
                    'mean_f1': np.mean([o['f1_score'] for o in class_objects]),
                    'class_match_rate': np.mean([o['class_match'] for o in class_objects])
                }

        return {
            'n_objects': len(object_metrics),
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'mean_f1': float(np.mean(f1_scores)),
            'class_match_rate': float(np.mean(class_matches)),
            'per_class_stats': class_stats
        }

    def get_best_worst_objects(self, object_metrics: Dict, n: int = 3) -> Dict:
        """Identify best and worst performing objects."""

        sorted_objects = sorted(object_metrics.values(), key=lambda x: x['accuracy'])

        return {
            'best': sorted_objects[-n:] if len(sorted_objects) >= n else sorted_objects,
            'worst': sorted_objects[:n] if len(sorted_objects) >= n else sorted_objects
        }
```

### Step 3: Create object_selector.py

```python
"""
Object Selection Interface
==========================
Interactive selection and analysis of individual objects.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Optional, Tuple, List

class ObjectSelector:
    """
    Interactive object selection and analysis interface.
    """

    def __init__(self, object_segmentation, ground_truth, predictions):
        """
        Initialize selector.

        Args:
            object_segmentation: ObjectSegmentation instance
            ground_truth: Ground truth array
            predictions: Predictions array
        """
        self.segmentation = object_segmentation
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.selected_object = None

    def display_object_grid(self, save_path: Optional[str] = None):
        """
        Display all 16 objects in a 4x4 grid.
        """
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(16):
            ax = axes[i]

            if i < self.segmentation.n_objects:
                obj_info = self.segmentation.objects.get(i)
                if obj_info:
                    # Create object visualization
                    vis = np.zeros(self.ground_truth.shape)
                    vis[obj_info['pixel_mask']] = 1

                    # Add bounding box
                    y1, y2, x1, x2 = obj_info['bounding_box']
                    vis_crop = vis[y1:y2, x1:x2]

                    ax.imshow(vis_crop, cmap='viridis')
                    ax.set_title(f"Obj {i} (Class {obj_info['class_id']})\n"
                                f"{obj_info['pixel_count']} pixels",
                                fontsize=10)

                    # Add object ID in corner
                    ax.text(0.05, 0.95, str(i), transform=ax.transAxes,
                           fontsize=14, fontweight='bold', color='red',
                           verticalalignment='top')
                else:
                    ax.axis('off')
            else:
                ax.axis('off')
                ax.set_title(f"Object {i}: Not found", fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle("Object Grid - Select Object ID for Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def select_object_interactive(self) -> Optional[int]:
        """
        Interactive object selection.

        Returns:
            Selected object ID or None
        """
        self.display_object_grid()

        try:
            selected_id = int(input("\nEnter object ID to analyze (0-15): "))
            if 0 <= selected_id < self.segmentation.n_objects:
                self.selected_object = selected_id
                return selected_id
            else:
                print(f"Invalid ID. Must be between 0 and {self.segmentation.n_objects-1}")
                return None
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None

    def analyze_single_object(self, object_id: int) -> Dict:
        """
        Comprehensive analysis of a single object.

        Args:
            object_id: ID of object to analyze

        Returns:
            Dictionary with detailed analysis
        """
        if object_id not in self.segmentation.objects:
            raise ValueError(f"Object {object_id} not found")

        obj_info = self.segmentation.objects[object_id]
        obj_mask = obj_info['pixel_mask']

        # Extract object data
        obj_true = self.ground_truth[obj_mask]
        obj_pred = self.predictions[obj_mask]

        # Calculate metrics
        accuracy = np.mean(obj_true == obj_pred)

        # Spatial consistency (how uniform are predictions)
        unique_preds, counts = np.unique(obj_pred, return_counts=True)
        spatial_consistency = np.max(counts) / len(obj_pred)

        # Edge vs interior analysis
        edge_accuracy, interior_accuracy = self._analyze_edge_vs_interior(obj_mask)

        # Confusion within object
        confusion = self._get_object_confusion(obj_true, obj_pred)

        analysis = {
            'object_id': object_id,
            'basic_info': obj_info,
            'classification_metrics': {
                'accuracy': float(accuracy),
                'spatial_consistency': float(spatial_consistency),
                'edge_accuracy': float(edge_accuracy),
                'interior_accuracy': float(interior_accuracy),
                'edge_interior_difference': float(edge_accuracy - interior_accuracy)
            },
            'prediction_analysis': {
                'unique_predictions': unique_preds.tolist(),
                'prediction_counts': counts.tolist(),
                'dominant_prediction': int(unique_preds[np.argmax(counts)]),
                'confusion_matrix': confusion
            },
            'spatial_analysis': {
                'centroid': obj_info['centroid'],
                'bounding_box': obj_info['bounding_box'],
                'area_pixels': obj_info['pixel_count'],
                'compactness': self._calculate_compactness(obj_mask)
            }
        }

        return analysis

    def _analyze_edge_vs_interior(self, obj_mask: np.ndarray) -> Tuple[float, float]:
        """Analyze accuracy at edges vs interior."""
        from scipy.ndimage import binary_erosion

        # Find edges (pixels on boundary)
        eroded = binary_erosion(obj_mask, iterations=1)
        edge_mask = obj_mask & ~eroded
        interior_mask = eroded

        # Calculate accuracies
        edge_accuracy = 0.0
        interior_accuracy = 0.0

        if np.any(edge_mask):
            edge_true = self.ground_truth[edge_mask]
            edge_pred = self.predictions[edge_mask]
            edge_accuracy = np.mean(edge_true == edge_pred)

        if np.any(interior_mask):
            interior_true = self.ground_truth[interior_mask]
            interior_pred = self.predictions[interior_mask]
            interior_accuracy = np.mean(interior_true == interior_pred)

        return edge_accuracy, interior_accuracy

    def _calculate_compactness(self, obj_mask: np.ndarray) -> float:
        """Calculate object compactness (circularity)."""
        # Find contour
        from scipy.ndimage import find_objects

        area = np.sum(obj_mask)
        bbox = find_objects(obj_mask)[0]
        height = bbox[0].stop - bbox[0].start
        width = bbox[1].stop - bbox[1].start

        # Compactness = area / (width * height)
        compactness = area / (width * height)

        return float(compactness)

    def _get_object_confusion(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Get confusion matrix for object."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(true_labels, pred_labels)

        return {
            'matrix': cm.tolist(),
            'true_classes': np.unique(true_labels).tolist(),
            'pred_classes': np.unique(pred_labels).tolist()
        }

    def visualize_object_analysis(self, object_id: int, save_path: Optional[str] = None):
        """
        Create comprehensive visualization for single object.
        """
        analysis = self.analyze_single_object(object_id)
        obj_info = self.segmentation.objects[object_id]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Object mask
        ax1 = axes[0, 0]
        vis1 = np.zeros(self.ground_truth.shape)
        vis1[obj_info['pixel_mask']] = 1
        y1, y2, x1, x2 = obj_info['bounding_box']
        ax1.imshow(vis1[y1:y2, x1:x2], cmap='viridis')
        ax1.set_title(f"Object {object_id} Mask")
        ax1.axis('off')

        # 2. Ground truth
        ax2 = axes[0, 1]
        gt_vis = self.ground_truth.copy()
        gt_vis[~obj_info['pixel_mask']] = -1
        ax2.imshow(gt_vis[y1:y2, x1:x2], cmap='tab10', vmin=-1)
        ax2.set_title(f"Ground Truth (Class {obj_info['class_id']})")
        ax2.axis('off')

        # 3. Predictions
        ax3 = axes[0, 2]
        pred_vis = self.predictions.copy()
        pred_vis[~obj_info['pixel_mask']] = -1
        ax3.imshow(pred_vis[y1:y2, x1:x2], cmap='tab10', vmin=-1)
        ax3.set_title(f"Predictions")
        ax3.axis('off')

        # 4. Accuracy map
        ax4 = axes[1, 0]
        acc_map = np.zeros(self.ground_truth.shape)
        correct_mask = (self.ground_truth == self.predictions) & obj_info['pixel_mask']
        incorrect_mask = (self.ground_truth != self.predictions) & obj_info['pixel_mask']
        acc_map[correct_mask] = 1
        acc_map[incorrect_mask] = -1
        ax4.imshow(acc_map[y1:y2, x1:x2], cmap='RdYlGn', vmin=-1, vmax=1)
        ax4.set_title(f"Accuracy Map ({analysis['classification_metrics']['accuracy']:.1%})")
        ax4.axis('off')

        # 5. Metrics text
        ax5 = axes[1, 1]
        ax5.axis('off')
        metrics_text = f"""
Object {object_id} Analysis
Class: {obj_info['class_id']} ({obj_info['class_name']})
Pixels: {obj_info['pixel_count']}

Accuracy: {analysis['classification_metrics']['accuracy']:.2%}
Spatial Consistency: {analysis['classification_metrics']['spatial_consistency']:.2%}
Edge Accuracy: {analysis['classification_metrics']['edge_accuracy']:.2%}
Interior Accuracy: {analysis['classification_metrics']['interior_accuracy']:.2%}

Dominant Prediction: {analysis['prediction_analysis']['dominant_prediction']}
Unique Predictions: {len(analysis['prediction_analysis']['unique_predictions'])}
        """
        ax5.text(0.1, 0.5, metrics_text, fontsize=10, transform=ax5.transAxes,
                verticalalignment='center', family='monospace')

        # 6. Prediction distribution
        ax6 = axes[1, 2]
        preds = analysis['prediction_analysis']['unique_predictions']
        counts = analysis['prediction_analysis']['prediction_counts']
        ax6.bar(preds, counts)
        ax6.set_xlabel('Predicted Class')
        ax6.set_ylabel('Pixel Count')
        ax6.set_title('Prediction Distribution')
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f"Object {object_id} Detailed Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()
```

### Step 4: Integrate into wavelengthselectionV2.py

Add after ground truth tracker initialization (around line 665):

```python
# ========================================================================
# V2.1 ENHANCEMENT: Object-Level Segmentation (Optional)
# ========================================================================
ENABLE_OBJECT_ANALYSIS = False  # Set to True to enable

if ENABLE_OBJECT_ANALYSIS:
    print("\n" + "=" * 80)
    print("OBJECT-LEVEL SEGMENTATION")
    print("=" * 80)

    from object_segmentation import ObjectSegmentation
    from object_metrics import ObjectLevelMetrics
    from object_selector import ObjectSelector

    # Perform object segmentation
    obj_segmentation = ObjectSegmentation(gt_tracker)
    n_objects = obj_segmentation.segment_objects(connectivity=8, min_size=100)

    if n_objects != 16:
        print(f"WARNING: Expected 16 objects, found {n_objects}")
        print("Continuing with {n_objects} objects...")

    # Save segmentation
    segmentation_file = metrics_dir / "object_segmentation.json"
    obj_segmentation.export_segmentation(segmentation_file)

    # Create object-level metrics calculator
    obj_metrics_calc = ObjectLevelMetrics(gt_tracker, obj_segmentation)

    # Store for later use in clustering
    object_analysis = {
        'segmentation': obj_segmentation,
        'metrics_calc': obj_metrics_calc,
        'enabled': True
    }
else:
    object_analysis = {'enabled': False}
```

### Step 5: Add Object Metrics to Clustering

In the clustering loop (around line 830):

```python
# After calculating supervised_metrics
if object_analysis['enabled']:
    # Calculate object-level metrics
    obj_metrics = object_analysis['metrics_calc'].calculate_object_metrics(cluster_map)

    # Save object metrics
    obj_metrics_file = experiment_folder / f"{config_name}_object_metrics.json"
    with open(obj_metrics_file, 'w') as f:
        json.dump(obj_metrics, f, indent=2)

    print(f"    Object-level metrics:")
    print(f"      Mean accuracy: {obj_metrics['aggregate']['mean_accuracy']:.4f}")
    print(f"      Std accuracy: {obj_metrics['aggregate']['std_accuracy']:.4f}")
    print(f"      Best object: {obj_metrics['aggregate']['max_accuracy']:.4f}")
    print(f"      Worst object: {obj_metrics['aggregate']['min_accuracy']:.4f}")

    # Store in results
    result['object_mean_accuracy'] = obj_metrics['aggregate']['mean_accuracy']
    result['object_std_accuracy'] = obj_metrics['aggregate']['std_accuracy']
```

### Step 6: Create Interactive Analysis Script

```python
"""
interactive_object_analysis.py
==============================
Interactive script for single object analysis.
"""

import sys
from pathlib import Path
import pickle
import numpy as np

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir / "wavelength_analysis"))

from ground_truth_tracker import GroundTruthTracker
from object_segmentation import ObjectSegmentation
from object_selector import ObjectSelector
from object_metrics import ObjectLevelMetrics

def main():
    """Run interactive object analysis."""

    print("=" * 80)
    print("INTERACTIVE OBJECT ANALYSIS")
    print("=" * 80)

    # Load saved ground truth tracker
    tracker_file = Path("path/to/ground_truth_tracker_state.pkl")
    tracker = GroundTruthTracker.load_state(tracker_file)

    # Load predictions (from a specific experiment)
    predictions_file = Path("path/to/cluster_map.npy")
    predictions = np.load(predictions_file)

    # Perform segmentation
    print("\n1. Segmenting objects...")
    segmentation = ObjectSegmentation(tracker)
    n_objects = segmentation.segment_objects()

    # Initialize selector
    selector = ObjectSelector(segmentation, tracker.ground_truth, predictions)

    # Interactive loop
    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("  1. Display object grid")
        print("  2. Select and analyze object")
        print("  3. Compare all objects")
        print("  4. Export results")
        print("  5. Exit")

        choice = input("\nSelect option (1-5): ")

        if choice == '1':
            selector.display_object_grid()

        elif choice == '2':
            obj_id = selector.select_object_interactive()
            if obj_id is not None:
                analysis = selector.analyze_single_object(obj_id)
                selector.visualize_object_analysis(obj_id)

                print(f"\nObject {obj_id} Summary:")
                print(f"  Accuracy: {analysis['classification_metrics']['accuracy']:.2%}")
                print(f"  Spatial Consistency: {analysis['classification_metrics']['spatial_consistency']:.2%}")

        elif choice == '3':
            metrics_calc = ObjectLevelMetrics(tracker, segmentation)
            all_metrics = metrics_calc.calculate_object_metrics(predictions)

            print("\nAll Objects Summary:")
            for obj_id, metrics in all_metrics['per_object'].items():
                print(f"  Object {obj_id}: {metrics['accuracy']:.2%} "
                      f"(Class {metrics['class_id']})")

        elif choice == '4':
            export_path = Path("object_analysis_results.json")
            # Export logic here
            print(f"Results exported to {export_path}")

        elif choice == '5':
            break

        else:
            print("Invalid option")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Unit Tests
```python
def test_object_segmentation():
    """Test object segmentation."""
    # Create test ground truth with known objects
    gt = np.zeros((100, 100), dtype=int)
    gt[10:30, 10:30] = 0  # Object 1
    gt[10:30, 50:70] = 0  # Object 2
    gt[50:70, 10:30] = 1  # Object 3
    gt[50:70, 50:70] = 1  # Object 4

    tracker = GroundTruthTracker(gt)
    segmentation = ObjectSegmentation(tracker)
    n_objects = segmentation.segment_objects(min_size=50)

    assert n_objects == 4, f"Expected 4 objects, got {n_objects}"
    assert len(segmentation.objects) == 4
```

---

## Expected Outputs

### Object Segmentation
- 16 individual objects identified
- 4 objects per class (4 classes total)
- Each object with unique ID (0-15)

### Object Metrics
- Per-object accuracy, precision, recall, F1
- Spatial consistency metrics
- Edge vs interior performance

### Visualizations
- 4x4 grid showing all objects
- Individual object analysis plots
- Performance heatmap across objects

---

## Important Considerations

1. **Connectivity**: Use 8-connectivity for diagonal connections
2. **Minimum Size**: Set appropriately to filter noise (e.g., 100 pixels)
3. **Object Ordering**: Consider sorting by centroid for consistent numbering
4. **Memory**: Object masks can be large, consider sparse storage
5. **Validation**: Verify 16 objects found, adjust parameters if needed

---

## Summary

This implementation guide provides:
1. Complete code modules for object-level analysis
2. Integration points with existing V2 pipeline
3. Interactive selection interface
4. Comprehensive metrics calculation
5. Testing and validation strategies

The object-level analysis will enable detailed per-object performance evaluation, supporting the research goal of understanding how wavelength selection affects individual lichen classification accuracy.