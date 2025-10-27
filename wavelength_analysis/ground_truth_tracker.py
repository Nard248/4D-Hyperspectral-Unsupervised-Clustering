"""
Ground Truth Tracker Module
===========================
Maintains pixel-level ground truth class information throughout the analysis pipeline.
Ensures every pixel's original class is preserved for accurate supervised metric calculation.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import pickle
import json


class GroundTruthTracker:
    """
    Tracks ground truth class labels at pixel level throughout the pipeline.
    Maintains mapping between pixels, their original classes, and ROI assignments.
    """

    def __init__(self, ground_truth: np.ndarray, class_names: Optional[List[str]] = None):
        """
        Initialize the ground truth tracker.

        Args:
            ground_truth: 2D array with class labels (-1 for background, 0-N for classes)
            class_names: Optional names for each class
        """
        self.ground_truth = ground_truth.copy()
        self.height, self.width = ground_truth.shape

        # Identify unique classes (excluding background -1)
        self.unique_classes = np.unique(ground_truth[ground_truth >= 0])
        self.n_classes = len(self.unique_classes)

        # Set class names
        if class_names is None:
            self.class_names = {i: f"Class_{i}" for i in self.unique_classes}
        else:
            self.class_names = {i: name for i, name in enumerate(class_names)}

        # Create pixel index for fast lookups
        self._build_pixel_index()

        # Initialize ROI mappings
        self.roi_mappings = {}

        # Track predictions for comparison
        self.predictions = None

        print(f"GroundTruthTracker initialized:")
        print(f"  Image shape: {self.height} x {self.width}")
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Classes: {self.unique_classes}")
        print(f"  Background pixels: {np.sum(ground_truth == -1):,}")
        print(f"  Labeled pixels: {np.sum(ground_truth >= 0):,}")

    def _build_pixel_index(self):
        """Build pixel-to-class index for fast lookups."""
        self.pixel_index = {}
        self.class_pixels = {cls: [] for cls in self.unique_classes}

        for y in range(self.height):
            for x in range(self.width):
                class_id = self.ground_truth[y, x]
                if class_id >= 0:  # Skip background
                    self.pixel_index[(x, y)] = class_id
                    self.class_pixels[class_id].append((x, y))

        # Convert to numpy arrays for efficiency
        for cls in self.unique_classes:
            self.class_pixels[cls] = np.array(self.class_pixels[cls])

    def get_pixel_class(self, x: int, y: int) -> int:
        """
        Get ground truth class for a specific pixel.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Class ID (-1 if background or out of bounds)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.ground_truth[y, x]
        return -1

    def add_roi_mapping(self, roi_id: str, coordinates: Tuple[int, int, int, int],
                       verify_single_class: bool = True) -> Dict:
        """
        Add ROI and map it to its ground truth class.

        Args:
            roi_id: Unique identifier for the ROI
            coordinates: (y_start, y_end, x_start, x_end)
            verify_single_class: If True, verify ROI contains only one class

        Returns:
            Dictionary with ROI mapping information
        """
        y_start, y_end, x_start, x_end = coordinates

        # Extract ground truth for this ROI
        roi_gt = self.ground_truth[y_start:y_end, x_start:x_end]

        # Find classes in this ROI (excluding background)
        roi_classes = np.unique(roi_gt[roi_gt >= 0])

        if len(roi_classes) == 0:
            raise ValueError(f"ROI {roi_id} contains only background pixels")

        if verify_single_class and len(roi_classes) > 1:
            # Calculate dominant class if multiple classes found
            class_counts = {cls: np.sum(roi_gt == cls) for cls in roi_classes}
            dominant_class = max(class_counts, key=class_counts.get)
            purity = class_counts[dominant_class] / np.sum(roi_gt >= 0)

            print(f"Warning: ROI {roi_id} contains {len(roi_classes)} classes")
            print(f"  Classes: {roi_classes}")
            print(f"  Dominant class: {dominant_class} (purity: {purity:.2%})")

            # Store warning but proceed with dominant class
            mapping = {
                'roi_id': roi_id,
                'coordinates': coordinates,
                'ground_truth_class': int(dominant_class),
                'all_classes': roi_classes.tolist(),
                'purity': float(purity),
                'pixel_count': int(np.sum(roi_gt >= 0)),
                'warning': 'Multiple classes detected'
            }
        else:
            # Single class ROI (ideal case)
            ground_truth_class = int(roi_classes[0])
            pixel_count = int(np.sum(roi_gt == ground_truth_class))

            mapping = {
                'roi_id': roi_id,
                'coordinates': coordinates,
                'ground_truth_class': ground_truth_class,
                'all_classes': [ground_truth_class],
                'purity': 1.0,
                'pixel_count': pixel_count,
                'class_name': self.class_names.get(ground_truth_class, f"Class_{ground_truth_class}")
            }

        self.roi_mappings[roi_id] = mapping
        return mapping

    def set_predictions(self, predictions: np.ndarray):
        """
        Set predicted labels for comparison with ground truth.

        Args:
            predictions: 2D array of predicted labels (same shape as ground_truth)
        """
        if predictions.shape != self.ground_truth.shape:
            raise ValueError(f"Predictions shape {predictions.shape} doesn't match ground truth {self.ground_truth.shape}")

        self.predictions = predictions.copy()

    def get_pixel_accuracy(self, mask: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate pixel-wise accuracy.

        Args:
            mask: Optional boolean mask for pixels to consider

        Returns:
            Dictionary with accuracy metrics
        """
        if self.predictions is None:
            raise ValueError("No predictions set. Call set_predictions() first.")

        # Create mask for valid pixels (non-background)
        valid_mask = self.ground_truth >= 0
        if mask is not None:
            valid_mask = valid_mask & mask

        # Get valid pixels
        gt_valid = self.ground_truth[valid_mask]
        pred_valid = self.predictions[valid_mask]

        # Calculate accuracy
        correct = np.sum(gt_valid == pred_valid)
        total = len(gt_valid)
        accuracy = correct / total if total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct_pixels': int(correct),
            'total_pixels': int(total),
            'error_rate': 1 - accuracy
        }

    def get_class_distribution(self) -> Dict:
        """
        Get distribution of pixels per class.

        Returns:
            Dictionary with class distribution statistics
        """
        distribution = {}
        total_labeled = np.sum(self.ground_truth >= 0)

        for cls in self.unique_classes:
            count = np.sum(self.ground_truth == cls)
            distribution[int(cls)] = {
                'name': self.class_names.get(cls, f"Class_{cls}"),
                'pixel_count': int(count),
                'percentage': float(count / total_labeled * 100) if total_labeled > 0 else 0,
                'coordinates': self.class_pixels[cls].tolist() if cls in self.class_pixels else []
            }

        # Add background statistics
        background_count = np.sum(self.ground_truth == -1)
        distribution[-1] = {
            'name': 'Background',
            'pixel_count': int(background_count),
            'percentage': float(background_count / self.ground_truth.size * 100)
        }

        return distribution

    def get_roi_statistics(self) -> Dict:
        """
        Get statistics about ROI mappings.

        Returns:
            Dictionary with ROI statistics
        """
        if not self.roi_mappings:
            return {'n_rois': 0, 'message': 'No ROIs mapped'}

        stats = {
            'n_rois': len(self.roi_mappings),
            'rois_per_class': {},
            'average_purity': 0,
            'perfect_rois': 0,
            'mixed_rois': 0
        }

        # Calculate statistics
        total_purity = 0
        for roi_id, mapping in self.roi_mappings.items():
            gt_class = mapping['ground_truth_class']

            # Count ROIs per class
            if gt_class not in stats['rois_per_class']:
                stats['rois_per_class'][gt_class] = []
            stats['rois_per_class'][gt_class].append(roi_id)

            # Track purity
            purity = mapping['purity']
            total_purity += purity

            if purity == 1.0:
                stats['perfect_rois'] += 1
            else:
                stats['mixed_rois'] += 1

        stats['average_purity'] = total_purity / len(self.roi_mappings)

        # Convert class IDs to names
        stats['rois_per_class'] = {
            self.class_names.get(cls, f"Class_{cls}"): roi_list
            for cls, roi_list in stats['rois_per_class'].items()
        }

        return stats

    def create_prediction_mask(self, predicted_labels: np.ndarray) -> np.ndarray:
        """
        Create a mask showing correct/incorrect predictions.

        Args:
            predicted_labels: 2D array of predicted labels

        Returns:
            2D array: 0=background, 1=correct, 2=incorrect
        """
        mask = np.zeros_like(self.ground_truth)

        # Background pixels
        mask[self.ground_truth == -1] = 0

        # Correct predictions
        correct = (self.ground_truth == predicted_labels) & (self.ground_truth >= 0)
        mask[correct] = 1

        # Incorrect predictions
        incorrect = (self.ground_truth != predicted_labels) & (self.ground_truth >= 0)
        mask[incorrect] = 2

        return mask

    def export_state(self, filepath: Union[str, Path]):
        """
        Export tracker state to file.

        Args:
            filepath: Path to save file (.pkl or .json)
        """
        filepath = Path(filepath)

        state = {
            'ground_truth': self.ground_truth,
            'class_names': self.class_names,
            'roi_mappings': self.roi_mappings,
            'unique_classes': self.unique_classes.tolist(),
            'n_classes': self.n_classes,
            'height': self.height,
            'width': self.width
        }

        if filepath.suffix == '.json':
            # Convert numpy arrays to lists for JSON
            state['ground_truth'] = self.ground_truth.tolist()
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
        else:
            # Use pickle for numpy arrays
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

        print(f"Tracker state exported to {filepath}")

    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> 'GroundTruthTracker':
        """
        Load tracker state from file.

        Args:
            filepath: Path to saved state file

        Returns:
            GroundTruthTracker instance
        """
        filepath = Path(filepath)

        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                state = json.load(f)
            state['ground_truth'] = np.array(state['ground_truth'])
            state['unique_classes'] = np.array(state['unique_classes'])
        else:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

        # Create tracker instance
        tracker = cls(state['ground_truth'])
        tracker.class_names = state['class_names']
        tracker.roi_mappings = state['roi_mappings']

        print(f"Tracker state loaded from {filepath}")
        return tracker