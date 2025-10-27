"""
Object-Wise Metrics Module for Wavelength Selection Pipeline V2
================================================================
This module calculates classification metrics for individual segmented objects,
enabling fine-grained performance analysis.

Author: Wavelength Selection Pipeline V2 Development
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, cohen_kappa_score, matthews_corrcoef)
from scipy.optimize import linear_sum_assignment
import logging
from object_segmentation import ObjectSegmentation, SegmentedObject

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectWiseMetrics:
    """
    Calculate classification metrics for individual objects.

    This class extends the supervised metrics calculation to work on a
    per-object basis, providing detailed performance analysis for each
    segmented object in the image.
    """

    def __init__(self, segmentation: ObjectSegmentation):
        """
        Initialize ObjectWiseMetrics.

        Args:
            segmentation: ObjectSegmentation instance with segmented objects
        """
        self.segmentation = segmentation
        self.object_metrics = {}
        self.aggregated_metrics = {}

    def _apply_hungarian_mapping(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply Hungarian algorithm to find optimal cluster-to-class mapping.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (mapped predictions, mapping dictionary)
        """
        # Get unique labels
        true_labels = np.unique(y_true)
        pred_labels = np.unique(y_pred)

        # Create cost matrix (negative accuracy for each mapping)
        n_true = len(true_labels)
        n_pred = len(pred_labels)
        cost_matrix = np.zeros((n_true, n_pred))

        for i, true_label in enumerate(true_labels):
            for j, pred_label in enumerate(pred_labels):
                mask = y_pred == pred_label
                if np.any(mask):
                    # Use negative accuracy as cost
                    cost_matrix[i, j] = -np.mean(y_true[mask] == true_label)

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create mapping
        mapping = {}
        for row, col in zip(row_indices, col_indices):
            mapping[pred_labels[col]] = true_labels[row]

        # Apply mapping
        mapped_pred = y_pred.copy()
        for pred_label, true_label in mapping.items():
            mapped_pred[y_pred == pred_label] = true_label

        return mapped_pred, mapping

    def calculate_object_metrics(self, ground_truth: np.ndarray,
                                predictions: np.ndarray,
                                apply_hungarian: bool = True) -> Dict[int, Dict]:
        """
        Calculate metrics for each individual object.

        Args:
            ground_truth: 2D array with true class labels
            predictions: 2D array with predicted class labels
            apply_hungarian: Whether to apply Hungarian algorithm for mapping

        Returns:
            Dictionary mapping object_id to metrics dictionary
        """
        logger.info(f"Calculating metrics for {self.segmentation.num_objects} objects")

        self.object_metrics = {}

        # Apply Hungarian mapping if requested (globally first)
        if apply_hungarian:
            # Flatten arrays for global Hungarian mapping
            gt_flat = ground_truth.flatten()
            pred_flat = predictions.flatten()

            # Apply Hungarian algorithm
            mapped_pred_flat, mapping = self._apply_hungarian_mapping(gt_flat, pred_flat)

            # Reshape back
            mapped_predictions = mapped_pred_flat.reshape(predictions.shape)
            logger.info(f"Applied Hungarian mapping: {mapping}")
        else:
            mapped_predictions = predictions

        # Calculate metrics for each object
        for obj in self.segmentation.objects:
            # Extract object pixels
            obj_ground_truth = ground_truth[obj.pixel_mask]
            obj_predictions = mapped_predictions[obj.pixel_mask]

            # Skip if object has no pixels (shouldn't happen)
            if len(obj_ground_truth) == 0:
                logger.warning(f"Object {obj.object_id} has no pixels")
                continue

            # Calculate metrics for this object
            try:
                metrics = self._calculate_single_object_metrics(
                    obj_ground_truth, obj_predictions
                )
                metrics['object_id'] = obj.object_id
                metrics['class_label'] = obj.class_label
                metrics['pixel_count'] = obj.pixel_count
                metrics['roi_id'] = obj.roi_id

                self.object_metrics[obj.object_id] = metrics

            except Exception as e:
                logger.error(f"Error calculating metrics for object {obj.object_id}: {e}")
                self.object_metrics[obj.object_id] = {
                    'object_id': obj.object_id,
                    'class_label': obj.class_label,
                    'pixel_count': obj.pixel_count,
                    'roi_id': obj.roi_id,
                    'error': str(e)
                }

        return self.object_metrics

    def _calculate_single_object_metrics(self, y_true: np.ndarray,
                                        y_pred: np.ndarray) -> Dict:
        """
        Calculate metrics for a single object's pixels.

        Args:
            y_true: True labels for object pixels
            y_pred: Predicted labels for object pixels

        Returns:
            Dictionary of metrics
        """
        # Get unique classes in this object
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'pixel_accuracy': np.mean(y_true == y_pred),  # Same as accuracy but explicit
            'total_pixels': len(y_true),
            'correct_pixels': np.sum(y_true == y_pred),
            'incorrect_pixels': np.sum(y_true != y_pred)
        }

        # Calculate per-class precision, recall, F1 if applicable
        if len(unique_true) > 1 or len(unique_pred) > 1:
            # Multi-class within object (shouldn't happen ideally)
            try:
                metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            except:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
        else:
            # Single class object (most common case)
            # For single-class objects, if all predictions match ground truth, perfect score
            if np.all(y_true == y_pred):
                metrics['precision'] = 1.0
                metrics['recall'] = 1.0
                metrics['f1'] = 1.0
            else:
                # Calculate based on majority class
                majority_pred = np.bincount(y_pred).argmax()
                correct_predictions = np.sum(y_pred == y_true[0])
                total_predictions = len(y_pred)

                metrics['precision'] = correct_predictions / total_predictions if total_predictions > 0 else 0
                metrics['recall'] = metrics['precision']  # Same for single-class
                metrics['f1'] = metrics['precision']  # Same for single-class

        # Additional metrics
        try:
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['cohen_kappa'] = 0.0
            metrics['mcc'] = 0.0

        # Class distribution info
        metrics['true_class'] = int(unique_true[0]) if len(unique_true) == 1 else -1
        metrics['predicted_classes'] = unique_pred.tolist()
        metrics['prediction_distribution'] = {
            int(cls): int(count) for cls, count in
            zip(*np.unique(y_pred, return_counts=True))
        }

        return metrics

    def aggregate_metrics_by_class(self) -> Dict[int, Dict]:
        """
        Aggregate object metrics by class label.

        Returns:
            Dictionary mapping class_label to aggregated metrics
        """
        if not self.object_metrics:
            logger.warning("No object metrics to aggregate")
            return {}

        # Group objects by class
        class_objects = {}
        for obj_id, metrics in self.object_metrics.items():
            if 'error' in metrics:
                continue

            class_label = metrics['class_label']
            if class_label not in class_objects:
                class_objects[class_label] = []
            class_objects[class_label].append(metrics)

        # Aggregate metrics for each class
        aggregated = {}
        for class_label, objects in class_objects.items():
            if not objects:
                continue

            # Calculate mean and std for each metric
            agg_metrics = {
                'class_label': class_label,
                'num_objects': len(objects),
                'total_pixels': sum(obj['total_pixels'] for obj in objects)
            }

            # Metrics to aggregate
            metric_names = ['accuracy', 'precision', 'recall', 'f1',
                          'cohen_kappa', 'mcc']

            for metric in metric_names:
                values = [obj[metric] for obj in objects if metric in obj]
                if values:
                    agg_metrics[f'{metric}_mean'] = np.mean(values)
                    agg_metrics[f'{metric}_std'] = np.std(values)
                    agg_metrics[f'{metric}_min'] = np.min(values)
                    agg_metrics[f'{metric}_max'] = np.max(values)

            aggregated[class_label] = agg_metrics

        self.aggregated_metrics = aggregated
        return aggregated

    def get_best_worst_objects(self, metric: str = 'accuracy',
                              n: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Get the best and worst performing objects by a specific metric.

        Args:
            metric: Metric to sort by
            n: Number of objects to return

        Returns:
            Tuple of (best_objects, worst_objects)
        """
        if not self.object_metrics:
            return [], []

        # Filter out objects with errors
        valid_objects = [
            metrics for metrics in self.object_metrics.values()
            if 'error' not in metrics and metric in metrics
        ]

        if not valid_objects:
            return [], []

        # Sort by metric
        sorted_objects = sorted(valid_objects, key=lambda x: x[metric], reverse=True)

        best = sorted_objects[:n]
        worst = sorted_objects[-n:] if len(sorted_objects) >= n else sorted_objects

        return best, worst

    def create_performance_matrix(self) -> pd.DataFrame:
        """
        Create a matrix showing performance of each object.

        Returns:
            DataFrame with objects as rows and metrics as columns
        """
        if not self.object_metrics:
            return pd.DataFrame()

        # Create list of records
        records = []
        for obj_id, metrics in self.object_metrics.items():
            if 'error' not in metrics:
                record = {
                    'Object_ID': obj_id,
                    'Class': metrics.get('class_label', -1),
                    'ROI': metrics.get('roi_id', 'Unknown'),
                    'Pixels': metrics.get('total_pixels', 0),
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'Kappa': metrics.get('cohen_kappa', 0),
                    'MCC': metrics.get('mcc', 0)
                }
                records.append(record)

        df = pd.DataFrame(records)

        # Sort by Object_ID
        if not df.empty:
            df = df.sort_values('Object_ID')

        return df

    def export_detailed_results(self, filename: str) -> None:
        """
        Export detailed object-wise results to CSV.

        Args:
            filename: Path to save the CSV file
        """
        df = self.create_performance_matrix()
        if not df.empty:
            df.to_csv(filename, index=False)
            logger.info(f"Exported object-wise metrics to {filename}")
        else:
            logger.warning("No metrics to export")

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics across all objects.

        Returns:
            Dictionary of summary statistics
        """
        if not self.object_metrics:
            return {}

        # Extract all metric values
        all_metrics = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa', 'mcc']

        for metric in metric_names:
            values = [
                obj[metric] for obj in self.object_metrics.values()
                if 'error' not in obj and metric in obj
            ]
            if values:
                all_metrics[metric] = values

        # Calculate summary statistics
        summary = {
            'total_objects': self.segmentation.num_objects,
            'objects_analyzed': len(self.object_metrics),
            'objects_with_errors': sum(1 for m in self.object_metrics.values() if 'error' in m)
        }

        for metric, values in all_metrics.items():
            summary[f'{metric}_global_mean'] = np.mean(values)
            summary[f'{metric}_global_std'] = np.std(values)
            summary[f'{metric}_global_min'] = np.min(values)
            summary[f'{metric}_global_max'] = np.max(values)
            summary[f'{metric}_global_median'] = np.median(values)

        return summary