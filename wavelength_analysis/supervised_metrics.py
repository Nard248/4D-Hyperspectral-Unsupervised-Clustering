"""
Supervised Metrics Module
=========================
Calculates comprehensive supervised learning metrics including precision, recall,
accuracy, F1-score, and confusion matrices with proper ground truth mapping.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import json


class SupervisedMetrics:
    """
    Comprehensive supervised learning metrics calculator with ground truth integration.
    """

    def __init__(self, ground_truth_tracker):
        """
        Initialize with a GroundTruthTracker instance.

        Args:
            ground_truth_tracker: Instance of GroundTruthTracker
        """
        self.tracker = ground_truth_tracker
        self.ground_truth = self.tracker.ground_truth
        self.unique_classes = self.tracker.unique_classes
        self.n_classes = self.tracker.n_classes

        # Storage for metrics
        self.current_metrics = None
        self.cluster_to_class_mapping = None

    def calculate_metrics(self, predictions: np.ndarray,
                         use_hungarian_mapping: bool = True) -> Dict:
        """
        Calculate comprehensive supervised metrics.

        Args:
            predictions: 2D array of predicted labels
            use_hungarian_mapping: If True, find optimal cluster-to-class mapping

        Returns:
            Dictionary containing all calculated metrics
        """
        # Validate predictions shape
        if predictions.shape != self.ground_truth.shape:
            raise ValueError(f"Predictions shape {predictions.shape} doesn't match ground truth {self.ground_truth.shape}")

        # Get valid pixels (non-background)
        valid_mask = (self.ground_truth >= 0) & (predictions >= 0)
        y_true = self.ground_truth[valid_mask]
        y_pred = predictions[valid_mask]

        if len(y_true) == 0:
            print("Warning: No valid pixels to calculate metrics")
            return {}

        # Apply Hungarian algorithm for optimal mapping if needed
        if use_hungarian_mapping:
            y_pred, mapping = self._apply_hungarian_mapping(y_true, y_pred)
            self.cluster_to_class_mapping = mapping
        else:
            self.cluster_to_class_mapping = None

        # Calculate all metrics
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # Per-class metrics with averaging strategies
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Agreement metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        # Per-class detailed metrics
        per_class_metrics = self._calculate_per_class_metrics(y_true, y_pred, cm)
        metrics['per_class'] = per_class_metrics

        # Pixel counts
        metrics['total_pixels'] = len(y_true)
        metrics['correct_pixels'] = np.sum(y_true == y_pred)
        metrics['error_rate'] = 1 - metrics['accuracy']

        # Store current metrics
        self.current_metrics = metrics

        return metrics

    def _apply_hungarian_mapping(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply Hungarian algorithm to find optimal cluster-to-class mapping.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (mapped_predictions, mapping_dict)
        """
        # Get unique labels
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        # Build cost matrix (negative of confusion matrix for maximization)
        n_true = len(unique_true)
        n_pred = len(unique_pred)
        cost_matrix = np.zeros((n_pred, n_true))

        for i, pred_label in enumerate(unique_pred):
            for j, true_label in enumerate(unique_true):
                cost_matrix[i, j] = -np.sum((y_pred == pred_label) & (y_true == true_label))

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create mapping
        mapping = {}
        for i, j in zip(row_ind, col_ind):
            if i < len(unique_pred) and j < len(unique_true):
                mapping[unique_pred[i]] = unique_true[j]

        # Apply mapping to predictions
        mapped_pred = y_pred.copy()
        for pred_label, true_label in mapping.items():
            mapped_pred[y_pred == pred_label] = true_label

        return mapped_pred, mapping

    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    cm: np.ndarray) -> Dict:
        """
        Calculate detailed per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            cm: Confusion matrix

        Returns:
            Dictionary with per-class metrics
        """
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class = {}

        for cls in unique_classes:
            cls_metrics = {}

            # Get true positives, false positives, false negatives
            cls_idx = np.where(unique_classes == cls)[0]
            if len(cls_idx) > 0:
                idx = cls_idx[0]

                if idx < cm.shape[0] and idx < cm.shape[1]:
                    tp = cm[idx, idx]
                    fp = np.sum(cm[:, idx]) - tp
                    fn = np.sum(cm[idx, :]) - tp
                    tn = np.sum(cm) - tp - fp - fn

                    # Calculate metrics
                    cls_metrics['true_positives'] = int(tp)
                    cls_metrics['false_positives'] = int(fp)
                    cls_metrics['false_negatives'] = int(fn)
                    cls_metrics['true_negatives'] = int(tn)

                    # Precision, Recall, F1
                    cls_metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    cls_metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

                    if cls_metrics['precision'] + cls_metrics['recall'] > 0:
                        cls_metrics['f1'] = 2 * (cls_metrics['precision'] * cls_metrics['recall']) / \
                                           (cls_metrics['precision'] + cls_metrics['recall'])
                    else:
                        cls_metrics['f1'] = 0

                    # Specificity
                    cls_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

                    # Support (number of true instances)
                    cls_metrics['support'] = int(np.sum(y_true == cls))

                    # Class name
                    cls_metrics['class_name'] = self.tracker.class_names.get(cls, f"Class_{cls}")

                    per_class[int(cls)] = cls_metrics

        return per_class

    def get_classification_report(self, predictions: np.ndarray,
                                 use_hungarian_mapping: bool = True) -> str:
        """
        Generate a detailed classification report.

        Args:
            predictions: 2D array of predicted labels
            use_hungarian_mapping: If True, apply optimal mapping

        Returns:
            Formatted classification report string
        """
        # Calculate metrics if not already done
        if self.current_metrics is None:
            self.calculate_metrics(predictions, use_hungarian_mapping)

        # Get valid pixels
        valid_mask = (self.ground_truth >= 0) & (predictions >= 0)
        y_true = self.ground_truth[valid_mask]
        y_pred = predictions[valid_mask]

        if use_hungarian_mapping and self.cluster_to_class_mapping:
            # Apply mapping
            for pred_label, true_label in self.cluster_to_class_mapping.items():
                y_pred[y_pred == pred_label] = true_label

        # Generate sklearn classification report
        target_names = [self.tracker.class_names.get(i, f"Class_{i}")
                       for i in sorted(np.unique(y_true))]

        report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

        # Add additional information
        report += "\n" + "="*60 + "\n"
        report += "Additional Metrics:\n"
        report += f"  Cohen's Kappa: {self.current_metrics['cohen_kappa']:.4f}\n"
        report += f"  Matthews Correlation: {self.current_metrics['matthews_corrcoef']:.4f}\n"
        report += f"  Balanced Accuracy: {self.current_metrics['balanced_accuracy']:.4f}\n"

        if self.cluster_to_class_mapping:
            report += "\nCluster to Class Mapping:\n"
            for cluster, cls in self.cluster_to_class_mapping.items():
                cls_name = self.tracker.class_names.get(cls, f"Class_{cls}")
                report += f"  Cluster {cluster} → {cls_name} (Class {cls})\n"

        return report

    def calculate_roi_metrics(self, predictions: np.ndarray) -> Dict:
        """
        Calculate metrics for each ROI region.

        Args:
            predictions: 2D array of predicted labels

        Returns:
            Dictionary with per-ROI metrics
        """
        roi_metrics = {}

        for roi_id, roi_info in self.tracker.roi_mappings.items():
            y_start, y_end, x_start, x_end = roi_info['coordinates']

            # Extract ROI regions
            roi_gt = self.ground_truth[y_start:y_end, x_start:x_end]
            roi_pred = predictions[y_start:y_end, x_start:x_end]

            # Get valid pixels
            valid_mask = roi_gt >= 0
            roi_gt_valid = roi_gt[valid_mask]
            roi_pred_valid = roi_pred[valid_mask]

            if len(roi_gt_valid) == 0:
                continue

            # Calculate ROI-specific metrics
            roi_metrics[roi_id] = {
                'ground_truth_class': roi_info['ground_truth_class'],
                'class_name': roi_info.get('class_name', f"Class_{roi_info['ground_truth_class']}"),
                'coordinates': roi_info['coordinates'],
                'accuracy': accuracy_score(roi_gt_valid, roi_pred_valid),
                'pixel_count': len(roi_gt_valid),
                'correct_pixels': np.sum(roi_gt_valid == roi_pred_valid),
                'dominant_prediction': int(np.bincount(roi_pred_valid).argmax()),
                'unique_predictions': np.unique(roi_pred_valid).tolist()
            }

            # Check if ROI prediction matches ground truth class
            roi_metrics[roi_id]['class_match'] = (
                roi_metrics[roi_id]['dominant_prediction'] == roi_info['ground_truth_class']
            )

        return roi_metrics

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of current metrics.

        Returns:
            Dictionary with summary statistics
        """
        if self.current_metrics is None:
            return {'error': 'No metrics calculated yet'}

        summary = {
            'overall': {
                'accuracy': self.current_metrics['accuracy'],
                'balanced_accuracy': self.current_metrics['balanced_accuracy'],
                'f1_weighted': self.current_metrics['f1_weighted'],
                'cohen_kappa': self.current_metrics['cohen_kappa'],
                'matthews_corrcoef': self.current_metrics['matthews_corrcoef']
            },
            'per_class_summary': {},
            'confusion_matrix_stats': {}
        }

        # Per-class summary
        if 'per_class' in self.current_metrics:
            for cls_id, cls_metrics in self.current_metrics['per_class'].items():
                summary['per_class_summary'][cls_metrics['class_name']] = {
                    'precision': cls_metrics['precision'],
                    'recall': cls_metrics['recall'],
                    'f1': cls_metrics['f1'],
                    'support': cls_metrics['support']
                }

        # Confusion matrix statistics
        cm = self.current_metrics['confusion_matrix']
        summary['confusion_matrix_stats'] = {
            'total_predictions': int(np.sum(cm)),
            'diagonal_sum': int(np.trace(cm)),
            'off_diagonal_sum': int(np.sum(cm) - np.trace(cm)),
            'class_distribution': {
                int(i): int(np.sum(cm[i, :])) for i in range(cm.shape[0])
            }
        }

        return summary

    def export_metrics(self, filepath: Union[str, Path], format: str = 'json'):
        """
        Export metrics to file.

        Args:
            filepath: Path to save file
            format: 'json', 'csv', or 'excel'
        """
        if self.current_metrics is None:
            print("No metrics to export. Calculate metrics first.")
            return

        filepath = Path(filepath)

        if format == 'json':
            # Convert numpy arrays to lists
            export_data = self._prepare_for_json(self.current_metrics)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

        elif format == 'csv':
            # Create DataFrame from metrics
            df = self._metrics_to_dataframe()
            df.to_csv(filepath, index=False)

        elif format == 'excel':
            # Create multiple sheets for comprehensive export
            with pd.ExcelWriter(filepath) as writer:
                # Overall metrics
                overall_df = pd.DataFrame([self.get_summary_statistics()['overall']])
                overall_df.to_excel(writer, sheet_name='Overall_Metrics', index=False)

                # Per-class metrics
                if 'per_class' in self.current_metrics:
                    per_class_df = pd.DataFrame(self.current_metrics['per_class']).T
                    per_class_df.to_excel(writer, sheet_name='Per_Class_Metrics')

                # Confusion matrix
                cm_df = pd.DataFrame(self.current_metrics['confusion_matrix'])
                cm_df.to_excel(writer, sheet_name='Confusion_Matrix')

        print(f"Metrics exported to {filepath}")

    def _prepare_for_json(self, data: Dict) -> Dict:
        """Convert numpy arrays to lists for JSON serialization."""
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, dict):
                json_data[key] = self._prepare_for_json(value)
            elif isinstance(value, (np.integer, np.floating)):
                json_data[key] = float(value)
            else:
                json_data[key] = value
        return json_data

    def _metrics_to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame format."""
        rows = []

        # Overall metrics row
        overall_row = {'metric_type': 'overall', **self.get_summary_statistics()['overall']}
        rows.append(overall_row)

        # Per-class rows
        if 'per_class' in self.current_metrics:
            for cls_id, cls_metrics in self.current_metrics['per_class'].items():
                cls_row = {
                    'metric_type': 'per_class',
                    'class_id': cls_id,
                    'class_name': cls_metrics['class_name'],
                    **{k: v for k, v in cls_metrics.items()
                       if k not in ['class_name', 'class_id']}
                }
                rows.append(cls_row)

        return pd.DataFrame(rows)