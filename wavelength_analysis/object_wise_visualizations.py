"""
Object-Wise Visualizations Module for Wavelength Selection Pipeline V2
======================================================================
This module provides visualization functions for per-object analysis results,
including individual object performance plots and comparative visualizations.

Author: Wavelength Selection Pipeline V2 Development
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from object_segmentation import ObjectSegmentation, SegmentedObject
from object_wise_metrics import ObjectWiseMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ObjectWiseVisualizations:
    """
    Create visualizations for object-wise analysis results.

    This class provides various plotting functions to visualize the performance
    of individual objects and compare metrics across objects.
    """

    def __init__(self, segmentation: ObjectSegmentation,
                 metrics: ObjectWiseMetrics,
                 figure_dpi: int = 150):
        """
        Initialize ObjectWiseVisualizations.

        Args:
            segmentation: ObjectSegmentation instance
            metrics: ObjectWiseMetrics instance with calculated metrics
            figure_dpi: DPI for saved figures
        """
        self.segmentation = segmentation
        self.metrics = metrics
        self.figure_dpi = figure_dpi

    def plot_object_boundaries_with_metrics(self, ground_truth: np.ndarray,
                                           predictions: np.ndarray,
                                           metric: str = 'accuracy',
                                           save_path: Optional[str] = None) -> None:
        """
        Plot object boundaries overlaid on the image with metric values.

        Args:
            ground_truth: Ground truth class labels
            predictions: Predicted class labels
            metric: Metric to display for each object
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot ground truth
        im1 = axes[0].imshow(ground_truth, cmap='tab20', interpolation='nearest')
        axes[0].set_title('Ground Truth with Object Boundaries')
        axes[0].axis('off')

        # Plot predictions
        im2 = axes[1].imshow(predictions, cmap='tab20', interpolation='nearest')
        axes[1].set_title('Predictions with Object Boundaries')
        axes[1].axis('off')

        # Plot metric heatmap
        metric_map = np.zeros_like(ground_truth, dtype=float)
        for obj_id, obj_metrics in self.metrics.object_metrics.items():
            if 'error' not in obj_metrics and metric in obj_metrics:
                obj = self.segmentation.get_object_by_id(obj_id)
                if obj:
                    metric_map[obj.pixel_mask] = obj_metrics[metric]

        im3 = axes[2].imshow(metric_map, cmap='RdYlGn', vmin=0, vmax=1,
                           interpolation='nearest')
        axes[2].set_title(f'{metric.capitalize()} by Object')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Add object boundaries and labels to all plots
        for ax in axes:
            for obj in self.segmentation.objects:
                # Get bounding box
                min_row, min_col, max_row, max_col = obj.bounding_box

                # Draw rectangle
                rect = patches.Rectangle((min_col, min_row),
                                        max_col - min_col,
                                        max_row - min_row,
                                        linewidth=2, edgecolor='white',
                                        facecolor='none', linestyle='--')
                ax.add_patch(rect)

                # Add object ID and metric value
                if obj.object_id in self.metrics.object_metrics:
                    obj_metrics = self.metrics.object_metrics[obj.object_id]
                    if 'error' not in obj_metrics and metric in obj_metrics:
                        metric_val = obj_metrics[metric]
                        label_text = f"#{obj.object_id}\n{metric_val:.2f}"
                    else:
                        label_text = f"#{obj.object_id}"
                else:
                    label_text = f"#{obj.object_id}"

                ax.text(min_col + 5, min_row + 15, label_text,
                       color='white', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='black', alpha=0.5))

        plt.suptitle(f'Object Segmentation and {metric.capitalize()} Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved object boundaries plot to {save_path}")

        plt.show()

    def plot_object_performance_bars(self, metrics_to_plot: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Create bar plots showing performance metrics for each object.

        Args:
            metrics_to_plot: List of metrics to include (default: all)
            save_path: Optional path to save the figure
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']

        # Prepare data
        df = self.metrics.create_performance_matrix()
        if df.empty:
            logger.warning("No data to plot")
            return

        # Create subplots
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 3 * n_metrics))
        if n_metrics == 1:
            axes = [axes]

        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            metric_col = metric.capitalize() if metric != 'f1' else 'F1'
            if metric_col not in df.columns:
                logger.warning(f"Metric {metric_col} not found in data")
                continue

            # Create bar plot
            ax = axes[idx]
            bars = ax.bar(df['Object_ID'], df[metric_col])

            # Color bars by class
            unique_classes = df['Class'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            class_colors = dict(zip(unique_classes, colors))

            for bar, class_label in zip(bars, df['Class']):
                bar.set_color(class_colors[class_label])

            # Formatting
            ax.set_xlabel('Object ID', fontsize=11)
            ax.set_ylabel(metric_col, fontsize=11)
            ax.set_title(f'{metric_col} by Object (Colored by Class)', fontsize=12)
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y')

            # Add horizontal line for mean
            mean_val = df[metric_col].mean()
            ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_val:.3f}')
            ax.legend()

            # Add value labels on bars
            for bar, val in zip(bars, df[metric_col]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Object-wise Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved performance bars to {save_path}")

        plt.show()

    def plot_class_aggregated_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot aggregated metrics by class.

        Args:
            save_path: Optional path to save the figure
        """
        # Get aggregated metrics
        aggregated = self.metrics.aggregate_metrics_by_class()
        if not aggregated:
            logger.warning("No aggregated metrics to plot")
            return

        # Prepare data for plotting
        classes = list(aggregated.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Extract mean and std for each class
            means = []
            stds = []
            for cls in classes:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                means.append(aggregated[cls].get(mean_key, 0))
                stds.append(aggregated[cls].get(std_key, 0))

            # Create bar plot with error bars
            x = np.arange(len(classes))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

            # Color bars
            colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            # Formatting
            ax.set_xlabel('Class', fontsize=11)
            ax.set_ylabel(f'{metric.capitalize()} (Mean ± Std)', fontsize=11)
            ax.set_title(f'{metric.capitalize()} by Class', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([f'Class {cls}' for cls in classes])
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)

        plt.suptitle('Class-Aggregated Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved class aggregated metrics to {save_path}")

        plt.show()

    def plot_object_confusion_matrix(self, object_id: int,
                                    ground_truth: np.ndarray,
                                    predictions: np.ndarray,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix for a specific object.

        Args:
            object_id: ID of the object to analyze
            ground_truth: Ground truth labels
            predictions: Predicted labels
            save_path: Optional path to save the figure
        """
        # Get object
        obj = self.segmentation.get_object_by_id(object_id)
        if obj is None:
            logger.error(f"Object {object_id} not found")
            return

        # Extract object pixels
        obj_gt = ground_truth[obj.pixel_mask]
        obj_pred = predictions[obj.pixel_mask]

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        unique_labels = np.unique(np.concatenate([obj_gt, obj_pred]))
        cm = confusion_matrix(obj_gt, obj_pred, labels=unique_labels)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_labels, yticklabels=unique_labels, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'Confusion Matrix for Object #{object_id}\n'
                    f'(Class {obj.class_label}, {obj.pixel_count} pixels)',
                    fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()

    def plot_performance_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create a heatmap showing all metrics for all objects.

        Args:
            save_path: Optional path to save the figure
        """
        # Get performance matrix
        df = self.metrics.create_performance_matrix()
        if df.empty:
            logger.warning("No data for heatmap")
            return

        # Select numeric columns for heatmap
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa', 'MCC']
        heatmap_data = df[metric_cols].T

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                   xticklabels=[f"Obj {id}" for id in df['Object_ID']],
                   yticklabels=metric_cols, ax=ax)

        ax.set_xlabel('Object ID', fontsize=11)
        ax.set_ylabel('Metric', fontsize=11)
        ax.set_title('Object-wise Performance Heatmap', fontsize=14, fontweight='bold')

        # Add class information as text above heatmap
        for idx, (obj_id, cls) in enumerate(zip(df['Object_ID'], df['Class'])):
            ax.text(idx + 0.5, -0.5, f"C{cls}", ha='center', va='bottom',
                   fontsize=8, rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved performance heatmap to {save_path}")

        plt.show()

    def plot_object_size_vs_accuracy(self, save_path: Optional[str] = None) -> None:
        """
        Plot relationship between object size and accuracy.

        Args:
            save_path: Optional path to save the figure
        """
        # Prepare data
        df = self.metrics.create_performance_matrix()
        if df.empty:
            logger.warning("No data to plot")
            return

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color by class
        unique_classes = df['Class'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))

        for cls, color in zip(unique_classes, colors):
            cls_data = df[df['Class'] == cls]
            ax.scatter(cls_data['Pixels'], cls_data['Accuracy'],
                      c=[color], s=100, alpha=0.7,
                      label=f'Class {cls}', edgecolors='black')

            # Add object IDs as labels
            for _, row in cls_data.iterrows():
                ax.annotate(f"{row['Object_ID']}", (row['Pixels'], row['Accuracy']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlabel('Object Size (pixels)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Object Size vs. Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])

        # Add trendline
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['Pixels'], df['Accuracy'])
        line = slope * df['Pixels'] + intercept
        ax.plot(df['Pixels'], line, 'r--', alpha=0.5,
               label=f'Trend (R²={r_value**2:.3f})')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved size vs accuracy plot to {save_path}")

        plt.show()

    def create_object_report_figure(self, object_id: int,
                                   ground_truth: np.ndarray,
                                   predictions: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive report figure for a single object.

        Args:
            object_id: ID of the object to analyze
            ground_truth: Ground truth labels
            predictions: Predicted labels
            save_path: Optional path to save the figure
        """
        # Get object and its metrics
        obj = self.segmentation.get_object_by_id(object_id)
        if obj is None or object_id not in self.metrics.object_metrics:
            logger.error(f"Object {object_id} not found or has no metrics")
            return

        obj_metrics = self.metrics.object_metrics[object_id]

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Object location in full image
        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(ground_truth, cmap='tab20', alpha=0.5)
        rect = patches.Rectangle((obj.bounding_box[1], obj.bounding_box[0]),
                                obj.bounding_box[3] - obj.bounding_box[1],
                                obj.bounding_box[2] - obj.bounding_box[0],
                                linewidth=3, edgecolor='red',
                                facecolor='none')
        ax1.add_patch(rect)
        ax1.set_title(f'Object #{object_id} Location (Class {obj.class_label})',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # 2. Zoomed object - ground truth
        ax2 = fig.add_subplot(gs[1, 0])
        obj_region_gt = ground_truth[obj.bounding_box[0]:obj.bounding_box[2],
                                     obj.bounding_box[1]:obj.bounding_box[3]]
        ax2.imshow(obj_region_gt, cmap='tab20', interpolation='nearest')
        ax2.set_title('Ground Truth', fontsize=11)
        ax2.axis('off')

        # 3. Zoomed object - predictions
        ax3 = fig.add_subplot(gs[1, 1])
        obj_region_pred = predictions[obj.bounding_box[0]:obj.bounding_box[2],
                                      obj.bounding_box[1]:obj.bounding_box[3]]
        ax3.imshow(obj_region_pred, cmap='tab20', interpolation='nearest')
        ax3.set_title('Predictions', fontsize=11)
        ax3.axis('off')

        # 4. Difference map
        ax4 = fig.add_subplot(gs[1, 2])
        diff_map = (obj_region_gt == obj_region_pred).astype(float)
        ax4.imshow(diff_map, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        ax4.set_title('Correct (Green) / Incorrect (Red)', fontsize=11)
        ax4.axis('off')

        # 5. Metrics bar chart
        ax5 = fig.add_subplot(gs[2, :2])
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'cohen_kappa', 'mcc']
        metric_values = [obj_metrics.get(m, 0) for m in metrics_to_show]
        metric_names = [m.replace('_', ' ').title() for m in metrics_to_show]

        bars = ax5.bar(metric_names, metric_values, color='skyblue', edgecolor='navy')
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_title('Performance Metrics', fontsize=11)
        ax5.set_ylim([0, 1.1])
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # 6. Object statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        stats_text = f"Object Statistics:\n" \
                    f"• Object ID: {object_id}\n" \
                    f"• True Class: {obj.class_label}\n" \
                    f"• ROI: {obj.roi_id or 'N/A'}\n" \
                    f"• Total Pixels: {obj.pixel_count}\n" \
                    f"• Correct Pixels: {obj_metrics.get('correct_pixels', 0)}\n" \
                    f"• Incorrect Pixels: {obj_metrics.get('incorrect_pixels', 0)}\n" \
                    f"• Centroid: ({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})"

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Detailed Analysis Report - Object #{object_id}',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved object report to {save_path}")

        plt.show()

    def plot_multi_configuration_comparison(self,
                                           config_results: Dict[str, pd.DataFrame],
                                           metric: str = 'accuracy',
                                           save_path: Optional[str] = None) -> None:
        """
        Compare object performance across multiple configurations.

        Args:
            config_results: Dictionary mapping config names to DataFrames
            metric: Metric to compare
            save_path: Optional path to save the figure
        """
        if not config_results:
            logger.warning("No configuration results to compare")
            return

        # Prepare data
        n_configs = len(config_results)
        n_objects = self.segmentation.num_objects

        fig, ax = plt.subplots(figsize=(14, 8))

        # Set up x-axis
        x = np.arange(n_objects)
        width = 0.8 / n_configs

        # Plot bars for each configuration
        for idx, (config_name, df) in enumerate(config_results.items()):
            if df.empty or metric.capitalize() not in df.columns:
                continue

            values = df[metric.capitalize()].values[:n_objects]
            offset = (idx - n_configs/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=config_name, alpha=0.8)

        ax.set_xlabel('Object ID', fontsize=11)
        ax.set_ylabel(f'{metric.capitalize()}', fontsize=11)
        ax.set_title(f'{metric.capitalize()} Comparison Across Configurations',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'#{i+1}' for i in range(n_objects)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Saved configuration comparison to {save_path}")

        plt.show()