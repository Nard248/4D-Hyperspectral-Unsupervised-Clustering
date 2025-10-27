"""
Supervised Learning Visualizations Module
==========================================
Creates separate, publication-quality visualizations for supervised learning metrics.
Each visualization is saved individually for maximum flexibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


class SupervisedVisualizations:
    """
    Creates individual visualizations for supervised learning metrics.
    """

    def __init__(self, output_dir: Optional[Path] = None, dpi: int = 300):
        """
        Initialize visualization module.

        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for saved figures
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix", save_name: str = "confusion_matrix.png"):
        """
        Create an enhanced confusion matrix visualization.

        Args:
            cm: Confusion matrix
            class_names: Names for each class
            title: Plot title
            save_name: Filename for saving
        """
        n_classes = cm.shape[0]
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize for color mapping
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Frequency', rotation=270, labelpad=20)

        # Set ticks and labels
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        # Add text annotations
        thresh = cm_normalized.max() / 2
        for i in range(n_classes):
            for j in range(n_classes):
                # Show both count and percentage
                text = f"{cm[i, j]:d}\n({cm_normalized[i, j]:.2%})"
                color = "white" if cm_normalized[i, j] > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

        # Labels and title
        ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Add grid for better readability
        ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_per_class_metrics(self, per_class_metrics: Dict,
                              title: str = "Per-Class Performance Metrics",
                              save_name: str = "per_class_metrics.png"):
        """
        Create bar plot for per-class precision, recall, and F1 scores.

        Args:
            per_class_metrics: Dictionary with per-class metrics
            title: Plot title
            save_name: Filename for saving
        """
        # Prepare data
        classes = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        support_values = []

        for cls_id, metrics in per_class_metrics.items():
            classes.append(metrics.get('class_name', f"Class {cls_id}"))
            precision_scores.append(metrics.get('precision', 0))
            recall_scores.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1', 0))
            support_values.append(metrics.get('support', 0))

        x = np.arange(len(classes))
        width = 0.25

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Precision, Recall, F1
        bars1 = ax1.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        bars2 = ax1.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        bars3 = ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes, rotation=45, ha='right')
        ax1.legend(loc='upper right')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')

        # Add horizontal line for average
        avg_f1 = np.mean(f1_scores)
        ax1.axhline(y=avg_f1, color='red', linestyle='--', alpha=0.5, label=f'Avg F1: {avg_f1:.3f}')

        # Plot 2: Support (number of samples per class)
        bars4 = ax2.bar(x, support_values, alpha=0.7, color='teal')
        ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Support (# samples)', fontsize=12, fontweight='bold')
        ax2.set_title('Class Distribution in Dataset', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_accuracy_heatmap(self, ground_truth: np.ndarray, predictions: np.ndarray,
                             title: str = "Spatial Accuracy Heatmap",
                             save_name: str = "accuracy_heatmap.png"):
        """
        Create a heatmap showing spatial distribution of correct/incorrect predictions.

        Args:
            ground_truth: 2D ground truth array
            predictions: 2D predictions array
            title: Plot title
            save_name: Filename for saving
        """
        # Create accuracy map: 0=background, 1=correct, 2=incorrect
        accuracy_map = np.zeros_like(ground_truth, dtype=float)
        background_mask = ground_truth == -1
        correct_mask = (ground_truth == predictions) & (~background_mask)
        incorrect_mask = (ground_truth != predictions) & (~background_mask)

        accuracy_map[background_mask] = np.nan
        accuracy_map[correct_mask] = 1
        accuracy_map[incorrect_mask] = 0

        fig, ax = plt.subplots(figsize=(12, 10))

        # Custom colormap: white for NaN, green for correct, red for incorrect
        colors = ['red', 'green']
        n_bins = 2
        cmap = LinearSegmentedColormap.from_list('accuracy', colors, N=n_bins)

        im = ax.imshow(accuracy_map, cmap=cmap, vmin=0, vmax=1, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Incorrect', 'Correct'])
        cbar.set_label('Prediction', rotation=270, labelpad=20)

        # Calculate and display accuracy statistics
        n_correct = np.sum(correct_mask)
        n_incorrect = np.sum(incorrect_mask)
        n_total = n_correct + n_incorrect
        accuracy = n_correct / n_total if n_total > 0 else 0

        ax.set_title(f'{title}\nAccuracy: {accuracy:.2%} ({n_correct:,}/{n_total:,} pixels)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_misclassification_patterns(self, ground_truth: np.ndarray, predictions: np.ndarray,
                                       class_names: Optional[List[str]] = None,
                                       title: str = "Misclassification Patterns",
                                       save_name: str = "misclassification_patterns.png"):
        """
        Visualize where each class is being misclassified.

        Args:
            ground_truth: 2D ground truth array
            predictions: 2D predictions array
            class_names: Names for each class
            title: Plot title
            save_name: Filename for saving
        """
        # Get unique classes
        unique_classes = np.unique(ground_truth[ground_truth >= 0])
        n_classes = len(unique_classes)

        if class_names is None:
            class_names = [f"Class {i}" for i in unique_classes]

        # Create subplots for each class
        cols = min(3, n_classes)
        rows = (n_classes + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        if n_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes

        for idx, (cls, ax) in enumerate(zip(unique_classes, axes)):
            # Create map for this class
            class_map = np.zeros_like(ground_truth, dtype=float)
            class_map[:] = np.nan

            # Mask for this class
            class_mask = ground_truth == cls

            # Correct predictions for this class
            correct = class_mask & (predictions == cls)
            class_map[correct] = 1

            # Misclassifications
            misclassified = class_mask & (predictions != cls)
            class_map[misclassified] = 0

            # Plot
            im = ax.imshow(class_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax.set_title(f'{class_names[idx]}', fontsize=11, fontweight='bold')
            ax.axis('off')

            # Add statistics
            n_class = np.sum(class_mask)
            n_correct = np.sum(correct)
            acc = n_correct / n_class if n_class > 0 else 0
            ax.text(0.02, 0.98, f'Acc: {acc:.1%}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Remove extra subplots if any
        for idx in range(n_classes, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_roi_performance(self, roi_metrics: Dict,
                            title: str = "ROI Classification Performance",
                            save_name: str = "roi_performance.png"):
        """
        Visualize performance metrics for each ROI.

        Args:
            roi_metrics: Dictionary with ROI metrics
            title: Plot title
            save_name: Filename for saving
        """
        if not roi_metrics:
            print("No ROI metrics to visualize")
            return

        # Prepare data
        roi_names = []
        accuracies = []
        class_matches = []
        pixel_counts = []

        for roi_id, metrics in roi_metrics.items():
            roi_names.append(roi_id)
            accuracies.append(metrics['accuracy'])
            class_matches.append(metrics['class_match'])
            pixel_counts.append(metrics['pixel_count'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: ROI Accuracies
        x = np.arange(len(roi_names))
        colors = ['green' if match else 'red' for match in class_matches]

        bars = ax1.bar(x, accuracies, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=9)

        ax1.set_xlabel('ROI', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(roi_names, rotation=45, ha='right')
        ax1.set_ylim([0, 1.1])
        ax1.grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Correct Class'),
                          Patch(facecolor='red', alpha=0.7, label='Wrong Class')]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Add average line
        avg_acc = np.mean(accuracies)
        ax1.axhline(y=avg_acc, color='blue', linestyle='--', alpha=0.5,
                   label=f'Average: {avg_acc:.2%}')

        # Plot 2: Pixel counts per ROI
        bars2 = ax2.bar(x, pixel_counts, alpha=0.7, color='teal')
        ax2.set_xlabel('ROI', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
        ax2.set_title('ROI Sizes', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(roi_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, count in zip(bars2, pixel_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count):,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_metric_comparison(self, metrics_dict: Dict,
                              title: str = "Overall Metrics Comparison",
                              save_name: str = "metrics_comparison.png"):
        """
        Create a radar plot comparing multiple metrics.

        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
            save_name: Filename for saving
        """
        # Filter metrics for radar plot (should be between 0 and 1)
        valid_metrics = {k: v for k, v in metrics_dict.items()
                        if isinstance(v, (int, float)) and 0 <= v <= 1}

        if len(valid_metrics) < 3:
            print("Not enough valid metrics for radar plot")
            return

        labels = list(valid_metrics.keys())
        values = list(valid_metrics.values())

        # Number of variables
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Draw the plot
        ax.plot(angles, values, 'o-', linewidth=2, label='Metrics', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)

        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.set_rlabel_position(0)

        # Add grid
        ax.grid(True)

        # Add title
        plt.title(title, size=14, fontweight='bold', pad=20)

        # Add value annotations
        for angle, value, label in zip(angles[:-1], values[:-1], labels):
            ax.text(angle, value + 0.05, f'{value:.3f}', size=9, ha='center', va='center')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_class_distribution_comparison(self, ground_truth: np.ndarray, predictions: np.ndarray,
                                          class_names: Optional[List[str]] = None,
                                          title: str = "Class Distribution Comparison",
                                          save_name: str = "class_distribution.png"):
        """
        Compare the distribution of classes in ground truth vs predictions.

        Args:
            ground_truth: Ground truth labels
            predictions: Predicted labels
            class_names: Names for each class
            title: Plot title
            save_name: Filename for saving
        """
        # Get valid pixels
        valid_mask = ground_truth >= 0
        gt_valid = ground_truth[valid_mask]
        pred_valid = predictions[valid_mask]

        # Get unique classes and counts
        gt_classes, gt_counts = np.unique(gt_valid, return_counts=True)
        pred_classes, pred_counts = np.unique(pred_valid, return_counts=True)

        # Ensure all classes are represented
        all_classes = np.unique(np.concatenate([gt_classes, pred_classes]))
        n_classes = len(all_classes)

        if class_names is None:
            class_names = [f"Class {i}" for i in all_classes]

        # Create count arrays
        gt_final = np.zeros(n_classes)
        pred_final = np.zeros(n_classes)

        for i, cls in enumerate(all_classes):
            if cls in gt_classes:
                gt_final[i] = gt_counts[np.where(gt_classes == cls)[0][0]]
            if cls in pred_classes:
                pred_final[i] = pred_counts[np.where(pred_classes == cls)[0][0]]

        # Create plot
        x = np.arange(n_classes)
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))

        bars1 = ax.bar(x - width/2, gt_final, width, label='Ground Truth', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_final, width, label='Predictions', alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Pixels', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_roi_overlay_with_accuracy(self, cluster_map: np.ndarray, ground_truth: np.ndarray,
                                      roi_regions: List[Dict], overall_accuracy: float,
                                      roi_metrics: Optional[Dict] = None,
                                      title: str = "Clustering with ROI Overlay and Accuracy",
                                      save_name: str = "roi_overlay_accuracy.png"):
        """
        Create ROI overlay visualization with accuracy metrics displayed.

        Args:
            cluster_map: 2D array of cluster labels
            ground_truth: 2D ground truth array
            roi_regions: List of ROI dictionaries with 'coords' and 'color' keys
            overall_accuracy: Overall accuracy value
            roi_metrics: Optional dictionary with ROI-specific metrics
            title: Plot title
            save_name: Filename for saving
        """
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle

        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Panel 1: Clustering result without overlay
        ax1 = axes[0]
        cluster_display = np.ma.masked_where(cluster_map == -1, cluster_map)
        im1 = ax1.imshow(cluster_display, cmap='tab10', interpolation='nearest')
        ax1.set_title('Clustering Result', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Add overall accuracy text
        ax1.text(0.02, 0.98, f'Overall Accuracy: {overall_accuracy:.2%}',
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Panel 2: Clustering with ROI boxes and accuracy metrics
        ax2 = axes[1]
        im2 = ax2.imshow(cluster_display, cmap='tab10', interpolation='nearest', alpha=0.8)

        # Draw ROI rectangles with accuracy metrics
        for roi in roi_regions:
            roi_name = roi['name']
            y_start, y_end, x_start, x_end = roi['coords']
            width = x_end - x_start
            height = y_end - y_start

            # Get ROI accuracy if available
            roi_acc = 'N/A'
            if roi_metrics and roi_name in roi_metrics:
                roi_acc = f"{roi_metrics[roi_name]['accuracy']:.1%}"

            # Draw rectangle
            rect = Rectangle((x_start, y_start), width, height,
                           linewidth=3, edgecolor=roi['color'],
                           facecolor='none', linestyle='-')
            ax2.add_patch(rect)

            # Add label with accuracy
            label_text = f"{roi_name}\nAcc: {roi_acc}"
            ax2.text(x_start + width/2, y_start - 5, label_text,
                    color=roi['color'], fontsize=10, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

            # Add accuracy inside ROI if large enough
            if height > 30 and width > 30:
                ax2.text(x_start + width/2, y_start + height/2, roi_acc,
                        color='white', fontsize=14, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=roi['color'], alpha=0.7))

        ax2.set_title(f'ROI Overlay (Overall Acc: {overall_accuracy:.2%})',
                     fontsize=14, fontweight='bold')
        ax2.axis('off')

        # Panel 3: Accuracy comparison by ROI
        ax3 = axes[2]

        if roi_metrics:
            roi_names = []
            accuracies = []
            colors = []

            for roi in roi_regions:
                if roi['name'] in roi_metrics:
                    roi_names.append(roi['name'])
                    accuracies.append(roi_metrics[roi['name']]['accuracy'])
                    colors.append(roi['color'])

            if roi_names:
                bars = ax3.bar(range(len(roi_names)), accuracies, color=colors, alpha=0.8)

                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

                ax3.set_xticks(range(len(roi_names)))
                ax3.set_xticklabels(roi_names, rotation=0)
                ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
                ax3.set_ylim([0, 1.1])
                ax3.set_title('ROI Accuracy Comparison', fontsize=14, fontweight='bold')

                # Add overall accuracy line
                ax3.axhline(y=overall_accuracy, color='red', linestyle='--',
                          label=f'Overall: {overall_accuracy:.2%}', linewidth=2)
                ax3.legend(loc='upper right')
                ax3.grid(True, alpha=0.3, axis='y')
        else:
            # If no ROI metrics, show overall accuracy text
            ax3.text(0.5, 0.5, f'Overall Accuracy\n{overall_accuracy:.2%}',
                    transform=ax3.transAxes, fontsize=20, fontweight='bold',
                    ha='center', va='center')
            ax3.axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def create_all_visualizations(self, metrics: Dict, ground_truth: np.ndarray,
                                 predictions: np.ndarray, roi_metrics: Optional[Dict] = None,
                                 roi_regions: Optional[List[Dict]] = None):
        """
        Create all available visualizations.

        Args:
            metrics: Dictionary containing all metrics
            ground_truth: Ground truth array
            predictions: Predictions array
            roi_metrics: Optional ROI metrics
            roi_regions: Optional ROI region definitions for overlay visualization
        """
        print("\nCreating supervised learning visualizations...")

        # Extract class names if available
        class_names = None
        if 'per_class' in metrics:
            class_names = [m.get('class_name', f"Class {i}")
                          for i, m in metrics['per_class'].items()]

        # 1. Confusion Matrix
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'], class_names)

        # 2. Per-class metrics
        if 'per_class' in metrics:
            self.plot_per_class_metrics(metrics['per_class'])

        # 3. Accuracy heatmap
        self.plot_accuracy_heatmap(ground_truth, predictions)

        # 4. Misclassification patterns
        self.plot_misclassification_patterns(ground_truth, predictions, class_names)

        # 5. ROI performance
        if roi_metrics:
            self.plot_roi_performance(roi_metrics)

        # 6. ROI overlay with accuracy (NEW)
        if roi_regions and 'accuracy' in metrics:
            self.plot_roi_overlay_with_accuracy(
                predictions, ground_truth, roi_regions,
                metrics['accuracy'], roi_metrics
            )

        # 7. Metrics comparison radar plot
        overall_metrics = {
            'Accuracy': metrics.get('accuracy', 0),
            'Balanced Accuracy': metrics.get('balanced_accuracy', 0),
            'Precision (Weighted)': metrics.get('precision_weighted', 0),
            'Recall (Weighted)': metrics.get('recall_weighted', 0),
            'F1 (Weighted)': metrics.get('f1_weighted', 0),
            'Cohen\'s Kappa': max(0, metrics.get('cohen_kappa', 0))  # Can be negative
        }
        self.plot_metric_comparison(overall_metrics)

        # 8. Class distribution comparison
        self.plot_class_distribution_comparison(ground_truth, predictions, class_names)

        print(f"All visualizations saved to: {self.output_dir}")

    def plot_combinations_vs_metrics(self, results_df: pd.DataFrame,
                                    metrics_to_plot: Optional[List[str]] = None,
                                    title_prefix: str = "Wavelength Combinations vs",
                                    save_name: str = "combinations_vs_metrics.png"):
        """
        Plot number of wavelength combinations vs multiple metrics.

        Args:
            results_df: DataFrame with columns including 'n_combinations_selected' and metric columns
            metrics_to_plot: List of metric column names to plot
            title_prefix: Prefix for plot titles
            save_name: Filename for saving
        """
        if metrics_to_plot is None:
            # Default metrics to plot
            metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted',
                             'f1_weighted', 'cohen_kappa', 'purity']

        # Filter to available metrics
        available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

        if not available_metrics:
            print("No metrics available to plot")
            return

        # Determine subplot layout
        n_metrics = len(available_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        # Plot each metric
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx] if n_metrics > 1 else axes[0]

            # Create scatter plot
            scatter = ax.scatter(results_df['n_combinations_selected'],
                               results_df[metric],
                               s=100, alpha=0.7, c=results_df[metric],
                               cmap='viridis', edgecolors='black', linewidth=1)

            # Add trend line
            z = np.polyfit(results_df['n_combinations_selected'], results_df[metric], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(results_df['n_combinations_selected'].min(),
                                results_df['n_combinations_selected'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r-", alpha=0.5, linewidth=2, label='Trend')

            # Highlight best point
            best_idx = results_df[metric].idxmax()
            ax.scatter(results_df.loc[best_idx, 'n_combinations_selected'],
                      results_df.loc[best_idx, metric],
                      s=200, color='red', marker='*', edgecolors='darkred',
                      linewidth=2, label=f'Best: {results_df.loc[best_idx, metric]:.3f}')

            # Labels and formatting
            ax.set_xlabel('Number of Wavelength Combinations', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{title_prefix} {metric.replace("_", " ").title()}',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

            # Add correlation coefficient
            corr = results_df['n_combinations_selected'].corr(results_df[metric])
            ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Remove extra subplots if any
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'Impact of Wavelength Selection on Performance Metrics',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_metrics_progression(self, results_df: pd.DataFrame,
                               primary_metric: str = 'accuracy',
                               secondary_metrics: Optional[List[str]] = None,
                               save_name: str = "metrics_progression.png"):
        """
        Plot the progression of metrics as number of combinations increases.

        Args:
            results_df: DataFrame with results
            primary_metric: Main metric to plot on primary y-axis
            secondary_metrics: Additional metrics to plot
            save_name: Filename for saving
        """
        if secondary_metrics is None:
            secondary_metrics = ['precision_weighted', 'recall_weighted', 'f1_weighted']

        # Sort by number of combinations
        df_sorted = results_df.sort_values('n_combinations_selected')

        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Primary metric on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Number of Wavelength Combinations', fontsize=12, fontweight='bold')
        ax1.set_ylabel(primary_metric.replace('_', ' ').title(),
                      color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(df_sorted['n_combinations_selected'], df_sorted[primary_metric],
                        'o-', color=color1, linewidth=2, markersize=8,
                        label=primary_metric.replace('_', ' ').title())
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Create second y-axis for data reduction
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Data Reduction (%)', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(df_sorted['n_combinations_selected'], df_sorted['data_reduction_pct'],
                        's--', color=color2, linewidth=2, markersize=6,
                        alpha=0.7, label='Data Reduction %')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Plot secondary metrics with different markers
        markers = ['v', '^', '<', '>', 'D', 'p']
        colors = ['tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        lines = line1 + line2

        for idx, metric in enumerate(secondary_metrics):
            if metric in df_sorted.columns and metric != primary_metric:
                line = ax1.plot(df_sorted['n_combinations_selected'], df_sorted[metric],
                              marker=markers[idx % len(markers)],
                              color=colors[idx % len(colors)],
                              linewidth=1.5, markersize=6, alpha=0.7,
                              label=metric.replace('_', ' ').title())
                lines += line

        # Add legend
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=10)

        # Mark best configuration
        best_idx = df_sorted[primary_metric].idxmax()
        best_combinations = df_sorted.loc[best_idx, 'n_combinations_selected']
        best_value = df_sorted.loc[best_idx, primary_metric]

        ax1.axvline(x=best_combinations, color='gray', linestyle=':', alpha=0.5)
        ax1.annotate(f'Best {primary_metric}: {best_value:.3f}\n@ {best_combinations} combinations',
                    xy=(best_combinations, best_value),
                    xytext=(best_combinations + 5, best_value - 0.02),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.title('Metrics Progression with Wavelength Selection',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")

    def plot_pareto_frontier(self, results_df: pd.DataFrame,
                           performance_metric: str = 'accuracy',
                           complexity_metric: str = 'n_combinations_selected',
                           save_name: str = "pareto_frontier.png"):
        """
        Plot Pareto frontier for performance vs complexity trade-off.

        Args:
            results_df: DataFrame with results
            performance_metric: Performance metric to maximize
            complexity_metric: Complexity metric to minimize
            save_name: Filename for saving
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create scatter plot
        scatter = ax.scatter(results_df[complexity_metric],
                           results_df[performance_metric],
                           s=150, alpha=0.6, c=results_df[performance_metric],
                           cmap='viridis', edgecolors='black', linewidth=1)

        # Identify Pareto frontier
        pareto_points = []
        for idx, row in results_df.iterrows():
            is_pareto = True
            for _, other in results_df.iterrows():
                # Check if 'other' dominates 'row'
                if (other[performance_metric] >= row[performance_metric] and
                    other[complexity_metric] <= row[complexity_metric] and
                    (other[performance_metric] > row[performance_metric] or
                     other[complexity_metric] < row[complexity_metric])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(idx)

        # Highlight Pareto optimal points
        if pareto_points:
            pareto_df = results_df.loc[pareto_points].sort_values(complexity_metric)
            ax.plot(pareto_df[complexity_metric], pareto_df[performance_metric],
                   'r-', linewidth=2, label='Pareto Frontier')
            ax.scatter(pareto_df[complexity_metric], pareto_df[performance_metric],
                      s=200, color='red', marker='D', edgecolors='darkred',
                      linewidth=2, label='Pareto Optimal', zorder=5)

        # Add labels for interesting points
        # Best performance
        best_perf_idx = results_df[performance_metric].idxmax()
        ax.annotate(f'Best {performance_metric}',
                   xy=(results_df.loc[best_perf_idx, complexity_metric],
                       results_df.loc[best_perf_idx, performance_metric]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

        # Minimal complexity
        min_complex_idx = results_df[complexity_metric].idxmin()
        ax.annotate(f'Minimal combinations',
                   xy=(results_df.loc[min_complex_idx, complexity_metric],
                       results_df.loc[min_complex_idx, performance_metric]),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2'))

        # Labels and formatting
        ax.set_xlabel('Number of Wavelength Combinations (Complexity)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{performance_metric.replace("_", " ").title()} (Performance)',
                     fontsize=12, fontweight='bold')
        ax.set_title('Performance vs Complexity Trade-off (Pareto Analysis)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(performance_metric.replace('_', ' ').title(),
                      rotation=270, labelpad=20)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_name}")