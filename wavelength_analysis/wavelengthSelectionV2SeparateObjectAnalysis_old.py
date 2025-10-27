"""
Wavelength Selection V2 with Separate Object Analysis
=====================================================
This pipeline extends V2 functionality with per-object performance analysis.
It segments individual objects spatially and calculates metrics for each object
separately, providing detailed insights into classification performance.

Author: Wavelength Selection Pipeline V2 Development
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import h5py
from scipy import stats
import warnings
from tqdm import tqdm
import os
import sys
import logging
import argparse
from datetime import datetime

# Import V2 components
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics
from supervised_visualizations import SupervisedVisualizations

# Import object-wise analysis components
from object_segmentation import ObjectSegmentation, SegmentedObject
from object_wise_metrics import ObjectWiseMetrics
from object_wise_visualizations import ObjectWiseVisualizations

# Import necessary functions from V2
from wavelengthselectionV2 import (
    select_informative_wavelengths_fixed,
    extract_wavelength_subset,
    create_roi_colormap,
    run_knn_clustering_pipeline_v2,
    run_kmeans_fallback_v2
)

# Import generated configs
from generated_configs import configurations

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WavelengthSelectionV2ObjectAnalysis:
    """
    Extended V2 Pipeline with Object-Wise Analysis.

    This class combines all V2 functionality with per-object performance analysis,
    providing detailed metrics and visualizations for each segmented object.
    """

    def __init__(self, data_path: str, mask_path: str, sample_name: str,
                 output_dir: str = "results/object_wise_analysis"):
        """
        Initialize the extended pipeline.

        Args:
            data_path: Path to the HDF5 data file
            mask_path: Path to the mask file
            sample_name: Name of the sample
            output_dir: Directory for saving results
        """
        self.data_path = data_path
        self.mask_path = mask_path
        self.sample_name = sample_name
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize trackers and analyzers
        self.ground_truth_tracker = None
        self.supervised_metrics = None  # Will initialize after ground truth setup
        self.supervised_viz = SupervisedVisualizations()

        # Object-wise analysis components
        self.object_segmentation = None
        self.object_metrics = None
        self.object_viz = None

        # Store results
        self.all_results = []
        self.object_results_per_config = {}

        logger.info(f"Initialized V2 Object Analysis Pipeline for {sample_name}")

    def setup_ground_truth(self) -> np.ndarray:
        """
        Set up ground truth tracking with ROI mappings.

        Returns:
            Ground truth array
        """
        # Load ground truth
        with h5py.File(self.data_path, 'r') as f:
            ground_truth = f['gt'][:]

        # Initialize ground truth tracker
        self.ground_truth_tracker = GroundTruthTracker(ground_truth.copy())

        # Now initialize supervised metrics with the tracker
        self.supervised_metrics = SupervisedMetrics(self.ground_truth_tracker)

        # Define ROI regions with their coordinates and expected classes
        roi_regions = [
            {'id': 'Yellow1', 'coordinates': (58, 160, 123, 123), 'expected_class': 1},
            {'id': 'Brown1', 'coordinates': (202, 185, 127, 110), 'expected_class': 2},
            {'id': 'White1', 'coordinates': (431, 188, 116, 110), 'expected_class': 3},
            {'id': 'Brown2', 'coordinates': (608, 199, 116, 117), 'expected_class': 2},
            {'id': 'Green1', 'coordinates': (763, 205, 103, 103), 'expected_class': 4},
            {'id': 'Yellow2', 'coordinates': (44, 354, 134, 137), 'expected_class': 1},
            {'id': 'Green2', 'coordinates': (239, 379, 103, 110), 'expected_class': 4},
            {'id': 'White2', 'coordinates': (416, 386, 117, 103), 'expected_class': 3},
            {'id': 'Brown3', 'coordinates': (594, 390, 117, 103), 'expected_class': 2},
            {'id': 'Yellow3', 'coordinates': (771, 397, 117, 110), 'expected_class': 1},
            {'id': 'Yellow4', 'coordinates': (30, 554, 134, 110), 'expected_class': 1},
            {'id': 'White3', 'coordinates': (226, 554, 117, 117), 'expected_class': 3},
            {'id': 'Green3', 'coordinates': (402, 565, 117, 110), 'expected_class': 4},
            {'id': 'Brown4', 'coordinates': (581, 568, 134, 103), 'expected_class': 2},
            {'id': 'Green4', 'coordinates': (757, 572, 117, 117), 'expected_class': 4},
            {'id': 'White4', 'coordinates': (913, 579, 110, 103), 'expected_class': 3},
        ]

        # Add ROI mappings
        for roi in roi_regions:
            self.ground_truth_tracker.add_roi_mapping(
                roi['id'], roi['coordinates'],
                expected_class=roi['expected_class']
            )

        # Initialize object segmentation
        logger.info("Performing object segmentation on ground truth...")
        self.object_segmentation = ObjectSegmentation(
            connectivity=8,
            min_object_size=100  # Minimum pixels for an object
        )

        # Segment objects
        segmented_objects = self.object_segmentation.segment_objects(
            ground_truth,
            background_value=0
        )

        # Assign ROI IDs to objects
        self.object_segmentation.assign_roi_to_objects(roi_regions)

        # Print segmentation summary
        stats = self.object_segmentation.get_object_statistics()
        logger.info(f"Segmented {stats['total_objects']} objects")
        logger.info(f"Objects per class: {stats['objects_per_class']}")
        logger.info(f"Mean object size: {stats['mean_object_size']:.0f} pixels")

        return ground_truth

    def run_configuration(self, config: Dict, ground_truth: np.ndarray,
                         config_index: int) -> Dict:
        """
        Run a single configuration and perform object-wise analysis.

        Args:
            config: Configuration dictionary
            ground_truth: Ground truth array
            config_index: Index of the configuration

        Returns:
            Results dictionary including object-wise metrics
        """
        logger.info(f"\nRunning configuration {config_index + 1}: {config['config_name']}")

        # Run wavelength selection
        wavelength_indices, purity_score = select_informative_wavelengths_fixed(
            self.data_path, self.mask_path, self.sample_name,
            config, verbose=False
        )

        # Load full data
        with h5py.File(self.data_path, 'r') as f:
            raw_data = f['data'][:]

        # Apply mask
        mask = np.load(self.mask_path)
        data = np.where(mask[..., np.newaxis], raw_data, 0)

        # Crop data and ground truth
        start_col = 467
        end_col = 1392
        data_cropped = data[:, start_col:end_col, :]
        ground_truth_cropped = ground_truth[:, start_col:end_col]

        # Update ground truth tracker with cropped version
        self.ground_truth_tracker.ground_truth = ground_truth_cropped.copy()

        # Extract wavelength subset
        selected_data = extract_wavelength_subset(data_cropped, wavelength_indices)

        # Run clustering
        n_clusters = config['params'].get('n_clusters', 16)

        try:
            # Try KNN clustering first
            cluster_map, metadata = run_knn_clustering_pipeline_v2(
                selected_data, n_clusters,
                roi_regions=self.ground_truth_tracker.roi_regions,
                ground_truth_tracker=self.ground_truth_tracker,
                scale_data=config['params'].get('scale_data', True),
                apply_pca=config['params'].get('apply_pca', False),
                random_state=42
            )
            algorithm_used = 'KNN'
        except Exception as e:
            logger.warning(f"KNN failed: {e}. Falling back to KMeans...")
            # Prepare data for KMeans
            df, valid_mask, metadata = self._prepare_data_for_clustering(selected_data)
            cluster_map = run_kmeans_fallback_v2(
                df, valid_mask, metadata, n_clusters,
                self.ground_truth_tracker, random_state=42
            )
            algorithm_used = 'KMeans'

        # Calculate supervised metrics (global)
        metrics = self.supervised_metrics.calculate_metrics(
            ground_truth_cropped, cluster_map
        )

        # Perform object-wise analysis
        logger.info("Performing object-wise analysis...")

        # Initialize object metrics calculator
        self.object_metrics = ObjectWiseMetrics(self.object_segmentation)

        # Calculate metrics for each object
        object_metrics_dict = self.object_metrics.calculate_object_metrics(
            ground_truth_cropped, cluster_map,
            apply_hungarian=True
        )

        # Aggregate metrics by class
        class_aggregated = self.object_metrics.aggregate_metrics_by_class()

        # Get best and worst performing objects
        best_objects, worst_objects = self.object_metrics.get_best_worst_objects(
            metric='accuracy', n=3
        )

        # Get summary statistics
        object_summary = self.object_metrics.get_summary_statistics()

        # Store results
        result = {
            'config_index': config_index,
            'config_name': config['config_name'],
            'n_wavelengths': len(wavelength_indices),
            'wavelength_indices': wavelength_indices,
            'algorithm': algorithm_used,
            'purity_score': purity_score,
            **metrics,  # Global metrics
            'object_metrics': object_metrics_dict,
            'class_aggregated': class_aggregated,
            'object_summary': object_summary,
            'best_objects': best_objects,
            'worst_objects': worst_objects,
            'cluster_map': cluster_map,
            'ground_truth': ground_truth_cropped
        }

        # Store object performance matrix for this config
        perf_matrix = self.object_metrics.create_performance_matrix()
        self.object_results_per_config[config['config_name']] = perf_matrix

        # Save object-wise results to CSV
        output_path = os.path.join(
            self.output_dir,
            f"object_metrics_{config['config_name'].replace(' ', '_')}.csv"
        )
        self.object_metrics.export_detailed_results(output_path)

        # Create visualizations
        self._create_object_visualizations(
            config, config_index, cluster_map, ground_truth_cropped
        )

        return result

    def _prepare_data_for_clustering(self, data: np.ndarray) -> Tuple:
        """Prepare data for clustering (helper method)."""
        h, w, d = data.shape
        data_reshaped = data.reshape(h * w, d)
        valid_mask = ~np.all(data_reshaped == 0, axis=1)
        df = pd.DataFrame(data_reshaped[valid_mask])

        metadata = {
            'height': h,
            'width': w,
            'depth': d,
            'valid_pixels': np.sum(valid_mask),
            'total_pixels': h * w
        }

        return df, valid_mask, metadata

    def _create_object_visualizations(self, config: Dict, config_index: int,
                                     cluster_map: np.ndarray,
                                     ground_truth: np.ndarray) -> None:
        """Create object-wise visualizations for a configuration."""
        # Initialize visualizer
        self.object_viz = ObjectWiseVisualizations(
            self.object_segmentation,
            self.object_metrics
        )

        # Create visualization subdirectory
        viz_dir = os.path.join(self.output_dir, "visualizations",
                              f"config_{config_index + 1}")
        os.makedirs(viz_dir, exist_ok=True)

        # 1. Object boundaries with metrics
        self.object_viz.plot_object_boundaries_with_metrics(
            ground_truth, cluster_map,
            metric='accuracy',
            save_path=os.path.join(viz_dir, "object_boundaries_accuracy.png")
        )

        # 2. Performance bars
        self.object_viz.plot_object_performance_bars(
            metrics_to_plot=['accuracy', 'precision', 'recall', 'f1'],
            save_path=os.path.join(viz_dir, "object_performance_bars.png")
        )

        # 3. Class aggregated metrics
        self.object_viz.plot_class_aggregated_metrics(
            save_path=os.path.join(viz_dir, "class_aggregated_metrics.png")
        )

        # 4. Performance heatmap
        self.object_viz.plot_performance_heatmap(
            save_path=os.path.join(viz_dir, "performance_heatmap.png")
        )

        # 5. Object size vs accuracy
        self.object_viz.plot_object_size_vs_accuracy(
            save_path=os.path.join(viz_dir, "size_vs_accuracy.png")
        )

        # 6. Create detailed reports for best and worst objects
        if self.object_metrics.object_metrics:
            # Best performing object
            best_objects, _ = self.object_metrics.get_best_worst_objects('accuracy', 1)
            if best_objects:
                best_id = best_objects[0]['object_id']
                self.object_viz.create_object_report_figure(
                    best_id, ground_truth, cluster_map,
                    save_path=os.path.join(viz_dir, f"best_object_{best_id}_report.png")
                )

            # Worst performing object
            _, worst_objects = self.object_metrics.get_best_worst_objects('accuracy', 1)
            if worst_objects:
                worst_id = worst_objects[0]['object_id']
                self.object_viz.create_object_report_figure(
                    worst_id, ground_truth, cluster_map,
                    save_path=os.path.join(viz_dir, f"worst_object_{worst_id}_report.png")
                )

        logger.info(f"Saved visualizations to {viz_dir}")

    def run_all_configurations(self, configurations: List[Dict],
                              max_configs: Optional[int] = None) -> None:
        """
        Run all configurations with object-wise analysis.

        Args:
            configurations: List of configuration dictionaries
            max_configs: Maximum number of configurations to run
        """
        # Set up ground truth and segment objects
        ground_truth = self.setup_ground_truth()

        # Limit configurations if specified
        configs_to_run = configurations[:max_configs] if max_configs else configurations

        logger.info(f"\nRunning {len(configs_to_run)} configurations...")

        # Run each configuration
        for i, config in enumerate(tqdm(configs_to_run, desc="Running configurations")):
            result = self.run_configuration(config, ground_truth, i)
            self.all_results.append(result)

        # Create summary visualizations
        self._create_summary_visualizations()

        # Export comprehensive results
        self._export_comprehensive_results()

        logger.info(f"\nCompleted all configurations. Results saved to {self.output_dir}")

    def _create_summary_visualizations(self) -> None:
        """Create summary visualizations across all configurations."""
        summary_dir = os.path.join(self.output_dir, "summary_visualizations")
        os.makedirs(summary_dir, exist_ok=True)

        # 1. Create DataFrame of global metrics
        global_metrics_df = pd.DataFrame([
            {
                'Configuration': r['config_name'],
                'Wavelengths': r['n_wavelengths'],
                'Accuracy': r['accuracy'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1': r['f1_score']
            }
            for r in self.all_results
        ])

        # 2. Plot global metrics progression
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

        for ax, metric in zip(axes.flatten(), metrics):
            ax.plot(global_metrics_df.index, global_metrics_df[metric], 'o-')
            ax.set_xlabel('Configuration Index')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Across Configurations')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

        plt.suptitle('Global Metrics Progression', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'global_metrics_progression.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Create object performance matrix across configurations
        self._create_object_performance_matrix_plot(summary_dir)

        # 4. Create best configuration summary
        self._create_best_configuration_summary(summary_dir)

        logger.info(f"Created summary visualizations in {summary_dir}")

    def _create_object_performance_matrix_plot(self, save_dir: str) -> None:
        """Create a matrix plot showing object performance across configurations."""
        # Collect accuracy for each object across all configurations
        n_objects = self.object_segmentation.num_objects
        n_configs = len(self.all_results)

        accuracy_matrix = np.zeros((n_objects, n_configs))

        for config_idx, result in enumerate(self.all_results):
            object_metrics = result['object_metrics']
            for obj_id, metrics in object_metrics.items():
                if 'error' not in metrics and 'accuracy' in metrics:
                    obj_idx = obj_id - 1  # Convert to 0-indexed
                    accuracy_matrix[obj_idx, config_idx] = metrics['accuracy']

        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))

        im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')

        # Set labels
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels([r['config_name'] for r in self.all_results],
                          rotation=45, ha='right')
        ax.set_yticks(range(n_objects))
        ax.set_yticklabels([f'Object {i+1}' for i in range(n_objects)])

        ax.set_xlabel('Configuration', fontsize=11)
        ax.set_ylabel('Object ID', fontsize=11)
        ax.set_title('Object Accuracy Across All Configurations', fontsize=14, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations for each cell (if not too many)
        if n_objects * n_configs <= 300:  # Only annotate if reasonable size
            for i in range(n_objects):
                for j in range(n_configs):
                    text = ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                                 ha='center', va='center', fontsize=6,
                                 color='white' if accuracy_matrix[i, j] < 0.5 else 'black')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'object_accuracy_matrix.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _create_best_configuration_summary(self, save_dir: str) -> None:
        """Identify and visualize the best configuration."""
        # Find best configuration by global accuracy
        best_idx = np.argmax([r['accuracy'] for r in self.all_results])
        best_result = self.all_results[best_idx]

        # Create summary figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Global metrics comparison
        ax1 = axes[0, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [best_result[m] for m in metrics]
        bars = ax1.bar(range(len(metrics)), metric_values, color='skyblue')
        ax1.set_xticks(range(len(metrics)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.set_ylim([0, 1.1])
        ax1.set_title(f'Best Configuration: {best_result["config_name"]}')
        for bar, val in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')

        # 2. Object accuracy distribution
        ax2 = axes[0, 1]
        object_accuracies = [m['accuracy'] for m in best_result['object_metrics'].values()
                            if 'error' not in m]
        ax2.hist(object_accuracies, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Number of Objects')
        ax2.set_title('Object Accuracy Distribution')
        ax2.axvline(np.mean(object_accuracies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(object_accuracies):.3f}')
        ax2.legend()

        # 3. Class performance
        ax3 = axes[1, 0]
        class_agg = best_result['class_aggregated']
        if class_agg:
            classes = list(class_agg.keys())
            class_accs = [class_agg[c]['accuracy_mean'] for c in classes]
            bars = ax3.bar(classes, class_accs, color='coral')
            ax3.set_xlabel('Class')
            ax3.set_ylabel('Mean Accuracy')
            ax3.set_title('Mean Accuracy by Class')
            ax3.set_ylim([0, 1.1])
            for bar, val in zip(bars, class_accs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')

        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = f"Configuration Summary:\n\n" \
                      f"Name: {best_result['config_name']}\n" \
                      f"Wavelengths Used: {best_result['n_wavelengths']}\n" \
                      f"Algorithm: {best_result['algorithm']}\n" \
                      f"Global Accuracy: {best_result['accuracy']:.3f}\n" \
                      f"Total Objects: {self.object_segmentation.num_objects}\n" \
                      f"Mean Object Accuracy: {np.mean(object_accuracies):.3f}\n" \
                      f"Std Object Accuracy: {np.std(object_accuracies):.3f}\n" \
                      f"Best Object: #{best_result['best_objects'][0]['object_id']} " \
                      f"({best_result['best_objects'][0]['accuracy']:.3f})\n" \
                      f"Worst Object: #{best_result['worst_objects'][0]['object_id']} " \
                      f"({best_result['worst_objects'][0]['accuracy']:.3f})"

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Best Configuration Analysis (Config #{best_idx + 1})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'best_configuration_summary.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _export_comprehensive_results(self) -> None:
        """Export comprehensive results to various formats."""
        # 1. Export global metrics summary
        global_summary = pd.DataFrame([
            {
                'Config_Index': r['config_index'],
                'Config_Name': r['config_name'],
                'N_Wavelengths': r['n_wavelengths'],
                'Algorithm': r['algorithm'],
                'Purity': r['purity_score'],
                'Global_Accuracy': r['accuracy'],
                'Global_Precision': r['precision'],
                'Global_Recall': r['recall'],
                'Global_F1': r['f1_score'],
                'Mean_Object_Accuracy': r['object_summary'].get('accuracy_global_mean', 0),
                'Std_Object_Accuracy': r['object_summary'].get('accuracy_global_std', 0)
            }
            for r in self.all_results
        ])

        global_summary.to_csv(
            os.path.join(self.output_dir, 'global_metrics_summary.csv'),
            index=False
        )

        # 2. Export detailed object metrics for all configurations
        all_object_metrics = []
        for result in self.all_results:
            config_name = result['config_name']
            for obj_id, metrics in result['object_metrics'].items():
                if 'error' not in metrics:
                    metrics_copy = metrics.copy()
                    metrics_copy['config_name'] = config_name
                    all_object_metrics.append(metrics_copy)

        detailed_df = pd.DataFrame(all_object_metrics)
        detailed_df.to_csv(
            os.path.join(self.output_dir, 'all_object_metrics_detailed.csv'),
            index=False
        )

        # 3. Export class aggregated metrics
        class_summary = []
        for result in self.all_results:
            for class_id, class_metrics in result['class_aggregated'].items():
                class_metrics_copy = class_metrics.copy()
                class_metrics_copy['config_name'] = result['config_name']
                class_summary.append(class_metrics_copy)

        class_df = pd.DataFrame(class_summary)
        class_df.to_csv(
            os.path.join(self.output_dir, 'class_aggregated_summary.csv'),
            index=False
        )

        # 4. Create and save a comprehensive report
        self._create_text_report()

        logger.info(f"Exported comprehensive results to {self.output_dir}")

    def _create_text_report(self) -> None:
        """Create a comprehensive text report."""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WAVELENGTH SELECTION V2 - OBJECT-WISE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample: {self.sample_name}\n")
            f.write(f"Total Configurations Tested: {len(self.all_results)}\n")
            f.write(f"Total Objects Segmented: {self.object_segmentation.num_objects}\n\n")

            # Object segmentation summary
            stats = self.object_segmentation.get_object_statistics()
            f.write("OBJECT SEGMENTATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Objects per class: {stats['objects_per_class']}\n")
            f.write(f"Mean object size: {stats['mean_object_size']:.0f} pixels\n")
            f.write(f"Object size range: {stats['min_object_size']:.0f} - {stats['max_object_size']:.0f} pixels\n\n")

            # Best configuration
            best_idx = np.argmax([r['accuracy'] for r in self.all_results])
            best_result = self.all_results[best_idx]

            f.write("BEST CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Configuration: {best_result['config_name']}\n")
            f.write(f"Index: {best_idx + 1}\n")
            f.write(f"Wavelengths Used: {best_result['n_wavelengths']}\n")
            f.write(f"Algorithm: {best_result['algorithm']}\n")
            f.write(f"Global Accuracy: {best_result['accuracy']:.4f}\n")
            f.write(f"Global Precision: {best_result['precision']:.4f}\n")
            f.write(f"Global Recall: {best_result['recall']:.4f}\n")
            f.write(f"Global F1-Score: {best_result['f1_score']:.4f}\n\n")

            # Object performance summary for best config
            f.write("OBJECT PERFORMANCE (Best Configuration)\n")
            f.write("-" * 40 + "\n")

            object_accuracies = [m['accuracy'] for m in best_result['object_metrics'].values()
                                if 'error' not in m]
            f.write(f"Mean Object Accuracy: {np.mean(object_accuracies):.4f}\n")
            f.write(f"Std Object Accuracy: {np.std(object_accuracies):.4f}\n")
            f.write(f"Min Object Accuracy: {np.min(object_accuracies):.4f}\n")
            f.write(f"Max Object Accuracy: {np.max(object_accuracies):.4f}\n\n")

            # Best and worst objects
            f.write("Top 3 Best Performing Objects:\n")
            for obj in best_result['best_objects']:
                f.write(f"  - Object #{obj['object_id']}: Accuracy={obj['accuracy']:.4f}, "
                       f"Class={obj['class_label']}, Pixels={obj['pixel_count']}\n")

            f.write("\nTop 3 Worst Performing Objects:\n")
            for obj in best_result['worst_objects']:
                f.write(f"  - Object #{obj['object_id']}: Accuracy={obj['accuracy']:.4f}, "
                       f"Class={obj['class_label']}, Pixels={obj['pixel_count']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Created text report at {report_path}")


def main(max_configs: Optional[int] = None):
    """
    Main function to run the V2 pipeline with object-wise analysis.

    Args:
        max_configs: Maximum number of configurations to run
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Wavelength Selection V2 with Object-Wise Analysis'
    )
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Maximum number of configurations to run')
    parser.add_argument('--output-dir', type=str,
                       default='results/object_wise_analysis',
                       help='Output directory for results')
    args = parser.parse_args()

    # Update max_configs if provided via command line
    if args.max_configs is not None:
        max_configs = args.max_configs

    # Set up paths
    data_path = r"F:\HS_DATA\Lichens\lichens_mini.h5"
    mask_path = r"F:\HS_DATA\Lichens\mask_mini.npy"
    sample_name = "Lichens"

    # Initialize pipeline
    pipeline = WavelengthSelectionV2ObjectAnalysis(
        data_path=data_path,
        mask_path=mask_path,
        sample_name=sample_name,
        output_dir=args.output_dir
    )

    # Run all configurations
    pipeline.run_all_configurations(configurations, max_configs=max_configs)

    logger.info("\nObject-wise analysis pipeline completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    # Run with optional configuration limit
    # To run all configurations: python wavelengthselectionV2SeparateObjectAnalysis.py
    # To run limited configs: python wavelengthselectionV2SeparateObjectAnalysis.py --max-configs 3
    import sys

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        main(max_configs=int(sys.argv[1]))
    else:
        main()