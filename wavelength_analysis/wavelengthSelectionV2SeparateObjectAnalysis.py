"""
Wavelength Selection V2 with Simple Object-Wise Analysis
=========================================================
Straightforward implementation that adds per-object metrics to V2 pipeline.

What this does:
1. Runs normal V2 wavelength selection and clustering
2. Spatially separates objects using connected components
3. Calculates metrics for each object individually
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import warnings
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
from scipy import ndimage
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import logging

# Import V2 components
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics
from supervised_visualizations import SupervisedVisualizations

# Import necessary functions from V2
from wavelengthselectionV2 import (
    select_informative_wavelengths_fixed,
    extract_wavelength_subset,
    run_knn_clustering_pipeline_v2,
    run_kmeans_fallback_v2,
    load_masked_data,
    extract_ground_truth_from_png
)

# Import the same configurations as V2
from generated_configs import configurations as wavelength_configs

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def segment_objects_spatially(ground_truth, background_value=0):
    """
    Segment objects based on spatial connectivity.
    Objects of the same class that are spatially separated are different objects.

    Args:
        ground_truth: 2D array with class labels
        background_value: Value for background pixels (default 0)

    Returns:
        Tuple of (list of boolean masks for each object, labeled object map)
    """
    # Create binary mask (non-background pixels)
    foreground_mask = ground_truth != background_value

    # Use connected components to find spatially separated regions
    labeled_objects, num_objects = ndimage.label(foreground_mask)

    # Create mask for each object
    object_masks = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        object_masks.append(obj_mask)

    logger.info(f"Found {num_objects} spatially separated objects")

    return object_masks, labeled_objects


def calculate_metrics_per_object(ground_truth, predictions, object_masks):
    """
    Calculate classification metrics for each spatially separated object.

    Args:
        ground_truth: 2D array with true labels
        predictions: 2D array with predicted labels
        object_masks: List of boolean masks for each object

    Returns:
        DataFrame with metrics for each object
    """
    results = []

    for obj_id, mask in enumerate(object_masks, 1):
        # Get pixels for this object
        y_true = ground_truth[mask]
        y_pred = predictions[mask]

        # Skip if empty
        if len(y_true) == 0:
            continue

        # Calculate metrics
        metrics = {
            'object_id': obj_id,
            'num_pixels': len(y_true),
            'true_class': int(np.unique(y_true)[0]) if len(np.unique(y_true)) == 1 else -1,
            'accuracy': accuracy_score(y_true, y_pred)
        }

        # Add precision, recall, F1 if applicable
        try:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            if len(unique_labels) > 1:
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                # Single class - perfect if all predictions match
                metrics['precision'] = 1.0 if np.all(y_pred == y_true) else 0.0
                metrics['recall'] = metrics['precision']
                metrics['f1'] = metrics['precision']
        except:
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0

        results.append(metrics)

    return pd.DataFrame(results)


def visualize_object_metrics(ground_truth, predictions, object_labels, metrics_df, config_name, save_dir):
    """
    Visualize objects and their metrics.

    Args:
        ground_truth: 2D ground truth array
        predictions: 2D predictions array
        object_labels: 2D array with object IDs
        metrics_df: DataFrame with object metrics
        config_name: Name of the configuration
        save_dir: Directory to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Ground truth
    axes[0, 0].imshow(ground_truth, cmap='tab20')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')

    # 2. Predictions
    axes[0, 1].imshow(predictions, cmap='tab20')
    axes[0, 1].set_title('Predictions')
    axes[0, 1].axis('off')

    # 3. Object segmentation
    axes[0, 2].imshow(object_labels, cmap='nipy_spectral')
    axes[0, 2].set_title(f'Object Segmentation ({len(metrics_df)} objects)')
    axes[0, 2].axis('off')

    # 4. Accuracy per object
    axes[1, 0].bar(metrics_df['object_id'], metrics_df['accuracy'])
    axes[1, 0].set_xlabel('Object ID')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy per Object')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].grid(True, alpha=0.3)

    # Add mean line
    mean_acc = metrics_df['accuracy'].mean()
    axes[1, 0].axhline(y=mean_acc, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_acc:.3f}')
    axes[1, 0].legend()

    # 5. Metrics heatmap
    metric_cols = ['accuracy', 'precision', 'recall', 'f1']
    heatmap_data = metrics_df[metric_cols].values.T
    im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1, 1].set_yticks(range(len(metric_cols)))
    axes[1, 1].set_yticklabels(metric_cols)
    axes[1, 1].set_xticks(range(len(metrics_df)))
    axes[1, 1].set_xticklabels(metrics_df['object_id'].values)
    axes[1, 1].set_xlabel('Object ID')
    axes[1, 1].set_title('Metrics Heatmap')
    plt.colorbar(im, ax=axes[1, 1])

    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"Object-Wise Summary:\n\n"
    summary_text += f"Configuration: {config_name}\n"
    summary_text += f"Total Objects: {len(metrics_df)}\n"
    summary_text += f"Mean Accuracy: {metrics_df['accuracy'].mean():.3f}\n"
    summary_text += f"Std Accuracy: {metrics_df['accuracy'].std():.3f}\n"

    if len(metrics_df) > 0:
        best_obj = metrics_df.loc[metrics_df['accuracy'].idxmax()]
        worst_obj = metrics_df.loc[metrics_df['accuracy'].idxmin()]
        summary_text += f"Best Object: #{int(best_obj['object_id'])} ({best_obj['accuracy']:.3f})\n"
        summary_text += f"Worst Object: #{int(worst_obj['object_id'])} ({worst_obj['accuracy']:.3f})"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Object-Wise Analysis: {config_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f'object_analysis_{config_name.replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved visualization to {save_path}")


def run_configuration_with_object_analysis(config, ground_truth, data_path, mask_path, sample_name, config_index, output_dir):
    """
    Run a single configuration and perform object-wise analysis.

    Args:
        config: Configuration dictionary
        ground_truth: Ground truth array (already cropped)
        data_path: Path to cropped data
        mask_path: Path to cropped mask
        sample_name: Sample name
        config_index: Index of the configuration
        output_dir: Output directory for results

    Returns:
        Dictionary with results
    """
    config_name = config.get('config_name', config.get('name', f'config_{config_index}'))

    logger.info(f"\n{'='*60}")
    logger.info(f"Configuration {config_index + 1}: {config_name}")
    logger.info(f"{'='*60}")

    # Run wavelength selection (returns tuple of 3 values like V2)
    wavelength_combinations, emission_wavelengths_only, selection_results = select_informative_wavelengths_fixed(
        str(data_path), str(mask_path), sample_name,
        config, verbose=False
    )

    # Use wavelength_combinations as the indices
    wavelength_indices = wavelength_combinations
    purity_score = selection_results.get('purity_score', 0.0) if selection_results else 0.0

    # Load cropped data
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)

    # Extract wavelength subset (data is already cropped)
    selected_data = extract_wavelength_subset(full_data, wavelength_indices, verbose=False)

    # Initialize ground truth tracker for V2 compatibility
    gt_tracker = GroundTruthTracker(ground_truth.copy())

    # Define ROI regions (from V2)
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

    # Add ROI mappings to tracker
    for roi in roi_regions:
        gt_tracker.add_roi_mapping(
            roi['id'], roi['coordinates']
        )

    # Run clustering
    n_clusters = config['params'].get('n_clusters', 16)

    try:
        # Try KNN clustering first
        cluster_map, metadata = run_knn_clustering_pipeline_v2(
            selected_data, n_clusters,
            roi_regions=roi_regions,
            ground_truth_tracker=gt_tracker,
            scale_data=config['params'].get('scale_data', True),
            apply_pca=config['params'].get('apply_pca', False),
            random_state=42
        )
        algorithm_used = 'KNN'
    except Exception as e:
        logger.warning(f"KNN failed: {e}. Falling back to KMeans...")
        # Prepare data for KMeans
        h, w, d = selected_data.shape
        data_reshaped = selected_data.reshape(h * w, d)
        valid_mask = ~np.all(data_reshaped == 0, axis=1)
        df = pd.DataFrame(data_reshaped[valid_mask])

        metadata = {
            'height': h,
            'width': w,
            'depth': d,
            'valid_pixels': np.sum(valid_mask),
            'total_pixels': h * w
        }

        cluster_map = run_kmeans_fallback_v2(
            df, valid_mask, metadata, n_clusters,
            gt_tracker, random_state=42
        )
        algorithm_used = 'KMeans'

    # Calculate global metrics using V2's supervised metrics
    supervised_metrics = SupervisedMetrics(gt_tracker)
    global_metrics = supervised_metrics.calculate_metrics(ground_truth, cluster_map)

    # OBJECT-WISE ANALYSIS
    logger.info("\nPerforming object-wise analysis...")

    # 1. Segment objects spatially
    object_masks, object_labels = segment_objects_spatially(ground_truth)

    # 2. Calculate metrics for each object
    object_metrics_df = calculate_metrics_per_object(ground_truth, cluster_map, object_masks)

    # 3. Save object metrics to CSV
    csv_path = os.path.join(output_dir, f'object_metrics_{config_name.replace(" ", "_")}.csv')
    object_metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved object metrics to {csv_path}")

    # 4. Create visualization
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    visualize_object_metrics(ground_truth, cluster_map, object_labels,
                            object_metrics_df, config_name, viz_dir)

    # 5. Print summary
    logger.info("\nObject-Wise Metrics Summary:")
    logger.info(f"  Number of objects: {len(object_metrics_df)}")
    logger.info(f"  Mean accuracy: {object_metrics_df['accuracy'].mean():.3f}")
    logger.info(f"  Std accuracy: {object_metrics_df['accuracy'].std():.3f}")

    if len(object_metrics_df) > 0:
        best_obj = object_metrics_df.loc[object_metrics_df['accuracy'].idxmax()]
        worst_obj = object_metrics_df.loc[object_metrics_df['accuracy'].idxmin()]
        logger.info(f"  Best object: #{int(best_obj['object_id'])} (accuracy={best_obj['accuracy']:.3f})")
        logger.info(f"  Worst object: #{int(worst_obj['object_id'])} (accuracy={worst_obj['accuracy']:.3f})")

    # Return results
    return {
        'config_name': config_name,
        'n_wavelengths': len(wavelength_indices),
        'algorithm': algorithm_used,
        'global_metrics': global_metrics,
        'object_metrics_df': object_metrics_df,
        'object_metrics_summary': {
            'num_objects': len(object_metrics_df),
            'mean_accuracy': object_metrics_df['accuracy'].mean(),
            'std_accuracy': object_metrics_df['accuracy'].std(),
            'min_accuracy': object_metrics_df['accuracy'].min(),
            'max_accuracy': object_metrics_df['accuracy'].max()
        }
    }


def create_summary_visualization(all_results, output_dir):
    """
    Create summary visualization across all configurations.

    Args:
        all_results: List of results from all configurations
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Global accuracy progression
    global_accuracies = [r['global_metrics']['accuracy'] for r in all_results]
    axes[0, 0].plot(range(len(all_results)), global_accuracies, 'o-')
    axes[0, 0].set_xlabel('Configuration Index')
    axes[0, 0].set_ylabel('Global Accuracy')
    axes[0, 0].set_title('Global Accuracy Across Configurations')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])

    # 2. Mean object accuracy progression
    mean_obj_accs = [r['object_metrics_summary']['mean_accuracy'] for r in all_results]
    axes[0, 1].plot(range(len(all_results)), mean_obj_accs, 'o-', color='green')
    axes[0, 1].set_xlabel('Configuration Index')
    axes[0, 1].set_ylabel('Mean Object Accuracy')
    axes[0, 1].set_title('Mean Object Accuracy Across Configurations')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])

    # 3. Object accuracy distribution for best config
    best_idx = np.argmax(global_accuracies)
    best_result = all_results[best_idx]
    axes[1, 0].hist(best_result['object_metrics_df']['accuracy'], bins=20,
                   color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_ylabel('Number of Objects')
    axes[1, 0].set_title(f'Object Accuracy Distribution (Best Config: {best_result["config_name"]})')
    axes[1, 0].axvline(best_result['object_metrics_summary']['mean_accuracy'],
                      color='red', linestyle='--', label='Mean')
    axes[1, 0].legend()

    # 4. Summary statistics
    axes[1, 1].axis('off')
    summary_text = "Overall Summary:\n\n"
    summary_text += f"Total configurations tested: {len(all_results)}\n"
    summary_text += f"Best configuration: {best_result['config_name']}\n"
    summary_text += f"Best global accuracy: {max(global_accuracies):.3f}\n"
    summary_text += f"Best mean object accuracy: {max(mean_obj_accs):.3f}\n\n"

    summary_text += f"Best Config Object Stats:\n"
    summary_text += f"  Objects: {best_result['object_metrics_summary']['num_objects']}\n"
    summary_text += f"  Mean accuracy: {best_result['object_metrics_summary']['mean_accuracy']:.3f}\n"
    summary_text += f"  Std accuracy: {best_result['object_metrics_summary']['std_accuracy']:.3f}\n"
    summary_text += f"  Range: [{best_result['object_metrics_summary']['min_accuracy']:.3f}, "
    summary_text += f"{best_result['object_metrics_summary']['max_accuracy']:.3f}]"

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Object-Wise Analysis Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'summary_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary visualization to {save_path}")


def main(max_configs=None):
    """
    Main function to run V2 pipeline with object-wise analysis.
    Uses the EXACT same data loading as wavelengthselectionV2.py

    Args:
        max_configs: Maximum number of configurations to run (None for all)
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Wavelength Selection V2 with Object-Wise Analysis'
    )
    parser.add_argument('--max-configs', type=int, default=None,
                       help='Maximum number of configurations to run')
    parser.add_argument('--output-dir', type=str,
                       default='results/object_wise_analysis',
                       help='Output directory for results')
    args = parser.parse_args()

    # Update max_configs if provided
    if args.max_configs is not None:
        max_configs = args.max_configs

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("="*60)
    logger.info("WAVELENGTH SELECTION V2 WITH OBJECT-WISE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # EXACT SAME DATA LOADING AS V2
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATA AND GROUND TRUTH (Same as V2)")
    print("=" * 80)

    # Define paths (SAME AS V2)
    base_dir = Path(__file__).parent.parent
    sample_name = "Lichens"
    data_path = base_dir / "data" / "processed" / sample_name / "lichens_data_masked.pkl"
    mask_path = base_dir / "data" / "processed" / sample_name / "lichens_mask.npy"
    png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")

    print("Loading data...")
    print(f"  Sample: {sample_name}")
    print(f"  Data: {data_path.name}")
    print(f"  Ground truth: {png_path.name}")

    # Load hyperspectral data
    print("\nLoading hyperspectral data...")
    full_data = load_masked_data(data_path)

    print(f"Data loaded")
    print(f"  Excitation wavelengths: {full_data['excitation_wavelengths']}")
    print(f"  Number of excitations: {len(full_data['excitation_wavelengths'])}")

    # Extract ground truth
    print("\nExtracting ground truth...")
    background_colors = [
        (24, 24, 24, 255),  # Dark gray background
        (168, 168, 168, 255)  # Light gray background
    ]

    ground_truth, color_mapping, lichen_colors = extract_ground_truth_from_png(
        png_path,
        background_colors=background_colors,
        target_shape=(1040, 1392)
    )

    n_true_classes = len(lichen_colors)
    print(f"Ground truth extracted")
    print(f"  Number of classes: {n_true_classes}")
    print(f"  Shape: {ground_truth.shape}")

    # Apply cropping (same as V2)
    sample_ex = str(full_data['excitation_wavelengths'][0])
    sample_shape = full_data['data'][sample_ex]['cube'].shape

    start_col = 1392 - 925
    end_col = 1392

    print(f"\nCropping data to horizontal range: {start_col} to {end_col}")
    print(f"Original spatial dimensions: {sample_shape[0]} x {sample_shape[1]}")
    print(f"New spatial dimensions: {sample_shape[0]} x {end_col - start_col}")

    # Create cropped version of full_data
    cropped_data = {
        'excitation_wavelengths': full_data['excitation_wavelengths'],
        'metadata': full_data.get('metadata', {}),
        'data': {}
    }

    # Crop each excitation wavelength's data
    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        original_cube = full_data['data'][ex_str]['cube']

        # Crop the spatial dimensions
        cropped_cube = original_cube[:, start_col:end_col, :]

        cropped_data['data'][ex_str] = {
            **full_data['data'][ex_str],
            'cube': cropped_cube
        }

    # Also crop the ground truth
    ground_truth_cropped = ground_truth[:, start_col:end_col]

    # Update working datasets
    full_data = cropped_data
    ground_truth = ground_truth_cropped

    print(f"Data successfully cropped")

    # Save cropped data for wavelength selection
    cropped_data_dir = base_dir / "data" / "processed" / sample_name / "temp_cropped"
    cropped_data_dir.mkdir(parents=True, exist_ok=True)

    cropped_data_path = cropped_data_dir / "lichens_data_cropped.pkl"
    with open(cropped_data_path, 'wb') as f:
        pickle.dump(full_data, f)

    cropped_mask_path = cropped_data_dir / "lichens_mask_cropped.npy"
    cropped_mask = np.ones(ground_truth.shape, dtype=bool)
    cropped_mask[ground_truth == -1] = False
    np.save(cropped_mask_path, cropped_mask)

    # Update paths for wavelength selection
    data_path = cropped_data_path
    mask_path = cropped_mask_path

    # Use V2's wavelength configs
    configs_to_run = wavelength_configs[:max_configs] if max_configs else wavelength_configs
    logger.info(f"Will run {len(configs_to_run)} configurations")

    # Run each configuration
    all_results = []
    for i, config in enumerate(configs_to_run):
        try:
            result = run_configuration_with_object_analysis(
                config, ground_truth_cropped, data_path, mask_path,
                sample_name, i, output_dir
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error in configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create summary visualization if we have results
    if all_results:
        create_summary_visualization(all_results, output_dir)

        # Save summary CSV
        summary_data = []
        for r in all_results:
            summary_data.append({
                'config_name': r['config_name'],
                'n_wavelengths': r['n_wavelengths'],
                'global_accuracy': r['global_metrics']['accuracy'],
                'num_objects': r['object_metrics_summary']['num_objects'],
                'mean_object_accuracy': r['object_metrics_summary']['mean_accuracy'],
                'std_object_accuracy': r['object_metrics_summary']['std_accuracy'],
                'min_object_accuracy': r['object_metrics_summary']['min_accuracy'],
                'max_object_accuracy': r['object_metrics_summary']['max_accuracy']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, 'summary_results.csv')
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary results to {summary_csv}")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()