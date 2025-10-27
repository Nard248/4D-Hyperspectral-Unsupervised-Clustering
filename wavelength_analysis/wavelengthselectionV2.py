"""
Wavelength Selection Pipeline V2
=================================
Enhanced version with integrated ground truth tracking and supervised learning metrics.
Maintains the same schema as the original but adds comprehensive metric tracking.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)
import torch
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

# Create results directory with subdirectories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = base_dir / "wavelength_analysis" / "validation_results_v2" / timestamp
results_dir.mkdir(parents=True, exist_ok=True)
visualizations_dir = results_dir / "visualizations"
visualizations_dir.mkdir(exist_ok=True)
supervised_viz_dir = results_dir / "supervised_visualizations"
supervised_viz_dir.mkdir(exist_ok=True)

# Create subdirectories
paper_results_dir = results_dir / "paper-results"
paper_results_dir.mkdir(exist_ok=True)
concat_data_dir = results_dir / "concat-data"
concat_data_dir.mkdir(exist_ok=True)
experiments_dir = results_dir / "experiments"
experiments_dir.mkdir(exist_ok=True)
analysis_summary_dir = results_dir / "analysis_summary"
analysis_summary_dir.mkdir(exist_ok=True)
metrics_dir = results_dir / "supervised_metrics"
metrics_dir.mkdir(exist_ok=True)

print("=" * 80)
print("WAVELENGTH SELECTION PIPELINE V2")
print("Enhanced with Ground Truth Tracking and Supervised Metrics")
print("=" * 80)
print(f"  Working directory: {base_dir}")
print(f"  Results directory: {results_dir}")
print(f"  Supervised visualizations: {supervised_viz_dir}")
print(f"  Metrics directory: {metrics_dir}")
print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Import required modules from original pipeline
from wavelength_analysis.core.config import AnalysisConfig
from wavelength_analysis.core.analyzer import WavelengthAnalyzer
from concatenation_clustering import (
    load_masked_data,
    concatenate_hyperspectral_data_improved,
    perform_clustering,
    reconstruct_cluster_map
)
from ground_truth_validation import (
    extract_ground_truth_from_png,
    calculate_clustering_accuracy
)

# Import KNN-related libraries
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import enhanced modules from original
from enhanced_difference_visualization import create_enhanced_difference_map, create_simple_difference_overlay
from performance_timing_tracker import TimingTracker, PerformanceTimer
from roi_visualization import create_roi_overlay_visualization, create_roi_analysis_report
from metrics_export import export_experiment_metrics, export_all_experiments_summary, export_experiment_csv
from paper_visualizations import create_ground_truth_difference_maps

# Import NEW V2 modules for ground truth tracking and supervised metrics
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics
from supervised_visualizations import SupervisedVisualizations

print("\nAll modules imported successfully (V2 with Ground Truth Tracking)")


# ============================================================================
# KEEP ORIGINAL FUNCTIONS (same as WavelengthSelectionFinal.py)
# ============================================================================

def select_informative_wavelengths_fixed(data_path, mask_path, sample_name, config_params, verbose=True):
    """
    FIXED VERSION: Returns actual wavelength combinations with excitation-emission pairs!
    (Same as original)
    """
    # Create model directory
    model_dir = base_dir / "results" / f"{sample_name}_wavelength_selection" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "autoencoder_model.pth"

    # Create configuration
    config = AnalysisConfig(
        sample_name=sample_name,
        data_path=str(data_path),
        mask_path=str(mask_path),
        model_path=str(model_path),
        dimension_selection_method=config_params.get('dimension_selection_method', 'activation'),
        perturbation_method=config_params.get('perturbation_method', 'percentile'),
        perturbation_magnitudes=config_params.get('perturbation_magnitudes', [10, 20, 30]),
        n_important_dimensions=config_params.get('n_important_dimensions', 15),
        n_bands_to_select=config_params.get('n_bands_to_select', 30),
        normalization_method=config_params.get('normalization_method', 'variance'),
        use_diversity_constraint=config_params.get('use_diversity_constraint', False),
        diversity_method=config_params.get('diversity_method', 'mmr'),
        lambda_diversity=config_params.get('lambda_diversity', 0.5),
        min_distance_nm=config_params.get('min_distance_nm', 15.0),
        output_dir=str(model_dir.parent / "output"),
        save_tiff_layers=False,
        save_visualizations=False,
        n_baseline_patches=10
    )

    if verbose:
        print(f"\nRunning wavelength selection: {config_params.get('name', 'unnamed')}")

    # Initialize analyzer
    analyzer = WavelengthAnalyzer(config)

    # Load data and model
    analyzer.load_data_and_model()

    # Run wavelength selection analysis
    results = analyzer.run_complete_analysis()

    if results is None or 'selected_bands' not in results:
        raise ValueError("Wavelength selection failed to return results")

    # Extract excitation-emission wavelength combinations
    selected_bands_raw = results['selected_bands']
    wavelength_combinations = []
    emission_wavelengths_only = []

    for band in selected_bands_raw:
        if isinstance(band, dict):
            excitation = band.get('excitation', 'unknown')
            emission = band.get('emission_wavelength', 'unknown')

            # Convert numpy to regular Python float/int
            if hasattr(excitation, 'item'):
                excitation = float(excitation.item())
            else:
                excitation = float(excitation)

            if hasattr(emission, 'item'):
                emission = float(emission.item())
            else:
                emission = float(emission)

            # Store the combination
            combination = {
                'excitation': excitation,
                'emission': emission,
                'combination_name': f"Ex{excitation:.0f}_Em{emission:.1f}"
            }
            wavelength_combinations.append(combination)

            # Also keep emission wavelengths for backward compatibility
            emission_wavelengths_only.append(emission)

    # Remove duplicates while preserving order
    seen_combinations = set()
    unique_combinations = []
    unique_emissions = []

    for combo, emission in zip(wavelength_combinations, emission_wavelengths_only):
        combo_key = (combo['excitation'], combo['emission'])
        if combo_key not in seen_combinations:
            seen_combinations.add(combo_key)
            unique_combinations.append(combo)
            unique_emissions.append(emission)

    if verbose:
        print(f"  Selected {len(unique_combinations)} unique wavelength combinations")
        if unique_combinations:
            print(f"  First few: {[c['combination_name'] for c in unique_combinations[:3]]}...")

    return unique_combinations, unique_emissions, results


def extract_wavelength_subset(full_data, wavelength_combinations, verbose=False):
    """
    Extract subset of data using selected excitation-emission wavelength combinations.
    (Same as original)
    """
    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': [],
        'selected_combinations': wavelength_combinations
    }

    total_bands_original = 0
    total_bands_selected = 0

    # Group combinations by excitation wavelength for efficient processing
    combos_by_excitation = {}
    for combo in wavelength_combinations:
        ex = combo['excitation']
        em = combo['emission']
        if ex not in combos_by_excitation:
            combos_by_excitation[ex] = []
        combos_by_excitation[ex].append(em)

    # Process each excitation wavelength
    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]

        original_wavelengths = np.array(ex_data['wavelengths'])
        original_cube = ex_data['cube']

        total_bands_original += len(original_wavelengths)

        # Check if this excitation has any selected combinations
        matching_emissions = None
        for combo_ex, emissions in combos_by_excitation.items():
            if abs(float(ex) - float(combo_ex)) < 1.0:  # 1nm tolerance
                matching_emissions = emissions
                break

        if matching_emissions is None:
            continue

        # Find indices of selected emission wavelengths
        selected_indices = []
        selected_wl_values = []

        for target_em in matching_emissions:
            target_em = float(target_em)

            # Find closest wavelength
            distances = np.abs(original_wavelengths - target_em)
            closest_idx = np.argmin(distances)

            # Only include if reasonably close (within 10 nm) and not duplicate
            if distances[closest_idx] < 10 and closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
                selected_wl_values.append(original_wavelengths[closest_idx])

        if selected_indices:
            # Extract only the selected bands
            subset_cube = original_cube[:, :, selected_indices]
            total_bands_selected += len(selected_indices)

            subset_data['data'][ex_str] = {
                'cube': subset_cube,
                'wavelengths': selected_wl_values,
                **{k: v for k, v in ex_data.items() if k not in ['cube', 'wavelengths']}
            }

            if ex not in subset_data['excitation_wavelengths']:
                subset_data['excitation_wavelengths'].append(ex)

    if verbose:
        reduction_pct = (1 - total_bands_selected / total_bands_original) * 100 if total_bands_original > 0 else 0
        print(f"  Data reduction: {total_bands_original} → {total_bands_selected} bands ({reduction_pct:.1f}% reduction)")
        print(f"  Selected {len(wavelength_combinations)} specific excitation-emission pairs")
        print(f"  Excitations with selected bands: {len(subset_data['excitation_wavelengths'])}")

    return subset_data


# Define ROI regions (same as original)
ROI_REGIONS = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},  # Red
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},  # Blue
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},  # Green
    {'name': 'Region 4', 'coords': (185, 225, 675, 700), 'color': '#FFFF00'},  # Yellow
]


def create_roi_colormap(roi_regions):
    """Create a custom colormap based on ROI region colors."""
    colors_list = []
    for roi in roi_regions:
        colors_list.append(roi['color'])
    return mcolors.ListedColormap(colors_list)


# ============================================================================
# ENHANCED V2 CLUSTERING FUNCTION WITH GROUND TRUTH TRACKING
# ============================================================================

def run_knn_clustering_pipeline_v2(data, n_clusters, roi_regions=None, ground_truth_tracker=None,
                                   random_state=42, export_concat_data=False, config_name=None):
    """
    V2: Enhanced KNN clustering with ground truth tracking.

    Args:
        data: Hyperspectral data dictionary
        n_clusters: Number of clusters
        roi_regions: ROI regions for training
        ground_truth_tracker: GroundTruthTracker instance for supervised metrics
        random_state: Random seed
        export_concat_data: If True, exports concatenated data
        config_name: Configuration name

    Returns:
        cluster_map: 2D array with cluster assignments
        metrics: Dictionary with clustering metrics
        n_features: Number of spectral features
        supervised_metrics: Dictionary with supervised learning metrics (if tracker provided)
    """
    if roi_regions is None:
        roi_regions = ROI_REGIONS

    # Step 1: Use improved data concatenation
    df, valid_mask, metadata = concatenate_hyperspectral_data_improved(
        data,
        global_normalize=True,
        normalization_method='global_percentile'
    )

    # Export concatenated data if requested
    if export_concat_data and config_name:
        concat_filename = concat_data_dir / f"{config_name}_concatenated_data.csv"
        df.to_csv(concat_filename, index=False)
        print(f"    Exported concatenated data to: {concat_filename.name}")

    # Step 2: Extract ROI training data
    spectral_cols = [col for col in df.columns if col not in ['x', 'y']]
    X_train_list = []
    y_train_list = []

    # Create label mapping
    label_mapping = {roi['name']: i for i, roi in enumerate(roi_regions)}

    # V2 ENHANCEMENT: Map ROIs to ground truth classes
    roi_to_gt_mapping = {}
    if ground_truth_tracker:
        for roi in roi_regions:
            roi_name = roi['name']
            roi_info = ground_truth_tracker.add_roi_mapping(
                roi_id=roi_name,
                coordinates=roi['coords'],
                verify_single_class=True
            )
            roi_to_gt_mapping[roi_name] = roi_info['ground_truth_class']
            print(f"      ROI '{roi_name}' mapped to GT class {roi_info['ground_truth_class']} "
                  f"({roi_info.get('class_name', 'Unknown')})")

    # Extract data from each ROI region
    for roi in roi_regions:
        roi_name = roi['name']
        y_start, y_end, x_start, x_end = roi['coords']

        # Find pixels within this ROI
        roi_mask = (
            (df['x'] >= x_start) & (df['x'] < x_end) &
            (df['y'] >= y_start) & (df['y'] < y_end)
        )

        roi_pixels = df[roi_mask]
        n_pixels = len(roi_pixels)

        if n_pixels > 0:
            # Extract spectral features
            roi_spectra = roi_pixels[spectral_cols].values

            # V2: Use ground truth class if available, otherwise use ROI index
            if roi_name in roi_to_gt_mapping:
                roi_labels = [roi_to_gt_mapping[roi_name]] * n_pixels
            else:
                roi_labels = [label_mapping[roi_name]] * n_pixels

            X_train_list.append(roi_spectra)
            y_train_list.extend(roi_labels)

    # Combine all ROI data
    if X_train_list:
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
    else:
        # Fallback to KMeans if no ROI data
        return run_kmeans_fallback_v2(df, valid_mask, metadata, n_clusters,
                                      ground_truth_tracker, random_state)

    # Step 3: Train KNN classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_model.fit(X_train_scaled, y_train)

    # Step 4: Apply to full image
    X_full = df[spectral_cols].values
    X_full_scaled = scaler.transform(X_full)
    predictions = knn_model.predict(X_full_scaled)

    # Step 5: Reconstruct cluster map
    height, width = metadata['height'], metadata['width']
    cluster_map = np.full((height, width), -1, dtype=int)

    # Fill in predictions for valid pixels
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = int(row['x']), int(row['y'])
        cluster_map[y, x] = predictions[i]

    # Step 6: Calculate clustering metrics
    unique_labels = np.unique(predictions)
    n_actual_clusters = len(unique_labels)

    # Calculate basic clustering metrics
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        n_samples = min(10000, len(X_full_scaled))
        sample_indices = np.random.choice(len(X_full_scaled), n_samples, replace=False)
        X_sample = X_full_scaled[sample_indices]
        pred_sample = predictions[sample_indices]

        if len(np.unique(pred_sample)) > 1:
            silhouette = silhouette_score(X_sample, pred_sample)
            davies_bouldin = davies_bouldin_score(X_sample, pred_sample)
            calinski_harabasz = calinski_harabasz_score(X_sample, pred_sample)
        else:
            silhouette = 0.0
            davies_bouldin = float('inf')
            calinski_harabasz = 0.0
    except:
        silhouette = 0.0
        davies_bouldin = float('inf')
        calinski_harabasz = 0.0

    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_clusters_found': n_actual_clusters,
        'cluster_counts': dict(zip(*np.unique(predictions, return_counts=True)))
    }

    # V2 ENHANCEMENT: Calculate supervised metrics if ground truth tracker is provided
    supervised_metrics = None
    if ground_truth_tracker:
        # Set predictions in tracker
        ground_truth_tracker.set_predictions(cluster_map)

        # Calculate supervised metrics
        metrics_calculator = SupervisedMetrics(ground_truth_tracker)
        supervised_metrics = metrics_calculator.calculate_metrics(
            cluster_map,
            use_hungarian_mapping=True
        )

        # Add ROI-specific metrics
        roi_metrics = metrics_calculator.calculate_roi_metrics(cluster_map)
        supervised_metrics['roi_metrics'] = roi_metrics

        # Get classification report
        report = metrics_calculator.get_classification_report(cluster_map)
        supervised_metrics['classification_report'] = report

    n_features = len(spectral_cols)

    return cluster_map, metrics, n_features, supervised_metrics


def run_kmeans_fallback_v2(df, valid_mask, metadata, n_clusters, ground_truth_tracker, random_state):
    """V2: Fallback to KMeans with ground truth tracking."""
    spectral_cols = [col for col in df.columns if col not in ['x', 'y']]
    X = df[spectral_cols].values

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    # Reconstruct cluster map
    height, width = metadata['height'], metadata['width']
    cluster_map = np.full((height, width), -1, dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        x, y = int(row['x']), int(row['y'])
        cluster_map[y, x] = labels[i]

    # Calculate metrics
    sample_size = min(10000, len(X))
    metrics = {
        'silhouette_score': silhouette_score(X, labels, sample_size=sample_size),
        'davies_bouldin_score': davies_bouldin_score(X, labels),
        'calinski_harabasz_score': calinski_harabasz_score(X, labels),
        'inertia': kmeans.inertia_
    }

    # Calculate supervised metrics if tracker provided
    supervised_metrics = None
    if ground_truth_tracker:
        ground_truth_tracker.set_predictions(cluster_map)
        metrics_calculator = SupervisedMetrics(ground_truth_tracker)
        supervised_metrics = metrics_calculator.calculate_metrics(cluster_map, use_hungarian_mapping=True)

    return cluster_map, metrics, len(spectral_cols), supervised_metrics


# Wrapper function for compatibility
def run_clustering_pipeline_v2(data, n_clusters, ground_truth_tracker=None, random_state=42,
                              export_concat_data=False, config_name=None):
    """V2: Wrapper function that includes ground truth tracking."""
    return run_knn_clustering_pipeline_v2(
        data, n_clusters,
        roi_regions=ROI_REGIONS,
        ground_truth_tracker=ground_truth_tracker,
        random_state=random_state,
        export_concat_data=export_concat_data,
        config_name=config_name
    )


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def main(max_configs=None):
    """Main execution function for V2 pipeline.

    Args:
        max_configs: Maximum number of configurations to run (None = run all)
    """

    print("\n" + "=" * 80)
    print("LOADING DATA AND GROUND TRUTH")
    print("=" * 80)

    # Define paths
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

    # Count total bands
    total_bands = 0
    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        total_bands += len(full_data['data'][ex_str]['wavelengths'])
    print(f"  Total spectral bands: {total_bands}")

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

    # Apply cropping (same as original)
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

    # Save cropped data for autoencoder training
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

    # ========================================================================
    # V2 ENHANCEMENT: Initialize Ground Truth Tracker
    # ========================================================================
    print("\n" + "=" * 80)
    print("INITIALIZING GROUND TRUTH TRACKER")
    print("=" * 80)

    # Get class names from color mapping (optional)
    class_names = [f"Lichen_Type_{i}" for i in range(n_true_classes)]

    # Initialize tracker
    gt_tracker = GroundTruthTracker(ground_truth, class_names)

    # Export tracker state for future use
    tracker_state_file = metrics_dir / "ground_truth_tracker_state.pkl"
    gt_tracker.export_state(tracker_state_file)

    # Get initial statistics
    class_distribution = gt_tracker.get_class_distribution()
    print("\nClass Distribution:")
    for cls_id, info in class_distribution.items():
        if cls_id >= 0:  # Skip background
            print(f"  {info['name']}: {info['pixel_count']:,} pixels ({info['percentage']:.2f}%)")

    # Initialize supervised visualization module
    supervised_viz = SupervisedVisualizations(output_dir=supervised_viz_dir, dpi=300)

    # ========================================================================
    # BASELINE: Full Data Clustering with Ground Truth Tracking
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: Clustering with Full Data (V2)")
    print("=" * 80)
    n_clusters = len(ROI_REGIONS)

    roi_colormap = create_roi_colormap(ROI_REGIONS)
    timing_tracker = TimingTracker()
    cluster_maps = {}

    with PerformanceTimer() as baseline_timer:
        cluster_map_full, metrics_full, n_features_full, supervised_metrics_full = run_clustering_pipeline_v2(
            full_data,
            n_clusters,
            ground_truth_tracker=gt_tracker,
            export_concat_data=True,
            config_name="BASELINE_FULL_DATA"
        )

    timing_tracker.record_clustering_full(
        baseline_timer.elapsed,
        n_features_full,
        "BASELINE"
    )

    cluster_maps['BASELINE_FULL_DATA'] = cluster_map_full.copy()

    # Calculate ground truth metrics (original method for comparison)
    baseline_metrics = calculate_clustering_accuracy(
        cluster_map_full,
        ground_truth,
        np.ones_like(ground_truth, dtype=bool)
    )

    print(f"\nBaseline Results:")
    print(f"  Features: {n_features_full}")
    print(f"  === Original Metrics ===")
    print(f"  Purity: {baseline_metrics['purity']:.4f}")
    print(f"  ARI: {baseline_metrics['adjusted_rand_score']:.4f}")
    print(f"  NMI: {baseline_metrics['normalized_mutual_info']:.4f}")

    if supervised_metrics_full:
        print(f"  === V2 Supervised Metrics ===")
        print(f"  Accuracy: {supervised_metrics_full['accuracy']:.4f}")
        print(f"  Precision (weighted): {supervised_metrics_full['precision_weighted']:.4f}")
        print(f"  Recall (weighted): {supervised_metrics_full['recall_weighted']:.4f}")
        print(f"  F1 (weighted): {supervised_metrics_full['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa: {supervised_metrics_full['cohen_kappa']:.4f}")

        # Create supervised visualizations for baseline
        supervised_viz.create_all_visualizations(
            supervised_metrics_full,
            ground_truth,
            cluster_map_full,
            supervised_metrics_full.get('roi_metrics'),
            roi_regions=ROI_REGIONS  # Pass ROI regions for overlay visualization
        )

        # Create standalone ROI overlay for baseline (saved to paper-results)
        paper_viz = SupervisedVisualizations(output_dir=paper_results_dir, dpi=300)
        paper_viz.plot_roi_overlay_with_accuracy(
            cluster_map=cluster_map_full,
            ground_truth=ground_truth,
            roi_regions=ROI_REGIONS,
            overall_accuracy=supervised_metrics_full['accuracy'],
            roi_metrics=supervised_metrics_full.get('roi_metrics'),
            title="BASELINE - Full Data",
            save_name="BASELINE_roi_overlay.png"
        )

        # Export supervised metrics
        baseline_metrics_file = metrics_dir / "baseline_supervised_metrics.json"
        sm = SupervisedMetrics(gt_tracker)
        sm.current_metrics = supervised_metrics_full
        sm.export_metrics(baseline_metrics_file, format='json')

    print(f"  ⏱️ Clustering time: {baseline_timer.elapsed:.2f}s")

    # ========================================================================
    # Run Wavelength Selection Configurations
    # ========================================================================
    from generated_configs import configurations

    # Determine how many configurations to run
    configs_to_run = configurations[:max_configs] if max_configs else configurations
    n_total_configs = len(configurations)
    n_configs_to_run = len(configs_to_run)

    print(f"\n{n_total_configs} configurations available")
    if max_configs:
        print(f"Running first {n_configs_to_run} configurations (max_configs={max_configs})")
    else:
        print(f"Running ALL {n_configs_to_run} configurations")

    # Initialize results storage
    results = []
    all_combinations = []

    print("\n" + "=" * 80)
    print("RUNNING WAVELENGTH SELECTION CONFIGURATIONS (V2)")
    print("=" * 80)

    for i, config in enumerate(tqdm(configs_to_run, desc="Running configurations")):  # Run selected configurations
        config_name = config['name']
        print(f"\n[{i + 1}/{n_configs_to_run}] Running: {config_name}")

        try:
            # Step 1: Wavelength selection
            with PerformanceTimer() as selection_timer:
                wavelength_combinations, emission_wavelengths_only, selection_results = select_informative_wavelengths_fixed(
                    data_path,
                    mask_path,
                    sample_name,
                    config,
                    verbose=False
                )

            timing_tracker.record_wavelength_selection(
                selection_timer.elapsed,
                len(wavelength_combinations),
                config_name
            )

            print(f"  Selected {len(wavelength_combinations)} wavelength combinations")
            print(f"  ⏱️ Selection time: {selection_timer.elapsed:.2f}s")

            # Step 2: Extract subset
            subset_data = extract_wavelength_subset(
                full_data,
                wavelength_combinations,
                verbose=True
            )

            # Step 3: Clustering with ground truth tracking
            with PerformanceTimer() as cluster_timer:
                cluster_map, metrics, n_features, supervised_metrics = run_clustering_pipeline_v2(
                    subset_data,
                    n_clusters,
                    ground_truth_tracker=gt_tracker,
                    export_concat_data=True,
                    config_name=config_name
                )

            timing_tracker.record_clustering_subset(
                cluster_timer.elapsed,
                n_features,
                config_name
            )

            cluster_maps[config_name] = cluster_map.copy()

            # Step 4: Calculate metrics
            gt_metrics = calculate_clustering_accuracy(
                cluster_map,
                ground_truth,
                np.ones_like(ground_truth, dtype=bool)
            )

            data_reduction_pct = (1 - n_features / n_features_full) * 100
            speedup = baseline_timer.elapsed / cluster_timer.elapsed if cluster_timer.elapsed > 0 else 0

            # Create experiment folder
            experiment_folder = experiments_dir / config_name
            experiment_folder.mkdir(exist_ok=True)

            # V2: Create supervised visualizations for this configuration
            if supervised_metrics:
                config_viz_dir = experiment_folder / "supervised_visualizations"
                config_viz_dir.mkdir(exist_ok=True)
                config_viz = SupervisedVisualizations(output_dir=config_viz_dir, dpi=300)
                config_viz.create_all_visualizations(
                    supervised_metrics,
                    ground_truth,
                    cluster_map,
                    supervised_metrics.get('roi_metrics'),
                    roi_regions=ROI_REGIONS  # Pass ROI regions for overlay visualization
                )

                # Create standalone ROI overlay for paper (saved to main experiment folder)
                config_viz.plot_roi_overlay_with_accuracy(
                    cluster_map=cluster_map,
                    ground_truth=ground_truth,
                    roi_regions=ROI_REGIONS,
                    overall_accuracy=supervised_metrics['accuracy'],
                    roi_metrics=supervised_metrics.get('roi_metrics'),
                    title=f"{config_name} - ROI Overlay with Accuracy",
                    save_name=f"{config_name}_roi_overlay_main.png"
                )

                # Also save to paper-results directory for easy access
                paper_viz = SupervisedVisualizations(output_dir=paper_results_dir, dpi=300)
                paper_viz.plot_roi_overlay_with_accuracy(
                    cluster_map=cluster_map,
                    ground_truth=ground_truth,
                    roi_regions=ROI_REGIONS,
                    overall_accuracy=supervised_metrics['accuracy'],
                    roi_metrics=supervised_metrics.get('roi_metrics'),
                    title=f"{config_name}",
                    save_name=f"{config_name}_roi_overlay.png"
                )

                # Export supervised metrics
                config_metrics_file = experiment_folder / f"{config_name}_supervised_metrics.json"
                sm = SupervisedMetrics(gt_tracker)
                sm.current_metrics = supervised_metrics
                sm.export_metrics(config_metrics_file, format='json')

                print(f"    V2 Metrics - Accuracy: {supervised_metrics['accuracy']:.4f}, "
                      f"F1: {supervised_metrics['f1_weighted']:.4f}")

            # Store results
            result = {
                'config_name': config_name,
                'n_combinations_selected': len(wavelength_combinations),
                'n_features': n_features,
                'data_reduction_pct': data_reduction_pct,
                # Original metrics
                'purity': gt_metrics['purity'],
                'ari': gt_metrics['adjusted_rand_score'],
                'nmi': gt_metrics['normalized_mutual_info'],
                # V2 Supervised metrics
                'accuracy': supervised_metrics.get('accuracy', 0) if supervised_metrics else 0,
                'precision_weighted': supervised_metrics.get('precision_weighted', 0) if supervised_metrics else 0,
                'recall_weighted': supervised_metrics.get('recall_weighted', 0) if supervised_metrics else 0,
                'f1_weighted': supervised_metrics.get('f1_weighted', 0) if supervised_metrics else 0,
                'cohen_kappa': supervised_metrics.get('cohen_kappa', 0) if supervised_metrics else 0,
                # Timing
                'selection_time': selection_timer.elapsed,
                'clustering_time': cluster_timer.elapsed,
                'speedup_factor': speedup
            }

            results.append(result)

            print(f"  Purity: {gt_metrics['purity']:.4f} | ARI: {gt_metrics['adjusted_rand_score']:.4f}")
            print(f"  ⏱️ Clustering time: {cluster_timer.elapsed:.2f}s | Speedup: {speedup:.2f}x")

        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                'config_name': config_name,
                'error': str(e)
            })

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Add baseline row
    baseline_row = {
        'config_name': 'BASELINE_FULL_DATA',
        'n_combinations_selected': total_bands,
        'n_features': n_features_full,
        'data_reduction_pct': 0.0,
        'purity': baseline_metrics['purity'],
        'ari': baseline_metrics['adjusted_rand_score'],
        'nmi': baseline_metrics['normalized_mutual_info'],
        'accuracy': supervised_metrics_full.get('accuracy', 0) if supervised_metrics_full else 0,
        'precision_weighted': supervised_metrics_full.get('precision_weighted', 0) if supervised_metrics_full else 0,
        'recall_weighted': supervised_metrics_full.get('recall_weighted', 0) if supervised_metrics_full else 0,
        'f1_weighted': supervised_metrics_full.get('f1_weighted', 0) if supervised_metrics_full else 0,
        'cohen_kappa': supervised_metrics_full.get('cohen_kappa', 0) if supervised_metrics_full else 0,
    }

    df_results = pd.concat([pd.DataFrame([baseline_row]), df_results], ignore_index=True)

    # Sort by accuracy (V2 metric)
    df_results = df_results.sort_values('accuracy', ascending=False)

    # Save to Excel
    excel_path = results_dir / "wavelength_selection_results_v2.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Results_V2', index=False)

    print(f"Results saved to: {excel_path}")

    # ========================================================================
    # Generate Summary Visualizations for All Experiments
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("=" * 80)

    summary_viz_dir = results_dir / "summary_visualizations"
    summary_viz_dir.mkdir(exist_ok=True)
    summary_viz = SupervisedVisualizations(output_dir=summary_viz_dir, dpi=300)

    # 1. Plot combinations vs multiple metrics (like the original purity plot)
    print("\n1. Creating combinations vs metrics plots...")
    summary_viz.plot_combinations_vs_metrics(
        df_results,
        metrics_to_plot=['accuracy', 'precision_weighted', 'recall_weighted',
                        'f1_weighted', 'cohen_kappa', 'purity'],
        save_name="combinations_vs_all_metrics.png"
    )

    # 2. Plot metrics progression
    print("2. Creating metrics progression plot...")
    summary_viz.plot_metrics_progression(
        df_results,
        primary_metric='accuracy',
        secondary_metrics=['f1_weighted', 'precision_weighted', 'recall_weighted'],
        save_name="metrics_progression.png"
    )

    # 3. Plot Pareto frontier for accuracy vs complexity
    print("3. Creating Pareto frontier analysis...")
    summary_viz.plot_pareto_frontier(
        df_results,
        performance_metric='accuracy',
        complexity_metric='n_combinations_selected',
        save_name="pareto_frontier_accuracy.png"
    )

    # Also create for F1 score
    summary_viz.plot_pareto_frontier(
        df_results,
        performance_metric='f1_weighted',
        complexity_metric='n_combinations_selected',
        save_name="pareto_frontier_f1.png"
    )

    # 4. Create individual plots for each metric (for paper)
    print("4. Creating individual metric plots for paper...")
    individual_metrics = ['accuracy', 'precision_weighted', 'recall_weighted',
                         'f1_weighted', 'cohen_kappa', 'purity']

    for metric in individual_metrics:
        if metric in df_results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Scatter plot
            scatter = ax.scatter(df_results['n_combinations_selected'],
                               df_results[metric],
                               s=120, alpha=0.7, c=df_results[metric],
                               cmap='viridis', edgecolors='black', linewidth=1)

            # Trend line
            z = np.polyfit(df_results['n_combinations_selected'], df_results[metric], 2)
            p = np.poly1d(z)
            x_trend = np.linspace(df_results['n_combinations_selected'].min(),
                                df_results['n_combinations_selected'].max(), 100)
            ax.plot(x_trend, p(x_trend), "r-", alpha=0.5, linewidth=2, label='Trend')

            # Highlight best and baseline
            best_idx = df_results[metric].idxmax()
            baseline_idx = df_results[df_results['config_name'] == 'BASELINE_FULL_DATA'].index[0] if 'BASELINE_FULL_DATA' in df_results['config_name'].values else None

            ax.scatter(df_results.loc[best_idx, 'n_combinations_selected'],
                      df_results.loc[best_idx, metric],
                      s=200, color='red', marker='*', edgecolors='darkred',
                      linewidth=2, label=f'Best: {df_results.loc[best_idx, metric]:.3f}', zorder=5)

            if baseline_idx is not None:
                ax.scatter(df_results.loc[baseline_idx, 'n_combinations_selected'],
                          df_results.loc[baseline_idx, metric],
                          s=200, color='blue', marker='s', edgecolors='darkblue',
                          linewidth=2, label=f'Baseline: {df_results.loc[baseline_idx, metric]:.3f}', zorder=5)

            ax.set_xlabel('Number of Wavelength Combinations', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_title(f'Wavelength Selection Impact on {metric.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=11)

            # Add statistics
            corr = df_results['n_combinations_selected'].corr(df_results[metric])
            ax.text(0.02, 0.02, f'Correlation: {corr:.3f}\nSamples: {len(df_results)}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.savefig(summary_viz_dir / f"combinations_vs_{metric}.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    Created plot for {metric}")

    # 5. Create comparison matrix plot
    print("5. Creating metrics comparison matrix...")
    metrics_for_matrix = ['accuracy', 'precision_weighted', 'recall_weighted',
                          'f1_weighted', 'cohen_kappa']
    available_metrics_matrix = [m for m in metrics_for_matrix if m in df_results.columns]

    if len(available_metrics_matrix) > 1:
        fig, axes = plt.subplots(len(available_metrics_matrix), len(available_metrics_matrix),
                                 figsize=(4*len(available_metrics_matrix), 4*len(available_metrics_matrix)))

        for i, metric1 in enumerate(available_metrics_matrix):
            for j, metric2 in enumerate(available_metrics_matrix):
                ax = axes[i, j] if len(available_metrics_matrix) > 1 else axes

                if i == j:
                    # Diagonal: histogram
                    ax.hist(df_results[metric1], bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel(metric1.replace('_', ' ').title())
                    ax.set_ylabel('Frequency')
                else:
                    # Off-diagonal: scatter
                    ax.scatter(df_results[metric2], df_results[metric1],
                             s=50, alpha=0.6, c=df_results['n_combinations_selected'],
                             cmap='viridis')
                    ax.set_xlabel(metric2.replace('_', ' ').title())
                    ax.set_ylabel(metric1.replace('_', ' ').title())

                    # Add correlation
                    corr = df_results[metric1].corr(df_results[metric2])
                    ax.text(0.02, 0.98, f'r={corr:.2f}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top')

                ax.grid(True, alpha=0.3)

        plt.suptitle('Metrics Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(summary_viz_dir / "metrics_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n✅ All summary visualizations saved to: {summary_viz_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("V2 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Supervised visualizations: {supervised_viz_dir}")
    print(f"Summary visualizations: {summary_viz_dir}")
    print(f"Metrics: {metrics_dir}")

    if not df_results.empty:
        print("\nTop configurations by accuracy:")
        print(df_results[['config_name', 'accuracy', 'f1_weighted', 'purity']].head())

        print("\nKey Statistics:")
        print(f"  Best accuracy: {df_results['accuracy'].max():.4f}")
        print(f"  Best F1 score: {df_results['f1_weighted'].max():.4f}")
        print(f"  Max data reduction: {df_results['data_reduction_pct'].max():.1f}%")
        print(f"  Configurations tested: {len(df_results)}")


if __name__ == "__main__":
    import sys

    # Check for command-line arguments
    if len(sys.argv) > 1:
        try:
            max_configs = int(sys.argv[1])
            print(f"Running with max_configs={max_configs}")
            main(max_configs=max_configs)
        except ValueError:
            print("Usage: python wavelengthselectionV2.py [max_configs]")
            print("  max_configs: Maximum number of configurations to run (optional)")
            sys.exit(1)
    else:
        # Run all configurations by default
        main()