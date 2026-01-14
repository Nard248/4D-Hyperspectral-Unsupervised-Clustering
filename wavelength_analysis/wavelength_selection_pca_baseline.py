"""
Wavelength Selection - PCA Baseline Comparison
===============================================
This script uses PCA-based wavelength selection instead of autoencoder+perturbation
to provide a baseline comparison. It generates the same outputs (Excel, metrics,
visualizations) as WavelengthSelectionV2-2.py for direct comparison.

Key Difference:
- ORIGINAL: Raw Data → Autoencoder → Perturbation → MMR Selection → Classification
- THIS SCRIPT: Raw Data → PCA Selection → Classification

All other components (data loading, clustering, metrics, visualizations) are identical.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from scipy.ndimage import label

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
results_dir = base_dir / "wavelength_analysis" / "pca_baseline_results" / timestamp
results_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories (same structure as V2)
visualizations_dir = results_dir / "visualizations"
visualizations_dir.mkdir(exist_ok=True)
supervised_viz_dir = results_dir / "supervised_visualizations"
supervised_viz_dir.mkdir(exist_ok=True)
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
print("WAVELENGTH SELECTION - PCA BASELINE COMPARISON")
print("=" * 80)
print(f"  Working directory: {base_dir}")
print(f"  Results directory: {results_dir}")
print(f"  Method: PCA-based selection (no autoencoder)")
print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Import required modules from original pipeline
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

# Import enhanced modules
from performance_timing_tracker import PerformanceTimer
from roi_visualization import create_roi_overlay_visualization, create_roi_analysis_report
from metrics_export import export_experiment_metrics, export_all_experiments_summary, export_experiment_csv

# Import V2 modules for ground truth tracking and supervised metrics
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics
from supervised_visualizations import SupervisedVisualizations

print("\nAll modules imported successfully (PCA Baseline Version)")


# ============================================================================
# PCA-BASED WAVELENGTH SELECTION FUNCTIONS
# ============================================================================

def select_wavelengths_pca(data_path, mask_path, n_bands_to_select, method='pca_loadings', verbose=True):
    """
    Select wavelength combinations using PCA instead of autoencoder.

    Args:
        data_path: Path to hyperspectral data
        mask_path: Path to mask
        n_bands_to_select: Number of wavelength combinations to select
        method: Selection method ('pca_loadings', 'variance', 'greedy_diversity')
        verbose: Print progress

    Returns:
        wavelength_combinations: List of selected excitation-emission pairs
        emission_wavelengths_only: List of emission wavelengths
        results: Dictionary with selection metadata
    """
    if verbose:
        print(f"\nRunning PCA-based wavelength selection (method: {method})")
        print(f"  Target bands: {n_bands_to_select}")

    # Load data
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)

    # Flatten hyperspectral data to 2D matrix (pixels × wavelength_combinations)
    # First pass: create mask from first excitation (identify valid/non-NaN pixels)
    first_ex = full_data['excitation_wavelengths'][0]
    first_cube = full_data['data'][str(first_ex)]['cube']

    # Valid pixels are those without NaN in the first emission
    first_slice = first_cube[:, :, 0]
    valid_mask_2d = ~np.isnan(first_slice)
    valid_mask_flat = valid_mask_2d.flatten()

    all_spectra = []
    wavelength_info = []  # Track which excitation-emission pair each column represents

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]
        cube = ex_data['cube']  # Shape: (height, width, n_emissions)
        wavelengths = ex_data['wavelengths']

        # Flatten spatial dimensions
        n_pixels = cube.shape[0] * cube.shape[1]
        n_emissions = cube.shape[2]

        flattened = cube.reshape(n_pixels, n_emissions)  # (pixels, emissions)

        # Filter out background pixels
        flattened_valid = flattened[valid_mask_flat, :]  # Only valid pixels

        # Add to collection
        all_spectra.append(flattened_valid)

        # Track wavelength info
        for em_idx, em_wavelength in enumerate(wavelengths):
            wavelength_info.append({
                'excitation': float(ex),
                'emission': float(em_wavelength),
                'exc_idx': len(wavelength_info) // len(wavelengths) if len(wavelength_info) > 0 else 0,
                'em_idx': em_idx
            })

    # Concatenate all excitations
    X = np.hstack(all_spectra)  # Shape: (n_valid_pixels, total_bands)

    if verbose:
        print(f"  Data matrix shape: {X.shape}")
        print(f"  Valid pixels: {np.sum(valid_mask_flat)} / {len(valid_mask_flat)}")
        print(f"  Total wavelength combinations: {len(wavelength_info)}")

    # Apply selection method
    if method == 'pca_loadings':
        selected_indices = select_by_pca_loadings(X, n_bands_to_select)
    elif method == 'variance':
        selected_indices = select_by_variance(X, n_bands_to_select)
    elif method == 'greedy_diversity':
        selected_indices = select_by_greedy_diversity(X, n_bands_to_select)
    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Convert indices to wavelength combinations
    wavelength_combinations = []
    emission_wavelengths_only = []

    for idx in selected_indices:
        info = wavelength_info[idx]
        combination = {
            'excitation': info['excitation'],
            'emission': info['emission'],
            'combination_name': f"Ex{info['excitation']:.0f}_Em{info['emission']:.1f}"
        }
        wavelength_combinations.append(combination)
        emission_wavelengths_only.append(info['emission'])

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

    results = {
        'method': method,
        'n_requested': n_bands_to_select,
        'n_selected': len(unique_combinations),
        'selected_indices': selected_indices
    }

    if verbose:
        print(f"  Selected {len(unique_combinations)} unique wavelength combinations")
        if unique_combinations:
            print(f"  First few: {[c['combination_name'] for c in unique_combinations[:3]]}...")

    return unique_combinations, unique_emissions, results


def select_by_pca_loadings(X, n_bands):
    """Select bands based on PCA component loadings."""
    # Standardize data
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Apply PCA
    n_components = min(n_bands, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X_std)

    # Get absolute loadings for each component
    loadings = np.abs(pca.components_)  # Shape: (n_components, n_features)

    # For each component, find feature with maximum loading
    selected_indices = set()

    for component_idx in range(n_components):
        max_idx = np.argmax(loadings[component_idx])
        selected_indices.add(max_idx)

        # If we don't have enough unique bands, get top k from each component
        if len(selected_indices) < n_bands:
            top_k = min(5, X.shape[1])  # Get top 5 from each component
            top_indices = np.argsort(loadings[component_idx])[-top_k:]
            selected_indices.update(top_indices)

    # Convert to sorted list
    selected_indices = sorted(list(selected_indices))[:n_bands]

    return selected_indices


def select_by_variance(X, n_bands):
    """Select bands with highest variance (simplest method)."""
    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[-n_bands:]
    return sorted(top_indices.tolist())


def select_by_greedy_diversity(X, n_bands, lambda_div=0.5):
    """
    Select bands greedily: balance high variance with low correlation to already selected.
    This mimics MMR logic but without autoencoder.
    """
    # Calculate variances
    variances = X.var(axis=0)

    # Normalize variances to [0, 1]
    var_norm = (variances - variances.min()) / (variances.max() - variances.min() + 1e-8)

    # Standardize for correlation calculation
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    selected_indices = []

    # Start with band with highest variance
    first_idx = np.argmax(variances)
    selected_indices.append(first_idx)

    # Greedily add remaining bands
    for _ in range(n_bands - 1):
        best_score = -np.inf
        best_idx = None

        for candidate_idx in range(X.shape[1]):
            if candidate_idx in selected_indices:
                continue

            # Relevance: variance
            relevance = var_norm[candidate_idx]

            # Diversity: average correlation to already selected
            correlations = []
            for selected_idx in selected_indices:
                corr = np.corrcoef(X_std[:, candidate_idx], X_std[:, selected_idx])[0, 1]
                correlations.append(abs(corr))

            avg_correlation = np.mean(correlations) if correlations else 0
            diversity = 1 - avg_correlation

            # MMR-style score
            score = lambda_div * relevance + (1 - lambda_div) * diversity

            if score > best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is not None:
            selected_indices.append(best_idx)

    return sorted(selected_indices)


def extract_wavelength_subset(full_data, wavelength_combinations, verbose=False):
    """
    Extract subset of data using selected excitation-emission wavelength combinations.
    (Same as V2-2 version)
    """
    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': [],
        'selected_combinations': wavelength_combinations
    }

    total_bands_original = 0
    total_bands_selected = 0

    # Group combinations by excitation wavelength
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
        selected_wavelengths = []

        for target_em in matching_emissions:
            # Find closest emission wavelength
            diffs = np.abs(original_wavelengths - target_em)
            closest_idx = np.argmin(diffs)

            if diffs[closest_idx] < 2.0:  # 2nm tolerance
                selected_indices.append(closest_idx)
                selected_wavelengths.append(original_wavelengths[closest_idx])

        if len(selected_indices) == 0:
            continue

        # Extract selected bands
        selected_cube = original_cube[:, :, selected_indices]

        # Store in subset
        subset_data['excitation_wavelengths'].append(ex)
        subset_data['data'][ex_str] = {
            'wavelengths': selected_wavelengths,
            'cube': selected_cube,
            'original_wavelengths': original_wavelengths.tolist(),
            'selected_indices': selected_indices
        }

        total_bands_selected += len(selected_indices)

    if verbose:
        reduction_pct = (1 - total_bands_selected / total_bands_original) * 100 if total_bands_original > 0 else 0
        print(f"  Data reduction: {total_bands_original} -> {total_bands_selected} bands ({reduction_pct:.1f}% reduction)")
        print(f"  Selected {len(wavelength_combinations)} specific excitation-emission pairs")
        print(f"  Excitations with selected bands: {len(subset_data['excitation_wavelengths'])}")

    return subset_data


# ============================================================================
# CLUSTERING PIPELINE (Same as V2-2)
# ============================================================================

# Define ROI regions (same as V2-2)
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
    return colors_list


def run_knn_clustering_pipeline_v2(data, n_clusters, roi_regions=None,
                                  ground_truth_tracker=None, random_state=42,
                                  export_concat_data=False, config_name=None):
    """
    Complete KNN clustering pipeline with ground truth tracking (V2).
    (Same as V2-2 version - imports from existing modules)
    """
    if roi_regions is None:
        roi_regions = ROI_REGIONS

    # Step 1: Concatenate data
    df, valid_mask, metadata = concatenate_hyperspectral_data_improved(
        data,
        global_normalize=True,
        normalization_method='global_percentile'
    )

    # Export concatenated data if requested
    if export_concat_data and config_name:
        concat_csv_path = concat_data_dir / f"{config_name}_concatenated_data.csv"
        df.to_csv(concat_csv_path, index=False)

    # Step 2: Run KNN clustering - pass DataFrame directly
    cluster_labels, metrics = perform_clustering(
        df,  # Pass DataFrame, not numpy array
        n_clusters=n_clusters,
        method='kmeans',
        random_state=random_state
    )

    # Get number of spectral features
    spectral_cols = [col for col in df.columns if col not in ['x', 'y', 'row', 'col']]
    n_features = len(spectral_cols)

    # Step 3: Reconstruct cluster map
    # Pass the DataFrame directly (reconstruct_cluster_map expects DataFrame with x, y columns)
    cluster_map = reconstruct_cluster_map(
        cluster_labels,
        df,  # Pass full DataFrame - function will extract x, y columns
        valid_mask,
        metadata
    )

    # Step 4: Supervised metrics if ground truth available
    supervised_metrics = None
    if ground_truth_tracker is not None:
        # SupervisedMetrics.calculate_metrics expects 2D predictions
        # (ground truth is already in the tracker)
        sm = SupervisedMetrics(ground_truth_tracker)
        supervised_metrics = sm.calculate_metrics(
            predictions=cluster_map,
            use_hungarian_mapping=True
        )

    return cluster_map, metrics, n_features, supervised_metrics


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

def main(selection_method='pca_loadings', max_configs=None):
    """
    Main execution function for PCA baseline pipeline.

    Args:
        selection_method: PCA method to use ('pca_loadings', 'variance', 'greedy_diversity')
        max_configs: Maximum number of configurations to run (None = run all)
    """

    print("\n" + "=" * 80)
    print(f"PCA BASELINE - SELECTION METHOD: {selection_method.upper()}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("LOADING DATA AND GROUND TRUTH")
    print("=" * 80)

    # Define paths (same as V2-2)
    sample_name = "Lichens"
    data_path = base_dir / "data" / "processed" / sample_name / "lichens_data_masked.pkl"
    mask_path = base_dir / "data" / "processed" / sample_name / "lichens_mask.npy"
    png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")

    print("Loading data...")
    print(f"  Sample: {sample_name}")
    print(f"  Data: {data_path.name}")
    print(f"  Ground truth: {png_path.name}")
    print(f"  Selection method: {selection_method}")

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
        (24, 24, 24, 255),
        (168, 168, 168, 255)
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

    # Apply cropping (same as V2-2)
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

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        original_cube = full_data['data'][ex_str]['cube']
        cropped_cube = original_cube[:, start_col:end_col, :]
        cropped_data['data'][ex_str] = {
            **full_data['data'][ex_str],
            'cube': cropped_cube
        }

    ground_truth_cropped = ground_truth[:, start_col:end_col]

    # Update working datasets
    full_data = cropped_data
    ground_truth = ground_truth_cropped

    print(f"Data successfully cropped")

    # Save cropped data
    cropped_data_dir = base_dir / "data" / "processed" / sample_name / "temp_cropped_pca"
    cropped_data_dir.mkdir(parents=True, exist_ok=True)

    cropped_data_path = cropped_data_dir / "lichens_data_cropped_pca.pkl"
    with open(cropped_data_path, 'wb') as f:
        pickle.dump(full_data, f)

    cropped_mask_path = cropped_data_dir / "lichens_mask_cropped_pca.npy"
    cropped_mask = np.ones(ground_truth.shape, dtype=bool)
    cropped_mask[ground_truth == -1] = False
    np.save(cropped_mask_path, cropped_mask)

    data_path = cropped_data_path
    mask_path = cropped_mask_path

    # Initialize Ground Truth Tracker
    print("\n" + "=" * 80)
    print("INITIALIZING GROUND TRUTH TRACKER")
    print("=" * 80)

    class_names = [f"Lichen_Type_{i}" for i in range(n_true_classes)]
    gt_tracker = GroundTruthTracker(ground_truth, class_names)

    tracker_state_file = metrics_dir / "ground_truth_tracker_state.pkl"
    gt_tracker.export_state(tracker_state_file)

    class_distribution = gt_tracker.get_class_distribution()
    print("\nClass Distribution:")
    for cls_id, info in class_distribution.items():
        if cls_id >= 0:
            print(f"  {info['name']}: {info['pixel_count']:,} pixels ({info['percentage']:.2f}%)")

    supervised_viz = SupervisedVisualizations(output_dir=supervised_viz_dir, dpi=300)

    # ========================================================================
    # BASELINE: Full Data Clustering
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: Clustering with Full Data (PCA Baseline)")
    print("=" * 80)
    n_clusters = len(ROI_REGIONS)

    roi_colormap = create_roi_colormap(ROI_REGIONS)
    cluster_maps = {}

    with PerformanceTimer() as baseline_timer:
        cluster_map_full, metrics_full, n_features_full, supervised_metrics_full = run_clustering_pipeline_v2(
            full_data,
            n_clusters,
            ground_truth_tracker=gt_tracker,
            export_concat_data=True,
            config_name="BASELINE_FULL_DATA"
        )

    cluster_maps['baseline'] = cluster_map_full

    print(f"\n[BASELINE RESULTS]")
    print(f"  Features: {n_features_full}")
    print(f"  Time: {baseline_timer.elapsed:.2f}s")
    print(f"  Purity: {metrics_full.get('purity', 0):.4f}")
    print(f"  ARI: {metrics_full.get('ari', 0):.4f}")
    print(f"  NMI: {metrics_full.get('nmi', 0):.4f}")

    if supervised_metrics_full:
        print(f"\n[SUPERVISED METRICS - Baseline]")
        print(f"  Accuracy: {supervised_metrics_full['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {supervised_metrics_full['balanced_accuracy']:.4f}")
        print(f"  Precision (weighted): {supervised_metrics_full['precision_weighted']:.4f}")
        print(f"  Recall (weighted): {supervised_metrics_full['recall_weighted']:.4f}")
        print(f"  F1 (weighted): {supervised_metrics_full['f1_weighted']:.4f}")
        print(f"  Cohen's Kappa: {supervised_metrics_full['cohen_kappa']:.4f}")

        # Create supervised visualizations for baseline
        try:
            supervised_viz.create_all_visualizations(
                supervised_metrics_full,
                ground_truth,
                cluster_map_full,
                supervised_metrics_full.get('roi_metrics'),
                roi_regions=ROI_REGIONS
            )
        except Exception as e:
            print(f"  Warning: Some visualizations failed: {e}")

        # Create paper visualizations
        try:
            paper_viz = SupervisedVisualizations(output_dir=paper_results_dir, dpi=300)
            paper_viz.plot_roi_overlay_with_accuracy(
                cluster_map=cluster_map_full,
                ground_truth=ground_truth,
                roi_regions=ROI_REGIONS,
                overall_accuracy=supervised_metrics_full['accuracy'],
                roi_metrics=supervised_metrics_full.get('roi_metrics'),
                title="BASELINE - Full Data (PCA Baseline)",
                save_name="BASELINE_roi_overlay.png"
            )

            paper_viz.plot_simple_classification(
                cluster_map=cluster_map_full,
                roi_regions=ROI_REGIONS,
                title="BASELINE - Classification",
                save_name="BASELINE_classification.png"
            )
        except Exception as e:
            print(f"  Warning: Paper visualizations failed: {e}")

        # Export baseline metrics
        baseline_metrics_file = metrics_dir / "baseline_supervised_metrics.json"
        sm = SupervisedMetrics(gt_tracker)
        sm.current_metrics = supervised_metrics_full
        sm.export_metrics(baseline_metrics_file, format='json')

    # Object-level analysis for baseline
    labeled_objects, num_objects = label(ground_truth >= 0)

    baseline_object_metrics = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled_objects == obj_id)
        obj_pixels = np.sum(obj_mask)

        if obj_pixels > 0:
            obj_gt = ground_truth[obj_mask]
            obj_pred = cluster_map_full[obj_mask]

            valid_indices = (obj_gt >= 0)
            if valid_indices.sum() > 0:
                obj_accuracy = accuracy_score(obj_gt[valid_indices], obj_pred[valid_indices])
                true_class = int(np.bincount(obj_gt[valid_indices].astype(int)).argmax())

                baseline_object_metrics.append({
                    'object_id': obj_id,
                    'num_pixels': int(obj_pixels),
                    'true_class': true_class,
                    'accuracy': float(obj_accuracy)
                })

    baseline_object_df = pd.DataFrame(baseline_object_metrics)

    # Save baseline experiment folder
    baseline_experiment_folder = experiments_dir / "BASELINE_FULL_DATA"
    baseline_experiment_folder.mkdir(exist_ok=True)

    baseline_object_csv = baseline_experiment_folder / "BASELINE_object_metrics.csv"
    baseline_object_df.to_csv(baseline_object_csv, index=False)

    # Create baseline visualizations
    try:
        baseline_viz = SupervisedVisualizations(output_dir=baseline_experiment_folder, dpi=300)

        baseline_viz.plot_enumerated_objects(
            ground_truth,
            labeled_objects,
            num_objects,
            title="Ground Truth with Enumerated Objects",
            save_name="ground_truth_enumerated_objects.png"
        )

        baseline_viz.plot_roi_overlay_with_object_accuracy(
            cluster_map_full,
            ground_truth,
            ROI_REGIONS,
            labeled_objects,
            baseline_object_metrics,
            supervised_metrics_full['accuracy'] if supervised_metrics_full else 0,
            title="BASELINE - ROI Overlay with Object Accuracy",
            save_name="BASELINE_roi_overlay_object_accuracy.png"
        )

        baseline_supervised_viz_dir = baseline_experiment_folder / "supervised_visualizations"
        baseline_supervised_viz_dir.mkdir(exist_ok=True)
        baseline_supervised_viz = SupervisedVisualizations(output_dir=baseline_supervised_viz_dir, dpi=300)
        baseline_supervised_viz.create_all_visualizations(
            supervised_metrics_full,
            ground_truth,
            cluster_map_full,
            supervised_metrics_full.get('roi_metrics') if supervised_metrics_full else None,
            roi_regions=ROI_REGIONS
        )

        baseline_viz.plot_simple_classification(
            cluster_map=cluster_map_full,
            roi_regions=ROI_REGIONS,
            title="BASELINE - Classification",
            save_name="BASELINE_classification.png"
        )

        baseline_viz.plot_roi_overlay_with_accuracy(
            cluster_map=cluster_map_full,
            ground_truth=ground_truth,
            roi_regions=ROI_REGIONS,
            overall_accuracy=supervised_metrics_full['accuracy'] if supervised_metrics_full else 0,
            roi_metrics=supervised_metrics_full.get('roi_metrics') if supervised_metrics_full else None,
            title="BASELINE - ROI Overlay",
            save_name="BASELINE_roi_overlay_main.png"
        )

        print(f"[SUCCESS] Baseline visualizations saved")
    except Exception as e:
        print(f"  Warning: Baseline visualizations failed: {e}")

    # ========================================================================
    # Run PCA-based Selection Configurations
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"RUNNING PCA-BASED WAVELENGTH SELECTION EXPERIMENTS")
    print(f"Method: {selection_method}")
    print("=" * 80)

    # Define configurations (similar band counts as V2-2)
    band_counts_to_test = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 50, 60, 70, 80, 90,
                           100, 110, 120, 130, 140, 150, 160, 170]

    configs_to_run = []
    for n_bands in band_counts_to_test:
        config_name = f"{selection_method}_{n_bands}bands"
        configs_to_run.append({
            'name': config_name,
            'n_bands': n_bands,
            'method': selection_method
        })

    if max_configs:
        configs_to_run = configs_to_run[:max_configs]

    print(f"\nTotal configurations to test: {len(configs_to_run)}")

    # Storage for results
    all_results = []
    all_object_metrics = []

    # Run each configuration
    for i, config in enumerate(tqdm(configs_to_run, desc="Running configurations")):
        config_name = config['name']
        n_bands = config['n_bands']

        print(f"\n{'='*80}")
        print(f"Configuration {i+1}/{len(configs_to_run)}: {config_name}")
        print(f"{'='*80}")

        try:
            # Select wavelengths using PCA
            with PerformanceTimer() as selection_timer:
                wavelength_combinations, emission_wavelengths, selection_results = select_wavelengths_pca(
                    data_path,
                    mask_path,
                    n_bands_to_select=n_bands,
                    method=selection_method,
                    verbose=True
                )

            # Extract subset
            subset_data = extract_wavelength_subset(full_data, wavelength_combinations, verbose=True)

            # Run clustering
            with PerformanceTimer() as clustering_timer:
                cluster_map, metrics, n_features, supervised_metrics = run_clustering_pipeline_v2(
                    subset_data,
                    n_clusters,
                    ground_truth_tracker=gt_tracker,
                    export_concat_data=True,
                    config_name=config_name
                )

            cluster_maps[config_name] = cluster_map

            # Calculate speedup
            speedup_factor = baseline_timer.elapsed / clustering_timer.elapsed if clustering_timer.elapsed > 0 else 1.0

            # Calculate data reduction
            data_reduction_pct = (1 - n_features / n_features_full) * 100 if n_features_full > 0 else 0

            # Store results
            result_row = {
                'config_name': config_name,
                'n_combinations_selected': len(wavelength_combinations),
                'n_features': n_features,
                'data_reduction_pct': data_reduction_pct,
                'purity': metrics.get('purity', 0),
                'ari': metrics.get('ari', 0),
                'nmi': metrics.get('nmi', 0),
                'selection_time': selection_timer.elapsed,
                'clustering_time': clustering_timer.elapsed,
                'speedup_factor': speedup_factor
            }

            # Add supervised metrics
            if supervised_metrics:
                result_row.update({
                    'accuracy': supervised_metrics['accuracy'],
                    'balanced_accuracy': supervised_metrics['balanced_accuracy'],
                    'precision_weighted': supervised_metrics['precision_weighted'],
                    'recall_weighted': supervised_metrics['recall_weighted'],
                    'f1_weighted': supervised_metrics['f1_weighted'],
                    'cohen_kappa': supervised_metrics['cohen_kappa']
                })

                print(f"\n[RESULTS - {config_name}]")
                print(f"  Accuracy: {supervised_metrics['accuracy']:.4f}")
                print(f"  F1 (weighted): {supervised_metrics['f1_weighted']:.4f}")
                print(f"  Data reduction: {data_reduction_pct:.1f}%")
                print(f"  Speedup: {speedup_factor:.2f}x")

            all_results.append(result_row)

            # Object-level metrics
            object_metrics = []
            for obj_id in range(1, num_objects + 1):
                obj_mask = (labeled_objects == obj_id)
                obj_pixels = np.sum(obj_mask)

                if obj_pixels > 0:
                    obj_gt = ground_truth[obj_mask]
                    obj_pred = cluster_map[obj_mask]

                    valid_indices = (obj_gt >= 0)
                    if valid_indices.sum() > 0:
                        obj_accuracy = accuracy_score(obj_gt[valid_indices], obj_pred[valid_indices])
                        true_class = int(np.bincount(obj_gt[valid_indices].astype(int)).argmax())

                        obj_metric = {
                            'object_id': obj_id,
                            'num_pixels': int(obj_pixels),
                            'true_class': true_class,
                            'accuracy': float(obj_accuracy),
                            'config_name': config_name,
                            'n_combinations_selected': len(wavelength_combinations),
                            'n_features': n_features,
                            'run_index': i
                        }
                        object_metrics.append(obj_metric)
                        all_object_metrics.append(obj_metric)

            # Save experiment outputs
            experiment_folder = experiments_dir / config_name
            experiment_folder.mkdir(exist_ok=True)

            # Save object metrics
            object_df = pd.DataFrame(object_metrics)
            object_csv = experiment_folder / f"{config_name}_object_metrics.csv"
            object_df.to_csv(object_csv, index=False)

            # Create visualizations
            if supervised_metrics:
                config_viz_dir = experiment_folder / "supervised_visualizations"
                config_viz_dir.mkdir(exist_ok=True)
                config_viz = SupervisedVisualizations(output_dir=config_viz_dir, dpi=300)
                config_viz.create_all_visualizations(
                    supervised_metrics,
                    ground_truth,
                    cluster_map,
                    supervised_metrics.get('roi_metrics'),
                    roi_regions=ROI_REGIONS
                )

                config_viz.plot_roi_overlay_with_accuracy(
                    cluster_map=cluster_map,
                    ground_truth=ground_truth,
                    roi_regions=ROI_REGIONS,
                    overall_accuracy=supervised_metrics['accuracy'],
                    roi_metrics=supervised_metrics.get('roi_metrics'),
                    title=f"{config_name} - ROI Overlay",
                    save_name=f"{config_name}_roi_overlay_main.png"
                )

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

                paper_viz.plot_simple_classification(
                    cluster_map=cluster_map,
                    roi_regions=ROI_REGIONS,
                    title=f"{config_name} - Classification",
                    save_name=f"{config_name}_classification.png"
                )

                # Export metrics
                config_metrics_file = experiment_folder / f"{config_name}_supervised_metrics.json"
                sm = SupervisedMetrics(gt_tracker)
                sm.current_metrics = supervised_metrics
                sm.export_metrics(config_metrics_file, format='json')

                # Additional visualizations
                config_viz.plot_simple_classification(
                    cluster_map=cluster_map,
                    roi_regions=ROI_REGIONS,
                    title=f"{config_name} - Classification",
                    save_name=f"{config_name}_classification.png"
                )

                config_viz.plot_roi_overlay_with_object_accuracy(
                    cluster_map,
                    ground_truth,
                    ROI_REGIONS,
                    labeled_objects,
                    object_metrics,
                    supervised_metrics['accuracy'],
                    title=f"{config_name} - ROI with Object Accuracy",
                    save_name=f"{config_name}_roi_overlay_object_accuracy.png"
                )

                config_viz.plot_enumerated_objects(
                    ground_truth,
                    labeled_objects,
                    num_objects,
                    title="Ground Truth with Enumerated Objects",
                    save_name="ground_truth_enumerated_objects.png"
                )

            print(f"[SUCCESS] {config_name} complete")

        except Exception as e:
            print(f"[ERROR] Configuration {config_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # Generate Summary Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(all_results)

    # Add baseline row
    baseline_row = {
        'config_name': 'BASELINE_FULL_DATA',
        'n_combinations_selected': total_bands,
        'n_features': n_features_full,
        'data_reduction_pct': 0,
        'purity': metrics_full.get('purity', 0),
        'ari': metrics_full.get('ari', 0),
        'nmi': metrics_full.get('nmi', 0),
        'accuracy': supervised_metrics_full['accuracy'] if supervised_metrics_full else 0,
        'balanced_accuracy': supervised_metrics_full['balanced_accuracy'] if supervised_metrics_full else 0,
        'precision_weighted': supervised_metrics_full['precision_weighted'] if supervised_metrics_full else 0,
        'recall_weighted': supervised_metrics_full['recall_weighted'] if supervised_metrics_full else 0,
        'f1_weighted': supervised_metrics_full['f1_weighted'] if supervised_metrics_full else 0,
        'cohen_kappa': supervised_metrics_full['cohen_kappa'] if supervised_metrics_full else 0,
        'selection_time': 0,
        'clustering_time': baseline_timer.elapsed,
        'speedup_factor': 1.0
    }

    df_results = pd.concat([pd.DataFrame([baseline_row]), df_results], ignore_index=True)

    # Sort by accuracy
    df_results = df_results.sort_values('accuracy', ascending=False)

    # Save to Excel
    excel_path = results_dir / "wavelength_selection_results_pca_baseline.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name=f'{selection_method}_results', index=False)

    print(f"Results saved to: {excel_path}")

    # Save object metrics
    if all_object_metrics:
        all_obj_df = pd.DataFrame(all_object_metrics)
        all_obj_csv = analysis_summary_dir / "all_object_metrics_across_configs.csv"
        all_obj_df.to_csv(all_obj_csv, index=False)

        per_config_summary = all_obj_df.groupby('config_name').agg(
            mean_object_accuracy=('accuracy', 'mean'),
            std_object_accuracy=('accuracy', 'std'),
            n_objects=('object_id', 'count')
        ).reset_index()
        per_config_csv = analysis_summary_dir / "per_config_object_metrics_summary.csv"
        per_config_summary.to_csv(per_config_csv, index=False)

        print(f"\nObject metrics saved to: {analysis_summary_dir}")

    # Print summary
    print("\n" + "=" * 80)
    print("PCA BASELINE PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Selection method: {selection_method}")
    print(f"Results directory: {results_dir}")
    print(f"Excel file: {excel_path}")

    if not df_results.empty:
        print("\nTop configurations by accuracy:")
        print(df_results[['config_name', 'accuracy', 'f1_weighted', 'n_features']].head())

        print("\nKey Statistics:")
        print(f"  Best accuracy: {df_results['accuracy'].max():.4f}")
        print(f"  Best F1 score: {df_results['f1_weighted'].max():.4f}")
        print(f"  Max data reduction: {df_results['data_reduction_pct'].max():.1f}%")
        print(f"  Configurations tested: {len(df_results)}")


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    selection_method = 'pca_loadings'  # Default
    max_configs = 10

    if len(sys.argv) > 1:
        selection_method = sys.argv[1]
        if selection_method not in ['pca_loadings', 'variance', 'greedy_diversity']:
            print("Invalid selection method. Choose from: pca_loadings, variance, greedy_diversity")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            max_configs = int(sys.argv[2])
        except ValueError:
            print("Invalid max_configs value")
            sys.exit(1)

    print(f"Running PCA baseline with method: {selection_method}")
    if max_configs:
        print(f"Limited to {max_configs} configurations")

    main(selection_method=selection_method, max_configs=max_configs)
