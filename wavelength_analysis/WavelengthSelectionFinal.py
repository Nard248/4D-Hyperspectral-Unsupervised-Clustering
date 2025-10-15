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
results_dir = base_dir / "wavelength_analysis" / "validation_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir.mkdir(parents=True, exist_ok=True)
visualizations_dir = results_dir / "visualizations"
visualizations_dir.mkdir(exist_ok=True)

# NEW: Create paper-results, concat-data, and experiments directories
paper_results_dir = results_dir / "paper-results"
paper_results_dir.mkdir(exist_ok=True)
concat_data_dir = results_dir / "concat-data"
concat_data_dir.mkdir(exist_ok=True)
experiments_dir = results_dir / "experiments"
experiments_dir.mkdir(exist_ok=True)
analysis_summary_dir = results_dir / "analysis_summary"
analysis_summary_dir.mkdir(exist_ok=True)

print("Environment setup completed")
print(f"  Working directory: {base_dir}")
print(f"  Results directory: {results_dir}")
print(f"  Paper results directory: {paper_results_dir}")
print(f"  Concatenated data directory: {concat_data_dir}")
print(f"  Experiments directory: {experiments_dir}")
print(f"  Analysis summary directory: {analysis_summary_dir}")
print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
# %%
# Import required modules
from wavelength_analysis.core.config import AnalysisConfig
from wavelength_analysis.core.analyzer import WavelengthAnalyzer
from concatenation_clustering import (
    load_masked_data,
    concatenate_hyperspectral_data_improved,  # Use improved version
    perform_clustering,
    reconstruct_cluster_map
)
from ground_truth_validation import (
    extract_ground_truth_from_png,
    calculate_clustering_accuracy
)

# Import KNN-related libraries for the new pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NEW: Import difference map and speed tracking modules
from enhanced_difference_visualization import create_enhanced_difference_map, create_simple_difference_overlay
from performance_timing_tracker import TimingTracker, PerformanceTimer

# NEW: Import ROI visualization and metrics export modules
from roi_visualization import create_roi_overlay_visualization, create_roi_analysis_report
from metrics_export import export_experiment_metrics, export_all_experiments_summary, export_experiment_csv
from paper_visualizations import create_ground_truth_difference_maps

print("All modules imported successfully (with KNN support + Enhanced visualization + ROI overlays + Metrics export)")

def select_informative_wavelengths_fixed(data_path, mask_path, sample_name, config_params, verbose=True):
    """
    FIXED VERSION: Returns actual wavelength combinations with excitation-emission pairs!

    Args:
        data_path: Path to data file
        mask_path: Path to mask file
        sample_name: Name of sample
        config_params: Configuration parameters
        verbose: Print progress
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
        # NEW: Diversity constraint parameters
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

    # FIXED: Extract excitation-emission wavelength combinations
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


def extract_wavelength_subset(full_data, emission_wavelengths_only, verbose=False):
    """
    Extract subset of data using selected emission wavelengths.
    Uses emission_wavelengths_only for backward compatibility.
    """
    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': full_data['excitation_wavelengths'],
        'selected_wavelengths': emission_wavelengths_only
    }

    total_bands_original = 0
    total_bands_selected = 0

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]

        original_wavelengths = np.array(ex_data['wavelengths'])
        original_cube = ex_data['cube']

        total_bands_original += len(original_wavelengths)

        # Find indices of selected wavelengths for this excitation
        selected_indices = []
        selected_wl_values = []

        for target_wl in emission_wavelengths_only:
            # Ensure target_wl is a number
            target_wl = float(target_wl)

            # Find closest wavelength
            distances = np.abs(original_wavelengths - target_wl)
            closest_idx = np.argmin(distances)

            # Only include if reasonably close (within 10 nm) and not duplicate
            if distances[closest_idx] < 10 and closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
                selected_wl_values.append(original_wavelengths[closest_idx])

        if selected_indices:
            subset_cube = original_cube[:, :, selected_indices]
            total_bands_selected += len(selected_indices)

            subset_data['data'][ex_str] = {
                'cube': subset_cube,
                'wavelengths': selected_wl_values,
                **{k: v for k, v in ex_data.items() if k not in ['cube', 'wavelengths']}
            }
        else:
            # Keep all bands if none selected
            subset_data['data'][ex_str] = ex_data
            total_bands_selected += len(original_wavelengths)

    if verbose:
        reduction_pct = (1 - total_bands_selected / total_bands_original) * 100 if total_bands_original > 0 else 0
        print(
            f"  Data reduction: {total_bands_original} → {total_bands_selected} bands ({reduction_pct:.1f}% reduction)")

    return subset_data


# Define ROI regions with specific colors for consistency
ROI_REGIONS = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},  # Red
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},  # Blue
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},  # Green
    {'name': 'Region 4', 'coords': (185, 225, 675, 700), 'color': '#FFFF00'},  # Yellow
]


def create_roi_colormap(roi_regions):
    """
    Create a custom colormap based on ROI region colors.
    Maps cluster IDs directly to ROI colors.
    """
    # Create colors list matching cluster IDs (0-3 for 4 regions)
    colors_list = []
    for roi in roi_regions:
        colors_list.append(roi['color'])

    # Add white for background/unassigned (-1) if needed
    # We'll handle -1 separately in visualization by masking

    return mcolors.ListedColormap(colors_list)


def run_knn_clustering_pipeline(data, n_clusters, roi_regions=None, random_state=42, export_concat_data=False,
                                config_name=None):
    """
    NEW: Run complete KNN-based clustering pipeline using improved data processing.
    This replaces the old KMeans-based approach with KNN classification.

    Args:
        data: Hyperspectral data dictionary
        n_clusters: Number of clusters (used for evaluation metrics)
        roi_regions: ROI regions for training (if None, uses default regions)
        random_state: Random seed for reproducibility
        export_concat_data: If True, exports the concatenated data to CSV
        config_name: Name of the configuration (for file naming)

    Returns:
        cluster_map: 2D array with cluster assignments
        metrics: Dictionary with clustering metrics
        n_features: Number of spectral features used
    """
    if roi_regions is None:
        roi_regions = ROI_REGIONS

    # Step 1: Use improved data concatenation (same as clustering_experiments.ipynb)
    df, valid_mask, metadata = concatenate_hyperspectral_data_improved(
        data,
        global_normalize=True,
        normalization_method='global_percentile'
    )

    # NEW: Export concatenated data if requested
    if export_concat_data and config_name:
        concat_filename = concat_data_dir / f"{config_name}_concatenated_data.csv"
        df.to_csv(concat_filename, index=False)
        print(f"    Exported concatenated data to: {concat_filename.name}")

    # Step 2: Extract ROI training data
    spectral_cols = [col for col in df.columns if col not in ['x', 'y']]
    X_train_list = []
    y_train_list = []

    # Create label mapping (ROI name -> numeric label)
    label_mapping = {roi['name']: i for i, roi in enumerate(roi_regions)}

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
            # Extract spectral features for this ROI
            roi_spectra = roi_pixels[spectral_cols].values
            roi_labels = [label_mapping[roi_name]] * n_pixels

            X_train_list.append(roi_spectra)
            y_train_list.extend(roi_labels)

    # Combine all ROI data
    if X_train_list:
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
    else:
        # Fallback: use KMeans clustering if no ROI training data available
        return run_kmeans_fallback(df, valid_mask, metadata, n_clusters, random_state)

    # Step 3: Train KNN classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Use simple KNN (no hyperparameter optimization for speed)
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

    # Step 6: Calculate clustering metrics (adapted for KNN)
    # Since KNN predictions might not match n_clusters exactly, we adapt the metrics
    unique_labels = np.unique(predictions)
    n_actual_clusters = len(unique_labels)

    # Calculate basic metrics using spectral features
    try:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # Sample data for metric calculation (to avoid memory issues)
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

    n_features = len(spectral_cols)

    return cluster_map, metrics, n_features


def run_kmeans_fallback(df, valid_mask, metadata, n_clusters, random_state):
    """
    Fallback to improved KMeans clustering when ROI training fails.
    """
    # Perform clustering using improved approach
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

    return cluster_map, metrics, len(spectral_cols)


# Keep old function name for backward compatibility
def run_clustering_pipeline(data, n_clusters, random_state=42, export_concat_data=False, config_name=None):
    """
    UPDATED: Now uses KNN-based clustering instead of KMeans.
    This maintains the same interface but uses the improved approach.
    """
    return run_knn_clustering_pipeline(data, n_clusters, roi_regions=ROI_REGIONS, random_state=random_state,
                                       export_concat_data=export_concat_data, config_name=config_name)


def format_wavelength_combinations(combinations):
    """
    Format wavelength combinations into readable strings for Excel.
    """
    if not combinations:
        return "None"

    # Group by excitation wavelength
    by_excitation = {}
    for combo in combinations:
        ex = combo['excitation']
        em = combo['emission']
        if ex not in by_excitation:
            by_excitation[ex] = []
        by_excitation[ex].append(em)

    # Create formatted strings
    formatted_parts = []
    for ex, emissions in sorted(by_excitation.items()):
        emissions_str = ", ".join([f"{em:.1f}" for em in sorted(emissions)])
        formatted_parts.append(f"Ex{ex:.0f}nm: [{emissions_str}]nm")

    return " | ".join(formatted_parts)


def create_misclassification_map(predicted_labels, ground_truth_labels, output_path, title="Misclassified Pixels"):
    """
    Create a visualization showing misclassified pixels in red on a white background.

    Args:
        predicted_labels: Predicted cluster labels (2D array)
        ground_truth_labels: Ground truth labels (2D array)
        output_path: Path to save the output image
        title: Title for the plot
    """
    from sklearn.metrics.cluster import contingency_matrix

    # Flatten arrays for processing
    pred_flat = predicted_labels.flatten()
    gt_flat = ground_truth_labels.flatten()

    # Find optimal mapping between predicted and ground truth labels using Hungarian algorithm
    # Build contingency matrix
    cont_matrix = contingency_matrix(gt_flat, pred_flat)

    # Use Hungarian algorithm to find optimal mapping
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-cont_matrix)

    # Create mapping dictionary
    label_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

    # Map predicted labels to ground truth space
    mapped_pred = np.vectorize(lambda x: label_mapping.get(x, -1))(predicted_labels)

    # Create misclassification mask (ignore background -1)
    # Create misclassification mask (ignore background -1)
    valid_mask = (ground_truth_labels != -1) & (predicted_labels != -1)
    misclassified = (mapped_pred != ground_truth_labels) & valid_mask

    # Calculate misclassification statistics
    n_total = valid_mask.sum()
    n_misclassified = misclassified.sum()
    misclassification_rate = (n_misclassified / n_total * 100) if n_total > 0 else 0

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create RGB image: white background, red misclassified pixels
    vis_image = np.ones((*misclassified.shape, 3))  # White background
    vis_image[misclassified] = [1, 0, 0]  # Red for misclassified
    vis_image[~valid_mask] = [0.9, 0.9, 0.9]  # Light gray for background

    ax.imshow(vis_image)
    ax.set_title(f'{title}\nMisclassified: {n_misclassified:,} pixels ({misclassification_rate:.2f}%)',
                 fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'n_total': n_total,
        'n_misclassified': n_misclassified,
        'misclassification_rate': misclassification_rate
    }


def export_ranking_data_to_excel(selection_results, config_name, output_path):
    """
    Export full ranking information (layer, excitation-emission, rank, coefficient) to Excel.

    Args:
        selection_results: Results from wavelength selection analysis
        config_name: Name of the configuration
        output_path: Path to save the Excel file
    """
    # Extract ranking information from selection results
    ranking_data = []

    # Check if dimension_rankings exists in results
    if 'dimension_rankings' in selection_results:
        rankings = selection_results['dimension_rankings']

        for rank, (dim_idx, score) in enumerate(rankings, start=1):
            # Try to get wavelength information
            if 'selected_bands' in selection_results:
                # Find the band information for this dimension
                bands = selection_results['selected_bands']
                if dim_idx < len(bands):
                    band_info = bands[dim_idx]
                    if isinstance(band_info, dict):
                        excitation = band_info.get('excitation', 'N/A')
                        emission = band_info.get('emission_wavelength', 'N/A')
                    else:
                        excitation = 'N/A'
                        emission = band_info
                else:
                    excitation = 'N/A'
                    emission = 'N/A'
            else:
                excitation = 'N/A'
                emission = 'N/A'

            ranking_data.append({
                'Rank': rank,
                'Layer_Index': dim_idx,
                'Excitation_nm': excitation,
                'Emission_nm': emission,
                'Coefficient': score,
                'Layer_Name': f"Ex{excitation}_Em{emission}" if excitation != 'N/A' else f"Layer_{dim_idx}"
            })

    # If dimension_rankings doesn't exist, try to extract from selected_bands
    elif 'selected_bands' in selection_results:
        bands = selection_results['selected_bands']
        for rank, band_info in enumerate(bands, start=1):
            if isinstance(band_info, dict):
                excitation = band_info.get('excitation', 'N/A')
                emission = band_info.get('emission_wavelength', 'N/A')
                score = band_info.get('importance_score', band_info.get('score', 0.0))
            else:
                excitation = 'N/A'
                emission = band_info
                score = 0.0

            ranking_data.append({
                'Rank': rank,
                'Layer_Index': rank - 1,
                'Excitation_nm': excitation,
                'Emission_nm': emission,
                'Coefficient': score,
                'Layer_Name': f"Ex{excitation}_Em{emission}" if excitation != 'N/A' else f"Layer_{rank - 1}"
            })

    # Create DataFrame
    if ranking_data:
        df_ranking = pd.DataFrame(ranking_data)
        df_ranking.to_excel(output_path, index=False, sheet_name='Layer_Rankings')
        return len(ranking_data)
    else:
        # Create empty DataFrame with expected columns
        df_ranking = pd.DataFrame(columns=['Rank', 'Layer_Index', 'Excitation_nm', 'Emission_nm', 'Coefficient', 'Layer_Name'])
        df_ranking.to_excel(output_path, index=False, sheet_name='Layer_Rankings')
        return 0


print("Core functions defined (UPDATED with KNN-based clustering and export functionality!)")
# %% md
## 3. Load Data and Ground Truth
# %%
# Define paths
sample_name = "Lichens"
data_path = base_dir / "data" / "processed" / sample_name / "lichens_data_masked.pkl"
# data_path = r"C:\Users\meloy\PycharmProjects\Capstone\data\processed\Lichens\data_cutoff_40nm_exposure_max_power_min.power_normalization.pk"
mask_path = base_dir / "data" / "processed" / sample_name / "lichens_mask.npy"
png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")

print("Loading data...")
print(f"  Sample: {sample_name}")
print(f"  Data: {data_path.name}")
print(f"  Ground truth: {png_path.name}")
# %%
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
# %%
# Visualize RGB-like image from the hyperspectral data
print("\nCreating RGB-like visualization from hyperspectral data...")


# Select wavelengths that approximate RGB channels
# We'll use different excitations and find wavelengths close to R(~650nm), G(~550nm), B(~450nm)

# Function to find closest wavelength index
def find_closest_wavelength(wavelengths, target):
    wavelengths = np.array(wavelengths)
    idx = np.argmin(np.abs(wavelengths - target))
    return idx


# Initialize RGB channels
rgb_image = None

# Try to find good approximations for RGB from available data
# We'll use excitation 365nm as it often gives good fluorescence across spectrum
excitation_for_rgb = '365.0'  # Middle excitation wavelength

if excitation_for_rgb in full_data['data']:
    cube_data = full_data['data'][excitation_for_rgb]['cube']
    wavelengths = full_data['data'][excitation_for_rgb]['wavelengths']

    # Find indices for RGB-like wavelengths
    # For fluorescence data, we might need to adjust these targets
    red_idx = find_closest_wavelength(wavelengths, 650)  # Red ~650nm
    green_idx = find_closest_wavelength(wavelengths, 550)  # Green ~550nm
    blue_idx = find_closest_wavelength(wavelengths, 450)  # Blue ~450nm

    # Extract the RGB channels
    red_channel = cube_data[:, :, red_idx]
    green_channel = cube_data[:, :, green_idx]
    blue_channel = cube_data[:, :, blue_idx]

    # Stack into RGB image
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)

    # Normalize and enhance brightness for better visibility
    # Normalize each channel independently to use full range
    for i in range(3):
        channel = rgb_image[:, :, i]
        # Remove outliers using percentile clipping
        vmin, vmax = np.percentile(channel[channel > 0], [2, 98]) if np.any(channel > 0) else (0, 1)
        rgb_image[:, :, i] = np.clip((channel - vmin) / (vmax - vmin + 1e-10), 0, 1)

    # Apply brightness enhancement
    brightness_factor = 2.5  # Increase brightness
    rgb_image = np.clip(rgb_image * brightness_factor, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Show enhanced RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title(
        f'RGB-like visualization (Ex: {excitation_for_rgb}nm)\nR≈{wavelengths[red_idx]:.1f}nm, G≈{wavelengths[green_idx]:.1f}nm, B≈{wavelengths[blue_idx]:.1f}nm')
    axes[0].axis('off')

    # Show with different enhancement for more detail
    # Apply gamma correction for better visibility of dark areas
    gamma = 0.5
    rgb_gamma = np.power(rgb_image, gamma)
    axes[1].imshow(rgb_gamma)
    axes[1].set_title(f'Enhanced with gamma correction (γ={gamma})')
    axes[1].axis('off')

    plt.suptitle(f'Hyperspectral Data Visualization - {sample_name} Sample', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"RGB visualization created using excitation {excitation_for_rgb}nm")
    print(f"  Red channel: {wavelengths[red_idx]:.1f}nm (index {red_idx})")
    print(f"  Green channel: {wavelengths[green_idx]:.1f}nm (index {green_idx})")
    print(f"  Blue channel: {wavelengths[blue_idx]:.1f}nm (index {blue_idx})")
    print(f"  Image shape: {rgb_image.shape}")
else:
    print(f"Warning: Excitation {excitation_for_rgb}nm not found in data")
    print(f"Available excitations: {full_data['excitation_wavelengths']}")
# %%
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

print("Current data shape information:")
sample_ex = str(full_data['excitation_wavelengths'][0])
sample_shape = full_data['data'][sample_ex]['cube'].shape
print(f"  Full spatial dimensions: {sample_shape[0]} x {sample_shape[1]} pixels")

# Set cropping parameters
# start_col = 1392 - 925
# end_col = 1392
start_col = 1392 - 925
end_col = 830

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

    # Crop the spatial dimensions (keeping all spectral bands)
    cropped_cube = original_cube[:, start_col:end_col, :]

    # Copy all metadata but replace cube with cropped version
    cropped_data['data'][ex_str] = {
        **full_data['data'][ex_str],
        'cube': cropped_cube
    }

# Also crop the ground truth mask to match
ground_truth_cropped = ground_truth[:, start_col:end_col]

# Update the working datasets
full_data = cropped_data
ground_truth = ground_truth_cropped

print(f"Data successfully cropped")
print(f"  New data shape: {cropped_cube.shape}")
print(f"  New ground truth shape: {ground_truth.shape}")

print(f"\nFinal working dimensions:")
working_shape = full_data['data'][sample_ex]['cube'].shape
print(f"  Spatial: {working_shape[0]} x {working_shape[1]} pixels")
print(f"  Ground truth: {ground_truth.shape}")
print(f"  Total pixels for analysis: {ground_truth.size:,}")
# %% md
## 4. Baseline: Full Data Clustering
# %%
print("=" * 80)
print("BASELINE: Clustering with Full Data")
print("=" * 80)
n_clusters = 4  # Using 4 clusters to match 4 ROI regions

# Create ROI colormap for consistent visualization
roi_colormap = create_roi_colormap(ROI_REGIONS)

# NEW: Initialize performance tracker and cluster maps storage
timing_tracker = TimingTracker()
cluster_maps = {}  # Store cluster maps for difference visualization
print("✅ Performance tracking and cluster map storage initialized")

# NEW: TIME THE BASELINE CLUSTERING
with PerformanceTimer() as baseline_timer:
    cluster_map_full, metrics_full, n_features_full = run_clustering_pipeline(
        full_data,
        n_clusters,  # Use n_clusters (4) instead of n_true_classes
        export_concat_data=True,
        config_name="BASELINE_FULL_DATA"
    )

# Record baseline timing
timing_tracker.record_clustering_full(
    baseline_timer.elapsed,
    n_features_full,
    "BASELINE"
)

# Save baseline cluster map for difference visualization
cluster_maps['BASELINE_FULL_DATA'] = cluster_map_full.copy()

# Calculate ground truth metrics
baseline_metrics = calculate_clustering_accuracy(
    cluster_map_full,
    ground_truth,
    np.ones_like(ground_truth, dtype=bool)
)

print(f"\nBaseline Results:")
print(f"  Features: {n_features_full}")
print(f"  Purity: {baseline_metrics['purity']:.4f}")
print(f"  ARI: {baseline_metrics['adjusted_rand_score']:.4f}")
print(f"  NMI: {baseline_metrics['normalized_mutual_info']:.4f}")
print(f"  Silhouette: {metrics_full['silhouette_score']:.4f}")
print(f"  ⏱️ Clustering time: {baseline_timer.elapsed:.2f}s")

# Save baseline visualization with ROI colors
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(ground_truth, cmap='tab10')
axes[0].set_title('Ground Truth')
axes[0].axis('off')

# Create a masked array for proper visualization (mask -1 values)
cluster_map_display = np.ma.masked_where(cluster_map_full == -1, cluster_map_full)
axes[1].imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
axes[1].set_title(f'Baseline (Purity: {baseline_metrics["purity"]:.3f})')
axes[1].axis('off')

plt.tight_layout()
plt.savefig(visualizations_dir / "baseline.png", dpi=150, bbox_inches='tight')
plt.show()

# NEW: Save baseline result image separately for paper
baseline_only_fig = plt.figure(figsize=(8, 8))
cluster_map_display = np.ma.masked_where(cluster_map_full == -1, cluster_map_full)
plt.imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
plt.title(f'Baseline Full Data\nPurity: {baseline_metrics["purity"]:.3f}', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(paper_results_dir / "baseline_full_data.png", dpi=300, bbox_inches='tight')
plt.close()
# %% md
## 5. Define Wavelength Selection Configurations
# %%
configurations = [
    # ============================================================================
    # SELECTED TOP 10 CONFIGURATIONS BASED ON EXPERIMENTAL RESULTS
    # Results from: wavelength_analysis/validation_results/20251012_212608
    # ============================================================================

    # ============================================================================
    # 🥇 BEST 5: Top-tier performance (Purity: 0.8668-0.8682)
    # ============================================================================
    # All use variance dimension selection + MMR diversity
    # These configurations achieve the highest purity scores

    {
        'name': 'mmr_lambda050_variance',  # BASELINE MMR - Balanced diversity/influence
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 7,
        'n_bands_to_select': 10,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5  # Balanced: equal weight to influence and diversity
    },

    {
        'name': 'mmr_lambda030_variance',  # LOW LAMBDA - Influence-focused
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 7,
        'n_bands_to_select': 10,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.3  # Favor influence over diversity
    },

    {
        'name': 'mmr_lambda070_variance',  # HIGH LAMBDA - Diversity-focused
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 7,
        'n_bands_to_select': 10,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.7  # Favor diversity over influence
    },

    {
        'name': 'mmr_11bands_lambda05',  # MORE BANDS - Testing 11 bands vs 10
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [12, 25, 40],
        'n_important_dimensions': 8,
        'n_bands_to_select': 11,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },

    {
        'name': 'mmr_perturbation_percentile',  # DIFFERENT PERTURBATION - percentile method
        'dimension_selection_method': 'variance',
        'perturbation_method': 'percentile',
        'perturbation_magnitudes': [10, 20, 35],
        'n_important_dimensions': 7,
        'n_bands_to_select': 10,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },

    # ============================================================================
    # 🥈 MEDIUM 3: Good performance with better data reduction (Purity: 0.8600-0.8654)
    # ============================================================================
    # Fewer bands (7-8) with acceptable performance loss (~2%)
    # Demonstrates efficiency gains (76-80% data reduction)

    {
        'name': 'hybrid_conservative_mmr',  # 8 BANDS - Conservative approach
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [10, 20, 30],
        'n_important_dimensions': 8,
        'n_bands_to_select': 8,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.45
    },

    {
        'name': 'mmr_8bands_lambda05',  # 8 BANDS - Balanced MMR
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 6,
        'n_bands_to_select': 8,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },

    {
        'name': 'mmr_7bands_lambda05',  # 7 BANDS - Maximum efficiency
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [20, 40, 60],
        'n_important_dimensions': 6,
        'n_bands_to_select': 7,
        'normalization_method': 'max_per_excitation',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },

    # ============================================================================
    # 🔻 NOT SO GOOD 2: Representative poor performers for validation (Purity: 0.7841-0.8145)
    # ============================================================================
    # Demonstrates that dimension selection method matters more than diversity strategy
    # Validates that activation and PCA methods underperform compared to variance

    {
        'name': 'mmr_activation_8bands',  # ACTIVATION METHOD - Best activation result (still poor)
        'dimension_selection_method': 'activation',
        'perturbation_method': 'percentile',
        'perturbation_magnitudes': [5, 10, 20],
        'n_important_dimensions': 8,
        'n_bands_to_select': 8,
        'normalization_method': 'variance',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },

    {
        'name': 'mmr_pca_lambda05',  # PCA METHOD - Representative PCA failure
        'dimension_selection_method': 'pca',
        'perturbation_method': 'absolute_range',
        'perturbation_magnitudes': [20, 40, 60],
        'n_important_dimensions': 8,
        'n_bands_to_select': 10,
        'normalization_method': 'variance',
        'use_diversity_constraint': True,
        'diversity_method': 'mmr',
        'lambda_diversity': 0.5
    },
]

print(f"Defined {len(configurations)} configurations to test:")
for config in configurations:
    print(f"{config['name']}: {config['n_bands_to_select']} bands, "
          f"{config['dimension_selection_method']} method")
# %% md
## 6. Run All Configurations
# %%
# Initialize results storage
results = []
all_combinations = []  # Store all wavelength combinations for detailed analysis

print("=" * 80)
print("RUNNING WAVELENGTH SELECTION CONFIGURATIONS")
print("=" * 80)

for i, config in enumerate(tqdm(configurations, desc="Testing configurations")):
    config_name = config['name']
    print(f"\n[{i + 1}/{len(configurations)}] Running: {config_name}")

    try:
        # Step 1: TIME WAVELENGTH SELECTION
        with PerformanceTimer() as selection_timer:
            wavelength_combinations, emission_wavelengths_only, selection_results = select_informative_wavelengths_fixed(
                data_path,
                mask_path,
                sample_name,
                config,
                verbose=False
            )

        # Record selection timing
        timing_tracker.record_wavelength_selection(
            selection_timer.elapsed,
            len(wavelength_combinations),
            config_name
        )

        print(f"  Selected {len(wavelength_combinations)} wavelength combinations")
        if wavelength_combinations:
            print(f"  Example: {wavelength_combinations[0]['combination_name']}")
        print(f"  ⏱️ Selection time: {selection_timer.elapsed:.2f}s")

        # Step 2: Extract subset (using emission wavelengths for compatibility)
        subset_data = extract_wavelength_subset(
            full_data,
            emission_wavelengths_only,
            verbose=True
        )

        # Step 3: TIME CLUSTERING
        with PerformanceTimer() as cluster_timer:
            cluster_map, metrics, n_features = run_clustering_pipeline(
                subset_data,
                n_clusters,  # Use n_clusters (4) instead of n_true_classes
                export_concat_data=True,  # NEW: Export concatenated data
                config_name=config_name  # NEW: Pass config name for file naming
            )

        # Record clustering timing
        timing_tracker.record_clustering_subset(
            cluster_timer.elapsed,
            n_features,
            config_name
        )

        # SAVE CLUSTER MAP for difference visualization
        cluster_maps[config_name] = cluster_map.copy()

        # Step 4: Validate against ground truth
        gt_metrics = calculate_clustering_accuracy(
            cluster_map,
            ground_truth,
            np.ones_like(ground_truth, dtype=bool)
        )

        # Calculate data reduction and timing metrics
        data_reduction_pct = (1 - n_features / n_features_full) * 100
        speedup = baseline_timer.elapsed / cluster_timer.elapsed if cluster_timer.elapsed > 0 else 0

        # Step 5: Save visualization (comparison)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ground_truth, cmap='tab10')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        # Mask -1 values for proper visualization
        cluster_map_full_display = np.ma.masked_where(cluster_map_full == -1, cluster_map_full)
        axes[1].imshow(cluster_map_full_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        axes[1].set_title(f'Baseline\nPurity: {baseline_metrics["purity"]:.3f}')
        axes[1].axis('off')

        # Mask -1 values for proper visualization
        cluster_map_display = np.ma.masked_where(cluster_map == -1, cluster_map)
        axes[2].imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        axes[2].set_title(f'{config_name}\nPurity: {gt_metrics["purity"]:.3f}')
        axes[2].axis('off')

        plt.suptitle(f'Configuration: {config_name} ({len(wavelength_combinations)} combinations)')
        plt.tight_layout()
        plt.savefig(visualizations_dir / f"{config_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # NEW: Save individual result image for paper
        individual_fig = plt.figure(figsize=(8, 8))
        cluster_map_display = np.ma.masked_where(cluster_map == -1, cluster_map)
        plt.imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        plt.title(f'{config_name}\n{len(wavelength_combinations)} wavelengths | Purity: {gt_metrics["purity"]:.3f}',
                  fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(paper_results_dir / f"{config_name}_result.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved individual result to paper-results/{config_name}_result.png")

        # NEW: Create experiment-specific folder and export all related images
        experiment_folder = experiments_dir / config_name
        experiment_folder.mkdir(exist_ok=True)

        # NEW: Export difference maps for this experiment
        # 1. Ground truth vs baseline
        print(f"    Creating difference maps...")
        diff_stats_baseline = create_misclassification_map(
            cluster_map_full,
            ground_truth,
            experiment_folder / f"{config_name}_diff_GT_vs_baseline.png",
            title=f"Ground Truth vs Baseline\n{config_name}"
        )
        print(f"      GT vs Baseline: {diff_stats_baseline['misclassification_rate']:.2f}% misclassified")

        # 2. Ground truth vs selected subset
        diff_stats_subset = create_misclassification_map(
            cluster_map,
            ground_truth,
            experiment_folder / f"{config_name}_diff_GT_vs_subset.png",
            title=f"Ground Truth vs Selected Subset\n{config_name}"
        )
        print(f"      GT vs Subset: {diff_stats_subset['misclassification_rate']:.2f}% misclassified")

        # NEW: Export full ranking data to Excel
        print(f"    Exporting ranking data to Excel...")
        ranking_excel_path = experiment_folder / f"{config_name}_layer_rankings.xlsx"
        n_rankings = export_ranking_data_to_excel(selection_results, config_name, ranking_excel_path)
        print(f"      Exported {n_rankings} layer rankings to Excel")

        # NEW: Copy/save all related images to experiment folder
        # Save clustering result
        individual_result_fig = plt.figure(figsize=(8, 8))
        cluster_map_display = np.ma.masked_where(cluster_map == -1, cluster_map)
        plt.imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        plt.title(f'{config_name}\n{len(wavelength_combinations)} wavelengths | Purity: {gt_metrics["purity"]:.3f}',
                  fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(experiment_folder / f"{config_name}_clustering_result.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save comparison plot (GT, Baseline, Subset)
        comparison_fig, comparison_axes = plt.subplots(1, 3, figsize=(18, 6))
        comparison_axes[0].imshow(ground_truth, cmap='tab10')
        comparison_axes[0].set_title('Ground Truth', fontsize=12)
        comparison_axes[0].axis('off')

        cluster_map_full_display = np.ma.masked_where(cluster_map_full == -1, cluster_map_full)
        comparison_axes[1].imshow(cluster_map_full_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        comparison_axes[1].set_title(f'Baseline\nPurity: {baseline_metrics["purity"]:.3f}', fontsize=12)
        comparison_axes[1].axis('off')

        cluster_map_display = np.ma.masked_where(cluster_map == -1, cluster_map)
        comparison_axes[2].imshow(cluster_map_display, cmap=roi_colormap, vmin=0, vmax=len(ROI_REGIONS) - 1)
        comparison_axes[2].set_title(f'{config_name}\nPurity: {gt_metrics["purity"]:.3f}', fontsize=12)
        comparison_axes[2].axis('off')

        plt.suptitle(f'Configuration: {config_name} ({len(wavelength_combinations)} combinations)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(experiment_folder / f"{config_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✅ All experiment images saved to: experiments/{config_name}/")

        # ==================================================================
        # NEW: Export ROI overlay visualizations
        # ==================================================================
        print(f"    Creating ROI overlay visualizations...")

        # ROI overlay with clustering result
        create_roi_overlay_visualization(
            cluster_map=cluster_map,
            roi_regions=ROI_REGIONS,
            output_path=experiment_folder / f"{config_name}_roi_overlay.png",
            title=f"{config_name} - Clustering with ROI Overlay"
        )

        # ROI analysis report
        create_roi_analysis_report(
            cluster_map=cluster_map,
            roi_regions=ROI_REGIONS,
            output_path=experiment_folder / f"{config_name}_roi_analysis.png"
        )

        # ==================================================================
        # NEW: Store results FIRST before using them
        # ==================================================================
        # Store results with detailed wavelength information AND TIMING DATA
        result = {
            'config_name': config_name,
            'dimension_method': config['dimension_selection_method'],
            'perturbation_method': config['perturbation_method'],
            'n_important_dims': config['n_important_dimensions'],
            'n_combinations_selected': len(wavelength_combinations),
            'n_features': n_features,
            'data_reduction_pct': data_reduction_pct,
            # Format wavelength combinations in readable format
            'wavelength_combinations': format_wavelength_combinations(wavelength_combinations),
            # Also keep the simple list for backward compatibility
            'emission_wavelengths_only': str(emission_wavelengths_only),
            'purity': gt_metrics['purity'],
            'ari': gt_metrics['adjusted_rand_score'],
            'nmi': gt_metrics['normalized_mutual_info'],
            'v_measure': gt_metrics['v_measure'],
            'homogeneity': gt_metrics['homogeneity'],
            'completeness': gt_metrics['completeness'],
            'silhouette': metrics['silhouette_score'],
            'davies_bouldin': metrics['davies_bouldin_score'],
            'calinski_harabasz': metrics['calinski_harabasz_score'],
            # NEW: Timing data
            'selection_time': selection_timer.elapsed,
            'clustering_time': cluster_timer.elapsed,
            'speedup_factor': speedup,
            'time_saved': baseline_timer.elapsed - cluster_timer.elapsed
        }

        # ==================================================================
        # NEW: Export ground truth difference maps (with proper label mapping)
        # ==================================================================
        print(f"    Creating ground truth difference maps with label mapping...")

        # CRITICAL FIX: Use mask that excludes background pixels
        valid_pixels_mask = (ground_truth != -1)

        gt_diff_stats = create_ground_truth_difference_maps(
            ground_truth=ground_truth,
            baseline_labels=cluster_map_full,
            optimized_labels=cluster_map,
            mask=valid_pixels_mask,  # FIXED: Only compare non-background pixels
            config_name=config_name,
            output_dir=experiment_folder,
            baseline_purity=baseline_metrics['purity'],
            optimized_purity=gt_metrics['purity']
        )

        print(f"      Baseline errors: {gt_diff_stats['baseline_wrong']:,}")
        print(f"      Optimized errors: {gt_diff_stats['optimized_wrong']:,}")
        print(f"      Noise reduction: {gt_diff_stats['noise_reduction']:,} pixels ({gt_diff_stats['noise_reduction_pct']:.1f}%)")

        # ==================================================================
        # NEW: Export comprehensive metrics to Excel
        # ==================================================================
        print(f"    Exporting comprehensive metrics...")

        # Prepare all data for export
        config_params_export = {
            **config,  # Include all config parameters
            'data_reduction_pct': data_reduction_pct,
            'n_features': n_features,
            'n_features_baseline': n_features_full
        }

        timing_data_export = {
            'selection_time': selection_timer.elapsed,
            'clustering_time': cluster_timer.elapsed,
            'speedup_factor': speedup,
            'time_saved': baseline_timer.elapsed - cluster_timer.elapsed
        }

        # Export to Excel
        metrics_excel_path = experiment_folder / f"{config_name}_comprehensive_metrics.xlsx"
        export_experiment_metrics(
            config_params=config_params_export,
            wavelength_combinations=wavelength_combinations,
            clustering_metrics=metrics,
            ground_truth_metrics=gt_metrics,
            timing_data=timing_data_export,
            output_path=metrics_excel_path
        )

        # CRITICAL FIX: Add gt_diff_stats to the result dictionary
        result.update(gt_diff_stats)

        # Also export quick CSV for review
        result_for_csv = {
            **result,  # Now includes gt_diff_stats
        }
        csv_path = experiment_folder / f"{config_name}_metrics_summary.csv"
        export_experiment_csv(result_for_csv, csv_path)

        print(f"    ✅ All exports complete for: {config_name}")

        # Store detailed wavelength combinations for this configuration
        config_combinations = []
        for combo in wavelength_combinations:
            config_combinations.append({
                'config_name': config_name,
                'excitation_nm': combo['excitation'],
                'emission_nm': combo['emission'],
                'combination_name': combo['combination_name']
            })
        all_combinations.extend(config_combinations)

        # Append result to results list (no duplicate definition needed)
        results.append(result)

        print(f"  Purity: {gt_metrics['purity']:.4f} | ARI: {gt_metrics['adjusted_rand_score']:.4f}")
        print(f"  ⏱️ Clustering time: {cluster_timer.elapsed:.2f}s")
        print(f"  ⚡ Speedup: {speedup:.2f}x")

    except Exception as e:
        print(f"Error: {str(e)}")
        results.append({
            'config_name': config_name,
            'error': str(e)
        })

print("\n" + "=" * 80)
print(f"Completed {len(results)} configurations")
print(f"Collected {len(all_combinations)} total wavelength combinations")
print(f"Exported concatenated data for each configuration to: {concat_data_dir}")
print(f"Saved individual clustering results to: {paper_results_dir}")

# NEW: Save cluster maps for difference visualization
cluster_maps_file = results_dir / "cluster_maps.pkl"
with open(cluster_maps_file, 'wb') as f:
    pickle.dump(cluster_maps, f)
print(f"✅ Saved {len(cluster_maps)} cluster maps to: {cluster_maps_file.name}")
# %% md
## 7. Compile and Save Results
# %%
# Create DataFrame from results
df_results = pd.DataFrame(results)

# Add baseline row
baseline_row = {
    'config_name': 'BASELINE_FULL_DATA',
    'dimension_method': 'N/A',
    'perturbation_method': 'N/A',
    'n_important_dims': 'N/A',
    'n_combinations_selected': total_bands,
    'n_features': n_features_full,
    'data_reduction_pct': 0.0,
    'wavelength_combinations': 'ALL_EXCITATION_EMISSION_PAIRS',
    'emission_wavelengths_only': 'ALL',
    'purity': baseline_metrics['purity'],
    'ari': baseline_metrics['adjusted_rand_score'],
    'nmi': baseline_metrics['normalized_mutual_info'],
    'v_measure': baseline_metrics['v_measure'],
    'homogeneity': baseline_metrics['homogeneity'],
    'completeness': baseline_metrics['completeness'],
    'silhouette': metrics_full['silhouette_score'],
    'davies_bouldin': metrics_full['davies_bouldin_score'],
    'calinski_harabasz': metrics_full['calinski_harabasz_score']
}

df_results = pd.concat([pd.DataFrame([baseline_row]), df_results], ignore_index=True)

# Sort by purity
df_results = df_results.sort_values('purity', ascending=False)

# Create detailed combinations DataFrame
df_combinations = pd.DataFrame(all_combinations)

print("Results DataFrames created:")
print(f"\nMain Results: {len(df_results)} configurations")
print(df_results[['config_name', 'n_combinations_selected', 'purity', 'ari', 'nmi']].head(10))

if not df_combinations.empty:
    print(f"\nDetailed Combinations: {len(df_combinations)} excitation-emission pairs")
    print(df_combinations.head())
# %%
# Save to Excel and CSV
excel_path = results_dir / "wavelength_selection_results.xlsx"
csv_path = results_dir / "wavelength_selection_results.csv"

# Save to Excel with multiple sheets
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Main results sheet
    df_results.to_excel(writer, sheet_name='Configuration_Results', index=False)

    # Detailed wavelength combinations sheet
    if not df_combinations.empty:
        df_combinations.to_excel(writer, sheet_name='Wavelength_Combinations', index=False)

    # Summary sheet
    best_purity_idx = df_results['purity'].idxmax()
    best_ari_idx = df_results['ari'].idxmax()
    best_nmi_idx = df_results['nmi'].idxmax()
    best_reduction_idx = df_results['data_reduction_pct'].idxmax()

    summary = pd.DataFrame({
        'Metric': ['Best Purity', 'Best ARI', 'Best NMI', 'Most Data Reduction'],
        'Configuration': [
            df_results.loc[best_purity_idx, 'config_name'],
            df_results.loc[best_ari_idx, 'config_name'],
            df_results.loc[best_nmi_idx, 'config_name'],
            df_results.loc[best_reduction_idx, 'config_name']
        ],
        'Value': [
            f"{df_results.loc[best_purity_idx, 'purity']:.4f}",
            f"{df_results.loc[best_ari_idx, 'ari']:.4f}",
            f"{df_results.loc[best_nmi_idx, 'nmi']:.4f}",
            f"{df_results.loc[best_reduction_idx, 'data_reduction_pct']:.1f}%"
        ],
        'Wavelength_Combinations': [
            str(df_results.loc[best_purity_idx, 'wavelength_combinations'])[:100] + "..." if len(
                str(df_results.loc[best_purity_idx, 'wavelength_combinations'])) > 100 else str(
                df_results.loc[best_purity_idx, 'wavelength_combinations']),
            str(df_results.loc[best_ari_idx, 'wavelength_combinations'])[:100] + "..." if len(
                str(df_results.loc[best_ari_idx, 'wavelength_combinations'])) > 100 else str(
                df_results.loc[best_ari_idx, 'wavelength_combinations']),
            str(df_results.loc[best_nmi_idx, 'wavelength_combinations'])[:100] + "..." if len(
                str(df_results.loc[best_nmi_idx, 'wavelength_combinations'])) > 100 else str(
                df_results.loc[best_nmi_idx, 'wavelength_combinations']),
            str(df_results.loc[best_reduction_idx, 'wavelength_combinations'])[:100] + "..." if len(
                str(df_results.loc[best_reduction_idx, 'wavelength_combinations'])) > 100 else str(
                df_results.loc[best_reduction_idx, 'wavelength_combinations'])
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

    # Wavelength frequency analysis sheet
    if not df_combinations.empty:
        # Count how often each excitation-emission combination appears
        combo_counts = df_combinations.groupby(['excitation_nm', 'emission_nm']).agg({
            'config_name': 'count',
            'combination_name': 'first'
        }).rename(columns={'config_name': 'frequency'}).sort_values('frequency', ascending=False)

        combo_counts.to_excel(writer, sheet_name='Combination_Frequency')

        # Excitation wavelength popularity
        ex_popularity = df_combinations.groupby('excitation_nm')['config_name'].count().sort_values(ascending=False)
        ex_popularity.to_excel(writer, sheet_name='Excitation_Popularity')

    # NEW: Add timing data sheet
    if 'selection_time' in df_results.columns:
        timing_cols = ['config_name', 'selection_time', 'clustering_time', 'speedup_factor', 'time_saved']
        timing_summary = df_results[timing_cols].copy()
        timing_summary.to_excel(writer, sheet_name='Performance_Timing', index=False)

# Save main results to CSV
df_results.to_csv(csv_path, index=False)

# Save combinations to separate CSV
if not df_combinations.empty:
    combinations_csv = results_dir / "wavelength_combinations_detailed.csv"
    df_combinations.to_csv(combinations_csv, index=False)
    print(f"  Combinations CSV: {combinations_csv}")

print(f"\nResults saved to:")
print(f"  Excel (multi-sheet): {excel_path}")
print(f"    - Configuration_Results: Main results with wavelength combinations")
if not df_combinations.empty:
    print(f"    - Wavelength_Combinations: Detailed excitation-emission pairs")
    print(f"    - Combination_Frequency: Most commonly selected wavelength pairs")
    print(f"    - Excitation_Popularity: Most useful excitation wavelengths")
print(f"    - Summary: Best configurations by metric")
print(f"  CSV: {csv_path}")
# %% md
## 8. Analysis and Visualization
# %%
# Create comprehensive comparison plot with improved visibility
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Increased from (15, 10)

# Plot 1: Purity vs Data Reduction
axes[0, 0].scatter(df_results['data_reduction_pct'], df_results['purity'], s=100, alpha=0.7)
axes[0, 0].set_xlabel('Data Reduction (%)', fontsize=11)
axes[0, 0].set_ylabel('Purity', fontsize=11)
axes[0, 0].set_title('Purity vs Data Reduction', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Metrics comparison bar plot
metrics_cols = ['purity', 'ari', 'nmi', 'v_measure']
top_5 = df_results.nlargest(5, 'purity')
x = np.arange(len(top_5))
width = 0.2

for i, metric in enumerate(metrics_cols):
    axes[0, 1].bar(x + i * width, top_5[metric], width, label=metric.upper())

axes[0, 1].set_xlabel('Configuration', fontsize=11)
axes[0, 1].set_ylabel('Score', fontsize=11)
axes[0, 1].set_title('Top 5 Configurations - Metrics Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(x + width * 1.5)
# Improved label visibility: use full names with smaller font and better rotation
axes[0, 1].set_xticklabels([c[:20] for c in top_5['config_name']], rotation=45, ha='right', fontsize=9)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Number of combinations vs Purity
axes[0, 2].scatter(df_results['n_combinations_selected'], df_results['purity'], s=100, alpha=0.7)
axes[0, 2].set_xlabel('Number of Wavelength Combinations Selected', fontsize=11)
axes[0, 2].set_ylabel('Purity', fontsize=11)
axes[0, 2].set_title('Combinations vs Purity', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Method comparison
valid_methods = df_results[df_results['dimension_method'] != 'N/A']
if not valid_methods.empty:
    method_results = valid_methods.groupby('dimension_method')['purity'].agg(['mean', 'std'])
    method_results.plot(kind='bar', y='mean', yerr='std', ax=axes[1, 0], legend=False)
    axes[1, 0].set_xlabel('Dimension Selection Method', fontsize=11)
    axes[1, 0].set_ylabel('Mean Purity', fontsize=11)
    axes[1, 0].set_title('Performance by Selection Method', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Purity improvement from baseline
baseline_purity = df_results[df_results['config_name'] == 'BASELINE_FULL_DATA']['purity'].values[0]
df_results['purity_improvement'] = ((df_results['purity'] - baseline_purity) / baseline_purity) * 100 if baseline_purity != 0 else 0

improvement_data = df_results[df_results['config_name'] != 'BASELINE_FULL_DATA'].sort_values('purity_improvement')
colors = ['green' if x >= 0 else 'red' for x in improvement_data['purity_improvement']]

axes[1, 1].barh(range(len(improvement_data)), improvement_data['purity_improvement'], color=colors, alpha=0.7)
axes[1, 1].set_yticks(range(len(improvement_data)))
# Improved label visibility: show more characters
axes[1, 1].set_yticklabels([c[:25] for c in improvement_data['config_name']], fontsize=9)
axes[1, 1].set_xlabel('Purity Improvement from Baseline (%)', fontsize=11)
axes[1, 1].set_title('Performance Change vs Baseline', fontsize=12, fontweight='bold')
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Correlation matrix
corr_cols = ['n_combinations_selected', 'data_reduction_pct', 'purity', 'ari', 'nmi', 'silhouette']
available_cols = [col for col in corr_cols if col in df_results.columns]
corr_data = df_results[available_cols].select_dtypes(include=[np.number])
corr_matrix = corr_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=axes[1, 2])
axes[1, 2].set_title('Metric Correlations', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(results_dir / "analysis_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# NEW: Export each of the 6 plots individually to analysis_summary folder
print("\n📊 Exporting individual analysis summary plots...")
plot_titles = [
    'purity_vs_data_reduction',
    'top5_metrics_comparison',
    'combinations_vs_purity',
    'performance_by_method',
    'purity_improvement',
    'metric_correlations'
]

for idx, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
    individual_fig = plt.figure(figsize=(8, 6))
    individual_ax = individual_fig.add_subplot(111)

    # Recreate each plot individually
    if idx == 0:  # Purity vs Data Reduction
        individual_ax.scatter(df_results['data_reduction_pct'], df_results['purity'], s=120, alpha=0.7)
        individual_ax.set_xlabel('Data Reduction (%)', fontsize=12)
        individual_ax.set_ylabel('Purity', fontsize=12)
        individual_ax.set_title('Purity vs Data Reduction', fontsize=14, fontweight='bold')
        individual_ax.grid(True, alpha=0.3)

    elif idx == 1:  # Metrics comparison
        for i_metric, metric in enumerate(metrics_cols):
            individual_ax.bar(x + i_metric * width, top_5[metric], width, label=metric.upper())
        individual_ax.set_xlabel('Configuration', fontsize=12)
        individual_ax.set_ylabel('Score', fontsize=12)
        individual_ax.set_title('Top 5 Configurations - Metrics Comparison', fontsize=14, fontweight='bold')
        individual_ax.set_xticks(x + width * 1.5)
        individual_ax.set_xticklabels([c[:20] for c in top_5['config_name']], rotation=45, ha='right', fontsize=10)
        individual_ax.legend(fontsize=10)
        individual_ax.grid(True, alpha=0.3)

    elif idx == 2:  # Combinations vs Purity
        individual_ax.scatter(df_results['n_combinations_selected'], df_results['purity'], s=120, alpha=0.7)
        individual_ax.set_xlabel('Number of Wavelength Combinations Selected', fontsize=12)
        individual_ax.set_ylabel('Purity', fontsize=12)
        individual_ax.set_title('Combinations vs Purity', fontsize=14, fontweight='bold')
        individual_ax.grid(True, alpha=0.3)

    elif idx == 3:  # Method comparison
        if not valid_methods.empty:
            method_results.plot(kind='bar', y='mean', yerr='std', ax=individual_ax, legend=False)
            individual_ax.set_xlabel('Dimension Selection Method', fontsize=12)
            individual_ax.set_ylabel('Mean Purity', fontsize=12)
            individual_ax.set_title('Performance by Selection Method', fontsize=14, fontweight='bold')
            individual_ax.tick_params(axis='x', rotation=45, labelsize=11)
        individual_ax.grid(True, alpha=0.3)

    elif idx == 4:  # Purity improvement
        individual_ax.barh(range(len(improvement_data)), improvement_data['purity_improvement'], color=colors, alpha=0.7)
        individual_ax.set_yticks(range(len(improvement_data)))
        individual_ax.set_yticklabels(improvement_data['config_name'], fontsize=10)
        individual_ax.set_xlabel('Purity Improvement from Baseline (%)', fontsize=12)
        individual_ax.set_title('Performance Change vs Baseline', fontsize=14, fontweight='bold')
        individual_ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        individual_ax.grid(True, alpha=0.3)

    elif idx == 5:  # Correlation matrix
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, ax=individual_ax)
        individual_ax.set_title('Metric Correlations', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(analysis_summary_dir / f"{plot_titles[idx]}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {plot_titles[idx]}.png")

print(f"  All 6 individual plots saved to: {analysis_summary_dir}")

# Additional plot: Wavelength combination popularity
if not df_combinations.empty:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Most popular excitation wavelengths
    ex_counts = df_combinations['excitation_nm'].value_counts()
    axes[0].bar(ex_counts.index.astype(str), ex_counts.values)
    axes[0].set_xlabel('Excitation Wavelength (nm)')
    axes[0].set_ylabel('Selection Frequency')
    axes[0].set_title('Most Popular Excitation Wavelengths')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # Top emission wavelengths
    em_counts = df_combinations['emission_nm'].value_counts().head(15)
    axes[1].barh(range(len(em_counts)), em_counts.values)
    axes[1].set_yticks(range(len(em_counts)))
    axes[1].set_yticklabels([f"{em:.1f} nm" for em in em_counts.index])
    axes[1].set_xlabel('Selection Frequency')
    axes[1].set_title('Top 15 Most Popular Emission Wavelengths')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "wavelength_popularity.png", dpi=300, bbox_inches='tight')
    plt.show()

print("Analysis plots saved")

print("=" * 80)
print("WAVELENGTH SELECTION VALIDATION - FINAL SUMMARY")
print("=" * 80)

print("\nBASELINE PERFORMANCE:")
print(f"  Features: {n_features_full}")
print(f"  Purity: {baseline_metrics['purity']:.4f}")
print(f"  ARI: {baseline_metrics['adjusted_rand_score']:.4f}")
print(f"  NMI: {baseline_metrics['normalized_mutual_info']:.4f}")

print("\n BEST CONFIGURATIONS:")
if not df_results.empty:
    print("\n  By Purity:")
    best_purity = df_results.loc[df_results['purity'].idxmax()]
    print(f"    Config: {best_purity['config_name']}")
    print(
        f"    Combinations: {best_purity['n_combinations_selected']} ({best_purity['data_reduction_pct']:.1f}% reduction)")
    print(f"    Purity: {best_purity['purity']:.4f}")
    print(f"    Wavelengths: {best_purity['wavelength_combinations'][:100]}...")

    print("\n  By ARI:")
    best_ari = df_results.loc[df_results['ari'].idxmax()]
    print(f"    Config: {best_ari['config_name']}")
    print(f"    Combinations: {best_ari['n_combinations_selected']} ({best_ari['data_reduction_pct']:.1f}% reduction)")
    print(f"    ARI: {best_ari['ari']:.4f}")

print("\n RESULTS SAVED TO:")
print(f"  Main Directory: {results_dir}")
print(f"  \n  Excel file: wavelength_selection_results.xlsx")
print(f"    - Configuration_Results: Main results with excitation-emission combinations")
if not df_combinations.empty:
    print(f"    - Wavelength_Combinations: Detailed wavelength pairs per configuration")
    print(f"    - Combination_Frequency: Most commonly selected wavelength pairs")
    print(f"    - Excitation_Popularity: Most useful excitation wavelengths")
print(f"  \n  CSV file: wavelength_selection_results.csv")
print(f"  \n  Visualizations: {visualizations_dir}")
print(f"    - {len(list(visualizations_dir.glob('*.png')))} comparison images saved")
print(f"  \n  Paper Results: {paper_results_dir}")
print(f"    - Individual clustering result images for each configuration (high DPI)")
print(f"    - {len(list(paper_results_dir.glob('*.png')))} individual result images")
print(f"  \n  Concatenated Data: {concat_data_dir}")
print(f"    - CSV files with concatenated hyperspectral data for each configuration")
print(f"    - {len(list(concat_data_dir.glob('*.csv')))} data files exported")

print("\n Pipeline completed successfully!")
print("\nKey Features for Paper Publication:")
print("   1. Individual clustering result images in paper-results folder (300 DPI)")
print("   2. Ground truth comparison difference maps for each configuration")
print("   3. ROI-based color consistency in clustering visualizations")
print("   4. Comprehensive experiment logs with step-by-step execution traces")
print("   5. Dimension/wavelength rankings (before & after diversity filtering)")
print("   6. Mathematical formulations in LaTeX format")
print("   7. Concatenated datasets exported to concat-data folder for further analysis")
print("   8. Excitation-emission wavelength combinations clearly tracked in Excel")

print("\n" + "=" * 80)
print("CREATING ENHANCED VISUALIZATIONS")
print("=" * 80)

print("\n📊 Creating difference map visualization...")

best_idx = df_results['purity'].idxmax()
best_config = df_results.loc[best_idx]
best_config_name = best_config['config_name']

print(f"  Best configuration: {best_config_name}")
print(f"  Purity: {best_config['purity']:.4f}")

if best_config_name in cluster_maps and 'BASELINE_FULL_DATA' in cluster_maps:
    print("\n  Generating comprehensive difference map...")

    mask = np.ones_like(ground_truth, dtype=bool)

    diff_stats = create_enhanced_difference_map(
        baseline_labels=cluster_maps['BASELINE_FULL_DATA'],
        optimized_labels=cluster_maps[best_config_name],
        mask=mask,
        output_path=visualizations_dir / "noise_reduction_analysis.png",
        title=f"Noise Reduction Analysis: Baseline vs {best_config_name}"
    )

    print(f"\n  ✅ Difference map created!")
    print(f"     Agreement: {diff_stats['agreement_pct']:.2f}%")
    print(f"     Changed pixels (noise reduced): {diff_stats['different_pixels']:,}")
    print(f"     Noise reduction rate: {diff_stats['noise_reduction_pct']:.2f}%")

    print("\n  Generating simple overlay...")
    create_simple_difference_overlay(
        baseline_labels=cluster_maps['BASELINE_FULL_DATA'],
        optimized_labels=cluster_maps[best_config_name],
        mask=mask,
        output_path=visualizations_dir / "noise_reduction_simple.png"
    )
else:
    print("  ⚠️ Cluster maps not available for difference visualization")

print("\n⏱️ Creating speed comparison visualization...")

from performance_timing_tracker import create_speed_comparison_visualization

create_speed_comparison_visualization(
    timing_tracker,
    output_path=visualizations_dir / "speed_performance_analysis.png",
    dpi=300
)

timing_csv = results_dir / "performance_timings.csv"
timing_tracker.save_to_csv(timing_csv)

print("\n" + "=" * 80)
print("📊 PERFORMANCE SUMMARY")
print("=" * 80)

stats = timing_tracker.get_summary_stats()

if 'clustering_full' in stats and 'clustering_subset' in stats:
    full_time = stats['clustering_full']['mean']
    subset_time = stats['clustering_subset']['mean']
    speedup = full_time / subset_time if subset_time > 0 else 0
    time_saved = full_time - subset_time
    time_saved_pct = (time_saved / full_time) * 100 if full_time > 0 else 0

    print(f"\n⚡ CLUSTERING PERFORMANCE:")
    print(f"  Full dataset: {full_time:.2f}s")
    print(f"  Optimized subset: {subset_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Time saved: {time_saved:.2f}s ({time_saved_pct:.1f}% reduction)")

if 'wavelength_selection' in stats:
    sel_stats = stats['wavelength_selection']
    print(f"\n📊 WAVELENGTH SELECTION:")
    print(f"  Average time: {sel_stats['mean']:.2f}s ± {sel_stats['std']:.2f}s")
    print(f"  Total time: {sel_stats['total']:.2f}s")
    print(f"  Number of runs: {sel_stats['count']}")

# Print best configuration with timing
print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Name: {best_config['config_name']}")
print(f"  Purity: {best_config['purity']:.4f}")
if 'selection_time' in best_config:
    print(f"  Selection time: {best_config['selection_time']:.2f}s")
if 'clustering_time' in best_config:
    print(f"  Clustering time: {best_config['clustering_time']:.2f}s")
if 'speedup_factor' in best_config:
    print(f"  Speedup: {best_config['speedup_factor']:.2f}x")

print("\n" + "=" * 80)
print("✅ ENHANCED VISUALIZATIONS COMPLETE")
print("=" * 80)

print(f"\n📁 Saved files:")
print(f"  • {visualizations_dir / 'noise_reduction_analysis.png'}")
print(f"  • {visualizations_dir / 'noise_reduction_simple.png'}")
print(f"  • {visualizations_dir / 'speed_performance_analysis.png'}")
print(f"  • {timing_csv}")
print(f"  • {cluster_maps_file}")

# ==================================================================
# NEW: Export comprehensive summary Excel for ALL experiments
# ==================================================================
print("\n" + "=" * 80)
print("EXPORTING COMPREHENSIVE SUMMARY")
print("=" * 80)

# Export summary of all experiments
summary_excel_path = results_dir / "ALL_EXPERIMENTS_SUMMARY.xlsx"
export_all_experiments_summary(
    all_results=results,
    output_path=summary_excel_path
)

print(f"\n✅ Comprehensive summary exported!")
print(f"   {summary_excel_path}")
print(f"   Contains: All results, timing data, top 10, wavelength info, statistics")

print("\n🎉 Ready for paper preparation!")