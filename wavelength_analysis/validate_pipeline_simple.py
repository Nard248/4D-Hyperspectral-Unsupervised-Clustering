"""
Simple validation script - runs 3 configurations to test the pipeline
Uses the exact same code as the working notebook
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Setup paths
base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")
sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "wavelength_analysis"))

# Import modules
from wavelength_analysis.core.config import AnalysisConfig
from wavelength_analysis.core.analyzer import WavelengthAnalyzer
from concatenation_clustering import (
    load_masked_data,
    concatenate_hyperspectral_data_improved
)
from ground_truth_validation import (
    extract_ground_truth_from_png,
    calculate_clustering_accuracy
)

# Results directory
RESULTS_DIR = Path("results/validation_test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PIPELINE VALIDATION - 3 CONFIGURATIONS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading data...")
sample_name = "Lichens"
data_path = base_dir / "data" / "processed" / sample_name / "lichens_data_masked.pkl"
mask_path = base_dir / "data" / "processed" / sample_name / "lichens_mask.npy"
png_path = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")

full_data = load_masked_data(data_path)
print(f"  Loaded {len(full_data['excitation_wavelengths'])} excitation wavelengths")

# Extract ground truth
ground_truth, _, _ = extract_ground_truth_from_png(
    png_path,
    background_colors=[(24, 24, 24, 255), (168, 168, 168, 255)],
    target_shape=(1040, 1392)
)

# Crop to working area
start_col = 1392 - 925
end_col = 1392
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

full_data = cropped_data
ground_truth = ground_truth[:, start_col:end_col]
print(f"  Cropped to {ground_truth.shape}")

# ============================================================================
# DEFINE ROI REGIONS
# ============================================================================

ROI_REGIONS = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150)},
    {'name': 'Region 2', 'coords': (175, 225, 250, 300)},
    {'name': 'Region 3', 'coords': (175, 225, 425, 475)},
    {'name': 'Region 4', 'coords': (175, 225, 650, 700)},
]

# ============================================================================
# CLUSTERING FUNCTION (copied from notebook)
# ============================================================================

def run_clustering(data, roi_regions):
    """Run KNN-based clustering"""

    # Step 1: Concatenate data
    df, valid_mask, metadata = concatenate_hyperspectral_data_improved(
        data,
        global_normalize=True,
        normalization_method='global_percentile'
    )

    # Step 2: Extract ROI training data
    spectral_cols = [col for col in df.columns if col not in ['x', 'y']]
    X_train_list = []
    y_train_list = []

    label_mapping = {roi['name']: i for i, roi in enumerate(roi_regions)}

    for roi in roi_regions:
        roi_name = roi['name']
        y_start, y_end, x_start, x_end = roi['coords']

        roi_mask = (
            (df['x'] >= x_start) & (df['x'] < x_end) &
            (df['y'] >= y_start) & (df['y'] < y_end)
        )

        roi_pixels = df[roi_mask]

        if len(roi_pixels) > 0:
            roi_spectra = roi_pixels[spectral_cols].values
            roi_labels = [label_mapping[roi_name]] * len(roi_pixels)

            X_train_list.append(roi_spectra)
            y_train_list.extend(roi_labels)

    if not X_train_list:
        print("  WARNING: No ROI data, falling back to KMeans")
        X = df[spectral_cols].values
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        predictions = kmeans.fit_predict(X)
    else:
        # Step 3: Train KNN
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn_model.fit(X_train_scaled, y_train)

        # Step 4: Predict
        X_full = df[spectral_cols].values
        X_full_scaled = scaler.transform(X_full)
        predictions = knn_model.predict(X_full_scaled)

    # Step 5: Reconstruct cluster map
    height, width = metadata['height'], metadata['width']
    cluster_map = np.full((height, width), -1, dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        x, y = int(row['x']), int(row['y'])
        cluster_map[y, x] = predictions[i]

    return cluster_map, len(spectral_cols)

# ============================================================================
# WAVELENGTH SELECTION FUNCTION (copied from notebook)
# ============================================================================

def select_wavelengths(data_path, mask_path, sample_name, config_params):
    """Select informative wavelengths"""

    model_dir = base_dir / "results" / f"{sample_name}_wavelength_selection" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "autoencoder_model.pth"

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
        output_dir=str(model_dir.parent / "output"),
        save_tiff_layers=False,
        save_visualizations=False,
        n_baseline_patches=10
    )

    analyzer = WavelengthAnalyzer(config)
    analyzer.load_data_and_model()
    results = analyzer.run_complete_analysis()

    if results is None or 'selected_bands' not in results:
        return None

    selected_bands_raw = results['selected_bands']
    emission_wavelengths = []

    for band in selected_bands_raw:
        if isinstance(band, dict):
            emission = band.get('emission_wavelength', 'unknown')
            if hasattr(emission, 'item'):
                emission = float(emission.item())
            else:
                emission = float(emission)
            emission_wavelengths.append(emission)

    # Remove duplicates
    unique_emissions = list(dict.fromkeys(emission_wavelengths))

    return unique_emissions

def extract_subset(full_data, selected_wavelengths):
    """Extract subset of data using selected wavelengths"""

    subset_data = {
        'data': {},
        'metadata': full_data.get('metadata', {}),
        'excitation_wavelengths': full_data['excitation_wavelengths']
    }

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]

        original_wavelengths = np.array(ex_data['wavelengths'])
        original_cube = ex_data['cube']

        selected_indices = []
        for target_wl in selected_wavelengths:
            distances = np.abs(original_wavelengths - target_wl)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 10 and closest_idx not in selected_indices:
                selected_indices.append(closest_idx)

        if selected_indices:
            subset_cube = original_cube[:, :, selected_indices]
            subset_data['data'][ex_str] = {
                'cube': subset_cube,
                'wavelengths': original_wavelengths[selected_indices].tolist(),
                **{k: v for k, v in ex_data.items() if k not in ['cube', 'wavelengths']}
            }
        else:
            subset_data['data'][ex_str] = ex_data

    return subset_data

# ============================================================================
# RUN BASELINE
# ============================================================================

print("\n2. BASELINE: Full data clustering")
baseline_map, baseline_features = run_clustering(full_data, ROI_REGIONS)
baseline_metrics = calculate_clustering_accuracy(
    baseline_map,
    ground_truth,
    np.ones_like(ground_truth, dtype=bool)
)

print(f"  Features: {baseline_features}")
print(f"  Purity: {baseline_metrics['purity']:.4f}")

# ============================================================================
# TEST 3 CONFIGURATIONS
# ============================================================================

test_configs = [
    {
        'name': 'Test1_Variance',
        'dimension_selection_method': 'variance',
        'perturbation_method': 'standard_deviation',
        'perturbation_magnitudes': [15, 30, 45],
        'n_important_dimensions': 10,
        'n_bands_to_select': 10,
        'normalization_method': 'max_per_excitation'
    },
    {
        'name': 'Test2_Activation',
        'dimension_selection_method': 'activation',
        'perturbation_method': 'percentile',
        'perturbation_magnitudes': [10, 20, 30],
        'n_important_dimensions': 10,
        'n_bands_to_select': 10,
        'normalization_method': 'variance'
    },
    {
        'name': 'Test3_PCA',
        'dimension_selection_method': 'pca',
        'perturbation_method': 'absolute_range',
        'perturbation_magnitudes': [20, 40, 60],
        'n_important_dimensions': 8,
        'n_bands_to_select': 10,
        'normalization_method': 'variance'
    }
]

results = []

for i, config in enumerate(test_configs):
    print(f"\n{i+1}. Testing: {config['name']}")

    try:
        # Select wavelengths
        selected_wls = select_wavelengths(data_path, mask_path, sample_name, config)

        if selected_wls is None:
            print("  ERROR: Wavelength selection failed")
            continue

        print(f"  Selected {len(selected_wls)} wavelengths")

        # Extract subset
        subset_data = extract_subset(full_data, selected_wls)

        # Cluster
        cluster_map, n_features = run_clustering(subset_data, ROI_REGIONS)

        # Evaluate
        metrics = calculate_clustering_accuracy(
            cluster_map,
            ground_truth,
            np.ones_like(ground_truth, dtype=bool)
        )

        data_reduction = (1 - n_features / baseline_features) * 100

        print(f"  Features: {n_features} (reduction: {data_reduction:.1f}%)")
        print(f"  Purity: {metrics['purity']:.4f}")
        print(f"  vs Baseline: {metrics['purity'] - baseline_metrics['purity']:+.4f}")

        results.append({
            'config': config['name'],
            'wavelengths': len(selected_wls),
            'features': n_features,
            'purity': metrics['purity'],
            'reduction_pct': data_reduction
        })

    except Exception as e:
        print(f"  ERROR: {str(e)}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)

if results:
    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string(index=False))

    # Save
    csv_path = RESULTS_DIR / "validation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # Quick plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(df)), df['purity'], alpha=0.7)
    ax.axhline(y=baseline_metrics['purity'], color='red', linestyle='--',
              label=f'Baseline ({baseline_metrics["purity"]:.3f})')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.set_ylabel('Purity Score')
    ax.set_title('Validation Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_plot.png", dpi=150)
    print(f"Plot saved to: {RESULTS_DIR / 'validation_plot.png'}")

    print("\n✓ Pipeline validated successfully!")
else:
    print("\n✗ No results generated")
