"""
Simple script to add PCA wavelength selection to V2-2.
This modifies wavelengthSelectionV2-2.py to support both autoencoder and PCA methods.
"""

import re

# Read the V2-2 file
with open('wavelengthSelectionV2-2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add PCA function after select_informative_wavelengths_fixed
pca_function = '''

def select_wavelengths_pca(data_path, mask_path, sample_name, config_params, verbose=True):
    """
    PCA-based wavelength selection (NO AUTOENCODER).
    Replaces autoencoder with simple PCA. Returns same format as select_informative_wavelengths_fixed.
    """
    import pickle
    from sklearn.decomposition import PCA

    n_bands_to_select = config_params.get('n_bands_to_select', 30)

    if verbose:
        print(f"\\nRunning PCA wavelength selection: {config_params.get('name', 'unnamed')}")
        print(f"  Target bands: {n_bands_to_select}")

    # Load data
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)

    # Create valid pixel mask
    first_ex = full_data['excitation_wavelengths'][0]
    first_cube = full_data['data'][str(first_ex)]['cube']
    valid_mask_flat = ~np.isnan(first_cube[:, :, 0]).flatten()

    # Flatten data
    all_spectra, wavelength_info = [], []
    for ex in full_data['excitation_wavelengths']:
        ex_data = full_data['data'][str(ex)]
        cube, wavelengths = ex_data['cube'], ex_data['wavelengths']
        flattened = cube.reshape(-1, cube.shape[2])
        all_spectra.append(flattened[valid_mask_flat, :])
        for em_idx, em_wavelength in enumerate(wavelengths):
            wavelength_info.append({
                'excitation': float(ex),
                'emission': float(em_wavelength)
            })

    X = np.hstack(all_spectra)
    if verbose:
        print(f"  Data shape: {X.shape}, Valid pixels: {np.sum(valid_mask_flat)}")

    # PCA selection
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    pca = PCA(n_components=min(n_bands_to_select, X.shape[1]))
    pca.fit(X_std)

    loadings = np.abs(pca.components_)
    selected_indices = []
    for comp_idx in range(pca.n_components_):
        for band_idx in np.argsort(loadings[comp_idx, :])[::-1]:
            if band_idx not in selected_indices and len(selected_indices) < n_bands_to_select:
                selected_indices.append(band_idx)
            if len(selected_indices) >= n_bands_to_select:
                break

    # Convert to wavelength combinations
    wavelength_combinations, emission_wavelengths_only = [], []
    seen_combinations = set()

    for idx in selected_indices:
        info = wavelength_info[idx]
        combo_key = (info['excitation'], info['emission'])
        if combo_key not in seen_combinations:
            seen_combinations.add(combo_key)
            combo = {
                'excitation': info['excitation'],
                'emission': info['emission'],
                'combination_name': f"Ex{info['excitation']:.0f}_Em{info['emission']:.1f}"
            }
            wavelength_combinations.append(combo)
            emission_wavelengths_only.append(info['emission'])

    if verbose:
        print(f"  Selected {len(wavelength_combinations)} unique combinations")

    results = {'selected_bands': wavelength_combinations, 'method': 'pca'}
    return wavelength_combinations, emission_wavelengths_only, results
'''

# Insert PCA function after select_informative_wavelengths_fixed
pattern = r'(    return unique_combinations, unique_emissions, results\n\n)(def extract_wavelength_subset)'
content = re.sub(pattern, r'\1' + pca_function + r'\n\2', content)

# 2. Modify main() signature to accept selection_method
content = re.sub(
    r'def main\(max_configs=None\):',
    'def main(max_configs=None, selection_method="autoencoder"):',
    content
)

# 3. Add conditional selection in main() loop
pattern = r'(\s+# Step 1: Wavelength selection\n\s+with PerformanceTimer\(\) as selection_timer:\n)(\s+wavelength_combinations, emission_wavelengths_only, selection_results = select_informative_wavelengths_fixed\()'
replacement = r'''\1\2                if selection_method == "pca":
                    wavelength_combinations, emission_wavelengths_only, selection_results = select_wavelengths_pca(
                        data_path, mask_path, sample_name, config, verbose=False
                    )
                else:
                    wavelength_combinations, emission_wavelengths_only, selection_results = select_informative_wavelengths_fixed('''

content = re.sub(pattern, replacement, content)

# 4. Update __main__ to parse selection method
main_section = '''if __name__ == "__main__":
    import sys

    selection_method = "autoencoder"  # default
    max_configs = None

    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["autoencoder", "pca"]:
            selection_method = sys.argv[1]
            if len(sys.argv) > 2:
                try:
                    max_configs = int(sys.argv[2])
                except ValueError:
                    print("Usage: python wavelengthselectionV2-2.py [autoencoder|pca] [max_configs]")
                    sys.exit(1)
        else:
            try:
                max_configs = int(sys.argv[1])
            except ValueError:
                print("Usage: python wavelengthselectionV2-2.py [autoencoder|pca] [max_configs]")
                sys.exit(1)

    print(f"Running with selection_method={selection_method}, max_configs={max_configs}")
    main(max_configs=max_configs, selection_method=selection_method)'''

content = re.sub(r'if __name__ == "__main__":.*?main\(\)', main_section, content, flags=re.DOTALL)

# Write modified file
with open('wavelengthSelectionV2-2-PCA.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("[SUCCESS] Created wavelengthSelectionV2-2-PCA.py with PCA support")
print("\\nUsage:")
print("  python wavelengthSelectionV2-2-PCA.py autoencoder 10  # Run autoencoder with 10 configs")
print("  python wavelengthSelectionV2-2-PCA.py pca 10          # Run PCA with 10 configs")
