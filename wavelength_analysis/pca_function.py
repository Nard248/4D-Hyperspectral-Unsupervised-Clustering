

def select_wavelengths_pca(data_path, mask_path, sample_name, config_params, verbose=True):
    """
    PCA-based wavelength selection (NO AUTOENCODER).

    Replaces autoencoder + perturbation with simple PCA on normalized data.
    Returns same format as select_informative_wavelengths_fixed for compatibility.
    """
    n_bands_to_select = config_params.get('n_bands_to_select', 30)

    if verbose:
        print(f"\nRunning PCA wavelength selection: {config_params.get('name', 'unnamed')}")
        print(f"  Target bands: {n_bands_to_select}")

    # Load data
    import pickle
    with open(data_path, 'rb') as f:
        full_data = pickle.load(f)

    # Create valid pixel mask from data (non-NaN pixels)
    first_ex = full_data['excitation_wavelengths'][0]
    first_cube = full_data['data'][str(first_ex)]['cube']
    first_slice = first_cube[:, :, 0]
    valid_mask_2d = ~np.isnan(first_slice)
    valid_mask_flat = valid_mask_2d.flatten()

    # Flatten hyperspectral data to 2D matrix (valid_pixels × wavelength_combinations)
    all_spectra = []
    wavelength_info = []

    for ex in full_data['excitation_wavelengths']:
        ex_str = str(ex)
        ex_data = full_data['data'][ex_str]
        cube = ex_data['cube']
        wavelengths = ex_data['wavelengths']

        # Flatten and filter valid pixels
        n_pixels = cube.shape[0] * cube.shape[1]
        n_emissions = cube.shape[2]
        flattened = cube.reshape(n_pixels, n_emissions)
        flattened_valid = flattened[valid_mask_flat, :]

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
        print(f"  Valid pixels: {np.sum(valid_mask_flat)}")

    # Run PCA
    from sklearn.decomposition import PCA
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    n_components = min(n_bands_to_select, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_std)

    # Select bands based on PCA loadings
    loadings = np.abs(pca.components_)
    selected_indices = []
    for comp_idx in range(n_components):
        # Get band with highest loading for this component
        band_idx = np.argmax(loadings[comp_idx, :])
        if band_idx not in selected_indices:
            selected_indices.append(band_idx)

        # If we need more bands, add second-highest, etc.
        if len(selected_indices) < n_bands_to_select:
            sorted_indices = np.argsort(loadings[comp_idx, :])[::-1]
            for idx in sorted_indices:
                if idx not in selected_indices and len(selected_indices) < n_bands_to_select:
                    selected_indices.append(idx)

    # Ensure we have exactly n_bands_to_select
    if len(selected_indices) < n_bands_to_select:
        # Add remaining bands by overall variance
        variances = np.var(X_std, axis=0)
        sorted_by_var = np.argsort(variances)[::-1]
        for idx in sorted_by_var:
            if idx not in selected_indices and len(selected_indices) < n_bands_to_select:
                selected_indices.append(idx)

    selected_indices = selected_indices[:n_bands_to_select]

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

    if verbose:
        print(f"  Selected {len(unique_combinations)} unique wavelength combinations")
        if unique_combinations:
            print(f"  First few: {[c['combination_name'] for c in unique_combinations[:3]]}...")

    # Create dummy results dict for compatibility
    results = {
        'selected_bands': unique_combinations,
        'method': 'pca',
        'n_components': n_components
    }

    return unique_combinations, unique_emissions, results

