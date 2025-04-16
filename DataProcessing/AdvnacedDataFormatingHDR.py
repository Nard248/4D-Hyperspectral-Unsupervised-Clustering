import numpy as np
import os
import glob
import re
import h5py
import datetime
from pathlib import Path


def read_hdr_file(hdr_file):
    """
    Read an ENVI header file

    Args:
        hdr_file: Path to the header file

    Returns:
        header_dict: Dictionary containing header information
    """
    header_dict = {}
    with open(hdr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                header_dict[key.strip()] = value.strip()
    return header_dict


def read_bin_file(bin_file, header_dict):
    """
    Read binary data file using header information

    Args:
        bin_file: Path to the binary file
        header_dict: Dictionary containing header information

    Returns:
        data: Numpy array containing the data
    """
    # Extract dimensions from header
    samples = int(header_dict.get('samples', 0))
    lines = int(header_dict.get('lines', 0))
    bands = int(header_dict.get('bands', 0))

    # Extract data type
    data_type_str = header_dict.get('data type', '4')  # Default to float32
    data_type_map = {
        '1': np.uint8,
        '2': np.int16,
        '3': np.int32,
        '4': np.float32,
        '5': np.float64,
        '6': np.complex64,
        '9': np.complex128,
        '12': np.uint16,
        '13': np.uint32,
        '14': np.int64,
        '15': np.uint64
    }
    data_type = data_type_map.get(data_type_str, np.float32)

    # Extract interleave format
    interleave = header_dict.get('interleave', 'bsq').lower()

    # Read binary data
    with open(bin_file, 'rb') as f:
        data = np.fromfile(f, dtype=data_type)

    # Reshape based on interleave format
    if interleave == 'bsq':  # Band Sequential
        data = data.reshape((bands, lines, samples))
    elif interleave == 'bil':  # Band Interleave by Line
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (1, 0, 2))  # Reorder to (bands, lines, samples)
    elif interleave == 'bip':  # Band Interleave by Pixel
        data = data.reshape((lines, samples, bands))
        data = np.transpose(data, (2, 0, 1))  # Reorder to (bands, lines, samples)

    return data


def read_hyperspectral_data(folder_path):
    """
    Read hyperspectral data from a folder

    Args:
        folder_path: Path to folder containing spectral_image_processed_image.hdr and .bin

    Returns:
        data: The hyperspectral data cube
        wavelengths: Array of wavelengths
        header: Dictionary containing header information
    """
    # Read the header file
    hdr_file = os.path.join(folder_path, 'spectral_image_processed_image.hdr')
    header = read_hdr_file(hdr_file)

    # Read the binary data file
    bin_file = os.path.join(folder_path, 'spectral_image_processed_image.bin')
    data = read_bin_file(bin_file, header)

    # Read the wavelengths file
    wavelengths_file = os.path.join(folder_path, 'spectral_image_wavelengths.csv')
    wavelengths = np.loadtxt(wavelengths_file, delimiter=',')

    return data, wavelengths, header


def extract_info_from_folder_name(folder_name):
    """
    Extract excitation wavelength and index from folder name

    Args:
        folder_name: Name of the folder (e.g., "Kiwi 2_03-25_300_1")

    Returns:
        excitation_wavelength: The excitation wavelength as integer
        index: The index number
    """
    # Extract excitation wavelength and index using regex
    match = re.search(r'_(\d+)_(\d+)$', folder_name)
    if match:
        excitation_wavelength = int(match.group(1))
        index = int(match.group(2))
        return excitation_wavelength, index
    return None, None


def organize_data_by_excitation(base_path):
    """
    Organize all hyperspectral data by excitation wavelength

    Args:
        base_path: Path to the base directory containing all folders

    Returns:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
    """
    organized_data = {}

    # Find all folders
    folders = glob.glob(os.path.join(base_path, "Kiwi 2_03-25_*_*"))

    for folder in folders:
        folder_name = os.path.basename(folder)
        excitation_wavelength, index = extract_info_from_folder_name(folder_name)

        if excitation_wavelength is not None:
            try:
                # Read data from this folder
                data, wavelengths, header = read_hyperspectral_data(folder)

                # Store in organized structure
                if excitation_wavelength not in organized_data:
                    organized_data[excitation_wavelength] = []

                organized_data[excitation_wavelength].append({
                    'data': data,
                    'wavelengths': wavelengths,
                    'header': header,
                    'index': index
                })
                print(f"Processed: {folder_name} - Shape: {data.shape}")
            except Exception as e:
                print(f"Error reading data from {folder}: {str(e)}")

    return organized_data


def apply_spectral_cutoff(data, wavelengths, excitation, cutoff_offset=30):
    """
    Apply spectral cutoff to remove excitation artifacts:
    1. Remove emission wavelengths < (excitation + cutoff_offset)
    2. Remove wavelengths in the second-order scattering region (2*excitation ± cutoff_offset)

    Args:
        data: Hyperspectral data cube (bands, height, width)
        wavelengths: Array of emission wavelengths
        excitation: Excitation wavelength
        cutoff_offset: Offset in nm (default: 30)

    Returns:
        filtered_data: Data after applying cutoff
        filtered_wavelengths: Wavelengths after applying cutoff
        cutoff_mask: Boolean mask showing which wavelengths were kept
    """
    # Create a mask to keep valid wavelengths
    keep_mask = np.ones(len(wavelengths), dtype=bool)

    # Remove wavelengths before excitation + cutoff_offset
    keep_mask = np.logical_and(keep_mask, wavelengths >= excitation + cutoff_offset)

    # Remove wavelengths in the second-order zone (2*excitation ± cutoff_offset)
    second_order_min = 2 * excitation - cutoff_offset
    second_order_max = 2 * excitation + cutoff_offset
    second_order_mask = np.logical_or(wavelengths < second_order_min, wavelengths > second_order_max)
    keep_mask = np.logical_and(keep_mask, second_order_mask)

    # Apply the mask
    filtered_data = data[keep_mask, :, :]
    filtered_wavelengths = wavelengths[keep_mask]

    return filtered_data, filtered_wavelengths, keep_mask


def extract_reflectance(data, wavelengths, excitation, method='interpolate', valid_range=(400, 500)):
    """
    Extract reflectance data (peaks where excitation~=emission)
    Using interpolation between the two closest emission wavelengths

    Args:
        data: Hyperspectral data cube (bands, height, width)
        wavelengths: Array of emission wavelengths
        excitation: Excitation wavelength
        method: Either 'closest' (just pick closest wavelength) or
                'interpolate' (interpolate between two closest wavelengths)
        valid_range: Tuple (min, max) defining the valid range for reflectance extraction

    Returns:
        reflectance: 2D array of reflectance data or None if excitation is out of valid range
        wavelength: The wavelength used for reflectance (or interpolated wavelength)
        indices: Indices of the wavelengths used (one or two depending on method)
        valid: Boolean indicating if the reflectance is valid (excitation in valid range)
    """
    # Check if excitation is within the valid range
    min_valid, max_valid = valid_range
    if excitation < min_valid or excitation > max_valid:
        # Return None if excitation is outside the valid range
        return None, excitation, None, False

    # Find the closest wavelength index
    closest_idx = np.argmin(np.abs(wavelengths - excitation))

    if method == 'closest':
        # Just return the closest wavelength's data
        reflectance = data[closest_idx, :, :]
        wavelength = wavelengths[closest_idx]
        return reflectance, wavelength, closest_idx, True

    elif method == 'interpolate':
        # Find the two nearest wavelengths (one on each side of excitation if possible)
        diff = wavelengths - excitation

        # If the closest wavelength is exactly the excitation, just return that
        if diff[closest_idx] == 0:
            reflectance = data[closest_idx, :, :]
            return reflectance, wavelengths[closest_idx], closest_idx, True

        # Find indices of wavelengths less than and greater than excitation
        less_indices = np.where(diff < 0)[0]
        greater_indices = np.where(diff > 0)[0]

        if len(less_indices) > 0 and len(greater_indices) > 0:
            # We have wavelengths on both sides of excitation
            # Find the closest one on each side
            lower_idx = less_indices[np.argmax(wavelengths[less_indices])]
            upper_idx = greater_indices[np.argmin(wavelengths[greater_indices])]

            # Get the wavelengths and corresponding data
            lower_wl = wavelengths[lower_idx]
            upper_wl = wavelengths[upper_idx]

            # Calculate weights for interpolation based on distance
            lower_weight = (upper_wl - excitation) / (upper_wl - lower_wl)
            upper_weight = (excitation - lower_wl) / (upper_wl - lower_wl)

            # Perform the weighted interpolation
            reflectance = lower_weight * data[lower_idx, :, :] + upper_weight * data[upper_idx, :, :]

            return reflectance, excitation, (lower_idx, upper_idx), True
        else:
            # We only have wavelengths on one side of excitation
            # Just return the closest one
            reflectance = data[closest_idx, :, :]
            return reflectance, wavelengths[closest_idx], closest_idx, True
    else:
        # Default to closest if method not recognized
        reflectance = data[closest_idx, :, :]
        return reflectance, wavelengths[closest_idx], closest_idx, True


def save_to_hdf5(organized_data, output_file, cutoff_offset=30):
    """
    Save organized hyperspectral data to HDF5 format with advanced processing

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output HDF5 file
        cutoff_offset: Offset in nm for spectral cutoff (default: 30)
    """
    with h5py.File(output_file, 'w') as f:
        # Dictionary to store reflectance data and coordinate information
        reflectance_dict = {}
        reflectance_wavelengths = {}
        reflectance_indices = {}

        # Create a group for each excitation wavelength
        for excitation_wavelength, data_list in organized_data.items():
            group_name = f'excitation_{excitation_wavelength}'
            group = f.create_group(group_name)

            # Store wavelengths (should be the same for all data at this excitation)
            wavelengths = data_list[0]['wavelengths']
            group.create_dataset('wavelengths', data=wavelengths)

            # Store each data cube with cutoff applied
            for data_item in data_list:
                index = data_item['index']
                data_cube = data_item['data']
                header = data_item['header']

                # Create a subgroup for this cube
                cube_group = group.create_group(f'cube_{index}')

                # Store original data
                cube_group.create_dataset('data', data=data_cube, compression='gzip', chunks=True)

                # Apply spectral cutoff to individual cube
                filtered_cube, filtered_wavelengths, cutoff_mask = apply_spectral_cutoff(
                    data_cube, wavelengths, excitation_wavelength, cutoff_offset)

                # Store filtered data
                cube_group.create_dataset('filtered_data', data=filtered_cube, compression='gzip', chunks=True)
                cube_group.create_dataset('filtered_wavelengths', data=filtered_wavelengths)
                cube_group.create_dataset('cutoff_mask', data=cutoff_mask)

                # Store header information as attributes
                for key, value in header.items():
                    try:
                        # Try to convert to appropriate types when possible
                        if key in ['samples', 'lines', 'bands', 'data type']:
                            cube_group.attrs[key] = int(value)
                        elif key in ['wavelength units']:
                            cube_group.attrs[key] = str(value)
                        else:
                            cube_group.attrs[key] = value
                    except:
                        # If conversion fails, store as string
                        cube_group.attrs[key] = str(value)

            # Calculate and store the average and summed cubes
            if len(data_list) > 1:
                # Average cube
                avg_cube = np.mean([item['data'] for item in data_list], axis=0)
                group.create_dataset('average_cube', data=avg_cube, compression='gzip', chunks=True)

                # Summed cube (new requirement)
                sum_cube = np.sum([item['data'] for item in data_list], axis=0)
                group.create_dataset('sum_cube', data=sum_cube, compression='gzip', chunks=True)

                # Apply spectral cutoff to average cube
                filtered_avg, filtered_wl, avg_mask = apply_spectral_cutoff(
                    avg_cube, wavelengths, excitation_wavelength, cutoff_offset)
                group.create_dataset('filtered_average_cube', data=filtered_avg, compression='gzip', chunks=True)
                group.create_dataset('filtered_wavelengths', data=filtered_wl)
                group.create_dataset('cutoff_mask', data=avg_mask)

                # Apply spectral cutoff to summed cube
                filtered_sum, _, _ = apply_spectral_cutoff(
                    sum_cube, wavelengths, excitation_wavelength, cutoff_offset)
                group.create_dataset('filtered_sum_cube', data=filtered_sum, compression='gzip', chunks=True)

                # Extract reflectance data from average cube (peaks where excitation=emission)
                refl_data, refl_wl, refl_idx = extract_reflectance(avg_cube, wavelengths, excitation_wavelength)
                group.create_dataset('reflectance', data=refl_data, compression='gzip', chunks=True)
                group.attrs['reflectance_wavelength'] = refl_wl
                group.attrs['reflectance_index'] = refl_idx

                # Store for creating the reflectance cube
                reflectance_dict[excitation_wavelength] = refl_data
                reflectance_wavelengths[excitation_wavelength] = refl_wl
                reflectance_indices[excitation_wavelength] = refl_idx

        # Create the ReflectanceCube by combining reflectance from all excitations
        if reflectance_dict:
            # Sort excitation wavelengths
            excitations = sorted(reflectance_dict.keys())

            # Check if all reflectance data has the same shape
            shapes = [reflectance_dict[ex].shape for ex in excitations]
            if len(set(shapes)) == 1:  # All shapes are the same
                # Stack reflectance data from all excitations
                reflectance_cube = np.stack([reflectance_dict[ex] for ex in excitations])

                # Create a group for the reflectance cube
                refl_group = f.create_group('reflectance_cube')
                refl_group.create_dataset('data', data=reflectance_cube, compression='gzip', chunks=True)

                # Store the excitation wavelengths as the "bands" of the reflectance cube
                refl_group.create_dataset('excitation_wavelengths', data=np.array(excitations))

                # Store the emission wavelength for each excitation
                refl_wls = np.array([reflectance_wavelengths[ex] for ex in excitations])
                refl_group.create_dataset('emission_wavelengths', data=refl_wls)

                # Store the emission wavelength indices
                refl_indices = np.array([reflectance_indices[ex] for ex in excitations])
                refl_group.create_dataset('emission_indices', data=refl_indices)
            else:
                print("Warning: Reflectance data has inconsistent shapes, skipping ReflectanceCube creation")

        # Store general metadata
        metadata = f.create_group('metadata')
        metadata.attrs['description'] = 'Hyperspectral image data with spectral cutoff and reflectance'
        metadata.attrs['date_created'] = str(datetime.datetime.now().isoformat())
        metadata.attrs['num_excitations'] = len(organized_data)
        metadata.attrs['cubes_per_excitation'] = len(data_list) if data_list else 0
        metadata.attrs['cutoff_offset'] = cutoff_offset


def read_from_hdf5(file_path):
    """
    Read hyperspectral data from HDF5 file

    Args:
        file_path: Path to the HDF5 file

    Returns:
        data_dict: Dictionary containing the hyperspectral data
    """
    data_dict = {}

    with h5py.File(file_path, 'r') as f:
        # Read each excitation group
        for group_name in f.keys():
            if group_name.startswith('excitation_'):
                # Extract excitation wavelength from group name
                excitation_wavelength = int(group_name.split('_')[1])

                group = f[group_name]
                excitation_data = {
                    'wavelengths': group['wavelengths'][:],
                    'cubes': {}
                }

                # Read individual cubes
                for dataset_name in group.keys():
                    if dataset_name.startswith('cube_'):
                        index = int(dataset_name.split('_')[1])
                        cube_group = group[dataset_name]
                        cube_data = cube_group['data'][:]

                        # Extract header information
                        header = {key: cube_group.attrs[key] for key in cube_group.attrs}

                        # Include filtered data if available
                        cube_dict = {
                            'data': cube_data,
                            'header': header
                        }

                        if 'filtered_data' in cube_group:
                            cube_dict['filtered_data'] = cube_group['filtered_data'][:]

                        if 'filtered_wavelengths' in cube_group:
                            cube_dict['filtered_wavelengths'] = cube_group['filtered_wavelengths'][:]

                        excitation_data['cubes'][index] = cube_dict
                    elif dataset_name == 'average_cube':
                        excitation_data['average_cube'] = group[dataset_name][:]
                    elif dataset_name == 'sum_cube':
                        excitation_data['sum_cube'] = group[dataset_name][:]
                    elif dataset_name == 'filtered_average_cube':
                        excitation_data['filtered_average_cube'] = group[dataset_name][:]
                    elif dataset_name == 'filtered_sum_cube':
                        excitation_data['filtered_sum_cube'] = group[dataset_name][:]
                    elif dataset_name == 'filtered_wavelengths':
                        excitation_data['filtered_wavelengths'] = group[dataset_name][:]
                    elif dataset_name == 'reflectance':
                        excitation_data['reflectance'] = group[dataset_name][:]

                # Get reflectance attributes
                if 'reflectance_wavelength' in group.attrs:
                    excitation_data['reflectance_wavelength'] = group.attrs['reflectance_wavelength']
                if 'reflectance_index' in group.attrs:
                    excitation_data['reflectance_index'] = group.attrs['reflectance_index']

                data_dict[excitation_wavelength] = excitation_data

        # Read the reflectance cube if available
        if 'reflectance_cube' in f:
            refl_group = f['reflectance_cube']
            reflectance_cube = {
                'data': refl_group['data'][:] if 'data' in refl_group else None,
                'excitation_wavelengths': refl_group['excitation_wavelengths'][
                                          :] if 'excitation_wavelengths' in refl_group else None,
                'emission_wavelengths': refl_group['emission_wavelengths'][
                                        :] if 'emission_wavelengths' in refl_group else None,
                'emission_indices': refl_group['emission_indices'][:] if 'emission_indices' in refl_group else None
            }
            data_dict['reflectance_cube'] = reflectance_cube

    return data_dict


def save_to_xarray(h5_file_path, output_file):
    """
    Save hyperspectral data to xarray format (NetCDF) for easier analysis

    Args:
        h5_file_path: Path to the HDF5 file
        output_file: Path to the output file (.nc for NetCDF)
    """
    try:
        import xarray as xr
    except ImportError:
        print("Error: xarray not installed. Please install with 'pip install xarray netCDF4'")
        return

    datasets = {}

    # Read the HDF5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Process each excitation wavelength
        for group_name in f:
            if group_name.startswith('excitation_'):
                excitation = int(group_name.split('_')[1])
                group = f[group_name]

                # Get emission wavelengths
                if 'wavelengths' in group:
                    emission_wls = group['wavelengths'][:]
                else:
                    continue  # Skip this group if no wavelengths

                # Process average cube
                if 'average_cube' in group:
                    avg_cube = group['average_cube'][:]

                    # Create xarray DataArray
                    avg_da = xr.DataArray(
                        avg_cube,
                        dims=['emission', 'y', 'x'],
                        coords={
                            'emission': emission_wls,
                            'y': np.arange(avg_cube.shape[1]),
                            'x': np.arange(avg_cube.shape[2]),
                            'excitation': excitation
                        },
                        name=f'average_cube_{excitation}'
                    )

                    datasets[f'average_cube_{excitation}'] = avg_da

                # Process sum cube (new requirement)
                if 'sum_cube' in group:
                    sum_cube = group['sum_cube'][:]

                    # Create xarray DataArray
                    sum_da = xr.DataArray(
                        sum_cube,
                        dims=['emission', 'y', 'x'],
                        coords={
                            'emission': emission_wls,
                            'y': np.arange(sum_cube.shape[1]),
                            'x': np.arange(sum_cube.shape[2]),
                            'excitation': excitation
                        },
                        name=f'sum_cube_{excitation}'
                    )

                    datasets[f'sum_cube_{excitation}'] = sum_da

                # Process filtered average cube
                if 'filtered_average_cube' in group and 'filtered_wavelengths' in group:
                    filtered_avg = group['filtered_average_cube'][:]
                    filtered_wls = group['filtered_wavelengths'][:]

                    # Create xarray DataArray
                    filtered_avg_da = xr.DataArray(
                        filtered_avg,
                        dims=['emission', 'y', 'x'],
                        coords={
                            'emission': filtered_wls,
                            'y': np.arange(filtered_avg.shape[1]),
                            'x': np.arange(filtered_avg.shape[2]),
                            'excitation': excitation
                        },
                        name=f'filtered_average_cube_{excitation}'
                    )

                    datasets[f'filtered_average_cube_{excitation}'] = filtered_avg_da

                # Process filtered sum cube
                if 'filtered_sum_cube' in group and 'filtered_wavelengths' in group:
                    filtered_sum = group['filtered_sum_cube'][:]
                    filtered_wls = group['filtered_wavelengths'][:]

                    # Create xarray DataArray
                    filtered_sum_da = xr.DataArray(
                        filtered_sum,
                        dims=['emission', 'y', 'x'],
                        coords={
                            'emission': filtered_wls,
                            'y': np.arange(filtered_sum.shape[1]),
                            'x': np.arange(filtered_sum.shape[2]),
                            'excitation': excitation
                        },
                        name=f'filtered_sum_cube_{excitation}'
                    )

                    datasets[f'filtered_sum_cube_{excitation}'] = filtered_sum_da

                # Process reflectance for this excitation
                if 'reflectance' in group:
                    refl_data = group['reflectance'][:]
                    refl_wl = group.attrs.get('reflectance_wavelength', 0)

                    # Create xarray DataArray
                    refl_da = xr.DataArray(
                        refl_data,
                        dims=['y', 'x'],
                        coords={
                            'y': np.arange(refl_data.shape[0]),
                            'x': np.arange(refl_data.shape[1]),
                            'excitation': excitation,
                            'emission': refl_wl
                        },
                        name=f'reflectance_{excitation}'
                    )

                    datasets[f'reflectance_{excitation}'] = refl_da

        # Process reflectance cube (combined from all excitations)
        if 'reflectance_cube' in f:
            refl_group = f['reflectance_cube']
            if 'data' in refl_group and 'excitation_wavelengths' in refl_group:
                refl_data = refl_group['data'][:]
                excitation_wls = refl_group['excitation_wavelengths'][:]
                emission_wls = refl_group['emission_wavelengths'][:] if 'emission_wavelengths' in refl_group else None

                # Create xarray DataArray
                refl_da = xr.DataArray(
                    refl_data,
                    dims=['excitation', 'y', 'x'],
                    coords={
                        'excitation': excitation_wls,
                        'emission': ('excitation', emission_wls) if emission_wls is not None else None,
                        'y': np.arange(refl_data.shape[1]),
                        'x': np.arange(refl_data.shape[2])
                    },
                    name='reflectance_cube'
                )

                datasets['reflectance_cube'] = refl_da

        # Combine all datasets into a single dataset
        if datasets:
            ds = xr.Dataset(datasets)

            # Add metadata
            if 'metadata' in f:
                for key in f['metadata'].attrs:
                    ds.attrs[key] = f['metadata'].attrs[key]

            # Save as NetCDF
            ds.to_netcdf(output_file)
            print(f"Data saved to {output_file} in xarray/NetCDF format")
            return ds
        else:
            print("No datasets were created. Check the HDF5 file structure.")
            return None


def save_to_pandas(h5_file_path, output_file):
    """
    Save hyperspectral data to pandas format (pickle)

    Args:
        h5_file_path: Path to the HDF5 file
        output_file: Path to the output file (.pkl)
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed. Please install with 'pip install pandas'")
        return

    # Read the HDF5 file
    data_dict = read_from_hdf5(h5_file_path)

    # Create a multi-level dictionary for pandas
    processed_data = {
        'average_cubes': {},
        'sum_cubes': {},
        'filtered_average_cubes': {},
        'filtered_sum_cubes': {},
        'reflectance_data': {},
        'reflectance_cube': None,
        'metadata': {}
    }

    # Extract metadata
    with h5py.File(h5_file_path, 'r') as f:
        if 'metadata' in f:
            for key in f['metadata'].attrs:
                processed_data['metadata'][key] = f['metadata'].attrs[key]

    # Process each excitation wavelength
    for excitation, data in data_dict.items():
        if excitation == 'reflectance_cube':
            continue  # Handle separately

        # Get emission wavelengths
        if 'wavelengths' in data:
            emission_wls = data['wavelengths']
        else:
            continue

        # Process average cube
        if 'average_cube' in data:
            avg_cube = data['average_cube']
            height, width = avg_cube.shape[1:]

            # Create a DataFrame from the average cube
            # Reshape to (height*width, bands)
            avg_flat = avg_cube.reshape(len(emission_wls), -1).T

            # Create MultiIndex columns
            columns = pd.MultiIndex.from_tuples([(emission, 'intensity') for emission in emission_wls],
                                                names=['emission', 'type'])

            # Create DataFrame
            avg_df = pd.DataFrame(avg_flat, columns=columns)

            # Add spatial information
            avg_df['y'] = np.repeat(np.arange(height), width)
            avg_df['x'] = np.tile(np.arange(width), height)

            # Set as index for efficient access
            avg_df.set_index(['y', 'x'], inplace=True)

            processed_data['average_cubes'][excitation] = avg_df

        # Process sum cube (similar approach)
        if 'sum_cube' in data:
            sum_cube = data['sum_cube']
            height, width = sum_cube.shape[1:]

            sum_flat = sum_cube.reshape(len(emission_wls), -1).T
            columns = pd.MultiIndex.from_tuples([(emission, 'intensity') for emission in emission_wls],
                                                names=['emission', 'type'])
            sum_df = pd.DataFrame(sum_flat, columns=columns)
            sum_df['y'] = np.repeat(np.arange(height), width)
            sum_df['x'] = np.tile(np.arange(width), height)
            sum_df.set_index(['y', 'x'], inplace=True)

            processed_data['sum_cubes'][excitation] = sum_df

        # Process filtered cubes (add similar code for filtered average & filtered sum)
        if 'filtered_average_cube' in data and 'filtered_wavelengths' in data:
            # Implementation similar to above with filtered data
            pass

        # Process reflectance
        if 'reflectance' in data:
            refl_data = data['reflectance']
            height, width = refl_data.shape

            # Create a DataFrame from reflectance
            refl_flat = refl_data.reshape(-1)
            refl_df = pd.DataFrame({
                'intensity': refl_flat,
                'y': np.repeat(np.arange(height), width),
                'x': np.tile(np.arange(width), height),
            })
            refl_df.set_index(['y', 'x'], inplace=True)

            processed_data['reflectance_data'][excitation] = refl_df

    # Process reflectance cube
    if 'reflectance_cube' in data_dict:
        refl_cube_data = data_dict['reflectance_cube']
        if refl_cube_data['data'] is not None:
            refl_data = refl_cube_data['data']
            excitation_wls = refl_cube_data['excitation_wavelengths']

            # Reshape to create a DataFrame
            num_ex, height, width = refl_data.shape
            refl_flat = refl_data.reshape(num_ex, -1).T

            # Create columns
            columns = pd.MultiIndex.from_tuples([(ex, 'reflectance') for ex in excitation_wls],
                                                names=['excitation', 'type'])

            # Create DataFrame
            refl_df = pd.DataFrame(refl_flat, columns=columns)
            refl_df['y'] = np.repeat(np.arange(height), width)
            refl_df['x'] = np.tile(np.arange(width), height)
            refl_df.set_index(['y', 'x'], inplace=True)

            processed_data['reflectance_cube'] = refl_df

    # Save the processed data
    pd.to_pickle(processed_data, output_file)
    print(f"Data saved to {output_file} in pandas format")

    return processed_data


def save_raw_to_xarray(organized_data, output_file):
    """
    Save only raw averaged data to xarray format (NetCDF)

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output file (.nc for NetCDF)
    """
    try:
        import xarray as xr
    except ImportError:
        print("Error: xarray not installed. Please install with 'pip install xarray netCDF4'")
        return

    # Create a list of excitation wavelengths
    excitations = sorted(organized_data.keys())

    # Create a dataset with excitation as a dimension
    datasets = {}
    data_vars = {}

    # Process each excitation wavelength
    for excitation in excitations:
        data_list = organized_data[excitation]

        # Get emission wavelengths (should be the same for all data at this excitation)
        emission_wls = data_list[0]['wavelengths']

        # Calculate average cube only
        if len(data_list) > 1:
            # Average cube
            avg_cube = np.mean([item['data'] for item in data_list], axis=0)

            # Create DataArray with excitation wavelength as a coordinate - this avoids the conflict
            data_vars[f'average_cube_{excitation}'] = xr.DataArray(
                avg_cube,
                dims=['emission', 'y', 'x'],
                coords={
                    'emission': emission_wls,
                    'y': np.arange(avg_cube.shape[1]),
                    'x': np.arange(avg_cube.shape[2])
                },
                attrs={'excitation_wavelength': excitation}
            )

    # Create the dataset from all data variables
    if data_vars:
        ds = xr.Dataset(data_vars)

        # Add global metadata
        ds.attrs['description'] = 'Raw averaged hyperspectral data without processing'
        ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
        ds.attrs['excitation_wavelengths'] = excitations

        # Save as NetCDF
        ds.to_netcdf(output_file)
        print(f"Raw data saved to {output_file} in xarray/NetCDF format")
        return ds
    else:
        print("No datasets were created.")
        return None


def save_raw_to_pandas(organized_data, output_file):
    """
    Save only raw averaged data to pandas format

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output file (.pkl)
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed. Please install with 'pip install pandas'")
        return

    # Create a dictionary to store the processed data
    processed_data = {
        'average_cubes': {},
        'metadata': {
            'description': 'Raw averaged hyperspectral data without processing',
            'date_created': str(datetime.datetime.now().isoformat())
        }
    }

    # Process each excitation wavelength
    for excitation, data_list in organized_data.items():
        # Get emission wavelengths
        emission_wls = data_list[0]['wavelengths']

        # Calculate average cube
        if len(data_list) > 1:
            avg_cube = np.mean([item['data'] for item in data_list], axis=0)
            height, width = avg_cube.shape[1:]

            # Create a DataFrame from the average cube
            # Reshape to (height*width, bands)
            avg_flat = avg_cube.reshape(len(emission_wls), -1).T

            # Create MultiIndex columns
            columns = pd.MultiIndex.from_tuples([(emission, 'intensity') for emission in emission_wls],
                                                names=['emission', 'type'])

            # Create DataFrame
            avg_df = pd.DataFrame(avg_flat, columns=columns)

            # Add spatial information
            avg_df['y'] = np.repeat(np.arange(height), width)
            avg_df['x'] = np.tile(np.arange(width), height)

            # Set as index for efficient access
            avg_df.set_index(['y', 'x'], inplace=True)

            processed_data['average_cubes'][excitation] = avg_df

    # Save the processed data
    pd.to_pickle(processed_data, output_file)
    print(f"Raw data saved to {output_file} in pandas format")

    return processed_data


def save_processed_to_xarray(organized_data, output_file, cutoff_offset=30, reflectance_range=(400, 500)):
    """
    Save processed data (with cutoffs, reflectance, etc.) to xarray format

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output file (.nc for NetCDF)
        cutoff_offset: Offset in nm for spectral cutoff (default: 30)
        reflectance_range: Valid range (min, max) for reflectance extraction
    """
    try:
        import xarray as xr
    except ImportError:
        print("Error: xarray not installed. Please install with 'pip install xarray netCDF4'")
        return

    # Create a list of excitation wavelengths
    excitations = sorted(organized_data.keys())

    # Create a dataset with excitation as a dimension
    data_vars = {}

    # Dictionary to store reflectance data and coordinate information for valid excitations
    reflectance_dict = {}
    reflectance_wavelengths = {}
    reflectance_indices = {}
    valid_excitations = []

    # Process each excitation wavelength
    for excitation in excitations:
        data_list = organized_data[excitation]

        # Get emission wavelengths (should be the same for all data at this excitation)
        emission_wls = data_list[0]['wavelengths']

        if len(data_list) > 1:
            # Calculate average and summed cubes
            avg_cube = np.mean([item['data'] for item in data_list], axis=0)
            sum_cube = np.sum([item['data'] for item in data_list], axis=0)

            # Apply spectral cutoff to average cube
            filtered_avg, filtered_wl, avg_mask = apply_spectral_cutoff(
                avg_cube, emission_wls, excitation, cutoff_offset)

            # Apply spectral cutoff to summed cube
            filtered_sum, _, _ = apply_spectral_cutoff(
                sum_cube, emission_wls, excitation, cutoff_offset)

            # Extract reflectance data (peaks where excitation=emission) only if in valid range
            refl_data, refl_wl, refl_idx, is_valid = extract_reflectance(
                avg_cube, emission_wls, excitation, method='interpolate', valid_range=reflectance_range)

            # Store reflectance data only if the excitation is within valid range
            if is_valid and refl_data is not None:
                reflectance_dict[excitation] = refl_data
                reflectance_wavelengths[excitation] = refl_wl
                reflectance_indices[excitation] = refl_idx
                valid_excitations.append(excitation)

            # Create xarray DataArrays - without using excitation as a coordinate
            # Average cube
            data_vars[f'average_cube_{excitation}'] = xr.DataArray(
                avg_cube,
                dims=['emission', 'y', 'x'],
                coords={
                    'emission': emission_wls,
                    'y': np.arange(avg_cube.shape[1]),
                    'x': np.arange(avg_cube.shape[2])
                },
                attrs={'excitation_wavelength': excitation}
            )

            # Sum cube
            data_vars[f'sum_cube_{excitation}'] = xr.DataArray(
                sum_cube,
                dims=['emission', 'y', 'x'],
                coords={
                    'emission': emission_wls,
                    'y': np.arange(sum_cube.shape[1]),
                    'x': np.arange(sum_cube.shape[2])
                },
                attrs={'excitation_wavelength': excitation}
            )

            # Filtered average cube
            data_vars[f'filtered_average_cube_{excitation}'] = xr.DataArray(
                filtered_avg,
                dims=['emission', 'y', 'x'],
                coords={
                    'emission': filtered_wl,
                    'y': np.arange(filtered_avg.shape[1]),
                    'x': np.arange(filtered_avg.shape[2])
                },
                attrs={'excitation_wavelength': excitation}
            )

            # Filtered sum cube
            data_vars[f'filtered_sum_cube_{excitation}'] = xr.DataArray(
                filtered_sum,
                dims=['emission', 'y', 'x'],
                coords={
                    'emission': filtered_wl,
                    'y': np.arange(filtered_sum.shape[1]),
                    'x': np.arange(filtered_sum.shape[2])
                },
                attrs={'excitation_wavelength': excitation}
            )

            # Add reflectance data only if valid
            if is_valid and refl_data is not None:
                data_vars[f'reflectance_{excitation}'] = xr.DataArray(
                    refl_data,
                    dims=['y', 'x'],
                    coords={
                        'y': np.arange(refl_data.shape[0]),
                        'x': np.arange(refl_data.shape[1])
                    },
                    attrs={
                        'excitation_wavelength': excitation,
                        'emission_wavelength': refl_wl,
                        'reflection_indices': str(refl_idx) if isinstance(refl_idx, tuple) else refl_idx,
                        'valid_reflectance': 1  # Convert boolean to integer (1=True, 0=False)
                    }
                )

    # Print a summary of valid excitations for reflectance
    print(
        f"Valid excitations for reflectance (within {reflectance_range[0]}-{reflectance_range[1]}nm): {valid_excitations}")

    # Create the ReflectanceCube by combining reflectance from valid excitations only
    if reflectance_dict:
        # Sort valid excitations
        valid_excitations.sort()

        # Check if we have any valid reflectance data
        if len(valid_excitations) > 0:
            # Check if all reflectance data has the same shape
            shapes = [reflectance_dict[ex].shape for ex in valid_excitations]
            if len(set(shapes)) == 1:  # All shapes are the same
                # Stack reflectance data from valid excitations
                reflectance_cube = np.stack([reflectance_dict[ex] for ex in valid_excitations])

                # Create xarray DataArray for reflectance cube - with excitation as a dimension
                data_vars['reflectance_cube'] = xr.DataArray(
                    reflectance_cube,
                    dims=['excitation_dim', 'y', 'x'],
                    coords={
                        'excitation_dim': valid_excitations,  # Use a different name to avoid conflict
                        'y': np.arange(reflectance_cube.shape[1]),
                        'x': np.arange(reflectance_cube.shape[2])
                    },
                    attrs={
                        'description': f'Reflectance cube extracted from peaks where excitation≈emission, valid range: {reflectance_range[0]}-{reflectance_range[1]}nm',
                        'emission_wavelengths': str([reflectance_wavelengths[ex] for ex in valid_excitations]),
                        'valid_range_min': reflectance_range[0],
                        'valid_range_max': reflectance_range[1]
                    }
                )
            else:
                print("Warning: Reflectance data has inconsistent shapes, skipping ReflectanceCube creation")
        else:
            print(
                f"Warning: No valid excitations found within the reflectance range ({reflectance_range[0]}-{reflectance_range[1]}nm)")

    # Create the dataset from all data variables
    if data_vars:
        ds = xr.Dataset(data_vars)

        # Add metadata
        ds.attrs['description'] = 'Processed hyperspectral data with spectral cutoff and reflectance extraction'
        ds.attrs['date_created'] = str(datetime.datetime.now().isoformat())
        ds.attrs['cutoff_offset'] = cutoff_offset
        ds.attrs['excitation_wavelengths'] = str(excitations)  # Convert list to string for netCDF compatibility
        ds.attrs['reflectance_range_min'] = reflectance_range[0]
        ds.attrs['reflectance_range_max'] = reflectance_range[1]
        ds.attrs['valid_reflectance_excitations'] = str(valid_excitations)  # Convert list to string

        # Save as NetCDF
        ds.to_netcdf(output_file)
        print(f"Processed data saved to {output_file} in xarray/NetCDF format")
        return ds
    else:
        print("No datasets were created.")
        return None


def save_processed_to_pandas(organized_data, output_file, cutoff_offset=30, reflectance_range=(400, 500)):
    """
    Save processed data (with cutoffs, reflectance, etc.) to pandas format

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output file (.pkl)
        cutoff_offset: Offset in nm for spectral cutoff (default: 30)
        reflectance_range: Valid range (min, max) for reflectance extraction (default: 400-500nm)
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed. Please install with 'pip install pandas'")
        return

    # Create a dictionary to store the processed data
    processed_data = {
        'average_cubes': {},
        'sum_cubes': {},
        'filtered_average_cubes': {},
        'filtered_sum_cubes': {},
        'reflectance_data': {},
        'reflectance_cube': None,
        'metadata': {
            'description': 'Processed hyperspectral data with spectral cutoff and reflectance extraction',
            'date_created': str(datetime.datetime.now().isoformat()),
            'cutoff_offset': cutoff_offset,
            'reflectance_range': reflectance_range  # Store the reflectance range in metadata
        }
    }

    # Dictionary to store reflectance data and coordinate information
    reflectance_dict = {}
    reflectance_wavelengths = {}
    reflectance_indices = {}
    valid_excitations = []  # Keep track of valid excitations

    # Process each excitation wavelength
    for excitation in sorted(organized_data.keys()):
        data_list = organized_data[excitation]

        # Get emission wavelengths
        emission_wls = data_list[0]['wavelengths']

        if len(data_list) > 1:
            # Calculate average and summed cubes
            avg_cube = np.mean([item['data'] for item in data_list], axis=0)
            sum_cube = np.sum([item['data'] for item in data_list], axis=0)

            # Apply spectral cutoff
            filtered_avg, filtered_wl, avg_mask = apply_spectral_cutoff(
                avg_cube, emission_wls, excitation, cutoff_offset)
            filtered_sum, _, _ = apply_spectral_cutoff(
                sum_cube, emission_wls, excitation, cutoff_offset)

            # Extract reflectance data (with validity check)
            refl_data, refl_wl, refl_idx, is_valid = extract_reflectance(
                avg_cube, emission_wls, excitation, method='interpolate', valid_range=reflectance_range)

            # Store reflectance data only if valid (excitation in valid range)
            if is_valid and refl_data is not None:
                reflectance_dict[excitation] = refl_data
                reflectance_wavelengths[excitation] = refl_wl
                reflectance_indices[excitation] = refl_idx
                valid_excitations.append(excitation)

            # Get dimensions
            height, width = avg_cube.shape[1:]

            # Create pandas DataFrames
            # Average cube
            avg_flat = avg_cube.reshape(len(emission_wls), -1).T
            columns = pd.MultiIndex.from_tuples([(emission, 'intensity') for emission in emission_wls],
                                                names=['emission', 'type'])
            avg_df = pd.DataFrame(avg_flat, columns=columns)
            avg_df['y'] = np.repeat(np.arange(height), width)
            avg_df['x'] = np.tile(np.arange(width), height)
            avg_df.set_index(['y', 'x'], inplace=True)
            processed_data['average_cubes'][excitation] = avg_df

            # Sum cube
            sum_flat = sum_cube.reshape(len(emission_wls), -1).T
            sum_df = pd.DataFrame(sum_flat, columns=columns)
            sum_df['y'] = np.repeat(np.arange(height), width)
            sum_df['x'] = np.tile(np.arange(width), height)
            sum_df.set_index(['y', 'x'], inplace=True)
            processed_data['sum_cubes'][excitation] = sum_df

            # Filtered average cube
            if len(filtered_wl) > 0:
                filtered_height, filtered_width = filtered_avg.shape[1:]
                filtered_flat = filtered_avg.reshape(len(filtered_wl), -1).T
                filtered_columns = pd.MultiIndex.from_tuples([(emission, 'intensity') for emission in filtered_wl],
                                                             names=['emission', 'type'])
                filtered_avg_df = pd.DataFrame(filtered_flat, columns=filtered_columns)
                filtered_avg_df['y'] = np.repeat(np.arange(filtered_height), filtered_width)
                filtered_avg_df['x'] = np.tile(np.arange(filtered_width), filtered_height)
                filtered_avg_df.set_index(['y', 'x'], inplace=True)
                processed_data['filtered_average_cubes'][excitation] = filtered_avg_df

                # Filtered sum cube
                filtered_sum_flat = filtered_sum.reshape(len(filtered_wl), -1).T
                filtered_sum_df = pd.DataFrame(filtered_sum_flat, columns=filtered_columns)
                filtered_sum_df['y'] = np.repeat(np.arange(filtered_height), filtered_width)
                filtered_sum_df['x'] = np.tile(np.arange(filtered_width), filtered_height)
                filtered_sum_df.set_index(['y', 'x'], inplace=True)
                processed_data['filtered_sum_cubes'][excitation] = filtered_sum_df

            # Add reflectance data to the output only if valid
            if is_valid and refl_data is not None:
                refl_flat = refl_data.reshape(-1)
                refl_df = pd.DataFrame({
                    'intensity': refl_flat,
                    'y': np.repeat(np.arange(height), width),
                    'x': np.tile(np.arange(width), height),
                })
                refl_df.set_index(['y', 'x'], inplace=True)
                refl_df.attrs = {'emission_wavelength': refl_wl}
                processed_data['reflectance_data'][excitation] = refl_df

    # Print a summary of valid excitations for reflectance
    print(
        f"Valid excitations for reflectance (within {reflectance_range[0]}-{reflectance_range[1]}nm): {valid_excitations}")
    processed_data['metadata']['valid_reflectance_excitations'] = valid_excitations

    # Create the reflectance cube from valid excitations only
    if valid_excitations:
        # Sort valid excitations
        valid_excitations.sort()

        # Check if all reflectance data has the same shape
        shapes = [reflectance_dict[ex].shape for ex in valid_excitations]
        if len(set(shapes)) == 1:  # All shapes are the same
            # Stack reflectance data from valid excitations only
            reflectance_cube = np.stack([reflectance_dict[ex] for ex in valid_excitations])

            # Create a DataFrame for the reflectance cube
            height, width = reflectance_cube.shape[1:]
            refl_flat = reflectance_cube.reshape(len(valid_excitations), -1).T

            # Create columns - only for valid excitations
            columns = pd.MultiIndex.from_tuples([(ex, 'reflectance') for ex in valid_excitations],
                                                names=['excitation', 'type'])

            # Create DataFrame
            refl_df = pd.DataFrame(refl_flat, columns=columns)
            refl_df['y'] = np.repeat(np.arange(height), width)
            refl_df['x'] = np.tile(np.arange(width), height)
            refl_df.set_index(['y', 'x'], inplace=True)

            # Add metadata to the reflectance cube DataFrame
            refl_df.attrs = {
                'excitation_wavelengths': valid_excitations,
                'emission_wavelengths': [reflectance_wavelengths[ex] for ex in valid_excitations],
                'reflectance_range': reflectance_range,
                'description': f'Reflectance cube extracted from peaks where excitation≈emission, valid range: {reflectance_range[0]}-{reflectance_range[1]}nm'
            }

            processed_data['reflectance_cube'] = refl_df
            processed_data['metadata']['reflectance_wavelengths'] = {ex: reflectance_wavelengths[ex] for ex in
                                                                     valid_excitations}
        else:
            print("Warning: Reflectance data has inconsistent shapes, skipping ReflectanceCube creation")
    else:
        print(
            f"Warning: No valid excitations found within the reflectance range ({reflectance_range[0]}-{reflectance_range[1]}nm)")

    # Save the processed data
    pd.to_pickle(processed_data, output_file)
    print(f"Processed data saved to {output_file} in pandas format")

    return processed_data

def print_data_summary(organized_data, cutoff_offset=30, reflectance_range=(400, 500)):
    """
    Print a summary of the data being processed

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        cutoff_offset: Offset in nm for spectral cutoff (default: 30)
        reflectance_range: Valid range (min, max) for reflectance extraction
    """
    print("\nData Summary:")
    print(f"Number of excitation wavelengths: {len(organized_data)}")

    # Dictionary to track reflectance data for summary
    reflectance_shapes = {}
    valid_excitations = []

    for excitation, data_list in organized_data.items():
        print(f"\nExcitation wavelength: {excitation} nm")
        print(f"Number of cubes: {len(data_list)}")

        # Get wavelengths
        if data_list:
            wavelengths = data_list[0]['wavelengths']
            print(f"Emission wavelengths range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

            # Calculate cutoff information
            _, filtered_wl, _ = apply_spectral_cutoff(
                data_list[0]['data'], wavelengths, excitation, cutoff_offset)

            if len(filtered_wl) > 0:
                print(f"Filtered emission wavelengths range: {filtered_wl.min():.1f} - {filtered_wl.max():.1f} nm")
                print(f"Removed {len(wavelengths) - len(filtered_wl)} of {len(wavelengths)} bands")

            # Calculate cube shapes
            if len(data_list) > 1:
                avg_cube = np.mean([item['data'] for item in data_list], axis=0)
                sum_cube = np.sum([item['data'] for item in data_list], axis=0)

                print(f"Average cube shape: {avg_cube.shape}")
                print(f"Sum cube shape: {sum_cube.shape}")

                # Apply cutoff to get filtered shapes
                filtered_avg, _, _ = apply_spectral_cutoff(
                    avg_cube, wavelengths, excitation, cutoff_offset)
                filtered_sum, _, _ = apply_spectral_cutoff(
                    sum_cube, wavelengths, excitation, cutoff_offset)

                print(f"Filtered average cube shape: {filtered_avg.shape}")
                print(f"Filtered sum cube shape: {filtered_sum.shape}")

                # Extract reflectance info - notice we now unpack 4 values instead of 3
                refl_data, refl_wl, refl_idx, is_valid = extract_reflectance(
                    avg_cube, wavelengths, excitation, method='interpolate', valid_range=reflectance_range)

                if is_valid and refl_data is not None:
                    print(f"Reflectance shape: {refl_data.shape}")
                    print(f"Reflectance wavelength: {refl_wl:.1f} nm")
                    reflectance_shapes[excitation] = refl_data.shape
                    valid_excitations.append(excitation)
                else:
                    print(
                        f"No valid reflectance data (excitation outside valid range {reflectance_range[0]}-{reflectance_range[1]}nm)")

    # Print reflectance cube info
    if valid_excitations:
        valid_excitations.sort()
        shapes = [reflectance_shapes[ex] for ex in valid_excitations]
        if len(set(shapes)) == 1:  # All shapes are the same
            print("\nReflectance Cube (valid excitations only):")
            print(f"Shape: ({len(valid_excitations)}, {shapes[0][0]}, {shapes[0][1]})")
            print(f"Valid excitation wavelengths: {valid_excitations}")
        else:
            print("\nReflectance Cube: Cannot be created due to inconsistent shapes among valid excitations")
    else:
        print(
            f"\nReflectance Cube: No valid excitations found in range {reflectance_range[0]}-{reflectance_range[1]}nm")


def main():
    """Main function to process hyperspectral data with enhanced features"""
    # Base path containing all the data folders
    base_path = r"C:\Users\meloy\Desktop\Files Arch\Kiwi 2"  # Update this path as needed

    # Organize the data
    print("Organizing hyperspectral data...")
    organized_data = organize_data_by_excitation(base_path)

    # Define cutoff offset and reflectance range
    cutoff_offset = 30
    reflectance_range = (400, 500)  # Valid range for reflectance extraction

    # Print data summary
    print_data_summary(organized_data, cutoff_offset, reflectance_range)

    # Save raw data (only averaged cubes, no processing)
    print("\nSaving raw data (averaged cubes only)...")
    raw_xarray_output = "kiwi_hyperspectral_raw.nc"
    raw_pandas_output = "kiwi_hyperspectral_raw.pkl"

    save_raw_to_xarray(organized_data, raw_xarray_output)
    save_raw_to_pandas(organized_data, raw_pandas_output)

    # Save processed data (with cutoffs, reflectance, etc.)
    print("\nSaving processed data (with spectral cutoff and reflectance extraction)...")
    processed_xarray_output = "kiwi_hyperspectral_processed.nc"
    processed_pandas_output = "kiwi_hyperspectral_processed.pkl"

    save_processed_to_xarray(organized_data, processed_xarray_output, cutoff_offset, reflectance_range)
    # Use the new function name to avoid any caching issues
    save_processed_to_pandas(organized_data, processed_pandas_output, cutoff_offset, reflectance_range)

    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()