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
    folders = glob.glob(os.path.join(base_path, "Ki2i Experiment 2_04-15_*_*"))
    print(folders)
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


def save_to_hdf5(organized_data, output_file):
    """
    Save organized hyperspectral data to HDF5 format

    Args:
        organized_data: Dictionary with excitation wavelength as keys and lists of data as values
        output_file: Path to the output HDF5 file
    """
    with h5py.File(output_file, 'w') as f:
        # Create a group for each excitation wavelength
        for excitation_wavelength, data_list in organized_data.items():
            group_name = f'excitation_{excitation_wavelength}'
            group = f.create_group(group_name)

            # Store wavelengths (should be the same for all data at this excitation)
            wavelengths = data_list[0]['wavelengths']
            group.create_dataset('wavelengths', data=wavelengths)

            # Store each data cube
            for data_item in data_list:
                index = data_item['index']
                data_cube = data_item['data']
                header = data_item['header']

                # Create a subgroup for this cube
                cube_group = group.create_group(f'cube_{index}')
                cube_group.create_dataset('data', data=data_cube, compression='gzip', chunks=True)

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

            # Create an average cube for noise reduction
            if len(data_list) > 1:
                avg_cube = np.mean([item['data'] for item in data_list], axis=0)
                group.create_dataset('average_cube', data=avg_cube, compression='gzip', chunks=True)

        # Store general metadata
        metadata = f.create_group('metadata')
        metadata.attrs['description'] = 'Hyperspectral image data from Kiwi experiment'
        metadata.attrs['date_created'] = str(datetime.datetime.now().isoformat())
        metadata.attrs['num_excitations'] = len(organized_data)
        metadata.attrs['cubes_per_excitation'] = 10


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

                        excitation_data['cubes'][index] = {
                            'data': cube_data,
                            'header': header
                        }
                    elif dataset_name == 'average_cube':
                        excitation_data['average_cube'] = group[dataset_name][:]

                data_dict[excitation_wavelength] = excitation_data

    return data_dict


def main():
    # Base path containing all the data folders
    base_path = r"C:\Users\meloy\Downloads\Kiwi Experiment 3\Kiwi Experiment 2"

    # Organize the data
    print("Organizing hyperspectral data...")
    organized_data = organize_data_by_excitation(base_path)

    # Save to HDF5
    output_file = "kiwi_hyperspectral_4d_data_test.h5"
    print(f"Saving data to {output_file}...")
    save_to_hdf5(organized_data, output_file)

    # Verify by reading data back
    print("Verifying data by reading it back...")
    data_dict = read_from_hdf5(output_file)

    # Print summary information
    print("\nData Summary:")
    print(f"Number of excitation wavelengths: {len(data_dict)}")

    for excitation, data in data_dict.items():
        print(f"\nExcitation wavelength: {excitation} nm")
        print(f"Number of cubes: {len(data['cubes'])}")

        if 'wavelengths' in data:
            print(f"Emission wavelengths range: {data['wavelengths'].min():.1f} - {data['wavelengths'].max():.1f} nm")

        if 'average_cube' in data:
            print(f"Average cube shape: {data['average_cube'].shape}")

        # Print info about the first cube
        if data['cubes']:
            first_cube_key = list(data['cubes'].keys())[0]
            first_cube = data['cubes'][first_cube_key]
            print(f"Sample cube shape (index {first_cube_key}): {first_cube['data'].shape}")


if __name__ == "__main__":
    main()