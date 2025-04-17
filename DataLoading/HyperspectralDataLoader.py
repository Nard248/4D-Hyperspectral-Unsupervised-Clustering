import numpy as np
import os
import glob
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class HyperspectralSimplified:
    """Simplified class for processing hyperspectral data with spectral cutoff"""

    def __init__(self, base_folder, cutoff_offset=30, reflectance_range=(400, 500)):
        """
        Initialize the processor

        Args:
            base_folder: Path to the folder containing the hyperspectral data subfolders
            cutoff_offset: Offset in nm to add to excitation wavelength for cutoff
            reflectance_range: Valid range for reflectance extraction
        """
        self.base_folder = base_folder
        self.cutoff_offset = cutoff_offset
        self.reflectance_range = reflectance_range
        self.data = {}  # Will hold processed data
        self.excitation_wavelengths = []
        self.metadata = {
            'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cutoff_offset': cutoff_offset,
            'reflectance_range': reflectance_range
        }

    def read_hdr_file(self, hdr_file):
        """Read an ENVI header file"""
        header_dict = {}
        with open(hdr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    header_dict[key.strip()] = value.strip()
        return header_dict

    def read_bin_file(self, bin_file, header_dict):
        """Read binary data file using header information"""
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

    def read_hyperspectral_data(self, folder_path):
        """Read hyperspectral data from a folder"""
        # Read the header file
        hdr_file = os.path.join(folder_path, 'spectral_image_processed_image.hdr')
        header = self.read_hdr_file(hdr_file)

        # Read the binary data file
        bin_file = os.path.join(folder_path, 'spectral_image_processed_image.bin')
        data = self.read_bin_file(bin_file, header)

        # Read the wavelengths file
        wavelengths_file = os.path.join(folder_path, 'spectral_image_wavelengths.csv')
        wavelengths = np.loadtxt(wavelengths_file, delimiter=',')

        return data, wavelengths, header

    def extract_info_from_folder_name(self, folder_name):
        """Extract excitation wavelength and index from folder name"""
        # Extract excitation wavelength and index using regex
        match = re.search(r'_(\d+)_(\d+)$', folder_name)
        if match:
            excitation_wavelength = int(match.group(1))
            index = int(match.group(2))
            return excitation_wavelength, index
        return None, None

    def apply_spectral_cutoff(self, data, wavelengths, excitation):
        """
        Apply spectral cutoff to remove excitation artifacts.
        Filters emission spectra to:
        1. Remove wavelengths < (excitation + cutoff_offset)
        2. Remove wavelengths in the second-order scattering region (2*excitation ± cutoff_offset)

        Args:
            data: Hyperspectral data cube (bands, height, width)
            wavelengths: Array of emission wavelengths
            excitation: Excitation wavelength

        Returns:
            filtered_data: Data after applying cutoff
            filtered_wavelengths: Wavelengths after applying cutoff
        """
        # Create a mask to keep valid wavelengths
        keep_mask = np.ones(len(wavelengths), dtype=bool)

        # 1. Remove wavelengths before excitation + cutoff_offset
        keep_mask = np.logical_and(keep_mask, wavelengths >= excitation + self.cutoff_offset)

        # 2. Remove wavelengths in the second-order zone (2*excitation ± cutoff_offset)
        second_order_min = 2 * excitation - self.cutoff_offset
        second_order_max = 2 * excitation + self.cutoff_offset
        second_order_mask = np.logical_or(wavelengths < second_order_min, wavelengths > second_order_max)
        keep_mask = np.logical_and(keep_mask, second_order_mask)

        # Apply the mask
        filtered_data = data[keep_mask, :, :]
        filtered_wavelengths = wavelengths[keep_mask]

        return filtered_data, filtered_wavelengths
    def extract_reflectance(self, data, wavelengths, excitation):
        """
        Extract reflectance data (peaks where excitation~=emission)
        Using interpolation between the two closest emission wavelengths

        Args:
            data: Hyperspectral data cube (bands, height, width)
            wavelengths: Array of emission wavelengths
            excitation: Excitation wavelength

        Returns:
            reflectance: 2D array of reflectance data or None if excitation is out of valid range
            valid: Boolean indicating if the reflectance is valid
        """
        # Check if excitation is within the valid range
        min_valid, max_valid = self.reflectance_range
        if excitation < min_valid or excitation > max_valid:
            # Return None if excitation is outside the valid range
            return None, False

        # Find the closest wavelength index
        closest_idx = np.argmin(np.abs(wavelengths - excitation))

        # Find indices of wavelengths less than and greater than excitation
        diff = wavelengths - excitation

        # If the closest wavelength is exactly the excitation, just return that
        if diff[closest_idx] == 0:
            reflectance = data[closest_idx, :, :]
            return reflectance, True

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

            return reflectance, True
        else:
            # We only have wavelengths on one side of excitation
            # Just return the closest one
            reflectance = data[closest_idx, :, :]
            return reflectance, True

    def process_data(self):
        """Process all hyperspectral data and create the simplified structure"""
        print(f"Processing data with cutoff offset: {self.cutoff_offset}nm...")

        # Find all data folders
        folders = glob.glob(os.path.join(self.base_folder, "Kiwi 2_03-25_*_*"))

        # Dictionary to organize data by excitation wavelength
        organized_data = {}

        # Process each folder
        for folder in folders:
            folder_name = os.path.basename(folder)
            excitation_wavelength, index = self.extract_info_from_folder_name(folder_name)

            if excitation_wavelength is not None:
                try:
                    # Read data from this folder
                    data, wavelengths, header = self.read_hyperspectral_data(folder)

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

        # Store excitation wavelengths
        self.excitation_wavelengths = sorted(organized_data.keys())
        self.metadata['excitation_wavelengths'] = self.excitation_wavelengths

        # Dictionary to store reflectance data
        reflectance_data_dict = {}
        valid_reflectance_excitations = []

        # Process each excitation wavelength
        for excitation in self.excitation_wavelengths:
            data_list = organized_data[excitation]

            # Skip if no data
            if not data_list:
                continue

            # Get wavelengths (should be the same for all data at this excitation)
            wavelengths = data_list[0]['wavelengths']

            # Create average cube
            avg_cube = np.mean([item['data'] for item in data_list], axis=0)

            # Apply spectral cutoff
            filtered_avg, filtered_wavelengths = self.apply_spectral_cutoff(avg_cube, wavelengths, excitation)

            # Store filtered data for this excitation
            self.data[str(excitation)] = {
                'cube': filtered_avg,
                'wavelengths': filtered_wavelengths,
                'excitation': excitation
            }

            # Extract reflectance data
            reflectance, is_valid = self.extract_reflectance(avg_cube, wavelengths, excitation)

            # Store valid reflectance data
            if is_valid and reflectance is not None:
                reflectance_data_dict[excitation] = reflectance
                valid_reflectance_excitations.append(excitation)

        # Create reflectance cube from valid excitations
        if valid_reflectance_excitations:
            valid_reflectance_excitations.sort()

            # Check if all reflectance data has the same shape
            shapes = [reflectance_data_dict[ex].shape for ex in valid_reflectance_excitations]
            if len(set(shapes)) == 1:  # All shapes are the same
                # Stack reflectance data
                reflectance_cube = np.stack([reflectance_data_dict[ex] for ex in valid_reflectance_excitations])

                # Store reflectance cube
                self.data['Reflectance'] = {
                    'cube': reflectance_cube,
                    'excitation_wavelengths': valid_reflectance_excitations
                }

                self.metadata['valid_reflectance_excitations'] = valid_reflectance_excitations
                print(f"Created reflectance cube with shape: {reflectance_cube.shape}")
            else:
                print("Warning: Reflectance data has inconsistent shapes, skipping reflectance cube creation")
        else:
            print(f"Warning: No valid excitations found within the reflectance range {self.reflectance_range}")

        return self.data

    def get_data_as_dataframe(self):
        """Convert the data structure to a pandas DataFrame"""
        # Create a dictionary that will be converted to a DataFrame
        df_data = {}

        # Add each excitation wavelength data
        for excitation_key, data_dict in self.data.items():
            if excitation_key != 'Reflectance':
                df_data[excitation_key] = data_dict['cube']

        # Add reflectance data if available
        if 'Reflectance' in self.data:
            df_data['Reflectance'] = self.data['Reflectance']['cube']

        # Create a DataFrame
        return pd.DataFrame(df_data)

    def visualize_cutoff(self, excitation_wavelength):
        """Visualize the effect of spectral cutoff for a specific excitation wavelength"""
        if str(excitation_wavelength) not in self.data:
            print(f"Excitation wavelength {excitation_wavelength} not found in data")
            return

        # Get the original data
        folders = glob.glob(os.path.join(self.base_folder, f"Kiwi 2_03-25_{excitation_wavelength}_*"))
        if not folders:
            print(f"No folders found for excitation wavelength {excitation_wavelength}")
            return

        # Read original data
        data, wavelengths, _ = self.read_hyperspectral_data(folders[0])

        # Get filtered data
        filtered_data = self.data[str(excitation_wavelength)]['cube']
        filtered_wavelengths = self.data[str(excitation_wavelength)]['wavelengths']

        # Calculate mean spectra
        full_mean = np.mean(data, axis=(1, 2))
        filtered_mean = np.mean(filtered_data, axis=(1, 2))

        # Create figure showing cutoff
        plt.figure(figsize=(12, 8))

        # Plot full spectrum with cutoff region highlighted
        plt.plot(wavelengths, full_mean, 'b-', label='Full Spectrum')

        # Highlight cutoff region
        cutoff_wavelength = excitation_wavelength + self.cutoff_offset
        plt.axvspan(excitation_wavelength - 5, cutoff_wavelength, color='r', alpha=0.2,
                    label=f'Cutoff Region (Ex: {excitation_wavelength}nm + {self.cutoff_offset}nm)')
        plt.axvline(x=cutoff_wavelength, color='r', linestyle='--')

        # Plot filtered spectrum
        plt.plot(filtered_wavelengths, filtered_mean, 'g-', linewidth=2, label='Filtered Spectrum')

        plt.xlabel('Emission Wavelength (nm)')
        plt.ylabel('Mean Signal')
        plt.title(f'Spectral Cutoff Effect - Excitation {excitation_wavelength}nm')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Highlight excitation wavelength
        plt.axvline(x=excitation_wavelength, color='orange', linestyle='-', label='Excitation Wavelength')

        plt.show()

    def save_to_pkl(self, output_file='hyperspectral_simplified.pkl'):
        """Save the processed data to a pickle file"""
        output_data = {
            'data': self.data,
            'metadata': self.metadata
        }

        pd.to_pickle(output_data, output_file)
        print(f"Data saved to {output_file}")


def main():
    # Base path containing the data folders
    base_folder = r"C:\Users\meloy\Desktop\Files Arch\Kiwi 2"  # Update this with your path

    # Create the processor with desired cutoff offset
    processor = HyperspectralSimplified(
        base_folder=base_folder,
        cutoff_offset=1,  # Adjust as needed
        reflectance_range=(400, 500)  # Adjust as needed
    )

    # Process the data
    data = processor.process_data()

    # Print a summary of the processed data
    print("\nProcessed Data Summary:")
    print(f"Number of excitation wavelengths: {len(processor.excitation_wavelengths)}")
    print(f"Excitation wavelengths: {processor.excitation_wavelengths}")

    for key, value in data.items():
        if key == 'Reflectance':
            if 'cube' in value:
                print(f"\nReflectance cube shape: {value['cube'].shape}")
                print(f"Valid excitation wavelengths: {value['excitation_wavelengths']}")
        else:
            print(f"\nExcitation {key} nm:")
            print(f"  Cube shape: {value['cube'].shape}")
            print(f"  Wavelengths: {value['wavelengths'].min():.1f} - {value['wavelengths'].max():.1f} nm")

    # Save data to pickle file
    processor.save_to_pkl('kiwi_simplified_data_cutoff_1.pkl')

    # Example of how to visualize cutoff for a specific excitation
    # Uncomment to use
    # processor.visualize_cutoff(310)  # Change to any available excitation wavelength


if __name__ == "__main__":
    main()