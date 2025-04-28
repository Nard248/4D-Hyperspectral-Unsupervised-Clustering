import numpy as np
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
import warnings
from typing import Optional, Tuple, Dict, List, Union, Any


class HyperspectralProcessor:
    """
    Comprehensive class for loading, processing, and normalizing hyperspectral data.
    Combines loading from .im3 files, dual cutoff (Rayleigh and second-order),
    exposure time normalization, and laser power normalization in one pipeline.
    """

    def __init__(self,
                 data_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 laser_power_excel: Optional[str] = None,
                 cutoff_offset: int = 30,
                 use_fiji: bool = True,
                 verbose: bool = True):
        """
        Initialize the hyperspectral processor.

        Args:
            data_path: Path to the directory containing .im3 files
            metadata_path: Path to the Excel file with exposure metadata
            laser_power_excel: Path to the Excel file with laser power measurements
            cutoff_offset: Offset in nm used for both cutoffs (Rayleigh and second-order)
            use_fiji: Whether to use ImageJ/Fiji for loading
            verbose: Whether to print processing progress
        """
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.laser_power_excel = laser_power_excel
        self.cutoff_offset = cutoff_offset
        self.use_fiji = use_fiji
        self.verbose = verbose

        # Import HyperspectralDataLoader
        try:
            from HyperspectralDataLoader import HyperspectralDataLoader
            self.loader_class = HyperspectralDataLoader
        except ImportError:
            raise ImportError("HyperspectralDataLoader not found. Make sure it's in your Python path.")

        # Patch the HyperspectralDataLoader class to use dual cutoff
        self._patch_loader_class()

        # Data containers
        self.raw_data = None
        self.data_with_cutoff = None
        self.data_exposure_normalized = None
        self.data_power_normalized = None
        self.laser_powers = None
        self.exposure_times = None

        # Results paths
        self.cutoff_file = None
        self.exposure_norm_file = None
        self.power_norm_file = None

        # Create the loader
        self.loader = self.loader_class(
            data_path=data_path,
            metadata_path=metadata_path,
            cutoff_offset=cutoff_offset,
            use_fiji=use_fiji,
            verbose=verbose
        )

    def _patch_loader_class(self):
        """
        Patch the HyperspectralDataLoader class to use dual cutoff
        (both Rayleigh and second-order).
        """

        def new_apply_spectral_cutoff(self, data, wavelengths, excitation):
            """
            Apply both Rayleigh and second-order spectral cutoffs.
            """
            # Convert wavelengths to numpy array if it's not already
            wavelengths_arr = np.array(wavelengths)

            # Create a mask to keep valid wavelengths
            keep_mask = np.ones(len(wavelengths_arr), dtype=bool)

            # 1. Apply Rayleigh cutoff - remove wavelengths below (excitation + rayleigh_offset)
            rayleigh_cutoff = excitation + self.cutoff_offset
            rayleigh_mask = wavelengths_arr >= rayleigh_cutoff
            keep_mask = np.logical_and(keep_mask, rayleigh_mask)

            # 2. Apply second-order cutoff - remove wavelengths in (2*excitation ± cutoff_offset)
            second_order_min = 2 * excitation - self.cutoff_offset
            second_order_max = 2 * excitation + self.cutoff_offset
            second_order_mask = np.logical_or(wavelengths_arr < second_order_min, wavelengths_arr > second_order_max)
            keep_mask = np.logical_and(keep_mask, second_order_mask)

            # Apply the combined mask to the third dimension (emission wavelengths)
            filtered_data = data[:, :, keep_mask]
            filtered_wavelengths = wavelengths_arr[keep_mask].tolist()

            if self.verbose:
                print(f"Applied dual cutoff for excitation {excitation}nm")
                print(f"Removed wavelengths below {rayleigh_cutoff}nm (Rayleigh cutoff)")
                print(
                    f"Removed wavelengths between {second_order_min}nm and {second_order_max}nm (second-order cutoff)")
                print(f"Original data shape: {data.shape}, filtered shape: {filtered_data.shape}")

            return filtered_data, filtered_wavelengths

        # Patch the method in the class
        self.loader_class.apply_spectral_cutoff = new_apply_spectral_cutoff

        if self.verbose:
            print("HyperspectralDataLoader patched to apply dual cutoff (Rayleigh and second-order)")

    def load_data(self, apply_cutoff: bool = True, pattern: str = "*.im3") -> Dict:
        """
        Load hyperspectral data from .im3 files with dual cutoff.

        Args:
            apply_cutoff: Whether to apply spectral cutoffs
            pattern: File pattern for hyperspectral data files

        Returns:
            Dictionary of processed data
        """
        # Load the data
        self.data_with_cutoff = self.loader.load_data(apply_cutoff=apply_cutoff, pattern=pattern)
        self.raw_data = self.loader.raw_data

        return self.data_with_cutoff

    def read_laser_power_excel(self, excel_path: Optional[str] = None) -> Dict[float, float]:
        """
        Read laser power measurements from Excel file.

        Args:
            excel_path: Path to Excel file (if None, uses self.laser_power_excel)

        Returns:
            Dictionary mapping excitation wavelengths to power values
        """
        if excel_path is None:
            excel_path = self.laser_power_excel

        if excel_path is None:
            raise ValueError("No laser power Excel file specified")

        # Read the Excel file
        df = pd.read_excel(excel_path)

        # Extract the columns
        if "Excitation Wavelength (nm)" in df.columns and "Average Power (W)" in df.columns:
            excitation_col = "Excitation Wavelength (nm)"
            power_col = "Average Power (W)"
        elif len(df.columns) >= 2:
            # If column names don't match, use the first two columns
            excitation_col = df.columns[0]
            power_col = df.columns[1]
        else:
            raise ValueError("Excel file doesn't have expected columns")

        # Create dictionary mapping wavelengths to power
        power_dict = {}
        for _, row in df.iterrows():
            excitation = float(row[excitation_col])
            power = float(row[power_col])
            power_dict[excitation] = power

        self.laser_powers = power_dict

        if self.verbose:
            print(f"Read laser powers for {len(power_dict)} excitation wavelengths from {excel_path}")

        return power_dict

    def _get_data_mapping(self, data_dict):
        """
        Helper function to handle different data dictionary structures.
        Returns the actual mapping of excitation wavelengths to data.
        """
        # First, check if this is a pickled file structure (with 'data' key at top level)
        if 'data' in data_dict and isinstance(data_dict['data'], dict):
            return data_dict['data']

        # Otherwise, assume the dict itself is the mapping
        return data_dict

    def normalize_by_exposure(self,
                              data_dict: Optional[Dict] = None,
                              reference_type: str = 'max',
                              output_file: Optional[str] = None) -> Dict:
        """
        Normalize hyperspectral data based on exposure time.

        Args:
            data_dict: Dictionary containing hyperspectral data (if None, uses self.data_with_cutoff)
            reference_type: Type of reference exposure time ('min', 'max', 'mean', or float value)
            output_file: Path to save the normalized data pickle file (optional)

        Returns:
            Dictionary containing normalized hyperspectral data
        """
        if data_dict is None:
            if self.data_with_cutoff is None:
                raise ValueError("No data available. Load data first with load_data()")
            data_dict = self.data_with_cutoff

        print(f"Normalizing hyperspectral data by exposure time using {reference_type} reference...")

        # Create a deep copy of the data
        normalized_data = copy.deepcopy(data_dict)

        # Get the actual mapping of excitation wavelengths to data
        data_mapping = self._get_data_mapping(data_dict)
        norm_data_mapping = self._get_data_mapping(normalized_data)

        # Print data structure for debugging
        if self.verbose:
            print(f"Data dictionary keys: {list(data_dict.keys())}")
            if 'data' in data_dict:
                print("Found 'data' key at top level")

        # Extract exposure times
        exposure_times = {}

        for ex_str in data_mapping.keys():
            # Try to get exposure time from different possible locations
            if 'raw' in data_mapping[ex_str] and 'expos_val' in data_mapping[ex_str]['raw']:
                exposure_times[ex_str] = data_mapping[ex_str]['raw']['expos_val']
            elif 'expos_val' in data_mapping[ex_str]:
                exposure_times[ex_str] = data_mapping[ex_str]['expos_val']

        if not exposure_times:
            raise ValueError("Could not find exposure time information in the data")

        print(f"Found exposure times for {len(exposure_times)} excitation wavelengths")
        self.exposure_times = exposure_times

        # Determine reference exposure time
        if reference_type == 'min':
            reference_exposure = min(exposure_times.values())
            print(f"Using minimum exposure time as reference: {reference_exposure}")
        elif reference_type == 'max':
            reference_exposure = max(exposure_times.values())
            print(f"Using maximum exposure time as reference: {reference_exposure}")
        elif reference_type == 'mean':
            reference_exposure = sum(exposure_times.values()) / len(exposure_times)
            print(f"Using mean exposure time as reference: {reference_exposure}")
        elif isinstance(reference_type, (int, float)):
            reference_exposure = float(reference_type)
            print(f"Using provided exposure time as reference: {reference_exposure}")
        else:
            raise ValueError("Invalid reference_type. Use 'min', 'max', 'mean', or a float value.")

        # Store normalization info in metadata
        if 'metadata' not in normalized_data:
            normalized_data['metadata'] = {}

        normalized_data['metadata']['exposure_normalization'] = {
            'reference_type': reference_type,
            'reference_exposure': reference_exposure,
            'original_exposures': exposure_times
        }

        # Create a plot of exposure times and normalization factors
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Original exposure times
        x = sorted([float(ex) for ex in exposure_times.keys()])
        y = [exposure_times[str(ex)] for ex in x]
        ax1.plot(x, y, 'o-', color='blue')
        ax1.set_xlabel('Excitation Wavelength (nm)')
        ax1.set_ylabel('Exposure Time')
        ax1.set_title('Exposure Time vs Excitation Wavelength')
        ax1.grid(True)

        # Plot 2: Normalization factors
        factors = [reference_exposure / exposure_times[str(ex)] for ex in x]
        ax2.plot(x, factors, 'o-', color='red')
        ax2.set_xlabel('Excitation Wavelength (nm)')
        ax2.set_ylabel('Normalization Factor')
        ax2.set_title(f'Exposure Normalization Factors ({reference_type} reference)')
        ax2.grid(True)

        plt.tight_layout()

        # Save the plot if output file is provided
        if output_file:
            plot_file = str(Path(output_file).with_suffix('.exposure_normalization.png'))
            plt.savefig(plot_file)
            if self.verbose:
                print(f"Exposure normalization plot saved to: {plot_file}")

        # Normalize each data cube
        print("Normalizing data cubes by exposure time...")
        for ex_str, exposure in exposure_times.items():
            # Calculate normalization factor
            normalization_factor = reference_exposure / exposure

            # Apply normalization
            original_cube = data_mapping[ex_str]['cube']
            norm_data_mapping[ex_str]['cube'] = original_cube * normalization_factor

            # Store normalization factor
            norm_data_mapping[ex_str]['exposure_normalization_factor'] = normalization_factor

            if self.verbose:
                print(f"  Normalized excitation {ex_str}nm (Exposure: {exposure}, Factor: {normalization_factor:.4f})")

        # Save the normalized data
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(normalized_data, f)
            self.exposure_norm_file = output_file
            print(f"Exposure normalized data saved to {output_file}")

        # Store the normalized data
        self.data_exposure_normalized = normalized_data

        return normalized_data

    def normalize_by_laser_power(self,
                                 data_dict: Optional[Dict] = None,
                                 laser_powers: Optional[Dict] = None,
                                 reference_type: str = 'max',
                                 output_file: Optional[str] = None) -> Dict:
        """
        Normalize hyperspectral data based on laser power.

        Args:
            data_dict: Dictionary containing hyperspectral data (if None, uses self.data_with_cutoff)
            laser_powers: Dictionary mapping excitation wavelengths to power values
                          (if None, uses self.laser_powers)
            reference_type: Type of reference power ('min', 'max', 'mean', or float value)
            output_file: Path to save the normalized data pickle file (optional)

        Returns:
            Dictionary containing normalized hyperspectral data
        """
        if data_dict is None:
            if self.data_exposure_normalized is not None:
                # Use exposure normalized data if available
                data_dict = self.data_exposure_normalized
            elif self.data_with_cutoff is not None:
                # Otherwise use cutoff data
                data_dict = self.data_with_cutoff
            else:
                raise ValueError("No data available. Load data first with load_data()")

        if laser_powers is None:
            if self.laser_powers is None:
                # Try to load from Excel file
                if self.laser_power_excel:
                    self.read_laser_power_excel()
                    laser_powers = self.laser_powers
                else:
                    raise ValueError("No laser power data. Call read_laser_power_excel() first.")
            else:
                laser_powers = self.laser_powers

        print(f"Normalizing hyperspectral data by laser power using {reference_type} reference...")

        # Create a deep copy of the data
        normalized_data = copy.deepcopy(data_dict)

        # Get the actual mapping of excitation wavelengths to data
        data_mapping = self._get_data_mapping(data_dict)
        norm_data_mapping = self._get_data_mapping(normalized_data)

        # Determine reference power
        if reference_type == 'min':
            reference_power = min(laser_powers.values())
            print(f"Using minimum laser power as reference: {reference_power:.8f} W")
        elif reference_type == 'max':
            reference_power = max(laser_powers.values())
            print(f"Using maximum laser power as reference: {reference_power:.8f} W")
        elif reference_type == 'mean':
            reference_power = sum(laser_powers.values()) / len(laser_powers)
            print(f"Using mean laser power as reference: {reference_power:.8f} W")
        elif isinstance(reference_type, (int, float)):
            reference_power = float(reference_type)
            print(f"Using provided power as reference: {reference_power:.8f} W")
        else:
            raise ValueError("Invalid reference_type. Use 'min', 'max', 'mean', or a float value.")

        # Store normalization info in metadata
        if 'metadata' not in normalized_data:
            normalized_data['metadata'] = {}

        normalized_data['metadata']['laser_power_normalization'] = {
            'reference_type': reference_type,
            'reference_power': reference_power,
            'laser_powers': laser_powers
        }

        # Create a plot of laser powers and normalization factors
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Original laser powers
        x = sorted(laser_powers.keys())
        y = [laser_powers[k] for k in x]
        ax1.plot(x, y, 'o-', color='blue')
        ax1.set_xlabel('Excitation Wavelength (nm)')
        ax1.set_ylabel('Laser Power (W)')
        ax1.set_title('Laser Power vs Excitation Wavelength')
        ax1.grid(True)

        # Plot 2: Normalization factors
        factors = [reference_power / laser_powers[k] for k in x]
        ax2.plot(x, factors, 'o-', color='red')
        ax2.set_xlabel('Excitation Wavelength (nm)')
        ax2.set_ylabel('Normalization Factor')
        ax2.set_title(f'Laser Power Normalization Factors ({reference_type} reference)')
        ax2.grid(True)

        plt.tight_layout()

        # Save the plot if output file is provided
        if output_file:
            plot_file = str(Path(output_file).with_suffix('.power_normalization.png'))
            plt.savefig(plot_file)
            if self.verbose:
                print(f"Laser power normalization plot saved to: {plot_file}")

        # Normalize each data cube
        print("Normalizing data cubes by laser power...")
        for ex_str in data_mapping.keys():
            excitation = float(ex_str)

            # Check if we have laser power for this excitation
            if excitation in laser_powers:
                laser_power = laser_powers[excitation]

                # Calculate normalization factor
                normalization_factor = reference_power / laser_power

                # Apply normalization
                original_cube = data_mapping[ex_str]['cube']
                norm_data_mapping[ex_str]['cube'] = original_cube * normalization_factor

                # Store normalization factor
                norm_data_mapping[ex_str]['laser_power_normalization_factor'] = normalization_factor

                if self.verbose:
                    print(
                        f"  Normalized excitation {ex_str}nm (Power: {laser_power:.8f} W, Factor: {normalization_factor:.4f})")
            else:
                print(f"  ⚠ No laser power data for excitation {ex_str}nm")

        # Save the normalized data
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump(normalized_data, f)
            self.power_norm_file = output_file
            print(f"Laser power normalized data saved to {output_file}")

        # Store the normalized data
        self.data_power_normalized = normalized_data

        return normalized_data

    def process_full_pipeline(self,
                              output_dir: Optional[str] = None,
                              exposure_reference: str = 'max',
                              power_reference: str = 'max',
                              create_parquet: bool = True,
                              sample_size: Optional[int] = None) -> Dict[str, str]:
        """
        Run the full pipeline: load data, apply cutoffs, normalize by exposure and laser power.

        Args:
            output_dir: Directory to save the output files
            exposure_reference: Reference type for exposure normalization ('min', 'max', 'mean')
            power_reference: Reference type for laser power normalization ('min', 'max', 'mean')
            create_parquet: Whether to create parquet files for analysis
            sample_size: Optional number of random pixels to sample for parquet files

        Returns:
            Dictionary mapping processing stages to their output file paths
        """
        # Validate paths
        if self.data_path is None:
            raise ValueError("data_path must be set")

        if self.metadata_path is None:
            raise ValueError("metadata_path must be set")

        if self.laser_power_excel is None:
            raise ValueError("laser_power_excel must be set")

        # Create output directory
        if output_dir is None:
            output_dir = Path("processed_data")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output file names
        cutoff_file = output_dir / f"data_dual_cutoff_{self.cutoff_offset}nm.pkl"
        exposure_norm_file = output_dir / f"data_cutoff_{self.cutoff_offset}nm_exposure_{exposure_reference}.pkl"
        power_norm_file = output_dir / f"data_cutoff_{self.cutoff_offset}nm_exposure_{exposure_reference}_power_{power_reference}.pkl"

        # Step 1: Load data with dual cutoff
        print(f"\n=== Step 1: Loading data with dual cutoff (offset: {self.cutoff_offset}nm) ===")
        self.load_data(apply_cutoff=True)

        # Save the data with dual cutoff
        pkl_data = {
            'data': self.data_with_cutoff,
            'raw_data': self.raw_data,
            'metadata': self.loader.metadata,
            'excitation_wavelengths': self.loader.excitation_wavelengths,
            'cutoff_offset': self.cutoff_offset
        }

        with open(str(cutoff_file), 'wb') as f:
            pickle.dump(pkl_data, f)

        self.cutoff_file = str(cutoff_file)
        print(f"Data with dual cutoff saved to: {cutoff_file}")

        # Load the pickle file to ensure correct structure
        with open(str(cutoff_file), 'rb') as f:
            loaded_data = pickle.load(f)

        # Step 2: Normalize by exposure time
        print(f"\n=== Step 2: Normalizing by exposure time ({exposure_reference} reference) ===")
        self.normalize_by_exposure(
            data_dict=loaded_data,  # Use loaded data to ensure correct structure
            reference_type=exposure_reference,
            output_file=str(exposure_norm_file)
        )

        # Load the exposure-normalized data
        with open(str(exposure_norm_file), 'rb') as f:
            exposure_normalized_data = pickle.load(f)

        # Step 3: Normalize by laser power
        print(f"\n=== Step 3: Normalizing by laser power ({power_reference} reference) ===")
        # First read laser powers if not already done
        if self.laser_powers is None:
            self.read_laser_power_excel()

        self.normalize_by_laser_power(
            data_dict=exposure_normalized_data,  # Use loaded exposure-normalized data
            reference_type=power_reference,
            output_file=str(power_norm_file)
        )

        # Create output file paths dictionary
        output_files = {
            'cutoff': str(cutoff_file),
            'exposure_normalized': str(exposure_norm_file),
            'power_normalized': str(power_norm_file)
        }

        # Step 4: Create parquet files if requested
        if create_parquet:
            print("\n=== Step 4: Creating parquet files ===")

            try:
                from HyperspectralDataLoader import load_data_and_create_df, save_dataframe

                # Create parquet for dual cutoff
                cutoff_parquet = str(cutoff_file).replace('.pkl', '.parquet')
                df_cutoff = load_data_and_create_df(str(cutoff_file), sample_size)
                save_dataframe(df_cutoff, cutoff_parquet)
                output_files['cutoff_parquet'] = cutoff_parquet

                # Create parquet for exposure normalized
                exposure_parquet = str(exposure_norm_file).replace('.pkl', '.parquet')
                df_exposure = load_data_and_create_df(str(exposure_norm_file), sample_size)
                save_dataframe(df_exposure, exposure_parquet)
                output_files['exposure_parquet'] = exposure_parquet

                # Create parquet for power normalized
                power_parquet = str(power_norm_file).replace('.pkl', '.parquet')
                df_power = load_data_and_create_df(str(power_norm_file), sample_size)
                save_dataframe(df_power, power_parquet)
                output_files['power_parquet'] = power_parquet

            except Exception as e:
                warnings.warn(f"Error creating parquet files: {str(e)}")

        print("\n=== Processing complete! ===")
        print(f"1. Data with dual cutoff: {cutoff_file}")
        print(f"2. Data normalized by exposure time ({exposure_reference} reference): {exposure_norm_file}")
        print(f"3. Data normalized by both exposure and laser power: {power_norm_file}")

        if create_parquet and 'power_parquet' in output_files:
            print(f"\nParquet files created for all processing stages.")
            print(f"Final parquet file (for modeling): {output_files['power_parquet']}")

        return output_files

    def print_summary(self) -> None:
        """Print a summary of the processing results."""
        print("\n=== Hyperspectral Processing Summary ===")

        # Data loading status
        print("\nData Status:")
        print(f"  Raw data loaded: {'Yes' if self.raw_data is not None else 'No'}")
        print(f"  Cutoff applied: {'Yes' if self.data_with_cutoff is not None else 'No'}")
        print(f"  Exposure normalized: {'Yes' if self.data_exposure_normalized is not None else 'No'}")
        print(f"  Power normalized: {'Yes' if self.data_power_normalized is not None else 'No'}")

        # Normalization info
        print("\nNormalization Information:")
        if self.exposure_times:
            print(f"  Exposure times: {len(self.exposure_times)} excitation wavelengths")
            print(
                f"  Exposure range: {min(self.exposure_times.values()):.2f} - {max(self.exposure_times.values()):.2f}")

        if self.laser_powers:
            print(f"  Laser powers: {len(self.laser_powers)} excitation wavelengths")
            print(
                f"  Laser power range: {min(self.laser_powers.values()):.8f}W - {max(self.laser_powers.values()):.8f}W")

        # Output files
        print("\nOutput Files:")
        if self.cutoff_file:
            print(f"  Cutoff data: {self.cutoff_file}")
        if self.exposure_norm_file:
            print(f"  Exposure normalized: {self.exposure_norm_file}")
        if self.power_norm_file:
            print(f"  Power normalized: {self.power_norm_file}")

        # Cutoff details
        if self.data_with_cutoff:
            print("\nCutoff Details:")
            print(f"  Rayleigh cutoff offset: {self.cutoff_offset}nm")
            print(f"  Second order cutoff offset: {self.cutoff_offset}nm")

            # Print dimensions from data_with_cutoff if available
            if self.loader.excitation_wavelengths:
                for ex in self.loader.excitation_wavelengths:
                    ex_str = str(ex)
                    if ex_str in self.data_with_cutoff:
                        cube = self.data_with_cutoff[ex_str]['cube']
                        wavelengths = self.data_with_cutoff[ex_str]['wavelengths']
                        print(f"\n  Excitation {ex}nm:")
                        print(f"    Data shape: {cube.shape}")
                        print(f"    Emission range: {min(wavelengths)}nm - {max(wavelengths)}nm")
                        print(f"    Number of emission bands: {len(wavelengths)}")


# Example usage
if __name__ == "__main__":
    processor = HyperspectralProcessor(
        data_path="../Data/Lime",
        metadata_path="../Data/Lime/metadata.xlsx",
        laser_power_excel="../Data/Lime/TLS Scans/average_power.xlsx",
        cutoff_offset=30,
        verbose=True
    )

    output_files = processor.process_full_pipeline(
        output_dir="Data/Lime Experiment/processed",
        exposure_reference="max",
        power_reference="min",
        create_parquet=True
    )

    processor.print_summary()