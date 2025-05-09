"""
Utility functions for hyperspectral data processing.
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import matplotlib.pyplot as plt


def load_data_from_pickle(file_path: str) -> Dict:
    """
    Load hyperspectral data from a pickle file.

    Args:
        file_path: Path to the pickle file

    Returns:
        Loaded data dictionary
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_data_to_pickle(data: Dict, file_path: str) -> None:
    """
    Save hyperspectral data to a pickle file.

    Args:
        data: Data dictionary to save
        file_path: Path to save the pickle file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def create_excitation_emission_dataframe(data_dict: Dict,
                                         sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Transform 4D hyperspectral data into a 2D dataframe.

    Args:
        data_dict: Dictionary containing hyperspectral data
        sample_size: Optional number of random pixels to sample (for large datasets)

    Returns:
        DataFrame with x, y coordinates and intensity values for each valid excitation-emission combination
    """
    # First, collect all valid excitation-emission combinations
    valid_combinations = []
    all_excitations = []

    # Check what excitations we actually have in the data
    for ex_str in data_dict['data'].keys():
        excitation = float(ex_str)
        all_excitations.append(excitation)

        # Get the valid emission wavelengths for this excitation
        emissions = data_dict['data'][ex_str]['wavelengths']

        # Add all valid combinations to our list
        for emission in emissions:
            col_name = f"{int(emission)}-{int(excitation)}"
            valid_combinations.append((excitation, emission, col_name))

    print(f"Found {len(all_excitations)} excitation wavelengths")
    print(f"Generated {len(valid_combinations)} valid excitation-emission combinations")

    # First, determine the dimensions of our data
    first_ex = str(all_excitations[0])
    cube_shape = data_dict['data'][first_ex]['cube'].shape
    height, width = cube_shape[0], cube_shape[1]

    print(f"Image dimensions: {height} x {width} pixels")
    total_pixels = height * width

    # Create a meshgrid of coordinates
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Flatten the coordinates
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    # Create initial dataframe with coordinates
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords
    })

    # If sample_size is provided, take a random sample of pixels
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} pixels out of {total_pixels}")

    print(f"Created initial dataframe with {len(df)} rows")

    # Fill in the intensity values for each valid combination
    for excitation, emission, col_name in valid_combinations:
        # Get the data cube for this excitation
        ex_str = str(excitation)
        cube = data_dict['data'][ex_str]['cube']
        wavelengths = data_dict['data'][ex_str]['wavelengths']

        # Find the index of this emission wavelength
        try:
            em_idx = wavelengths.index(emission)

            # Extract the intensity values for this emission wavelength
            # For the sampled rows only
            if sample_size is not None and sample_size < total_pixels:
                # Get the x, y coordinates of the sampled pixels
                sampled_coords = df[['x', 'y']].values
                # Extract intensity values for these coordinates
                intensities = [cube[y, x, em_idx] for x, y in zip(sampled_coords[:, 1], sampled_coords[:, 0])]
                df[col_name] = intensities
            else:
                # Extract for all pixels - flatten in the same order as the coordinates
                intensities = cube[:, :, em_idx].flatten()
                df[col_name] = intensities

        except ValueError:
            # This emission wavelength doesn't exist for this excitation
            # Skip it as requested instead of adding NaN values
            continue

    print(f"Final dataframe has {len(df.columns)} columns")
    return df


def load_data_and_create_df(pickle_file: str, sample_size: Optional[int] = None,
                           preserve_full_data: bool = True) -> pd.DataFrame:
    """
    Load data from pickle file and create the dataframe.

    Args:
        pickle_file: Path to the pickle file
        sample_size: Optional number of random pixels to sample
        preserve_full_data: Whether to preserve all data dimensions and features

    Returns:
        Transformed dataframe with all data preserved if preserve_full_data=True
    """
    # Load the data
    print(f"Loading data from {pickle_file}...")
    data_dict = load_data_from_pickle(pickle_file)

    # Create the dataframe
    return create_excitation_emission_dataframe(data_dict, sample_size, preserve_full_data)

def save_dataframe(df: pd.DataFrame, output_file: str) -> None:
    """
    Save the dataframe to a file based on its extension.

    Args:
        df: DataFrame to save
        output_file: Path to save the dataframe
    """
    print(f"Saving dataframe to {output_file}")

    # Determine file extension and save accordingly
    ext = Path(output_file).suffix
    if ext == '.csv':
        df.to_csv(output_file, index=False)
    elif ext == '.parquet':
        df.to_parquet(output_file, index=False)
    elif ext == '.pkl' or ext == '.pickle':
        df.to_pickle(output_file)
    else:
        print(f"Unrecognized extension {ext}, saving as pickle")
        df.to_pickle(output_file)

    print(f"Saved dataframe with {len(df)} rows and {len(df.columns)} columns")


def create_masked_excitation_emission_dataframe(data_dict: Dict,
                                                sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Transform 4D hyperspectral data into a 2D dataframe, excluding masked pixels (NaN values).

    Args:
        data_dict: Dictionary containing hyperspectral data
        sample_size: Optional number of random pixels to sample from non-masked pixels

    Returns:
        DataFrame with x, y coordinates and intensity values for non-masked pixels
    """
    # First, collect all valid excitation-emission combinations
    valid_combinations = []
    all_excitations = []

    # Check what excitations we actually have in the data
    for ex_str in data_dict['data'].keys():
        excitation = float(ex_str)
        all_excitations.append(excitation)

        # Get the valid emission wavelengths for this excitation
        emissions = data_dict['data'][ex_str]['wavelengths']

        # Add all valid combinations to our list
        for emission in emissions:
            col_name = f"{int(emission)}-{int(excitation)}"
            valid_combinations.append((excitation, emission, col_name))

    print(f"Found {len(all_excitations)} excitation wavelengths")
    print(f"Generated {len(valid_combinations)} valid excitation-emission combinations")

    # Create coordinate arrays
    first_ex = str(all_excitations[0])
    cube_shape = data_dict['data'][first_ex]['cube'].shape
    height, width = cube_shape[0], cube_shape[1]
    print(f"Image dimensions: {height} x {width} pixels")

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    flat_x_coords = x_coords.flatten()
    flat_y_coords = y_coords.flatten()

    # Create a mask to identify non-NaN pixels (pixels to keep)
    first_cube = data_dict['data'][first_ex]['cube']
    first_band = first_cube[:, :, 0]  # Use the first band to determine mask
    valid_mask = ~np.isnan(first_band)  # NaN values indicate masked pixels

    # Check if any masking has been applied
    if not np.any(np.isnan(first_band)):
        print("No masked pixels detected (no NaN values found)")
    else:
        # Flatten the mask and filter coordinates
        flat_mask = valid_mask.flatten()
        flat_x_coords = flat_x_coords[flat_mask]
        flat_y_coords = flat_y_coords[flat_mask]

        print(f"Identified {np.sum(~valid_mask)} masked pixels, keeping {np.sum(valid_mask)} pixels")

    # Create initial dataframe with filtered coordinates
    df = pd.DataFrame({
        'x': flat_x_coords,
        'y': flat_y_coords
    })

    # If sample_size is provided, take a random sample of pixels
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} pixels out of {len(df.index)} non-masked pixels")

    print(f"Created initial dataframe with {len(df)} rows")

    # Now, fill in the intensity values for each valid combination
    for excitation, emission, col_name in valid_combinations:
        # Get the data cube for this excitation
        ex_str = str(excitation)
        cube = data_dict['data'][ex_str]['cube']
        wavelengths = data_dict['data'][ex_str]['wavelengths']

        # Find the index of this emission wavelength
        try:
            em_idx = wavelengths.index(emission)

            # Extract the intensity values for these coordinates - vectorized approach
            # Convert coordinates to integers and make sure they're within bounds
            y_indices = df['y'].astype(int).values
            x_indices = df['x'].astype(int).values

            # Get the intensities for these coordinates at the specified emission wavelength
            df[col_name] = cube[y_indices, x_indices, em_idx]

        except ValueError:
            # This emission wavelength doesn't exist for this excitation
            # Skip it instead of adding NaN values
            continue

    print(f"Final dataframe has {len(df.columns)} columns")
    return df


def load_masked_data_and_create_df(pickle_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load hyperspectral data from pickle file and create a dataframe,
    excluding masked pixels (indicated by NaN values).

    Args:
        pickle_file: Path to the pickle file
        sample_size: Optional number of random pixels to sample from non-masked pixels

    Returns:
        Transformed dataframe with masked pixels excluded
    """
    # Load the data
    data_dict = load_data_from_pickle(pickle_file)

    # Create the dataframe excluding masked pixels
    return create_masked_excitation_emission_dataframe(data_dict, sample_size)