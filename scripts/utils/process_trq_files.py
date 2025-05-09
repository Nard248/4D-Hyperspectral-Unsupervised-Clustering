import os
import re
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import Optional, Union


def parse_trq_file(file_path: str) -> pd.DataFrame:
    """
    Parse a TRQ file and extract wavelength and power data.

    Args:
        file_path (str): Path to the TRQ file

    Returns:
        pandas.DataFrame: DataFrame with wavelength and power data
    """
    data_started = False
    wavelengths = []
    powers = []

    with open(file_path, 'r') as f:
        for line in f:
            if data_started:
                # Skip the header line with column names
                if line.strip() and not line.startswith('X\t'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        wavelength = float(parts[0])
                        power = float(parts[1])
                        wavelengths.append(wavelength)
                        powers.append(power)
            elif line.strip() == '//DATA//':
                data_started = True

    # Create a DataFrame with the extracted data
    df = pd.DataFrame({
        'Wavelength': wavelengths,
        'Power': powers
    })

    return df


def process_trq_files(folder_path: Union[str, Path]) -> pd.DataFrame:
    """
    Process all TRQ files in a folder and calculate average power for each wavelength.

    Args:
        folder_path (str or Path): Path to the folder containing TRQ files

    Returns:
        pandas.DataFrame: DataFrame with wavelength and average power data
    """
    # Convert to Path object for better cross-platform compatibility
    folder_path = Path(folder_path)

    # Find all TRQ files in the folder
    trq_files = list(folder_path.glob('*.TRQ'))

    if not trq_files:
        raise ValueError(f"No TRQ files found in {folder_path}")

    # Parse each TRQ file and store the data
    all_data = []
    for file_path in trq_files:
        df = parse_trq_file(str(file_path))
        all_data.append(df)

    # Combine all data and group by wavelength
    combined_data = pd.concat(all_data)

    # Round wavelengths to the nearest 10 to handle slight variations
    combined_data['Wavelength_Rounded'] = (combined_data['Wavelength'] / 10).round() * 10

    # Group by rounded wavelength and calculate average power
    result = combined_data.groupby('Wavelength_Rounded')['Power'].mean().reset_index()
    result.rename(columns={'Wavelength_Rounded': 'Excitation Wavelength (nm)',
                           'Power': 'Average Power (W)'}, inplace=True)

    # Sort by wavelength
    result = result.sort_values('Excitation Wavelength (nm)')

    return result


def save_power_data(data: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Save the processed power data to an Excel file.

    Args:
        data (pd.DataFrame): DataFrame with wavelength and power data
        output_path (str or Path): Path to save the Excel file
    """
    # Convert to Path object for better handling
    output_path = Path(output_path)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Excel
    data.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


def process_and_save(folder_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Process TRQ files and save the results to an Excel file.
    This is the main function to use when importing this module.

    Args:
        folder_path (str or Path): Path to the folder containing TRQ files
        output_path (str or Path, optional): Path to save the Excel file.
                                           If None, returns DataFrame without saving.

    Returns:
        pandas.DataFrame: DataFrame with wavelength and average power data
    """
    # Process the files
    result = process_trq_files(folder_path)

    # Save if output path is provided
    if output_path:
        save_power_data(result, output_path)

    return result


def main():
    """
    Main function for command-line usage.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process TRQ files and calculate average power for each wavelength.')
    parser.add_argument('folder_path', help='Path to the folder containing TRQ files')
    parser.add_argument('--output', '-o', default='average_power.xlsx',
                        help='Output Excel file name (default: average_power.xlsx)')

    args = parser.parse_args()

    try:
        # Process and save
        process_and_save(args.folder_path, args.output)
        print(f"Successfully processed {len(list(Path(args.folder_path).glob('*.TRQ')))} TRQ files.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0