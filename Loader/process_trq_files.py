import os
import re
import pandas as pd
import numpy as np
import glob

def parse_trq_file(file_path):
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

def process_trq_files(folder_path):
    """
    Process all TRQ files in a folder and calculate average power for each wavelength.

    Args:
        folder_path (str): Path to the folder containing TRQ files

    Returns:
        pandas.DataFrame: DataFrame with wavelength and average power data
    """
    # Find all TRQ files in the folder
    trq_files = glob.glob(os.path.join(folder_path, '*.TRQ'))

    if not trq_files:
        raise ValueError(f"No TRQ files found in {folder_path}")

    # Parse each TRQ file and store the data
    all_data = []
    for file_path in trq_files:
        df = parse_trq_file(file_path)
        all_data.append(df)

    # Combine all data and group by wavelength
    combined_data = pd.concat(all_data)

    # Round wavelengths to the nearest 10 to handle slight variations
    # The issue description mentions wavelengths from 300 to 500 with step size of 10nm
    combined_data['Wavelength_Rounded'] = (combined_data['Wavelength'] / 10).round() * 10

    # Group by rounded wavelength and calculate average power
    result = combined_data.groupby('Wavelength_Rounded')['Power'].mean().reset_index()
    result.rename(columns={'Wavelength_Rounded': 'Excitation Wavelength (nm)', 'Power': 'Average Power (W)'}, inplace=True)

    # Sort by wavelength
    result = result.sort_values('Excitation Wavelength (nm)')

    return result

def main():
    """
    Main function to process TRQ files and create an Excel file with the results.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process TRQ files and calculate average power for each wavelength.')
    parser.add_argument('folder_path', help='Path to the folder containing TRQ files')
    parser.add_argument('--output', '-o', default='average_power.xlsx', help='Output Excel file name (default: average_power.xlsx)')

    args = parser.parse_args()

    try:
        # Process TRQ files
        result = process_trq_files(args.folder_path)

        # Save the result to an Excel file
        result.to_excel(args.output, index=False)

        print(f"Successfully processed {len(glob.glob(os.path.join(args.folder_path, '*.TRQ')))} TRQ files.")
        print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == '__main__':
    main()
