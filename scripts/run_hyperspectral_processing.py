"""
Main script for running the hyperspectral data processing pipeline.
"""

from pathlib import Path
from data_processing import HyperspectralProcessor


def main():
    """
    Main function to run the hyperspectral data processing pipeline.
    Uses relative paths for better cross-platform compatibility.
    """
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = project_root / "data" / "raw" / "Lime"
    metadata_path = data_path / "metadata.xlsx"
    laser_power_excel = data_path / "TLS Scans" / "average_power.xlsx"
    output_dir = project_root / "data" / "processed" / "Lime"

    processor = HyperspectralProcessor(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        laser_power_excel=str(laser_power_excel),
        cutoff_offset=40,
        verbose=True
    )

    output_files = processor.process_full_pipeline(
        output_dir=str(output_dir),
        exposure_reference="max",
        power_reference="min",
        create_parquet=False,
        preserve_full_data=True
    )

    processor.print_summary()

    return output_files


if __name__ == "__main__":
    main()