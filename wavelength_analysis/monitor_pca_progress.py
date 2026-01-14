import os
import time
import glob

results_dir = "validation_results_v2"

# Find the latest results directory
dirs = sorted([d for d in os.listdir(results_dir) if d.startswith("2025")])
if not dirs:
    print("No results directories found")
    exit(1)

latest_dir = os.path.join(results_dir, dirs[-1])
print(f"Monitoring: {latest_dir}")
print("="*80)

# Count experiments
experiments_dir = os.path.join(latest_dir, "experiments")
if os.path.exists(experiments_dir):
    experiments = [d for d in os.listdir(experiments_dir) if d.startswith("mmr_")]
    experiments.sort()

    total_expected = 43  # 3-30 (28 configs) + 30-180 step 10 (15 configs)
    completed = len(experiments)

    print(f"\nProgress: {completed}/{total_expected} configurations completed")
    print(f"Percentage: {completed/total_expected*100:.1f}%")

    if experiments:
        print(f"\nLatest completed: {experiments[-1]}")

        # Extract band numbers
        band_numbers = []
        for exp in experiments:
            try:
                n_bands = int(exp.split("_")[-1].replace("bands", ""))
                band_numbers.append(n_bands)
            except:
                pass

        if band_numbers:
            print(f"\nCompleted band counts: {sorted(band_numbers)}")
            print(f"Range: {min(band_numbers)} to {max(band_numbers)} bands")

    # Check for Excel file
    excel_file = os.path.join(latest_dir, "wavelength_selection_results_v2.xlsx")
    if os.path.exists(excel_file):
        print(f"\nExcel file exists: {excel_file}")
        import pandas as pd
        df = pd.read_excel(excel_file)
        print(f"Excel entries: {len(df)} rows")
else:
    print("Experiments directory not found yet")

print("\n" + "="*80)
