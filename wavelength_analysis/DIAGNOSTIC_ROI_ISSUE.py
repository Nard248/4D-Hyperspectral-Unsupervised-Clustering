"""
DIAGNOSTIC SCRIPT: ROI Coordinate Issue After Cropping
========================================================
This script checks if ROI coordinates are correct after data cropping.
"""

import numpy as np

# Original ROI coordinates (before cropping)
ROI_REGIONS_ORIGINAL = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},  # Red
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},  # Blue
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},  # Green
    {'name': 'Region 4', 'coords': (175, 225, 650, 700), 'color': '#FFFF00'},  # Yellow
]

# Cropping parameters from your code
start_col = 1392 - 925  # = 467
end_col = 1392
new_width = end_col - start_col  # = 925

print("="*80)
print("DIAGNOSTIC: ROI COORDINATES VS CROPPED DATA")
print("="*80)

print(f"\nOriginal image width: 1392 pixels")
print(f"Cropping from column {start_col} to {end_col}")
print(f"New cropped width: {new_width} pixels (x coordinates: 0 to {new_width-1})")

print("\n" + "="*80)
print("CHECKING ROI COORDINATES")
print("="*80)

for roi in ROI_REGIONS_ORIGINAL:
    y_start, y_end, x_start, x_end = roi['coords']

    print(f"\n{roi['name']} ({roi['color']}):")
    print(f"  Original coordinates: x={x_start}-{x_end}, y={y_start}-{y_end}")

    # Check if this ROI is within the cropped region
    if x_start >= start_col and x_end <= end_col:
        # ROI is fully within cropped region - need to adjust coordinates
        new_x_start = x_start - start_col
        new_x_end = x_end - start_col
        print(f"  [OK] ROI is within cropped region")
        print(f"  Adjusted coordinates: x={new_x_start}-{new_x_end}, y={y_start}-{y_end}")
        print(f"  --> SHOULD USE: ({y_start}, {y_end}, {new_x_start}, {new_x_end})")
    elif x_end <= start_col:
        print(f"  [ERROR] ROI is COMPLETELY LEFT of cropped region (x_end={x_end} < start_col={start_col})")
        print(f"  --> ROI LOST IN CROPPING - NO TRAINING DATA!")
    elif x_start >= end_col:
        print(f"  [ERROR] ROI is COMPLETELY RIGHT of cropped region (x_start={x_start} >= end_col={end_col})")
        print(f"  --> ROI LOST IN CROPPING - NO TRAINING DATA!")
    else:
        print(f"  [WARNING] ROI is PARTIALLY in cropped region")
        overlap_start = max(x_start, start_col)
        overlap_end = min(x_end, end_col)
        new_x_start = overlap_start - start_col
        new_x_end = overlap_end - start_col
        print(f"  Overlap: x={overlap_start}-{overlap_end} in original coords")
        print(f"  Adjusted coordinates: x={new_x_start}-{new_x_end}, y={y_start}-{y_end}")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

# Count how many ROIs are in the cropped region
rois_in_crop = 0
rois_lost = []
for roi in ROI_REGIONS_ORIGINAL:
    y_start, y_end, x_start, x_end = roi['coords']
    if x_start >= start_col and x_end <= end_col:
        rois_in_crop += 1
    else:
        rois_lost.append(roi['name'])

print(f"\n[OK] ROIs within cropped region: {rois_in_crop} / {len(ROI_REGIONS_ORIGINAL)}")
if rois_lost:
    print(f"[ERROR] ROIs lost/outside cropped region: {', '.join(rois_lost)}")
    print(f"\n[PROBLEM IDENTIFIED]:")
    print(f"   The code is using ORIGINAL ROI coordinates after cropping the data!")
    print(f"   ROIs outside the crop have NO TRAINING DATA for KNN.")
    print(f"   This causes the Hungarian algorithm to fail for those classes.")

print("\n" + "="*80)
print("SOLUTION: CORRECTED ROI_REGIONS FOR CROPPED DATA")
print("="*80)

print("\nROI_REGIONS = [")
for roi in ROI_REGIONS_ORIGINAL:
    y_start, y_end, x_start, x_end = roi['coords']

    if x_start >= start_col and x_end <= end_col:
        new_x_start = x_start - start_col
        new_x_end = x_end - start_col
        print(f"    {{'name': '{roi['name']}', 'coords': ({y_start}, {y_end}, {new_x_start}, {new_x_end}), 'color': '{roi['color']}'}},  # {roi['name']}")
    elif x_end > start_col and x_start < end_col:
        # Partial overlap
        overlap_start = max(x_start, start_col)
        overlap_end = min(x_end, end_col)
        new_x_start = overlap_start - start_col
        new_x_end = overlap_end - start_col
        print(f"    {{'name': '{roi['name']}', 'coords': ({y_start}, {y_end}, {new_x_start}, {new_x_end}), 'color': '{roi['color']}'}},  # {roi['name']} (PARTIAL)")
print("]")

print("\n" + "="*80)
