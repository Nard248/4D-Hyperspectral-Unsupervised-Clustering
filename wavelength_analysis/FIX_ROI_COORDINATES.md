# FIX: ROI Coordinates After Data Cropping

## Problem

The code crops the data from columns 467-1392 (925 pixels), but continues using the ORIGINAL ROI coordinates. This causes 3 out of 4 ROIs to be lost or have insufficient training data.

## Root Cause

In `WavelengthSelectionFinal.py`:

1. **Line 254-259**: `ROI_REGIONS` defined with original coordinates
2. **Line 511-544**: Data cropped from column 467 to 1392
3. **Line 322-329**: KNN training extracts ROIs using ORIGINAL coordinates on CROPPED data

## Current State After Cropping

| ROI | Original X | Cropped X | Status | Training Data |
|-----|-----------|-----------|--------|---------------|
| Region 1 (Red) | 100-150 | LOST | Outside crop | **NONE** |
| Region 2 (Blue) | 250-300 | LOST | Outside crop | **NONE** |
| Region 3 (Green) | 425-475 | 0-8 | Partial | **8 pixels** |
| Region 4 (Yellow) | 650-700 | 183-233 | Full | **50x50 pixels** |

## Solution

### Option 1: Update ROI_REGIONS After Cropping

Replace the ROI_REGIONS definition (after line 544 where cropping happens):

```python
# After cropping, update ROI coordinates to match new coordinate system
start_col = 1392 - 925  # = 467

# Adjust ROI regions for cropped data
ROI_REGIONS = []
ROI_REGIONS_ORIGINAL = [
    {'name': 'Region 1', 'coords': (175, 225, 100, 150), 'color': '#FF0000'},
    {'name': 'Region 2', 'coords': (175, 225, 250, 300), 'color': '#0000FF'},
    {'name': 'Region 3', 'coords': (175, 225, 425, 475), 'color': '#00FF00'},
    {'name': 'Region 4', 'coords': (175, 225, 650, 700), 'color': '#FFFF00'},
]

for roi in ROI_REGIONS_ORIGINAL:
    y_start, y_end, x_start, x_end = roi['coords']

    # Check if ROI is within cropped region
    if x_start >= start_col and x_end <= end_col:
        # Adjust coordinates
        new_x_start = x_start - start_col
        new_x_end = x_end - start_col
        ROI_REGIONS.append({
            'name': roi['name'],
            'coords': (y_start, y_end, new_x_start, new_x_end),
            'color': roi['color']
        })
        print(f"✓ {roi['name']}: x={x_start}-{x_end} → x={new_x_start}-{new_x_end} (after crop)")
    elif x_end > start_col and x_start < end_col:
        # Partial overlap
        overlap_start = max(x_start, start_col)
        overlap_end = min(x_end, end_col)
        new_x_start = overlap_start - start_col
        new_x_end = overlap_end - start_col
        ROI_REGIONS.append({
            'name': roi['name'],
            'coords': (y_start, y_end, new_x_start, new_x_end),
            'color': roi['color']
        })
        print(f"⚠ {roi['name']}: PARTIAL - x={new_x_start}-{new_x_end} (only {new_x_end-new_x_start} pixels)")
    else:
        print(f"✗ {roi['name']}: LOST (outside cropped region)")

print(f"\n✓ Using {len(ROI_REGIONS)} ROIs for training (out of {len(ROI_REGIONS_ORIGINAL)} original)")
```

### Option 2: Don't Crop the Data (or Crop Less)

If you need all 4 ROIs, either:
1. Don't crop the data at all
2. Crop less (e.g., from column 0 to 925 instead of 467 to 1392)

### Option 3: Define New ROI Coordinates for Cropped Data

If the ground truth has different ROI locations after cropping, you need to visually inspect the cropped ground truth and define new ROI coordinates that match where the colored regions actually are.

## Next Steps

1. **Check your ground truth**: After cropping, where are the 4 colored regions actually located?
2. **Match ROI coordinates to ground truth**: The ROI_REGIONS coordinates must match where the colored regions are in the cropped ground truth
3. **Verify KNN training**: After fixing, check that all 4 ROIs have sufficient training data

## Quick Fix for Testing

Replace lines 254-259 in `WavelengthSelectionFinal.py` with:

```python
# ROI regions adjusted for cropped data (columns 467-1392)
ROI_REGIONS = [
    # Region 3 (Green) - only 8 pixels remain after crop
    {'name': 'Region 3', 'coords': (175, 225, 0, 8), 'color': '#00FF00'},
    # Region 4 (Yellow) - fully within crop
    {'name': 'Region 4', 'coords': (175, 225, 183, 233), 'color': '#FFFF00'},
]
```

But this only gives you 2 ROIs. You need to either not crop, or find where all 4 ROIs are in the cropped data.
