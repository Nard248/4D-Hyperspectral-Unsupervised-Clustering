# ROI Overlay Visualization with Accuracy Metrics

## Feature Summary
Enhanced the V2 pipeline with ROI overlay visualizations that display accuracy metrics directly on top of clustering results. This provides immediate visual feedback on how well each ROI is being classified.

## What Was Added

### 1. **Enhanced Visualization Function**
- `plot_roi_overlay_with_accuracy()` in `supervised_visualizations.py`
- Creates a 3-panel visualization:
  - **Panel 1**: Clustering result with overall accuracy displayed
  - **Panel 2**: ROI boxes overlaid on clustering with individual accuracy metrics
  - **Panel 3**: Bar chart comparing ROI accuracies vs overall accuracy

### 2. **Integration in V2 Pipeline**
- Automatically generates ROI overlay visualizations for:
  - **Baseline** (full data) clustering
  - **Each configuration** tested
- Saves visualizations to multiple locations:
  - `supervised_visualizations/` - with all other metrics
  - `paper-results/` - for easy access and paper inclusion
  - `experiments/[config]/` - with configuration-specific results

### 3. **Key Features**
- **Accuracy labels on ROIs**: Each ROI box shows its accuracy percentage
- **Color-coded boundaries**: ROI regions use consistent colors (Red, Blue, Green, Yellow)
- **Overall vs ROI comparison**: Visual comparison of overall accuracy vs individual ROIs
- **Inside-ROI accuracy display**: Large ROIs show accuracy inside the region
- **Comparison line**: Red dashed line shows overall accuracy for reference

## Output Locations

```
validation_results_v2/
├── paper-results/
│   ├── BASELINE_roi_overlay.png         # Baseline ROI overlay
│   ├── [config]_roi_overlay.png         # Per-configuration overlays
│   └── ...
├── supervised_visualizations/
│   └── roi_overlay_accuracy.png         # Main directory overlay
└── experiments/[config]/
    ├── supervised_visualizations/
    │   └── roi_overlay_accuracy.png     # Configuration-specific
    └── [config]_roi_overlay_main.png    # Standalone version
```

## Usage Examples

### 1. **In V2 Pipeline (Automatic)**
```python
# Automatically generated when running:
python wavelengthselectionV2.py
```

### 2. **Standalone Usage**
```python
from supervised_visualizations import SupervisedVisualizations

# Create visualizer
viz = SupervisedVisualizations(output_dir=output_path, dpi=300)

# Create ROI overlay with accuracy
viz.plot_roi_overlay_with_accuracy(
    cluster_map=predictions,
    ground_truth=ground_truth,
    roi_regions=ROI_REGIONS,
    overall_accuracy=accuracy,
    roi_metrics=roi_metrics,
    title="Experiment Name",
    save_name="roi_overlay.png"
)
```

### 3. **Demo Script**
```python
# Run the demo to see example visualizations:
python demo_roi_overlay.py
```

## Visual Elements

### Panel 1: Clustering Result
- Full clustering visualization
- Overall accuracy displayed in top-left corner
- White background box for readability

### Panel 2: ROI Overlay
- Semi-transparent clustering result
- Bold colored ROI boundaries
- ROI names above each region
- Accuracy percentage for each ROI
- Inside-ROI accuracy display (if space permits)
- Title includes overall accuracy

### Panel 3: Accuracy Comparison
- Bar chart with ROI colors
- Individual ROI accuracies
- Red dashed line for overall accuracy
- Legend indicating overall accuracy
- Grid for easy reading

## Benefits

1. **Immediate Visual Feedback**: See at a glance how well each ROI is classified
2. **Spatial Understanding**: Understand where classification errors occur
3. **Publication Ready**: High-quality figures for papers (300 DPI)
4. **Comparative Analysis**: Easy comparison between configurations
5. **Performance Tracking**: Track improvements across experiments

## Testing

Two test scripts are provided:
1. `test_roi_overlay_v2.py` - Quick integration test
2. `demo_roi_overlay.py` - Full demonstration with sample data

Both confirm the feature is working correctly.

## Paper Usage

For publications, use the images from `paper-results/`:
- Clean titles without extra text
- 300 DPI resolution
- Consistent formatting across all experiments
- Direct accuracy metrics visible on the figure

## Notes

- ROI regions must be defined with coordinates and colors
- Accuracy metrics are calculated using the SupervisedMetrics module
- Hungarian algorithm ensures optimal cluster-to-class mapping
- Background pixels (class -1) are excluded from accuracy calculations
- Each ROI should ideally contain a single ground truth class for best results

## Future Enhancements

Potential additions:
- Confusion matrix overlay on ROIs
- Precision/Recall display option
- Animated transitions between configurations
- 3D visualization for spectral dimension
- Export to interactive HTML format