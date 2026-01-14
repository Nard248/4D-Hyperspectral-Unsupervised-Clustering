# Wavelength Selection V2 Architecture Documentation

## Overview
The V2 pipeline enhances the original wavelength selection system with comprehensive ground truth tracking and supervised learning metrics. This enables accurate performance measurement based on known class labels throughout the entire analysis pipeline.

## Key Features

### 1. **Pixel-Level Ground Truth Preservation**
- Every pixel maintains its original ground truth class label throughout the pipeline
- Enables accurate supervised learning metrics calculation at any stage
- Supports both class-level (4 types) and future object-level (16 objects) analysis

### 2. **ROI-to-Class Mapping**
- Automatically maps each ROI to its corresponding ground truth class
- Verifies ROI purity (ensures ROIs contain only one class)
- Tracks mapping relationships for validation

### 3. **Comprehensive Supervised Metrics**
- **Accuracy Metrics**: Overall accuracy, balanced accuracy, per-class accuracy
- **Precision/Recall/F1**: Micro, macro, and weighted averaging
- **Agreement Metrics**: Cohen's Kappa, Matthews Correlation Coefficient
- **Confusion Matrices**: With optimal cluster-to-class mapping via Hungarian algorithm
- **ROI-specific Metrics**: Performance analysis for each ROI region

### 4. **Publication-Quality Visualizations**
Each visualization is saved separately for maximum flexibility:
- Confusion matrices with normalized frequencies
- Per-class performance bar charts
- Spatial accuracy heatmaps
- Misclassification pattern analysis
- ROI performance comparisons
- Metric radar plots
- Class distribution comparisons

## Architecture Components

### Core Modules

#### `ground_truth_tracker.py`
**Purpose**: Maintains pixel-level ground truth throughout the pipeline

**Key Classes**:
- `GroundTruthTracker`: Core tracking class

**Key Methods**:
```python
# Initialize with ground truth array
tracker = GroundTruthTracker(ground_truth, class_names)

# Add ROI mapping
roi_info = tracker.add_roi_mapping(roi_id, coordinates, verify_single_class=True)

# Set predictions for comparison
tracker.set_predictions(predictions)

# Get pixel accuracy
accuracy = tracker.get_pixel_accuracy()

# Export/import state
tracker.export_state(filepath)
loaded_tracker = GroundTruthTracker.load_state(filepath)
```

#### `supervised_metrics.py`
**Purpose**: Calculates comprehensive supervised learning metrics

**Key Classes**:
- `SupervisedMetrics`: Metric calculation class

**Key Methods**:
```python
# Initialize with tracker
metrics_calc = SupervisedMetrics(tracker)

# Calculate all metrics
metrics = metrics_calc.calculate_metrics(predictions, use_hungarian_mapping=True)

# Get classification report
report = metrics_calc.get_classification_report(predictions)

# Calculate ROI-specific metrics
roi_metrics = metrics_calc.calculate_roi_metrics(predictions)

# Export metrics
metrics_calc.export_metrics(filepath, format='json')
```

#### `supervised_visualizations.py`
**Purpose**: Creates individual, publication-quality visualizations

**Key Classes**:
- `SupervisedVisualizations`: Visualization generator

**Key Methods**:
```python
# Initialize visualizer
viz = SupervisedVisualizations(output_dir, dpi=300)

# Create individual plots
viz.plot_confusion_matrix(cm, class_names)
viz.plot_per_class_metrics(per_class_metrics)
viz.plot_accuracy_heatmap(ground_truth, predictions)
viz.plot_misclassification_patterns(ground_truth, predictions)
viz.plot_roi_performance(roi_metrics)
viz.plot_metric_comparison(metrics_dict)

# Create all visualizations at once
viz.create_all_visualizations(metrics, ground_truth, predictions, roi_metrics)
```

#### `wavelengthselectionV2.py`
**Purpose**: Main pipeline integrating all V2 enhancements

**Key Enhancements**:
- Initializes `GroundTruthTracker` at pipeline start
- Maps ROIs to ground truth classes during clustering
- Calculates supervised metrics alongside traditional clustering metrics
- Generates comprehensive visualization suite
- Exports detailed metrics for analysis

## Data Flow

```
1. Load Ground Truth PNG
   ↓
2. Initialize GroundTruthTracker
   ├── Extract class labels
   ├── Build pixel index
   └── Store class distribution
   ↓
3. ROI Selection & Mapping
   ├── Define ROI regions
   ├── Map each ROI to GT class
   └── Verify ROI purity
   ↓
4. Clustering/Classification
   ├── Perform KNN classification
   ├── Generate predictions
   └── Reconstruct spatial map
   ↓
5. Metric Calculation
   ├── Apply Hungarian mapping
   ├── Calculate supervised metrics
   ├── Generate confusion matrix
   └── Compute per-class/ROI metrics
   ↓
6. Visualization & Export
   ├── Create individual plots
   ├── Export metrics (JSON/Excel)
   └── Save state for reproducibility
```

## Usage Example

```python
# 1. Initialize ground truth tracker
from ground_truth_tracker import GroundTruthTracker
from supervised_metrics import SupervisedMetrics
from supervised_visualizations import SupervisedVisualizations

# Load ground truth
ground_truth = load_ground_truth_from_png(png_path)
tracker = GroundTruthTracker(ground_truth, class_names)

# 2. Map ROIs to classes
for roi in roi_regions:
    roi_info = tracker.add_roi_mapping(
        roi_id=roi['name'],
        coordinates=roi['coords'],
        verify_single_class=True
    )

# 3. Run clustering
predictions = run_clustering_pipeline(data)

# 4. Calculate supervised metrics
tracker.set_predictions(predictions)
metrics_calc = SupervisedMetrics(tracker)
metrics = metrics_calc.calculate_metrics(predictions)

# 5. Generate visualizations
viz = SupervisedVisualizations(output_dir)
viz.create_all_visualizations(metrics, ground_truth, predictions)

# 6. Export results
metrics_calc.export_metrics('results.json', format='json')
tracker.export_state('tracker_state.pkl')
```

## Key Advantages

1. **Complete Traceability**: Every pixel's ground truth is preserved
2. **Accurate Metrics**: Supervised learning metrics provide true performance measures
3. **ROI Validation**: Ensures ROIs map correctly to single classes
4. **Optimal Mapping**: Hungarian algorithm finds best cluster-to-class assignment
5. **Flexible Analysis**: Supports pixel, class, and ROI-level metrics
6. **Clean Architecture**: New modules don't modify existing code
7. **Reproducible**: State persistence enables result reproduction

## Output Files

### Metrics Files
- `supervised_metrics.json`: Complete metrics in JSON format
- `supervised_metrics.xlsx`: Multi-sheet Excel with detailed breakdowns
- `classification_report.txt`: Sklearn-style classification report
- `tracker_state.pkl`: Saved tracker state for reproducibility

### Visualization Files
- `confusion_matrix.png`: Normalized confusion matrix
- `per_class_metrics.png`: Precision/Recall/F1 bar charts
- `accuracy_heatmap.png`: Spatial accuracy visualization
- `misclassification_patterns.png`: Per-class error patterns
- `roi_performance.png`: ROI-specific accuracy analysis
- `metrics_comparison.png`: Radar plot of overall metrics
- `class_distribution.png`: GT vs predicted class distributions

## Performance Metrics Explained

### Pixel-Level Metrics
- **Accuracy**: Proportion of correctly classified pixels
- **Balanced Accuracy**: Average of per-class recall scores

### Class-Level Metrics
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of true instances per class

### Agreement Metrics
- **Cohen's Kappa**: Agreement corrected for chance (-1 to 1, higher is better)
- **Matthews Correlation**: Correlation between predicted and true labels (-1 to 1)

### Averaging Strategies
- **Micro**: Calculate metrics globally
- **Macro**: Calculate per-class then average (unweighted)
- **Weighted**: Calculate per-class then average (weighted by support)

## Future Extensions

### Object-Level Analysis (Planned)
- Separate 16 individual objects within 4 classes
- Track object boundaries using connected components
- Calculate per-object performance metrics
- Generate object-wise accuracy heatmaps

### Additional Metrics
- IoU (Intersection over Union) per class
- Dice coefficient
- Boundary accuracy metrics
- Spatial autocorrelation analysis

## Running the V2 Pipeline

```bash
# Run the full V2 pipeline
python wavelengthselectionV2.py

# Test V2 components
python test_v2_pipeline.py

# Run on specific configuration
python wavelengthselectionV2.py --config config_name
```

## Validation

The V2 pipeline has been validated with:
- Unit tests for each component
- Integration tests with sample data
- Comparison with original pipeline metrics
- ROI purity verification
- Metric consistency checks

## Conclusion

The V2 architecture provides a robust framework for accurate performance measurement in hyperspectral wavelength selection. By maintaining ground truth throughout the pipeline and calculating comprehensive supervised metrics, it enables rigorous validation of clustering results and supports publication-quality analysis.