"""
Test Script for Object-Wise Analysis Pipeline
==============================================
This script tests the V2 pipeline with object-wise analysis functionality.

Author: Wavelength Selection Pipeline V2 Development
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_object_segmentation():
    """Test the object segmentation module."""
    logger.info("\n" + "="*60)
    logger.info("Testing Object Segmentation Module")
    logger.info("="*60)

    try:
        from object_segmentation import ObjectSegmentation

        # Create a synthetic ground truth image with 4 objects
        ground_truth = np.zeros((100, 100), dtype=int)

        # Object 1: Class 1, top-left
        ground_truth[10:30, 10:30] = 1

        # Object 2: Class 2, top-right
        ground_truth[10:35, 60:85] = 2

        # Object 3: Class 3, bottom-left
        ground_truth[60:90, 15:40] = 3

        # Object 4: Class 1, bottom-right (same class as object 1)
        ground_truth[65:85, 65:90] = 1

        # Initialize segmentation
        segmenter = ObjectSegmentation(connectivity=8, min_object_size=50)

        # Segment objects
        objects = segmenter.segment_objects(ground_truth, background_value=0)

        logger.info(f"✓ Successfully segmented {len(objects)} objects")

        # Get statistics
        stats = segmenter.get_object_statistics()
        logger.info(f"✓ Object statistics:")
        logger.info(f"  - Total objects: {stats['total_objects']}")
        logger.info(f"  - Mean object size: {stats['mean_object_size']:.1f} pixels")
        logger.info(f"  - Objects per class: {stats['objects_per_class']}")

        # Test object extraction
        obj_pixels = segmenter.extract_object_pixels(ground_truth, object_id=1)
        logger.info(f"✓ Extracted pixels for object 1: {len(obj_pixels)} pixels")

        return True

    except Exception as e:
        logger.error(f"✗ Object segmentation test failed: {e}")
        return False


def test_object_metrics():
    """Test the object-wise metrics calculation."""
    logger.info("\n" + "="*60)
    logger.info("Testing Object-Wise Metrics Module")
    logger.info("="*60)

    try:
        from object_segmentation import ObjectSegmentation
        from object_wise_metrics import ObjectWiseMetrics

        # Create synthetic data
        np.random.seed(42)
        ground_truth = np.zeros((100, 100), dtype=int)
        predictions = np.zeros((100, 100), dtype=int)

        # Create 4 objects with varying accuracy
        # Object 1: Perfect prediction
        ground_truth[10:30, 10:30] = 1
        predictions[10:30, 10:30] = 1

        # Object 2: 80% accuracy
        ground_truth[10:35, 60:85] = 2
        predictions[10:35, 60:85] = 2
        predictions[10:15, 60:65] = 3  # Some misclassification

        # Object 3: 60% accuracy
        ground_truth[60:90, 15:40] = 3
        predictions[60:90, 15:40] = 3
        predictions[70:80, 20:30] = 1  # More misclassification

        # Object 4: Poor prediction
        ground_truth[65:85, 65:90] = 1
        predictions[65:85, 65:90] = 2  # Completely wrong

        # Segment objects
        segmenter = ObjectSegmentation(connectivity=8, min_object_size=50)
        objects = segmenter.segment_objects(ground_truth)

        # Calculate metrics
        metrics_calc = ObjectWiseMetrics(segmenter)
        object_metrics = metrics_calc.calculate_object_metrics(
            ground_truth, predictions, apply_hungarian=True
        )

        logger.info(f"✓ Calculated metrics for {len(object_metrics)} objects")

        # Check individual object metrics
        for obj_id, metrics in object_metrics.items():
            if 'error' not in metrics:
                logger.info(f"  Object {obj_id}: Accuracy={metrics['accuracy']:.3f}, "
                          f"Class={metrics['class_label']}")

        # Test aggregation
        class_agg = metrics_calc.aggregate_metrics_by_class()
        logger.info(f"✓ Aggregated metrics by {len(class_agg)} classes")

        # Test best/worst identification
        best, worst = metrics_calc.get_best_worst_objects('accuracy', n=2)
        logger.info(f"✓ Identified {len(best)} best and {len(worst)} worst objects")

        # Test performance matrix creation
        perf_matrix = metrics_calc.create_performance_matrix()
        logger.info(f"✓ Created performance matrix with shape {perf_matrix.shape}")

        return True

    except Exception as e:
        logger.error(f"✗ Object metrics test failed: {e}")
        return False


def test_object_visualizations():
    """Test the object-wise visualization functions."""
    logger.info("\n" + "="*60)
    logger.info("Testing Object-Wise Visualizations Module")
    logger.info("="*60)

    try:
        from object_segmentation import ObjectSegmentation
        from object_wise_metrics import ObjectWiseMetrics
        from object_wise_visualizations import ObjectWiseVisualizations

        # Create synthetic data
        np.random.seed(42)
        ground_truth = np.zeros((100, 100), dtype=int)
        predictions = np.zeros((100, 100), dtype=int)

        # Add some objects
        ground_truth[10:30, 10:30] = 1
        predictions[10:30, 10:30] = 1

        ground_truth[10:35, 60:85] = 2
        predictions[10:35, 60:85] = 2

        ground_truth[60:90, 15:40] = 3
        predictions[60:90, 15:40] = 3

        ground_truth[65:85, 65:90] = 1
        predictions[65:85, 65:90] = 1

        # Add some noise to predictions
        noise_mask = np.random.random((100, 100)) > 0.9
        predictions[noise_mask] = np.random.randint(0, 4, size=np.sum(noise_mask))

        # Segment and calculate metrics
        segmenter = ObjectSegmentation(connectivity=8, min_object_size=50)
        objects = segmenter.segment_objects(ground_truth)

        metrics_calc = ObjectWiseMetrics(segmenter)
        object_metrics = metrics_calc.calculate_object_metrics(ground_truth, predictions)

        # Initialize visualizer
        viz = ObjectWiseVisualizations(segmenter, metrics_calc)

        # Create test output directory
        test_output_dir = "results/test_object_wise"
        os.makedirs(test_output_dir, exist_ok=True)

        # Test visualization functions (without display)
        plt.ioff()  # Turn off interactive mode

        # Test object boundaries plot
        viz.plot_object_boundaries_with_metrics(
            ground_truth, predictions, metric='accuracy',
            save_path=os.path.join(test_output_dir, "test_boundaries.png")
        )
        plt.close('all')
        logger.info("✓ Created object boundaries visualization")

        # Test performance bars
        viz.plot_object_performance_bars(
            save_path=os.path.join(test_output_dir, "test_bars.png")
        )
        plt.close('all')
        logger.info("✓ Created performance bars visualization")

        # Test heatmap
        viz.plot_performance_heatmap(
            save_path=os.path.join(test_output_dir, "test_heatmap.png")
        )
        plt.close('all')
        logger.info("✓ Created performance heatmap")

        plt.ion()  # Turn interactive mode back on

        return True

    except Exception as e:
        logger.error(f"✗ Object visualizations test failed: {e}")
        return False


def test_full_pipeline_integration():
    """Test the full integrated pipeline with a small configuration."""
    logger.info("\n" + "="*60)
    logger.info("Testing Full Pipeline Integration")
    logger.info("="*60)

    try:
        # Check if data files exist
        data_path = r"F:\HS_DATA\Lichens\lichens_mini.h5"
        mask_path = r"F:\HS_DATA\Lichens\mask_mini.npy"

        if not os.path.exists(data_path) or not os.path.exists(mask_path):
            logger.warning("Data files not found. Skipping integration test.")
            logger.info("To run full integration test, ensure data files are available at:")
            logger.info(f"  - {data_path}")
            logger.info(f"  - {mask_path}")
            return None

        from wavelengthselectionV2SeparateObjectAnalysis import WavelengthSelectionV2ObjectAnalysis
        from generated_configs import wavelength_configs

        # Initialize pipeline
        pipeline = WavelengthSelectionV2ObjectAnalysis(
            data_path=data_path,
            mask_path=mask_path,
            sample_name="Lichens_Test",
            output_dir="results/test_object_wise_integration"
        )

        # Run with just 1 configuration for testing
        test_configs = wavelength_configs[:1]
        pipeline.run_all_configurations(test_configs, max_configs=1)

        logger.info("✓ Full pipeline integration test completed successfully")
        logger.info(f"  Results saved to: results/test_object_wise_integration")

        return True

    except Exception as e:
        logger.error(f"✗ Pipeline integration test failed: {e}")
        logger.error(f"  Error details: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("OBJECT-WISE ANALYSIS PIPELINE TEST SUITE")
    logger.info("="*60)

    # Track test results
    results = {
        "Object Segmentation": test_object_segmentation(),
        "Object Metrics": test_object_metrics(),
        "Object Visualizations": test_object_visualizations(),
        "Full Pipeline Integration": test_full_pipeline_integration()
    }

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            logger.info(f"✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            logger.error(f"✗ {test_name}: FAILED")
            failed += 1
        else:
            logger.warning(f"○ {test_name}: SKIPPED")
            skipped += 1

    logger.info(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        logger.info("\n✓ All tests passed successfully!")
        logger.info("\nThe object-wise analysis pipeline is ready to use.")
        logger.info("To run the full pipeline, use:")
        logger.info("  python wavelengthselectionV2SeparateObjectAnalysis.py")
        logger.info("\nTo limit configurations, use:")
        logger.info("  python wavelengthselectionV2SeparateObjectAnalysis.py --max-configs 3")
    else:
        logger.error(f"\n✗ {failed} tests failed. Please check the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)