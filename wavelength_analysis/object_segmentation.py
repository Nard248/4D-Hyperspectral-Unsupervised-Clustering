"""
Object Segmentation Module for Wavelength Selection Pipeline V2
================================================================
This module provides functionality to segment and identify individual objects
in hyperspectral images using connected components analysis.

Author: Wavelength Selection Pipeline V2 Development
Date: 2025
"""

import numpy as np
from scipy import ndimage
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentedObject:
    """Represents a single segmented object with its properties."""
    object_id: int
    pixel_mask: np.ndarray  # Boolean mask for this object's pixels
    pixel_count: int
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    class_label: Optional[int] = None
    roi_id: Optional[str] = None


class ObjectSegmentation:
    """
    Handles spatial segmentation of objects in hyperspectral images.

    This class identifies and separates individual objects based on spatial
    connectivity, enabling per-object analysis of classification metrics.
    """

    def __init__(self, connectivity: int = 8, min_object_size: int = 50):
        """
        Initialize the ObjectSegmentation module.

        Args:
            connectivity: Connectivity for connected components (4 or 8)
            min_object_size: Minimum number of pixels for an object to be considered
        """
        self.connectivity = connectivity
        self.min_object_size = min_object_size
        self.objects = []
        self.object_map = None
        self.num_objects = 0

    def segment_objects(self, ground_truth: np.ndarray,
                        background_value: int = 0) -> List[SegmentedObject]:
        """
        Segment individual objects from ground truth using connected components.

        Args:
            ground_truth: 2D array with class labels
            background_value: Value representing background pixels to ignore

        Returns:
            List of SegmentedObject instances
        """
        logger.info("Starting object segmentation...")

        # Create binary mask excluding background
        foreground_mask = ground_truth != background_value

        # Define connectivity structure
        if self.connectivity == 4:
            structure = np.array([[0, 1, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]])
        else:  # 8-connectivity
            structure = np.ones((3, 3))

        # Perform connected components analysis
        labeled_array, num_features = ndimage.label(foreground_mask, structure=structure)

        logger.info(f"Found {num_features} potential objects")

        # Extract individual objects
        self.objects = []
        self.object_map = np.zeros_like(ground_truth, dtype=np.int32)
        object_counter = 1

        for obj_id in range(1, num_features + 1):
            # Create mask for this object
            obj_mask = labeled_array == obj_id
            pixel_count = np.sum(obj_mask)

            # Skip small objects
            if pixel_count < self.min_object_size:
                logger.debug(f"Skipping object {obj_id} with only {pixel_count} pixels")
                continue

            # Get object properties
            rows, cols = np.where(obj_mask)
            centroid = (np.mean(rows), np.mean(cols))
            bounding_box = (np.min(rows), np.min(cols), np.max(rows), np.max(cols))

            # Get the most common class label for this object
            obj_labels = ground_truth[obj_mask]
            unique_labels, counts = np.unique(obj_labels, return_counts=True)
            # Filter out background
            non_bg_mask = unique_labels != background_value
            if np.any(non_bg_mask):
                unique_labels = unique_labels[non_bg_mask]
                counts = counts[non_bg_mask]
                class_label = unique_labels[np.argmax(counts)]
            else:
                class_label = background_value

            # Create SegmentedObject
            seg_obj = SegmentedObject(
                object_id=object_counter,
                pixel_mask=obj_mask,
                pixel_count=pixel_count,
                centroid=centroid,
                bounding_box=bounding_box,
                class_label=class_label
            )

            self.objects.append(seg_obj)
            self.object_map[obj_mask] = object_counter
            object_counter += 1

        self.num_objects = len(self.objects)
        logger.info(f"Segmented {self.num_objects} objects after filtering")

        return self.objects

    def get_object_by_id(self, object_id: int) -> Optional[SegmentedObject]:
        """Get a specific object by its ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_objects_by_class(self, class_label: int) -> List[SegmentedObject]:
        """Get all objects belonging to a specific class."""
        return [obj for obj in self.objects if obj.class_label == class_label]

    def get_object_statistics(self) -> Dict:
        """
        Get statistics about the segmented objects.

        Returns:
            Dictionary containing object statistics
        """
        if not self.objects:
            return {}

        pixel_counts = [obj.pixel_count for obj in self.objects]
        class_labels = [obj.class_label for obj in self.objects]

        unique_classes, class_counts = np.unique(class_labels, return_counts=True)

        stats = {
            'total_objects': self.num_objects,
            'mean_object_size': np.mean(pixel_counts),
            'std_object_size': np.std(pixel_counts),
            'min_object_size': np.min(pixel_counts),
            'max_object_size': np.max(pixel_counts),
            'objects_per_class': dict(zip(unique_classes, class_counts)),
            'class_distribution': {
                int(cls): float(count / self.num_objects)
                for cls, count in zip(unique_classes, class_counts)
            }
        }

        return stats

    def extract_object_pixels(self, data: np.ndarray, object_id: int) -> np.ndarray:
        """
        Extract pixels belonging to a specific object from any data array.

        Args:
            data: 2D or 3D data array
            object_id: ID of the object to extract

        Returns:
            Array of pixels belonging to the object
        """
        obj = self.get_object_by_id(object_id)
        if obj is None:
            raise ValueError(f"Object with ID {object_id} not found")

        if data.ndim == 2:
            return data[obj.pixel_mask]
        elif data.ndim == 3:
            return data[obj.pixel_mask, :]
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    def create_object_visualization_map(self) -> np.ndarray:
        """
        Create a visualization map with different colors for each object.

        Returns:
            RGB array for visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Create colormap
        cmap = plt.cm.get_cmap('tab20')

        # Create RGB image
        vis_map = np.zeros((*self.object_map.shape, 3))

        for obj in self.objects:
            # Get color for this object
            color = cmap(obj.object_id / (self.num_objects + 1))[:3]
            vis_map[obj.pixel_mask] = color

        return vis_map

    def assign_roi_to_objects(self, roi_regions: List[Dict]) -> None:
        """
        Assign ROI IDs to objects based on their location.

        Args:
            roi_regions: List of ROI dictionaries with 'id' and 'coordinates'
        """
        for obj in self.objects:
            centroid_row, centroid_col = obj.centroid

            for roi in roi_regions:
                x, y, w, h = roi['coordinates']
                if x <= centroid_col < x + w and y <= centroid_row < y + h:
                    obj.roi_id = roi['id']
                    break

    def get_object_summary(self) -> List[Dict]:
        """
        Get a summary of all objects as a list of dictionaries.

        Returns:
            List of dictionaries containing object information
        """
        summary = []
        for obj in self.objects:
            summary.append({
                'object_id': obj.object_id,
                'class_label': obj.class_label,
                'pixel_count': obj.pixel_count,
                'centroid': obj.centroid,
                'bounding_box': obj.bounding_box,
                'roi_id': obj.roi_id
            })
        return summary