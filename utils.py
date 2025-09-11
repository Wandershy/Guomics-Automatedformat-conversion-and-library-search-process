"""
Utility functions for image registration.
Includes morphological operations, feature extraction, and geometric computations.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure
from typing import Tuple, Dict, List


def morphological_cleanup(mask: np.ndarray, 
                         kernel_size: int = 5, 
                         operations: List[str] = ['close', 'open']) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask array (0-1)
        kernel_size: Size of morphological kernel
        operations: List of operations to apply ('open', 'close', 'erode', 'dilate')
    
    Returns:
        Cleaned binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.astype(np.uint8)
    
    for op in operations:
        if op == 'open':
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == 'close':
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == 'erode':
            result = cv2.erode(result, kernel, iterations=1)
        elif op == 'dilate':
            result = cv2.dilate(result, kernel, iterations=1)
    
    return result.astype(bool)


def extract_contours(mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """
    Extract contours from binary mask.
    
    Args:
        mask: Binary mask array
        min_area: Minimum contour area to keep
    
    Returns:
        List of contours as numpy arrays
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            filtered_contours.append(contour)
    
    return filtered_contours


def compute_geometric_features(mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute geometric features from binary mask.
    
    Args:
        mask: Binary mask array
    
    Returns:
        Dictionary containing geometric features:
        - centroid: Center of mass
        - principal_axis: Principal axis direction
        - bounding_box: [min_row, min_col, max_row, max_col]
        - area: Total area
        - orientation: Principal axis angle in radians
    """
    # Find connected components
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)
    
    if not props:
        return {
            'centroid': np.array([0.0, 0.0]),
            'principal_axis': np.array([1.0, 0.0]),
            'bounding_box': np.array([0, 0, mask.shape[0], mask.shape[1]]),
            'area': 0.0,
            'orientation': 0.0
        }
    
    # Use largest component
    largest_component = max(props, key=lambda x: x.area)
    
    # Centroid
    centroid = np.array(largest_component.centroid)
    
    # Principal axis and orientation
    orientation = largest_component.orientation
    principal_axis = np.array([np.cos(orientation), np.sin(orientation)])
    
    # Bounding box
    bbox = largest_component.bbox  # (min_row, min_col, max_row, max_col)
    bounding_box = np.array(bbox)
    
    return {
        'centroid': centroid,
        'principal_axis': principal_axis,
        'bounding_box': bounding_box,
        'area': float(largest_component.area),
        'orientation': orientation
    }


def compute_moments(mask: np.ndarray) -> Dict[str, float]:
    """
    Compute image moments for shape analysis.
    
    Args:
        mask: Binary mask array
    
    Returns:
        Dictionary containing moments and derived features
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    moments = cv2.moments(mask_uint8)
    
    # Centroids
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
    else:
        cx, cy = 0, 0
    
    # Central moments
    mu20 = moments['mu20']
    mu02 = moments['mu02']
    mu11 = moments['mu11']
    
    # Orientation angle
    if mu20 != mu02:
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    else:
        theta = 0 if mu11 == 0 else np.pi/4
    
    return {
        'area': moments['m00'],
        'centroid_x': cx,
        'centroid_y': cy,
        'orientation': theta,
        'mu20': mu20,
        'mu02': mu02,
        'mu11': mu11
    }


def compute_hausdorff_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two contours.
    
    Args:
        contour1: First contour points
        contour2: Second contour points
    
    Returns:
        Hausdorff distance
    """
    def directed_hausdorff(points1, points2):
        distances = []
        for p1 in points1:
            min_dist = float('inf')
            for p2 in points2:
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        return max(distances)
    
    # Reshape contours if needed
    if len(contour1.shape) == 3:
        contour1 = contour1.reshape(-1, 2)
    if len(contour2.shape) == 3:
        contour2 = contour2.reshape(-1, 2)
    
    # Compute bidirectional Hausdorff distance
    dist1 = directed_hausdorff(contour1, contour2)
    dist2 = directed_hausdorff(contour2, contour1)
    
    return max(dist1, dist2)


def apply_transformation(image: np.ndarray, 
                        scale: float, 
                        translation: Tuple[float, float], 
                        rotation: float = 0.0) -> np.ndarray:
    """
    Apply geometric transformation to image.
    
    Args:
        image: Input image
        scale: Scaling factor
        translation: Translation (tx, ty)
        rotation: Rotation angle in radians
    
    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D(center, np.degrees(rotation), scale)
    M[0, 2] += translation[0]
    M[1, 2] += translation[1]
    
    # Apply transformation
    if len(image.shape) == 3:
        transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    else:
        transformed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
    
    return transformed


def compute_overlap_metrics(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
    """
    Compute overlap metrics between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Dictionary containing overlap metrics:
        - dice: Dice coefficient
        - jaccard: Jaccard index
        - overlap: Simple overlap ratio
        - sensitivity: True positive rate
        - specificity: True negative rate
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # Intersection and union
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    
    # Basic metrics
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    mask1_area = np.sum(mask1_bool)
    mask2_area = np.sum(mask2_bool)
    
    # Dice coefficient
    dice = (2.0 * intersection_area) / (mask1_area + mask2_area) if (mask1_area + mask2_area) > 0 else 0.0
    
    # Jaccard index
    jaccard = intersection_area / union_area if union_area > 0 else 0.0
    
    # Simple overlap
    overlap = intersection_area / mask1_area if mask1_area > 0 else 0.0
    
    # Sensitivity (True Positive Rate)
    sensitivity = intersection_area / mask2_area if mask2_area > 0 else 0.0
    
    # Specificity (True Negative Rate)
    total_pixels = mask1.size
    true_negatives = total_pixels - union_area
    false_positives = mask1_area - intersection_area
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    
    return {
        'dice': dice,
        'jaccard': jaccard,
        'overlap': overlap,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'intersection_area': float(intersection_area),
        'union_area': float(union_area)
    }