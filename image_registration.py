"""
HE image registration algorithm for aligning HE stained images with mask matrices.
Implements contour-based registration using scaling and translation operations.
"""

import numpy as np
import cv2
from scipy import optimize
from typing import Tuple, Dict, Optional, List
from utils import (
    morphological_cleanup, extract_contours, compute_geometric_features,
    apply_transformation, compute_overlap_metrics, compute_hausdorff_distance
)


class ImageRegistration:
    """
    Main class for HE image registration with mask matrices.
    """
    
    def __init__(self, 
                 morphology_kernel_size: int = 5,
                 min_contour_area: int = 100,
                 max_iterations: int = 1000):
        """
        Initialize registration parameters.
        
        Args:
            morphology_kernel_size: Size of morphological operations kernel
            min_contour_area: Minimum area for contour filtering
            max_iterations: Maximum iterations for optimization
        """
        self.morphology_kernel_size = morphology_kernel_size
        self.min_contour_area = min_contour_area
        self.max_iterations = max_iterations
        
        # Store registration results
        self.registration_result = None
        
    def preprocess_masks(self, 
                        complete_mask: np.ndarray, 
                        target_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess masks using morphological operations.
        
        Args:
            complete_mask: Complete mask matrix
            target_mask: Target mask matrix
        
        Returns:
            Tuple of preprocessed masks
        """
        # Clean up masks
        clean_complete = morphological_cleanup(
            complete_mask, 
            self.morphology_kernel_size,
            ['close', 'open']
        )
        
        clean_target = morphological_cleanup(
            target_mask, 
            self.morphology_kernel_size,
            ['close', 'open']
        )
        
        return clean_complete, clean_target
    
    def coarse_registration(self, 
                          complete_mask: np.ndarray, 
                          target_mask: np.ndarray) -> Dict[str, float]:
        """
        Perform coarse registration based on geometric features.
        
        Args:
            complete_mask: Complete mask matrix
            target_mask: Target mask matrix
        
        Returns:
            Dictionary with initial transformation parameters
        """
        # Extract geometric features
        complete_features = compute_geometric_features(complete_mask)
        target_features = compute_geometric_features(target_mask)
        
        # Calculate initial translation based on centroids
        centroid_diff = target_features['centroid'] - complete_features['centroid']
        initial_translation = (float(centroid_diff[1]), float(centroid_diff[0]))  # (tx, ty)
        
        # Calculate initial scale based on bounding boxes
        complete_bbox = complete_features['bounding_box']
        target_bbox = target_features['bounding_box']
        
        complete_height = complete_bbox[2] - complete_bbox[0]
        complete_width = complete_bbox[3] - complete_bbox[1]
        target_height = target_bbox[2] - target_bbox[0]
        target_width = target_bbox[3] - target_bbox[1]
        
        scale_y = target_height / complete_height if complete_height > 0 else 1.0
        scale_x = target_width / complete_width if complete_width > 0 else 1.0
        initial_scale = float(np.mean([scale_x, scale_y]))
        
        # Calculate initial rotation based on principal axes
        angle_diff = target_features['orientation'] - complete_features['orientation']
        initial_rotation = float(angle_diff)
        
        return {
            'scale': initial_scale,
            'translation': initial_translation,
            'rotation': initial_rotation
        }
    
    def compute_registration_cost(self, 
                                 params: np.ndarray, 
                                 complete_mask: np.ndarray, 
                                 target_mask: np.ndarray) -> float:
        """
        Compute registration cost function.
        
        Args:
            params: [scale, tx, ty, rotation] transformation parameters
            complete_mask: Complete mask matrix
            target_mask: Target mask matrix
        
        Returns:
            Registration cost (lower is better)
        """
        scale, tx, ty, rotation = params
        
        # Apply transformation to complete mask
        transformed_mask = apply_transformation(
            complete_mask.astype(np.uint8), 
            scale, 
            (tx, ty), 
            rotation
        )
        transformed_mask = transformed_mask.astype(bool)
        
        # Compute overlap metrics
        metrics = compute_overlap_metrics(transformed_mask, target_mask)
        
        # Cost function: maximize Dice coefficient (minimize negative Dice)
        cost = 1.0 - metrics['dice']
        
        # Add penalty for extreme transformations to ensure stability
        scale_penalty = 0.1 * abs(scale - 1.0) ** 2
        rotation_penalty = 0.01 * (rotation ** 2)
        
        return cost + scale_penalty + rotation_penalty
    
    def fine_registration(self, 
                         complete_mask: np.ndarray, 
                         target_mask: np.ndarray, 
                         initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fine registration using optimization.
        
        Args:
            complete_mask: Complete mask matrix
            target_mask: Target mask matrix
            initial_params: Initial transformation parameters
        
        Returns:
            Optimized transformation parameters
        """
        # Initial parameter vector [scale, tx, ty, rotation]
        x0 = np.array([
            initial_params['scale'],
            initial_params['translation'][0],
            initial_params['translation'][1],
            initial_params['rotation']
        ])
        
        # Parameter bounds
        bounds = [
            (0.5, 2.0),      # scale: 0.5x to 2x
            (-100, 100),     # tx: translation limits
            (-100, 100),     # ty: translation limits  
            (-np.pi/4, np.pi/4)  # rotation: ±45 degrees
        ]
        
        # Optimization
        try:
            result = optimize.minimize(
                self.compute_registration_cost,
                x0,
                args=(complete_mask, target_mask),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.max_iterations}
            )
            
            if result.success:
                optimized_params = result.x
            else:
                print(f"Optimization failed: {result.message}")
                optimized_params = x0
                
        except Exception as e:
            print(f"Optimization error: {e}")
            optimized_params = x0
        
        return {
            'scale': float(optimized_params[0]),
            'translation': (float(optimized_params[1]), float(optimized_params[2])),
            'rotation': float(optimized_params[3])
        }
    
    def register_image(self, 
                      he_image: np.ndarray, 
                      complete_mask: np.ndarray, 
                      target_mask: np.ndarray) -> Dict:
        """
        Main registration function.
        
        Args:
            he_image: HE stained image [H, W, 3]
            complete_mask: Complete mask matrix [H, W]
            target_mask: Target mask matrix [H, W]
        
        Returns:
            Dictionary containing registration results
        """
        print("Starting image registration...")
        
        # Validate input shapes
        if he_image.shape[:2] != complete_mask.shape or complete_mask.shape != target_mask.shape:
            raise ValueError("Input dimensions must match")
        
        # Preprocess masks
        print("Preprocessing masks...")
        clean_complete, clean_target = self.preprocess_masks(complete_mask, target_mask)
        
        # Coarse registration
        print("Performing coarse registration...")
        coarse_params = self.coarse_registration(clean_complete, clean_target)
        print(f"Initial parameters: {coarse_params}")
        
        # Fine registration
        print("Performing fine registration...")
        fine_params = self.fine_registration(clean_complete, clean_target, coarse_params)
        print(f"Optimized parameters: {fine_params}")
        
        # Apply final transformation to HE image
        print("Applying transformation to HE image...")
        registered_he = apply_transformation(
            he_image, 
            fine_params['scale'],
            fine_params['translation'],
            fine_params['rotation']
        )
        
        # Apply transformation to complete mask for evaluation
        registered_mask = apply_transformation(
            clean_complete.astype(np.uint8),
            fine_params['scale'],
            fine_params['translation'],
            fine_params['rotation']
        ).astype(bool)
        
        # Compute final quality metrics
        print("Computing quality metrics...")
        quality_metrics = compute_overlap_metrics(registered_mask, clean_target)
        
        # Store results
        self.registration_result = {
            'registered_he_image': registered_he,
            'registered_mask': registered_mask,
            'original_complete_mask': clean_complete,
            'target_mask': clean_target,
            'transformation_parameters': fine_params,
            'initial_parameters': coarse_params,
            'quality_metrics': quality_metrics
        }
        
        print("Registration completed successfully!")
        print(f"Final Dice coefficient: {quality_metrics['dice']:.4f}")
        print(f"Final Jaccard index: {quality_metrics['jaccard']:.4f}")
        
        return self.registration_result
    
    def get_transformation_matrix(self, params: Dict[str, float]) -> np.ndarray:
        """
        Get 2D transformation matrix from parameters.
        
        Args:
            params: Transformation parameters
        
        Returns:
            3x3 transformation matrix
        """
        scale = params['scale']
        tx, ty = params['translation']
        rotation = params['rotation']
        
        # Create transformation matrix
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        
        T = np.array([
            [scale * cos_r, -scale * sin_r, tx],
            [scale * sin_r,  scale * cos_r, ty],
            [0,              0,              1]
        ])
        
        return T


def create_sample_data(shape: Tuple[int, int] = (620, 930)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sample data for testing registration algorithm.
    
    Args:
        shape: Image shape (height, width)
    
    Returns:
        Tuple of (he_image, complete_mask, target_mask)
    """
    h, w = shape
    
    # Create synthetic HE image
    he_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Create elliptical complete mask
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Create ellipse
    a, b = h // 4, w // 4  # Semi-axes
    ellipse_mask = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
    
    complete_mask = ellipse_mask.astype(np.uint8)
    
    # Create target mask (shifted and scaled version)
    shift_x, shift_y = 50, 30
    scale_factor = 1.2
    
    # Apply transformation to create target
    M = cv2.getRotationMatrix2D((center_x, center_y), 15, scale_factor)
    M[0, 2] += shift_x
    M[1, 2] += shift_y
    
    target_mask = cv2.warpAffine(complete_mask, M, (w, h), flags=cv2.INTER_NEAREST)
    
    # Add some noise to make it more realistic
    noise = np.random.random((h, w)) < 0.05
    target_mask = np.logical_xor(target_mask, noise).astype(np.uint8)
    
    return he_image, complete_mask, target_mask