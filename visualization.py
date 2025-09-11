"""
Visualization functions for image registration results.
Provides comprehensive visualization of registration process and outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from typing import Dict, Tuple, Optional, List


class RegistrationVisualizer:
    """
    Class for visualizing image registration results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
    def plot_registration_overview(self, registration_result: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive overview plot of registration results.
        
        Args:
            registration_result: Dictionary containing registration results
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        he_image = registration_result['registered_he_image']
        registered_mask = registration_result['registered_mask']
        original_mask = registration_result['original_complete_mask']
        target_mask = registration_result['target_mask']
        quality_metrics = registration_result['quality_metrics']
        
        # Original HE image (create from registered by reverse transform if needed)
        axes[0, 0].imshow(he_image)
        axes[0, 0].set_title('HE Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Original complete mask
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title('Original Complete Mask', fontsize=14)
        axes[0, 1].axis('off')
        
        # Target mask
        axes[0, 2].imshow(target_mask, cmap='gray')
        axes[0, 2].set_title('Target Mask', fontsize=14)
        axes[0, 2].axis('off')
        
        # Registered mask
        axes[1, 0].imshow(registered_mask, cmap='gray')
        axes[1, 0].set_title('Registered Mask', fontsize=14)
        axes[1, 0].axis('off')
        
        # Overlay comparison
        overlay = self.create_overlay_image(registered_mask, target_mask)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Red: Registered, Green: Target)', fontsize=14)
        axes[1, 1].axis('off')
        
        # Quality metrics
        axes[1, 2].axis('off')
        metrics_text = f"""Registration Quality Metrics:

Dice Coefficient: {quality_metrics['dice']:.4f}
Jaccard Index: {quality_metrics['jaccard']:.4f}
Overlap Ratio: {quality_metrics['overlap']:.4f}
Sensitivity: {quality_metrics['sensitivity']:.4f}
Specificity: {quality_metrics['specificity']:.4f}

Intersection Area: {quality_metrics['intersection_area']:.0f}
Union Area: {quality_metrics['union_area']:.0f}"""
        
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, 
                        verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Quality Assessment', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_transformation_parameters(self, registration_result: Dict, save_path: Optional[str] = None):
        """
        Visualize transformation parameters.
        
        Args:
            registration_result: Dictionary containing registration results
            save_path: Optional path to save the figure
        """
        initial_params = registration_result['initial_parameters']
        final_params = registration_result['transformation_parameters']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        params = ['scale', 'translation', 'rotation']
        initial_values = [
            initial_params['scale'],
            np.linalg.norm(initial_params['translation']),
            np.degrees(initial_params['rotation'])
        ]
        final_values = [
            final_params['scale'],
            np.linalg.norm(final_params['translation']),
            np.degrees(final_params['rotation'])
        ]
        
        x = np.arange(len(params))
        width = 0.35
        
        axes[0].bar(x - width/2, initial_values, width, label='Initial', alpha=0.8)
        axes[0].bar(x + width/2, final_values, width, label='Final', alpha=0.8)
        axes[0].set_xlabel('Parameters')
        axes[0].set_ylabel('Values')
        axes[0].set_title('Transformation Parameters Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Scale', 'Translation\n(magnitude)', 'Rotation\n(degrees)'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Translation components
        tx_values = [initial_params['translation'][0], final_params['translation'][0]]
        ty_values = [initial_params['translation'][1], final_params['translation'][1]]
        
        axes[1].bar(['Initial', 'Final'], tx_values, alpha=0.8, label='TX')
        axes[1].bar(['Initial', 'Final'], ty_values, alpha=0.8, label='TY', bottom=tx_values)
        axes[1].set_title('Translation Components')
        axes[1].set_ylabel('Translation (pixels)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Parameter evolution (if available)
        axes[2].plot([0, 1], [initial_params['scale'], final_params['scale']], 'o-', label='Scale')
        axes[2].plot([0, 1], [np.degrees(initial_params['rotation']), 
                              np.degrees(final_params['rotation'])], 's-', label='Rotation (deg)')
        axes[2].set_xlabel('Optimization Step')
        axes[2].set_ylabel('Parameter Value')
        axes[2].set_title('Parameter Evolution')
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['Initial', 'Final'])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_overlay_image(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Create RGB overlay of two binary masks.
        
        Args:
            mask1: First mask (will be shown in red channel)
            mask2: Second mask (will be shown in green channel)
        
        Returns:
            RGB overlay image
        """
        overlay = np.zeros((*mask1.shape, 3), dtype=np.uint8)
        overlay[..., 0] = mask1.astype(np.uint8) * 255  # Red channel
        overlay[..., 1] = mask2.astype(np.uint8) * 255  # Green channel
        return overlay
    
    def plot_contour_comparison(self, 
                               registration_result: Dict, 
                               save_path: Optional[str] = None):
        """
        Plot contour comparison between registered and target masks.
        
        Args:
            registration_result: Dictionary containing registration results
            save_path: Optional path to save the figure
        """
        registered_mask = registration_result['registered_mask']
        target_mask = registration_result['target_mask']
        
        # Extract contours
        registered_contours, _ = cv2.findContours(
            registered_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        target_contours, _ = cv2.findContours(
            target_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Contour overlay
        axes[0].imshow(np.zeros_like(registered_mask), cmap='gray')
        
        for contour in registered_contours:
            contour = contour.reshape(-1, 2)
            axes[0].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, label='Registered')
        
        for contour in target_contours:
            contour = contour.reshape(-1, 2)
            axes[0].plot(contour[:, 0], contour[:, 1], 'g-', linewidth=2, label='Target')
        
        axes[0].set_title('Contour Comparison')
        axes[0].axis('equal')
        axes[0].legend()
        
        # Error map
        difference = np.abs(registered_mask.astype(int) - target_mask.astype(int))
        im = axes[1].imshow(difference, cmap='hot')
        axes[1].set_title('Difference Map')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_registration_report(self, registration_result: Dict, save_path: Optional[str] = None):
        """
        Create a comprehensive registration report with multiple visualizations.
        
        Args:
            registration_result: Dictionary containing registration results
            save_path: Optional base path to save figures (will append suffixes)
        """
        print("Creating registration report...")
        
        # Overview plot
        overview_path = f"{save_path}_overview.png" if save_path else None
        self.plot_registration_overview(registration_result, overview_path)
        
        # Transformation parameters
        params_path = f"{save_path}_parameters.png" if save_path else None
        self.plot_transformation_parameters(registration_result, params_path)
        
        # Contour comparison
        contour_path = f"{save_path}_contours.png" if save_path else None
        self.plot_contour_comparison(registration_result, contour_path)
        
        # Print summary
        self.print_registration_summary(registration_result)
    
    def print_registration_summary(self, registration_result: Dict):
        """
        Print a summary of registration results.
        
        Args:
            registration_result: Dictionary containing registration results
        """
        params = registration_result['transformation_parameters']
        metrics = registration_result['quality_metrics']
        
        print("\n" + "="*60)
        print("REGISTRATION SUMMARY")
        print("="*60)
        
        print(f"Final Transformation Parameters:")
        print(f"  Scale Factor: {params['scale']:.4f}")
        print(f"  Translation: ({params['translation'][0]:.2f}, {params['translation'][1]:.2f}) pixels")
        print(f"  Rotation: {np.degrees(params['rotation']):.2f} degrees")
        
        print(f"\nQuality Assessment:")
        print(f"  Dice Coefficient: {metrics['dice']:.4f}")
        print(f"  Jaccard Index: {metrics['jaccard']:.4f}")
        print(f"  Overlap Ratio: {metrics['overlap']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        
        # Registration quality assessment
        if metrics['dice'] >= 0.8:
            quality = "Excellent"
        elif metrics['dice'] >= 0.6:
            quality = "Good"
        elif metrics['dice'] >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"\nOverall Registration Quality: {quality}")
        print("="*60)


def plot_sample_data(he_image: np.ndarray, 
                    complete_mask: np.ndarray, 
                    target_mask: np.ndarray, 
                    save_path: Optional[str] = None):
    """
    Visualize sample input data.
    
    Args:
        he_image: HE stained image
        complete_mask: Complete mask matrix
        target_mask: Target mask matrix
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(he_image)
    axes[0].set_title('HE Image')
    axes[0].axis('off')
    
    axes[1].imshow(complete_mask, cmap='gray')
    axes[1].set_title('Complete Mask')
    axes[1].axis('off')
    
    axes[2].imshow(target_mask, cmap='gray')
    axes[2].set_title('Target Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()