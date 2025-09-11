# HE Image Registration Algorithm

A comprehensive image registration algorithm for aligning HE (Hematoxylin-Eosin) stained images with mask matrices using scaling, translation, and rotation operations.

## Overview

This implementation provides automatic registration of HE stained images with corresponding mask matrices through contour-based registration. The algorithm includes preprocessing, feature extraction, coarse and fine registration, quality assessment, and comprehensive visualization capabilities.

## Features

- **Morphological Preprocessing**: Cleanup and enhancement of mask data
- **Geometric Feature Extraction**: Centroid, principal axis, bounding box calculations
- **Two-Stage Registration**: Coarse alignment followed by fine optimization
- **Quality Assessment**: Dice coefficient, Jaccard index, overlap metrics
- **Comprehensive Visualization**: Registration process and results visualization
- **Parameter Optimization**: Automatic parameter tuning with bounds checking

## File Structure

```
├── image_registration.py     # Main registration algorithm
├── utils.py                 # Utility functions (morphology, features, metrics)
├── visualization.py         # Visualization and reporting tools
├── example.py              # Usage examples and testing
├── requirements.txt        # Python dependencies
└── README_ImageRegistration.md   # This documentation
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Import the modules:
```python
from image_registration import ImageRegistration
from visualization import RegistrationVisualizer
```

## Quick Start

### Basic Usage

```python
import numpy as np
from image_registration import ImageRegistration, create_sample_data
from visualization import RegistrationVisualizer

# Load or create sample data
he_image, complete_mask, target_mask = create_sample_data(shape=(620, 930))

# Initialize registration algorithm
registrator = ImageRegistration(
    morphology_kernel_size=5,    # Morphological operations kernel size
    min_contour_area=100,        # Minimum contour area threshold
    max_iterations=1000          # Maximum optimization iterations
)

# Perform registration
result = registrator.register_image(
    he_image=he_image,          # HE stained image [H, W, 3]
    complete_mask=complete_mask, # Complete mask matrix [H, W]
    target_mask=target_mask      # Target mask matrix [H, W]
)

# Visualize results
visualizer = RegistrationVisualizer()
visualizer.create_registration_report(result, save_path='registration_results')
```

### Working with Real Data

```python
# Example for loading real medical images
import cv2
import numpy as np

# Load HE image (RGB, shape: [H, W, 3])
he_image = cv2.imread('path/to/he_image.tif')
he_image = cv2.cvtColor(he_image, cv2.COLOR_BGR2RGB)

# Load masks (binary, shape: [H, W])
complete_mask = cv2.imread('path/to/complete_mask.tif', cv2.IMREAD_GRAYSCALE)
target_mask = cv2.imread('path/to/target_mask.tif', cv2.IMREAD_GRAYSCALE)

# Ensure binary masks (0 or 1)
complete_mask = (complete_mask > 127).astype(np.uint8)
target_mask = (target_mask > 127).astype(np.uint8)

# Perform registration
registrator = ImageRegistration()
result = registrator.register_image(he_image, complete_mask, target_mask)
```

## Algorithm Details

### 1. Preprocessing

The algorithm starts with morphological cleanup of the input masks:

- **Closing**: Fill small holes and gaps
- **Opening**: Remove small noise artifacts
- **Kernel Size**: Configurable morphological kernel size

### 2. Feature Extraction

Geometric features are computed for both masks:

- **Centroid**: Center of mass calculation
- **Principal Axis**: Main orientation direction
- **Bounding Box**: Spatial extent information
- **Area**: Total tissue area

### 3. Coarse Registration

Initial alignment based on geometric features:

- **Translation**: Align centroids
- **Scale**: Match bounding box dimensions
- **Rotation**: Align principal axes

### 4. Fine Registration

Optimization-based refinement:

- **Cost Function**: Negative Dice coefficient
- **Optimization Method**: L-BFGS-B with bounds
- **Parameters**: Scale (0.5-2.0x), Translation (±100px), Rotation (±45°)

### 5. Quality Assessment

Multiple metrics for registration evaluation:

- **Dice Coefficient**: 2×intersection/(area1 + area2)
- **Jaccard Index**: intersection/union
- **Overlap Ratio**: intersection/area1
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate

## Configuration Parameters

### ImageRegistration Class

```python
ImageRegistration(
    morphology_kernel_size=5,    # Morphological operations kernel size
    min_contour_area=100,        # Minimum contour area for filtering
    max_iterations=1000          # Maximum optimization iterations
)
```

### Parameter Guidelines

- **morphology_kernel_size**: 3-9 (larger for noisier images)
- **min_contour_area**: 50-500 (depends on image resolution)
- **max_iterations**: 100-2000 (more for higher accuracy)

## Output Structure

The registration result contains:

```python
{
    'registered_he_image': np.ndarray,      # Transformed HE image
    'registered_mask': np.ndarray,          # Transformed complete mask
    'original_complete_mask': np.ndarray,   # Cleaned original mask
    'target_mask': np.ndarray,              # Cleaned target mask
    'transformation_parameters': {          # Final transformation
        'scale': float,                     # Scale factor
        'translation': (float, float),      # Translation (tx, ty)
        'rotation': float                   # Rotation angle (radians)
    },
    'initial_parameters': dict,             # Coarse registration result
    'quality_metrics': {                    # Registration quality
        'dice': float,                      # Dice coefficient
        'jaccard': float,                   # Jaccard index
        'overlap': float,                   # Overlap ratio
        'sensitivity': float,               # True positive rate
        'specificity': float,               # True negative rate
        'intersection_area': float,         # Intersection area
        'union_area': float                 # Union area
    }
}
```

## Visualization Features

### Registration Overview
- Original and registered images
- Mask overlays and comparisons
- Quality metrics summary

### Parameter Analysis
- Transformation parameter evolution
- Initial vs. final parameter comparison
- Parameter sensitivity analysis

### Contour Comparison
- Contour overlay visualization
- Difference maps
- Registration error analysis

## Performance Characteristics

### Typical Performance
- **Small images** (300×400): ~0.5 seconds
- **Medium images** (620×930): ~2-5 seconds  
- **Large images** (1000×1000): ~5-15 seconds

### Quality Expectations
- **Excellent**: Dice > 0.8 (typical for clean data)
- **Good**: Dice > 0.6 (acceptable for clinical use)
- **Fair**: Dice > 0.4 (may need parameter tuning)
- **Poor**: Dice < 0.4 (check input data quality)

## Troubleshooting

### Common Issues

1. **Low Registration Quality**
   - Increase morphology kernel size
   - Adjust min_contour_area threshold
   - Check input mask quality

2. **Optimization Failures**
   - Reduce max_iterations for faster execution
   - Check parameter bounds appropriateness
   - Coarse registration often sufficient

3. **Memory Issues**
   - Process smaller image tiles
   - Reduce image resolution for testing
   - Use lower precision data types

### Input Data Requirements

- **HE Image**: RGB format, shape [H, W, 3]
- **Masks**: Binary format (0/1), shape [H, W]
- **Dimensions**: All inputs must have matching spatial dimensions
- **Data Type**: uint8 for images, bool/uint8 for masks

## Examples

See `example.py` for comprehensive usage examples including:
- Basic registration demonstration
- Parameter sensitivity analysis
- Performance benchmarking
- Custom data integration templates

## Dependencies

- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- scikit-image >= 0.18.0

## Contributing

This implementation follows standard Python coding practices:
- PEP 8 style guidelines
- Comprehensive docstrings
- Type hints where appropriate
- Modular design for extensibility

## License

This code is provided as part of the Guomics project for academic and research purposes.