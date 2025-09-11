"""
Example usage and test code for HE image registration algorithm.
Demonstrates the complete registration pipeline with sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
from image_registration import ImageRegistration, create_sample_data
from visualization import RegistrationVisualizer, plot_sample_data
import time


def main():
    """
    Main function demonstrating the image registration pipeline.
    """
    print("HE Image Registration Algorithm Demo")
    print("="*50)
    
    # Create sample data
    print("Creating sample data...")
    he_image, complete_mask, target_mask = create_sample_data(shape=(620, 930))
    
    print(f"HE image shape: {he_image.shape}")
    print(f"Complete mask shape: {complete_mask.shape}")
    print(f"Target mask shape: {target_mask.shape}")
    
    # Visualize input data
    print("\nVisualizing input data...")
    plot_sample_data(he_image, complete_mask, target_mask)
    
    # Initialize registration algorithm
    print("\nInitializing registration algorithm...")
    registrator = ImageRegistration(
        morphology_kernel_size=7,
        min_contour_area=500,
        max_iterations=1000
    )
    
    # Perform registration
    print("\nPerforming image registration...")
    start_time = time.time()
    
    registration_result = registrator.register_image(
        he_image=he_image,
        complete_mask=complete_mask,
        target_mask=target_mask
    )
    
    end_time = time.time()
    print(f"\nRegistration completed in {end_time - start_time:.2f} seconds")
    
    # Create visualizations
    print("\nGenerating visualization results...")
    visualizer = RegistrationVisualizer()
    visualizer.create_registration_report(registration_result)
    
    return registration_result


def test_with_custom_data():
    """
    Test function for using custom data (template for real usage).
    """
    print("\nTemplate for custom data usage:")
    print("="*40)
    
    # Template code for loading real data
    print("""
# Example of how to use with real data:

import numpy as np
from image_registration import ImageRegistration
from visualization import RegistrationVisualizer

# Load your data (replace with actual loading code)
# he_image = load_he_image('path/to/he_image.tif')  # Shape: [620, 930, 3]
# complete_mask = load_mask('path/to/complete_mask.tif')  # Shape: [620, 930]
# target_mask = load_mask('path/to/target_mask.tif')  # Shape: [620, 930]

# Initialize registration
registrator = ImageRegistration(
    morphology_kernel_size=5,  # Adjust based on your data
    min_contour_area=100,      # Adjust based on your data
    max_iterations=1000
)

# Perform registration
result = registrator.register_image(he_image, complete_mask, target_mask)

# Visualize results
visualizer = RegistrationVisualizer()
visualizer.create_registration_report(result, save_path='registration_results')

# Access results
registered_he = result['registered_he_image']
transformation_params = result['transformation_parameters']
quality_metrics = result['quality_metrics']
""")


def test_parameter_sensitivity():
    """
    Test sensitivity to different parameter settings.
    """
    print("\nTesting parameter sensitivity...")
    print("="*40)
    
    # Create test data
    he_image, complete_mask, target_mask = create_sample_data(shape=(620, 930))
    
    # Test different kernel sizes
    kernel_sizes = [3, 5, 7, 9]
    results = {}
    
    for kernel_size in kernel_sizes:
        print(f"\nTesting with kernel size: {kernel_size}")
        
        registrator = ImageRegistration(
            morphology_kernel_size=kernel_size,
            min_contour_area=100,
            max_iterations=500  # Reduce for faster testing
        )
        
        try:
            result = registrator.register_image(he_image, complete_mask, target_mask)
            results[kernel_size] = result['quality_metrics']['dice']
            print(f"Dice coefficient: {results[kernel_size]:.4f}")
        except Exception as e:
            print(f"Failed with kernel size {kernel_size}: {e}")
            results[kernel_size] = 0.0
    
    # Plot sensitivity results
    plt.figure(figsize=(10, 6))
    plt.bar(kernel_sizes, [results[k] for k in kernel_sizes])
    plt.xlabel('Kernel Size')
    plt.ylabel('Dice Coefficient')
    plt.title('Parameter Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return results


def benchmark_performance():
    """
    Benchmark registration performance with different image sizes.
    """
    print("\nBenchmarking performance...")
    print("="*30)
    
    sizes = [(300, 400), (620, 930), (1000, 1000)]
    times = []
    
    for size in sizes:
        print(f"\nTesting with size: {size}")
        
        # Create test data
        he_image, complete_mask, target_mask = create_sample_data(shape=size)
        
        # Initialize registration
        registrator = ImageRegistration(max_iterations=100)  # Reduce for faster benchmarking
        
        # Time the registration
        start_time = time.time()
        try:
            result = registrator.register_image(he_image, complete_mask, target_mask)
            end_time = time.time()
            registration_time = end_time - start_time
            times.append(registration_time)
            
            print(f"Registration time: {registration_time:.2f} seconds")
            print(f"Dice coefficient: {result['quality_metrics']['dice']:.4f}")
            
        except Exception as e:
            print(f"Failed with size {size}: {e}")
            times.append(float('inf'))
    
    # Plot benchmark results
    size_labels = [f"{s[0]}x{s[1]}" for s in sizes]
    
    plt.figure(figsize=(10, 6))
    plt.bar(size_labels, times)
    plt.xlabel('Image Size')
    plt.ylabel('Registration Time (seconds)')
    plt.title('Performance Benchmark')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return dict(zip(size_labels, times))


if __name__ == "__main__":
    # Run main demonstration
    main_result = main()
    
    # Test with parameter sensitivity
    sensitivity_results = test_parameter_sensitivity()
    
    # Benchmark performance
    benchmark_results = benchmark_performance()
    
    # Show template for custom data
    test_with_custom_data()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nTo use this algorithm with your own data:")
    print("1. Replace the sample data creation with your data loading code")
    print("2. Adjust the registration parameters as needed")
    print("3. Save the results using the visualization tools")
    print("\nFor questions or issues, refer to the documentation in each module.")