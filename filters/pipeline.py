"""
pipeline.py - Complete Filter Pipeline

Applies all 5 filters in sequence to an image.
"""

import numpy as np

from filters.grayscale import grayscale
from filters.gaussian_blur import gaussian_blur
from filters.edge_detection import edge_detection
from filters.sharpen import sharpen
from filters.brightness import adjust_brightness


def apply_all_filters(image: np.ndarray, brightness_value: int = 30) -> np.ndarray:
    """
    Apply the complete 5-stage filter pipeline to an image.
    
    Pipeline order:
        1. Grayscale Conversion - Convert RGB to single channel
        2. Gaussian Blur - Reduce noise
        3. Edge Detection (Sobel) - Find edges
        4. Sharpening - Enhance edges
        5. Brightness Adjustment - Final touch-up
    
    Args:
        image: Input RGB image as numpy array with shape (H, W, 3)
        brightness_value: Value for brightness adjustment (default: 30)
    
    Returns:
        Processed grayscale image as numpy array with shape (H, W)
    
    Notes:
        - This function is designed to be called by worker processes
          in parallel implementations
        - Each stage builds on the previous one
        - The order matters: blurring before edge detection reduces noise
    
    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> img = np.array(Image.open('photo.jpg'))
        >>> processed = apply_all_filters(img, brightness_value=30)
        >>> processed.shape  # (H, W) grayscale
    """
    # Stage 1: Convert to grayscale
    # This reduces 3 channels to 1, simplifying subsequent operations
    result = grayscale(image)
    
    # Stage 2: Apply Gaussian blur
    # Smooths the image and reduces noise before edge detection
    result = gaussian_blur(result)
    
    # Stage 3: Detect edges using Sobel
    # Finds areas of high intensity change (edges)
    result = edge_detection(result)
    
    # Stage 4: Sharpen the image
    # Enhances the detected edges for better visibility
    result = sharpen(result)
    
    # Stage 5: Adjust brightness
    # Final adjustment to make the result more visible
    result = adjust_brightness(result, brightness_value)
    
    return result


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing complete filter pipeline...")
    
    # Create a test RGB image with some structure
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add some features to test
    # Vertical edge
    test_image[:, 25:50, :] = 200
    # Horizontal edge
    test_image[25:50, :, :] = 150
    # Bright spot
    test_image[60:70, 60:70, :] = 255
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Input image dtype: {test_image.dtype}")
    
    # Apply full pipeline
    result = apply_all_filters(test_image, brightness_value=30)
    
    print(f"\nOutput image shape: {result.shape}")
    print(f"Output image dtype: {result.dtype}")
    print(f"Output value range: [{result.min()}, {result.max()}]")
    
    # Test with different brightness values
    for bv in [-50, 0, 50, 100]:
        r = apply_all_filters(test_image, brightness_value=bv)
        print(f"Brightness {bv:+4d}: range [{r.min():3d}, {r.max():3d}]")
    
    # Test with random image
    random_image = np.random.randint(0, 256, size=(50, 50, 3), dtype=np.uint8)
    result_random = apply_all_filters(random_image)
    assert result_random.shape == (50, 50), "Shape transformation failed"
    assert result_random.dtype == np.uint8, "Dtype mismatch"
    print(f"\nRandom image test: OK")
    
    print("\nAll pipeline tests passed!")
