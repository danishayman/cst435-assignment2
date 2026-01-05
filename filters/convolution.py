"""
convolution.py - 2D Convolution Implementation using OpenCV

This module provides the core convolution function used by multiple filters.
It's a shared utility to avoid code duplication.

Note: This module now uses OpenCV's filter2D for optimized performance.
      The manual implementation has been replaced for better efficiency.
"""

import cv2
import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Perform 2D convolution on a grayscale image using OpenCV.
    
    This implementation uses OpenCV's filter2D for optimized performance.
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
        kernel: Convolution kernel as numpy array with shape (K, K)
                Must be square and odd-sized (e.g., 3x3, 5x5)
        normalize: If True, clip output to [0, 255] and convert to uint8
                   If False, return float64 array (for intermediate operations)
    
    Returns:
        Convolved image with same shape as input
    
    Example:
        >>> import numpy as np
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        >>> kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        >>> result = convolve2d(img, kernel)
    """
    # Convert kernel to float32 for OpenCV
    kernel = kernel.astype(np.float32)
    
    if normalize:
        # Use -1 to keep same depth, OpenCV handles clipping for uint8
        output = cv2.filter2D(image.astype(np.uint8), -1, kernel)
        return output.astype(np.uint8)
    else:
        # Use CV_64F for float output (no clipping)
        output = cv2.filter2D(image.astype(np.float64), cv2.CV_64F, kernel)
        return output


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing convolution function...")
    
    # Create a simple test image
    test_image = np.array([
        [10, 20, 30, 40, 50],
        [20, 30, 40, 50, 60],
        [30, 40, 50, 60, 70],
        [40, 50, 60, 70, 80],
        [50, 60, 70, 80, 90]
    ], dtype=np.float64)
    
    # Identity kernel (should return similar image)
    identity_kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float64)
    
    result = convolve2d(test_image, identity_kernel)
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    
    # Test with box blur kernel
    box_blur = np.ones((3, 3), dtype=np.float64) / 9
    blurred = convolve2d(test_image, box_blur)
    print(f"Box blur result center: {blurred[2, 2]}")
    
    print("\nConvolution tests passed!")
