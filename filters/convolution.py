"""
convolution.py - 2D Convolution Implementation

This module provides the core convolution function used by multiple filters.
It's a shared utility to avoid code duplication.

"""

import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Perform 2D convolution on a grayscale image.
    
    This is a manual implementation for educational purposes.
    Uses zero-padding to maintain image dimensions.
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
        kernel: Convolution kernel as numpy array with shape (K, K)
                Must be square and odd-sized (e.g., 3x3, 5x5)
        normalize: If True, clip output to [0, 255] and convert to uint8
                   If False, return float64 array (for intermediate operations)
    
    Returns:
        Convolved image with same shape as input
    
    Notes:
        - This implementation prioritizes clarity over speed
        - For production, use scipy.ndimage.convolve or cv2.filter2D
    
    Example:
        >>> import numpy as np
        >>> img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        >>> kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        >>> result = convolve2d(img, kernel)
    """
    # Get dimensions
    img_h, img_w = image.shape
    k_size = kernel.shape[0]
    pad = k_size // 2  # Padding size for 'same' output
    
    # Pad the image with zeros
    padded = np.pad(image.astype(np.float64), pad, mode='constant', constant_values=0)
    
    # Initialize output array
    output = np.zeros((img_h, img_w), dtype=np.float64)
    
    # Perform convolution using sliding window
    for i in range(img_h):
        for j in range(img_w):
            # Extract the region of interest
            region = padded[i:i + k_size, j:j + k_size]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    if normalize:
        # Clip to valid range and convert to uint8
        output = np.clip(output, 0, 255)
        return output.astype(np.uint8)
    else:
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
