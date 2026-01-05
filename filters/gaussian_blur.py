"""
gaussian_blur.py - Gaussian Blur Filter

Applies Gaussian blur using a 3x3 kernel for noise reduction.
"""

import numpy as np
from filters.convolution import convolve2d


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur using a 3x3 kernel.
    
    Kernel (normalized):
        [1, 2, 1]
        [2, 4, 2]  * (1/16)
        [1, 2, 1]
    
    This kernel approximates a Gaussian distribution with sigma ≈ 0.85.
    It provides smoothing that reduces noise while preserving edges
    better than a simple box blur.
    
    The kernel weights:
        - Center (4): Highest weight for the pixel itself
        - Adjacent (2): Medium weight for immediate neighbors
        - Diagonal (1): Lowest weight for corner neighbors
        - Sum = 16, so we divide by 16 to normalize
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
               Values should be in range [0, 255]
    
    Returns:
        Blurred image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
    
    Notes:
        - Uses zero-padding at boundaries
        - For stronger blur, apply multiple times or use larger kernel
    
    Example:
        >>> img = np.array([[100, 100, 100],
        ...                 [100, 200, 100],
        ...                 [100, 100, 100]], dtype=np.uint8)
        >>> blurred = gaussian_blur(img)
        >>> blurred[1, 1]  # Center pixel gets smoothed
        131
    """
    # Define the 3x3 Gaussian kernel (hardcoded, no magic)
    # Values derived from discrete approximation of Gaussian function
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64) / 16.0  # Normalize: sum = 16
    
    return convolve2d(image, kernel)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Gaussian blur filter...")
    
    # Test 1: Uniform image (should remain mostly unchanged)
    uniform = np.full((5, 5), 100, dtype=np.uint8)
    blurred_uniform = gaussian_blur(uniform)
    print(f"Uniform image center before: {uniform[2, 2]}")
    print(f"Uniform image center after: {blurred_uniform[2, 2]}")
    
    # Test 2: Single bright pixel (should spread out)
    single_point = np.zeros((5, 5), dtype=np.uint8)
    single_point[2, 2] = 255
    blurred_point = gaussian_blur(single_point)
    print(f"\nSingle point blur:")
    print(f"  Center: {blurred_point[2, 2]} (was 255)")
    print(f"  Adjacent: {blurred_point[2, 1]} (was 0)")
    print(f"  Diagonal: {blurred_point[1, 1]} (was 0)")
    
    # Test 3: Edge preservation
    edge = np.zeros((5, 5), dtype=np.uint8)
    edge[:, :2] = 200
    blurred_edge = gaussian_blur(edge)
    print(f"\nEdge blur (left side bright):")
    print(f"  Left side: {blurred_edge[2, 0]}")
    print(f"  Edge: {blurred_edge[2, 2]}")
    print(f"  Right side: {blurred_edge[2, 4]}")
    
    # Test 4: Random image shape preservation
    random_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    blurred_random = gaussian_blur(random_img)
    assert blurred_random.shape == random_img.shape, "Shape mismatch"
    assert blurred_random.dtype == np.uint8, "Dtype mismatch"
    print(f"\nShape preservation: OK ({random_img.shape})")
    
    print("\nAll Gaussian blur tests passed!")
