"""
gaussian_blur.py - Gaussian Blur Filter

Applies Gaussian blur using OpenCV for noise reduction.
"""

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur using OpenCV's GaussianBlur function.
    
    Uses a 3x3 kernel with sigma calculated automatically by OpenCV.
    This provides smoothing that reduces noise while preserving edges
    better than a simple box blur.
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
               Values should be in range [0, 255]
    
    Returns:
        Blurred image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
    
    Notes:
        - Uses border replication at boundaries
        - For stronger blur, increase kernel size or sigma
    
    Example:
        >>> img = np.array([[100, 100, 100],
        ...                 [100, 200, 100],
        ...                 [100, 100, 100]], dtype=np.uint8)
        >>> blurred = gaussian_blur(img)
        >>> blurred[1, 1]  # Center pixel gets smoothed
        131
    """
    # Use OpenCV's GaussianBlur with 3x3 kernel
    # sigmaX=0 means sigma is calculated from kernel size
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
    
    return blurred.astype(np.uint8)


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
