"""
sharpen.py - Image Sharpening Filter

Enhances edges and details using a sharpening kernel.
"""

import numpy as np
from filters.convolution import convolve2d


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the image using a sharpening kernel.
    
    Sharpening Kernel:
        [ 0, -1,  0]
        [-1,  5, -1]
        [ 0, -1,  0]
    
    How it works:
        - The kernel enhances edges by amplifying differences between
          a pixel and its neighbors
        - Center weight (5) > sum of neighbor weights (4), which
          amplifies high-frequency components
        - Negative weights subtract the blurred version from the
          original, enhancing edges
    
    This is equivalent to: sharpened = original + (original - blurred)
    Or: sharpened = 2 * original - blurred (unsharp masking concept)
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
               Values should be in range [0, 255]
    
    Returns:
        Sharpened image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
    
    Notes:
        - Can amplify noise along with edges
        - Output is clipped to [0, 255] to prevent overflow
        - For subtle sharpening, blend result with original
    
    Example:
        >>> # Blurry edge
        >>> img = np.array([[100, 110, 120, 130, 140],
        ...                 [100, 110, 120, 130, 140],
        ...                 [100, 110, 120, 130, 140]], dtype=np.uint8)
        >>> sharp = sharpen(img)
        >>> # Transitions become more pronounced
    """
    # Define sharpening kernel (hardcoded)
    # This kernel is derived from the Laplacian operator
    # combined with the original image
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)
    
    return convolve2d(image, kernel)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing sharpening filter...")
    
    # Test 1: Uniform image (should remain largely unchanged)
    uniform = np.full((5, 5), 100, dtype=np.uint8)
    sharp_uniform = sharpen(uniform)
    print(f"Uniform image:")
    print(f"  Before: {uniform[2, 2]}")
    print(f"  After: {sharp_uniform[2, 2]}")
    
    # Test 2: Blurry gradient (should become sharper)
    gradient = np.array([
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90],
        [50, 60, 70, 80, 90]
    ], dtype=np.uint8)
    sharp_gradient = sharpen(gradient)
    print(f"\nGradient sharpening:")
    print(f"  Original center row: {gradient[2, :]}")
    print(f"  Sharpened center row: {sharp_gradient[2, :]}")
    
    # Test 3: Soft edge becomes harder
    soft_edge = np.array([
        [50,  75, 100, 125, 150],
        [50,  75, 100, 125, 150],
        [50,  75, 100, 125, 150],
        [50,  75, 100, 125, 150],
        [50,  75, 100, 125, 150]
    ], dtype=np.uint8)
    sharp_edge = sharpen(soft_edge)
    print(f"\nSoft edge sharpening:")
    print(f"  Original: {soft_edge[2, :]}")
    print(f"  Sharpened: {sharp_edge[2, :]}")
    
    # Test 4: Check for clipping (no overflow)
    bright = np.full((5, 5), 250, dtype=np.uint8)
    bright[2, 2] = 255
    sharp_bright = sharpen(bright)
    assert sharp_bright.max() <= 255, "Overflow detected"
    assert sharp_bright.min() >= 0, "Underflow detected"
    print(f"\nClipping check: OK (max={sharp_bright.max()}, min={sharp_bright.min()})")
    
    # Test 5: Shape and dtype
    random_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    sharp_random = sharpen(random_img)
    assert sharp_random.shape == random_img.shape, "Shape mismatch"
    assert sharp_random.dtype == np.uint8, "Dtype mismatch"
    print(f"Shape preservation: OK")
    
    print("\nAll sharpening tests passed!")
