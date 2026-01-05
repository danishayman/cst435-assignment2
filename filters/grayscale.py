"""
grayscale.py - Grayscale Conversion Filter

Converts RGB images to grayscale using OpenCV.

"""

import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using OpenCV.
    
    Uses the standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    
    This formula is based on human perception of brightness, where green
    contributes most to perceived luminance, followed by red, then blue.
    These coefficients are derived from the ITU-R BT.601 standard.
    
    Args:
        image: Input RGB image as numpy array with shape (H, W, 3)
               Values should be in range [0, 255]
    
    Returns:
        Grayscale image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
    
    Example:
        >>> rgb_img = np.array([[[255, 0, 0], [0, 255, 0]]])  # Red, Green pixels
        >>> gray = grayscale(rgb_img)
        >>> gray.shape
        (1, 2)
        >>> gray[0, 0]  # Red pixel: 0.299 * 255 ≈ 76
        76
        >>> gray[0, 1]  # Green pixel: 0.587 * 255 ≈ 150
        150
    
    Notes:
        - If input is already grayscale (2D), returns as-is
        - If input has alpha channel (RGBA), alpha is dropped
    """
    # Handle already grayscale images
    if len(image.shape) == 2:
        return image.astype(np.uint8)
    
    # Handle RGBA images (drop alpha channel)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Use OpenCV for grayscale conversion (expects RGB input)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return gray.astype(np.uint8)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing grayscale filter...")
    
    # Test 1: RGB image
    rgb_image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]]  # White, Gray, Black
    ], dtype=np.uint8)
    
    gray = grayscale(rgb_image)
    print(f"RGB input shape: {rgb_image.shape}")
    print(f"Grayscale output shape: {gray.shape}")
    print(f"Red pixel value: {gray[0, 0]} (expected ~76)")
    print(f"Green pixel value: {gray[0, 1]} (expected ~150)")
    print(f"Blue pixel value: {gray[0, 2]} (expected ~29)")
    print(f"White pixel value: {gray[1, 0]} (expected 255)")
    print(f"Black pixel value: {gray[1, 2]} (expected 0)")
    
    # Test 2: Already grayscale
    gray_input = np.array([[100, 150], [200, 50]], dtype=np.uint8)
    gray_output = grayscale(gray_input)
    assert gray_input.shape == gray_output.shape, "Grayscale passthrough failed"
    print(f"\nGrayscale passthrough: OK")
    
    # Test 3: RGBA image
    rgba_image = np.random.randint(0, 256, size=(10, 10, 4), dtype=np.uint8)
    gray_rgba = grayscale(rgba_image)
    assert gray_rgba.shape == (10, 10), "RGBA handling failed"
    print(f"RGBA handling: OK")
    
    print("\nAll grayscale tests passed!")
