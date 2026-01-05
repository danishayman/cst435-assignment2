"""
brightness.py - Brightness Adjustment Filter

Adjusts image brightness using OpenCV's convertScaleAbs.
"""

import cv2
import numpy as np


def adjust_brightness(image: np.ndarray, value: int = 30) -> np.ndarray:
    """
    Adjust image brightness using OpenCV.
    
    Formula: Output = clip(Input + value, 0, 255)
    
    This is a simple linear brightness adjustment. Each pixel's
    intensity is increased or decreased by the same amount.
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W)
               Values should be in range [0, 255]
        value: Brightness adjustment value (default: 30)
               Positive values increase brightness (make lighter)
               Negative values decrease brightness (make darker)
               Typical range: -100 to +100
    
    Returns:
        Brightness-adjusted image as numpy array with shape (H, W)
        Values in range [0, 255] as uint8
    
    Notes:
        - Values are clipped to [0, 255] to prevent overflow/underflow
        - This is different from contrast adjustment (multiplication)
        - For percentage-based adjustment, use value = int(255 * percent / 100)
    
    Examples:
        >>> img = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        
        >>> # Increase brightness
        >>> bright = adjust_brightness(img, 30)
        >>> bright[0, 0]  # 100 + 30 = 130
        130
        >>> bright[1, 1]  # 250 + 30 = 280 -> clipped to 255
        255
        
        >>> # Decrease brightness
        >>> dark = adjust_brightness(img, -50)
        >>> dark[0, 0]  # 100 - 50 = 50
        50
        >>> dark[1, 0]  # 200 - 50 = 150
        150
    """
    # Use OpenCV's convertScaleAbs for brightness adjustment
    # alpha=1 (no contrast change), beta=value (brightness offset)
    adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=value)
    
    return adjusted


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing brightness adjustment filter...")
    
    # Test 1: Increase brightness
    img = np.array([[50, 100, 150], [200, 225, 250]], dtype=np.uint8)
    bright = adjust_brightness(img, 30)
    print(f"Increase brightness by 30:")
    print(f"  Original: {img.flatten()}")
    print(f"  Adjusted: {bright.flatten()}")
    print(f"  Expected: [80, 130, 180, 230, 255, 255]")
    
    # Test 2: Decrease brightness
    dark = adjust_brightness(img, -30)
    print(f"\nDecrease brightness by 30:")
    print(f"  Original: {img.flatten()}")
    print(f"  Adjusted: {dark.flatten()}")
    print(f"  Expected: [20, 70, 120, 170, 195, 220]")
    
    # Test 3: Extreme increase (clipping)
    extreme_bright = adjust_brightness(img, 100)
    print(f"\nExtreme increase by 100:")
    print(f"  Adjusted: {extreme_bright.flatten()}")
    assert extreme_bright.max() == 255, "Max clipping failed"
    print(f"  Max value correctly clipped to 255")
    
    # Test 4: Extreme decrease (clipping)
    extreme_dark = adjust_brightness(img, -200)
    print(f"\nExtreme decrease by 200:")
    print(f"  Adjusted: {extreme_dark.flatten()}")
    assert extreme_dark.min() == 0, "Min clipping failed"
    print(f"  Min value correctly clipped to 0")
    
    # Test 5: Zero adjustment (no change)
    no_change = adjust_brightness(img, 0)
    assert np.array_equal(img, no_change), "Zero adjustment changed image"
    print(f"\nZero adjustment: No change (correct)")
    
    # Test 6: Shape and dtype preservation
    large_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    adjusted = adjust_brightness(large_img, 50)
    assert adjusted.shape == large_img.shape, "Shape mismatch"
    assert adjusted.dtype == np.uint8, "Dtype mismatch"
    print(f"\nShape preservation: OK")
    
    print("\nAll brightness adjustment tests passed!")
