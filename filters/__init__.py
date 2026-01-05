"""
filters - Image Filter Package for Parallel Processing Pipeline

This package contains all image filter functions that form the processing pipeline.
Each filter is implemented in its own module for modularity and clarity.

Filter Pipeline Order:
    1. Grayscale Conversion
    2. Gaussian Blur (3x3)
    3. Edge Detection (Sobel)
    4. Sharpening
    5. Brightness Adjustment

Usage:
    from filters import grayscale, gaussian_blur, edge_detection, sharpen, adjust_brightness
    from filters import apply_all_filters
"""

# Import all filter functions for easy access
from filters.grayscale import grayscale
from filters.gaussian_blur import gaussian_blur
from filters.edge_detection import edge_detection
from filters.sharpen import sharpen
from filters.brightness import adjust_brightness
from filters.convolution import convolve2d
from filters.pipeline import apply_all_filters

# Define public API
__all__ = [
    'grayscale',
    'gaussian_blur',
    'edge_detection',
    'sharpen',
    'adjust_brightness',
    'convolve2d',
    'apply_all_filters'
]
