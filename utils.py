"""
utils.py - Utility Functions for Image I/O and Path Management

This module provides helper functions for loading, saving, and
managing image files in the parallel processing pipeline.

Functions:
    - get_image_paths: Collect image file paths from a directory
    - load_image: Load a single image as NumPy array
    - save_image: Save a NumPy array as an image file
    - ensure_directory: Create directory if it doesn't exist
    - get_output_path: Generate output path for processed image
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def get_image_paths(input_dir: str, limit: Optional[int] = None) -> List[str]:
    """
    Collect all image file paths from a directory (recursively).
    
    Args:
        input_dir: Path to directory containing images
        limit: Maximum number of images to return (None for all)
    
    Returns:
        List of absolute paths to image files
    
    Raises:
        ValueError: If input directory does not exist
    
    Example:
        >>> paths = get_image_paths("data/food-101-subset", limit=100)
        >>> len(paths)
        100
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    image_paths = []
    
    # Walk through directory recursively
    for root, _, files in os.walk(input_path):
        for filename in files:
            # Skip macOS resource fork files (start with "._")
            if filename.startswith('._'):
                continue
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                image_paths.append(full_path)
    
    # Sort for reproducibility
    image_paths.sort()
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]
    
    return image_paths


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image file as a NumPy array (RGB format).
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as NumPy array with shape (H, W, 3) for RGB
        or (H, W) for grayscale, dtype uint8
    
    Raises:
        FileNotFoundError: If image file does not exist
        IOError: If image cannot be loaded
    
    Example:
        >>> img = load_image("data/food-101-subset/apple_pie/image1.jpg")
        >>> img.shape
        (512, 512, 3)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Open with Pillow and convert to RGB
        with Image.open(image_path) as img:
            # Convert to RGB if not already (handles grayscale, RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to NumPy array
            return np.array(img, dtype=np.uint8)
    
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a NumPy array as an image file.
    
    Args:
        image: Image as NumPy array (grayscale or RGB)
        output_path: Path where to save the image
    
    Raises:
        IOError: If image cannot be saved
    
    Notes:
        - Creates parent directories if they don't exist
        - Supports grayscale (H, W) and RGB (H, W, 3) arrays
    """
    # Ensure parent directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_directory(output_dir)
    
    try:
        # Determine image mode based on array shape
        if len(image.shape) == 2:
            mode = 'L'  # Grayscale
        elif image.shape[2] == 3:
            mode = 'RGB'
        elif image.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Create PIL Image and save
        pil_image = Image.fromarray(image.astype(np.uint8), mode=mode)
        pil_image.save(output_path)
    
    except Exception as e:
        raise IOError(f"Failed to save image to {output_path}: {e}")


def ensure_directory(directory: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to directory to create
    
    Notes:
        Creates parent directories as needed (like mkdir -p)
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_output_path(input_path: str, input_dir: str, output_dir: str, 
                    suffix: str = "_processed") -> str:
    """
    Generate the output path for a processed image.
    
    Preserves the relative directory structure from input to output.
    
    Args:
        input_path: Original image path
        input_dir: Base input directory
        output_dir: Base output directory
        suffix: Suffix to add to filename (default: "_processed")
    
    Returns:
        Full output path for the processed image
    
    Example:
        >>> get_output_path(
        ...     "data/food/apple/img1.jpg",
        ...     "data/food",
        ...     "output/processed"
        ... )
        'output/processed/apple/img1_processed.jpg'
    """
    # Get relative path from input directory
    input_path_obj = Path(input_path)
    input_dir_obj = Path(input_dir)
    
    try:
        relative_path = input_path_obj.relative_to(input_dir_obj)
    except ValueError:
        # If input_path is not relative to input_dir, use just the filename
        relative_path = Path(input_path_obj.name)
    
    # Build output path
    output_path = Path(output_dir) / relative_path
    
    # Add suffix to filename (before extension)
    stem = output_path.stem
    ext = output_path.suffix
    output_path = output_path.parent / f"{stem}{suffix}{ext}"
    
    return str(output_path)


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image without fully loading it.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with image information:
            - path: Full path to image
            - filename: Just the filename
            - size: (width, height) tuple
            - mode: Image mode (RGB, L, etc.)
            - format: Image format (JPEG, PNG, etc.)
    """
    with Image.open(image_path) as img:
        return {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'size': img.size,  # (width, height)
            'mode': img.mode,
            'format': img.format
        }


def prepare_task_args(input_dir: str, output_dir: str, 
                      limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Prepare a list of (input_path, output_path) tuples for processing.
    
    This function is useful for preparing arguments for parallel workers.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output images
        limit: Maximum number of images to process
    
    Returns:
        List of (input_path, output_path) tuples
    
    Example:
        >>> tasks = prepare_task_args("data/food", "output/processed", limit=50)
        >>> len(tasks)
        50
        >>> tasks[0]
        ('data/food/apple/img1.jpg', 'output/processed/apple/img1_processed.jpg')
    """
    image_paths = get_image_paths(input_dir, limit=limit)
    
    task_args = []
    for input_path in image_paths:
        output_path = get_output_path(input_path, input_dir, output_dir)
        task_args.append((input_path, output_path))
    
    return task_args


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test ensure_directory
    test_dir = "test_output_temp"
    ensure_directory(test_dir)
    assert os.path.exists(test_dir), "Directory creation failed"
    print(f"✓ ensure_directory: Created {test_dir}")
    
    # Test with synthetic image
    test_image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    test_path = os.path.join(test_dir, "test_image.png")
    
    # Test save_image
    save_image(test_image, test_path)
    assert os.path.exists(test_path), "Image save failed"
    print(f"✓ save_image: Saved to {test_path}")
    
    # Test load_image
    loaded = load_image(test_path)
    assert loaded.shape == test_image.shape, "Image shape mismatch"
    print(f"✓ load_image: Loaded with shape {loaded.shape}")
    
    # Test get_output_path
    output = get_output_path(
        "data/food/apple/img1.jpg",
        "data/food",
        "output/processed"
    )
    expected = str(Path("output/processed/apple/img1_processed.jpg"))
    print(f"✓ get_output_path: {output}")
    
    # Cleanup
    os.remove(test_path)
    os.rmdir(test_dir)
    print(f"✓ Cleanup completed")
    
    print("\nAll utility tests passed!")
