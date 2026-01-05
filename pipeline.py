"""
pipeline.py - Sequential Image Processing Pipeline

This module provides the sequential (non-parallel) implementation of
the image processing pipeline. It serves as:
    1. The baseline for benchmarking parallel implementations
    2. A reference implementation for correctness verification

Functions:
    - process_single_image: Process one image through all filters
    - run_sequential_pipeline: Process all images sequentially
"""

import time
from typing import List, Tuple, Optional, Dict, Any

from filters import apply_all_filters
from utils import (
    load_image, 
    save_image, 
    get_image_paths, 
    get_output_path,
    ensure_directory
)


def process_single_image(input_path: str, output_path: str, 
                         brightness_value: int = 30) -> Dict[str, Any]:
    """
    Process a single image through the complete filter pipeline.
    
    This function:
        1. Loads the image from disk
        2. Applies all 5 filters in sequence
        3. Saves the processed image to disk
    
    Args:
        input_path: Path to the input image
        output_path: Path where processed image will be saved
        brightness_value: Brightness adjustment value (default: 30)
    
    Returns:
        Dictionary containing:
            - input_path: Original image path
            - output_path: Processed image path
            - success: Boolean indicating success
            - error: Error message if failed, None otherwise
            - processing_time: Time taken in seconds
    
    Notes:
        This function is designed to be picklable for use with
        multiprocessing. It handles its own I/O to minimize
        data transfer between processes.
    """
    result = {
        'input_path': input_path,
        'output_path': output_path,
        'success': False,
        'error': None,
        'processing_time': 0.0
    }
    
    start_time = time.perf_counter()
    
    try:
        # Load image
        image = load_image(input_path)
        
        # Apply filter pipeline
        processed = apply_all_filters(image, brightness_value)
        
        # Save processed image
        save_image(processed, output_path)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    finally:
        result['processing_time'] = time.perf_counter() - start_time
    
    return result


def process_image_task(task: Tuple[str, str, int]) -> Dict[str, Any]:
    """
    Wrapper for process_single_image that accepts a tuple of arguments.
    
    This is useful for Pool.map() and similar functions that expect
    a single argument.
    
    Args:
        task: Tuple of (input_path, output_path, brightness_value)
    
    Returns:
        Result dictionary from process_single_image
    """
    input_path, output_path, brightness_value = task
    return process_single_image(input_path, output_path, brightness_value)


def run_sequential_pipeline(input_dir: str, output_dir: str,
                            limit: Optional[int] = None,
                            brightness_value: int = 30,
                            verbose: bool = True) -> Dict[str, Any]:
    """
    Run the image processing pipeline sequentially (no parallelism).
    
    This serves as the baseline for benchmarking parallel implementations.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for processed images
        limit: Maximum number of images to process (None for all)
        brightness_value: Brightness adjustment value (default: 30)
        verbose: Print progress information (default: True)
    
    Returns:
        Dictionary containing:
            - total_images: Number of images processed
            - successful: Number of successful processings
            - failed: Number of failed processings
            - total_time: Total wall-clock time in seconds
            - avg_time_per_image: Average time per image in seconds
            - results: List of individual result dictionaries
    
    Example:
        >>> results = run_sequential_pipeline(
        ...     "data/food-101-subset",
        ...     "output/sequential",
        ...     limit=50
        ... )
        >>> print(f"Processed {results['total_images']} images in {results['total_time']:.2f}s")
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Get list of images to process
    image_paths = get_image_paths(input_dir, limit=limit)
    total_images = len(image_paths)
    
    if verbose:
        print(f"Sequential Pipeline")
        print(f"==================")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Images to process: {total_images}")
        print()
    
    # Process images
    results = []
    successful = 0
    failed = 0
    
    start_time = time.perf_counter()
    
    for i, input_path in enumerate(image_paths, 1):
        # Generate output path
        output_path = get_output_path(input_path, input_dir, output_dir)
        
        # Process image
        result = process_single_image(input_path, output_path, brightness_value)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
            if verbose:
                print(f"  ERROR: {result['input_path']}: {result['error']}")
        
        # Progress update
        if verbose and (i % 10 == 0 or i == total_images):
            elapsed = time.perf_counter() - start_time
            print(f"  Progress: {i}/{total_images} images ({elapsed:.2f}s elapsed)")
    
    total_time = time.perf_counter() - start_time
    avg_time = total_time / total_images if total_images > 0 else 0
    
    summary = {
        'total_images': total_images,
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'results': results
    }
    
    if verbose:
        print()
        print(f"Summary")
        print(f"-------")
        print(f"Successful: {successful}/{total_images}")
        print(f"Failed: {failed}/{total_images}")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time per image: {avg_time:.4f} seconds")
    
    return summary


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import os
    import numpy as np
    from utils import save_image, ensure_directory
    
    print("Testing sequential pipeline...")
    
    # Create test directory structure
    test_input_dir = "test_data_temp"
    test_output_dir = "test_output_temp"
    
    ensure_directory(test_input_dir)
    
    # Create some test images
    num_test_images = 5
    for i in range(num_test_images):
        test_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        save_image(test_img, os.path.join(test_input_dir, f"test_{i}.png"))
    
    print(f"Created {num_test_images} test images")
    
    # Run sequential pipeline
    results = run_sequential_pipeline(
        test_input_dir,
        test_output_dir,
        verbose=True
    )
    
    # Verify results
    assert results['total_images'] == num_test_images
    assert results['successful'] == num_test_images
    assert results['failed'] == 0
    
    print("\nSequential pipeline test passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_input_dir)
    shutil.rmtree(test_output_dir)
    print("Cleanup completed")
