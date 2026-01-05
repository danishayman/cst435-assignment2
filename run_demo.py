"""
run_demo.py - Demo Script for Testing the Pipeline

This script creates synthetic test images and runs the complete
benchmark to verify everything works correctly.

Usage:
    python run_demo.py
"""

import os
import sys
import shutil
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_image, ensure_directory
from benchmark import run_benchmark, print_summary_table, export_results_to_csv


def create_test_images(output_dir: str, num_images: int = 50, 
                       image_size: tuple = (256, 256)) -> str:
    """
    Create synthetic test images for benchmarking.
    
    Args:
        output_dir: Directory to save test images
        num_images: Number of images to create
        image_size: Size of each image (height, width)
    
    Returns:
        Path to the created test directory
    """
    print(f"Creating {num_images} test images ({image_size[0]}x{image_size[1]})...")
    
    ensure_directory(output_dir)
    
    for i in range(num_images):
        # Create random RGB image
        img = np.random.randint(0, 256, size=(image_size[0], image_size[1], 3), dtype=np.uint8)
        
        # Save image
        filename = f"test_image_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(img, filepath)
        
        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{num_images} images")
    
    print(f"Test images saved to: {output_dir}")
    return output_dir


def run_demo(num_images: int = 30, cleanup: bool = True):
    """
    Run a complete demo of the pipeline.
    
    Args:
        num_images: Number of test images to create and process
        cleanup: Whether to cleanup test data after completion
    """
    print("="*70)
    print("PARALLEL IMAGE PROCESSING PIPELINE - DEMO")
    print("="*70)
    print()
    
    # Paths
    test_data_dir = "demo_test_images"
    output_dir = "demo_output"
    
    try:
        # Step 1: Create test images
        print("STEP 1: Creating test images")
        print("-"*40)
        create_test_images(test_data_dir, num_images=num_images, image_size=(256, 256))
        print()
        
        # Step 2: Run benchmark
        print("STEP 2: Running benchmark")
        print("-"*40)
        results = run_benchmark(
            input_dir=test_data_dir,
            output_base_dir=output_dir,
            process_counts=[1, 2, 4],
            image_limit=num_images,
            verbose=True
        )
        
        # Step 3: Print summary
        print_summary_table(results)
        
        # Step 4: Export results
        csv_path = os.path.join(output_dir, "demo_results.csv")
        export_results_to_csv(results, csv_path)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print(f"\nResults exported to: {csv_path}")
        print(f"Processed images saved to: {output_dir}/")
        
        # Return results for inspection
        return results
        
    finally:
        if cleanup:
            print("\nCleaning up test data...")
            if os.path.exists(test_data_dir):
                shutil.rmtree(test_data_dir)
            print("Cleanup complete. Output files preserved.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pipeline demo')
    parser.add_argument('--images', '-n', type=int, default=30,
                        help='Number of test images (default: 30)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep test images after completion')
    
    args = parser.parse_args()
    
    run_demo(num_images=args.images, cleanup=not args.no_cleanup)
