"""
run_demo.py - Demo Script for Testing the Pipeline

This script uses images from the data/food-101-subset folder and runs 
the complete benchmark to verify everything works correctly.

Usage:
    python run_demo.py
"""

import os
import sys
import shutil
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import save_image, ensure_directory, get_image_paths
from benchmark import run_benchmark, print_summary_table, export_results_to_csv


def run_demo(num_images: int = 1000, cleanup: bool = True):
    """
    Run a complete demo of the pipeline using images from data folder.
    
    Args:
        num_images: Number of images to process from data folder (default: 1000)
        cleanup: Whether to cleanup output data after completion
    """
    print("="*70)
    print("PARALLEL IMAGE PROCESSING PIPELINE - DEMO")
    print("="*70)
    print()
    
    # Paths
    data_dir = "data/food-101-subset"
    output_dir = "output"
    
    # Check if data directory exists and has images
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run select_random_images.py first to create the dataset.")
        return None
    
    # Count available images
    image_paths = get_image_paths(data_dir)
    if len(image_paths) == 0:
        print(f"Error: No images found in {data_dir}")
        print("Please run select_random_images.py first to create the dataset.")
        return None
    
    print(f"Found {len(image_paths)} images in {data_dir}")
    print()
    
    try:
        # Run benchmark
        print("Running benchmark on food-101 subset images")
        print("-"*40)
        results = run_benchmark(
            input_dir=data_dir,
            output_base_dir=output_dir,
            process_counts=[1, 2, 4],
            image_limit=num_images,
            verbose=True
        )
        
        # Print summary
        print_summary_table(results)
        
        # Export results
        csv_path = os.path.join(output_dir, "benchmark_results.csv")
        export_results_to_csv(results, csv_path)
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        
        # Print explanations for both implementations
        print()
        print("=" * 60)
        print("PID Observation & Core Allocation Analysis")
        print("=" * 60)
        
        print()
        print("MULTIPROCESSING (mp_version.py):")
        print("-" * 40)
        print("PID Observation:")
        print("  - Each worker has a DIFFERENT Process ID (PID)")
        print("  - This is because multiprocessing creates separate processes")
        print("  - Each process has its own memory space and Python interpreter")
        print()
        print("Core Allocation:")
        print("  - Each process can run on different CPU cores simultaneously")
        print("  - Bypasses Python's Global Interpreter Lock (GIL)")
        print("  - Achieves TRUE parallel execution for CPU-bound tasks")
        print("  - Ideal for computationally intensive operations like image processing")
        
        print()
        print("THREADPOOLEXECUTOR (futures_version.py):")
        print("-" * 40)
        print("PID Observation:")
        print("  - All threads share the SAME Process ID (PID)")
        print("  - This is because ThreadPoolExecutor uses threads, not processes")
        print("  - Threads exist within a single process's memory space")
        print()
        print("Core Allocation:")
        print("  - Threads may run on different CPU cores")
        print("  - However, Python's Global Interpreter Lock (GIL) limits")
        print("    true parallel execution for CPU-bound tasks")
        print("  - Only one thread can execute Python bytecode at a time")
        print("  - Best suited for I/O-bound tasks (file operations, network)")
        print("=" * 60)
        
        print(f"\nResults exported to: {csv_path}")
        print(f"Processed images saved to: {output_dir}/")
        
        # Return results for inspection
        return results
        
    finally:
        if cleanup:
            print("\nNote: Output files preserved in output/ folder.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pipeline demo')
    parser.add_argument('--images', '-n', type=int, default=1000,
                        help='Number of images to process (default: 1000)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep output after completion')
    
    args = parser.parse_args()
    
    run_demo(num_images=args.images, cleanup=not args.no_cleanup)
