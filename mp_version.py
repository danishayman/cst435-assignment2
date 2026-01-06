"""
mp_version.py - Multiprocessing Implementation

This module implements the parallel image processing pipeline using
Python's built-in multiprocessing module with Pool.

Key Features:
    - Uses multiprocessing.Pool for worker management
    - Explicit control over process count
    - Uses Pool.map() for parallel task distribution
    - Process-based parallelism (bypasses GIL)

Paradigm: Low-level multiprocessing with explicit Pool management
"""

import multiprocessing as mp
import os
import time
from typing import List, Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from filters import apply_all_filters
from utils import (
    load_image,
    save_image,
    get_image_paths,
    get_output_path,
    ensure_directory
)


def _get_cpu_core_id() -> int:
    """Get the current CPU core ID that this process is running on."""
    if PSUTIL_AVAILABLE:
        try:
            p = psutil.Process(os.getpid())
            return p.cpu_num()
        except (AttributeError, psutil.Error):
            return -1
    return -1


# =============================================================================
# Worker Function (Must be at module level for pickling)
# =============================================================================

def _process_image_worker(args: tuple) -> Dict[str, Any]:
    """
    Worker function that processes a single image.
    
    This function is designed to be called by Pool.map().
    It must be defined at module level (not inside a class or function)
    to be picklable by multiprocessing.
    
    Args:
        args: Tuple of (input_path, output_path, brightness_value)
    
    Returns:
        Dictionary containing:
            - input_path: Original image path
            - output_path: Processed image path
            - success: Boolean indicating success
            - error: Error message if failed
            - processing_time: Time taken in seconds
            - worker_pid: Process ID of the worker
            - cpu_core: CPU core ID where the process executed
    """
    input_path, output_path, brightness_value = args
    
    # Capture worker info at start
    pid = mp.current_process().pid
    cpu_core = _get_cpu_core_id()
    
    result = {
        'input_path': input_path,
        'output_path': output_path,
        'success': False,
        'error': None,
        'processing_time': 0.0,
        'worker_pid': pid,
        'cpu_core': cpu_core
    }
    
    start_time = time.perf_counter()
    
    try:
        # Load image from disk
        image = load_image(input_path)
        
        # Apply the complete filter pipeline
        processed = apply_all_filters(image, brightness_value)
        
        # Save processed image to disk
        save_image(processed, output_path)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    finally:
        result['processing_time'] = time.perf_counter() - start_time
    
    return result


# =============================================================================
# Main Parallel Pipeline
# =============================================================================

def run_multiprocessing_pipeline(input_dir: str, output_dir: str,
                                  num_processes: Optional[int] = None,
                                  limit: Optional[int] = None,
                                  brightness_value: int = 30,
                                  verbose: bool = True) -> Dict[str, Any]:
    """
    Run the image processing pipeline using multiprocessing.Pool.
    
    This implementation uses Python's multiprocessing module for
    process-based parallelism. Each worker process handles complete
    images independently.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for processed images
        num_processes: Number of worker processes (default: CPU count)
        limit: Maximum number of images to process (None for all)
        brightness_value: Brightness adjustment value (default: 30)
        verbose: Print progress information (default: True)
    
    Returns:
        Dictionary containing:
            - total_images: Number of images processed
            - successful: Number of successful processings
            - failed: Number of failed processings
            - total_time: Total wall-clock time in seconds
            - avg_time_per_image: Average time per image
            - num_processes: Number of worker processes used
            - results: List of individual result dictionaries
    
    Example:
        >>> results = run_multiprocessing_pipeline(
        ...     "data/food-101-subset",
        ...     "output/mp",
        ...     num_processes=4,
        ...     limit=100
        ... )
        >>> print(f"Speedup: {baseline_time / results['total_time']:.2f}x")
    """
    # Determine number of processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Get list of images to process
    image_paths = get_image_paths(input_dir, limit=limit)
    total_images = len(image_paths)
    
    if verbose:
        print(f"Multiprocessing Pipeline")
        print(f"========================")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Images to process: {total_images}")
        print(f"Worker processes: {num_processes}")
        print()
    
    # Prepare task arguments: (input_path, output_path, brightness_value)
    tasks = []
    for input_path in image_paths:
        output_path = get_output_path(input_path, input_dir, output_dir)
        tasks.append((input_path, output_path, brightness_value))
    
    # Start timing
    start_time = time.perf_counter()
    
    # Create pool and process images in parallel
    # Using Pool as a context manager ensures proper cleanup
    with mp.Pool(processes=num_processes) as pool:
        # map() blocks until all tasks are complete
        # It preserves order of results
        results = pool.map(_process_image_worker, tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Count successes and failures
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    avg_time = total_time / total_images if total_images > 0 else 0
    
    # Collect unique worker PIDs (for verification)
    unique_workers = set(r['worker_pid'] for r in results)
    
    summary = {
        'total_images': total_images,
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'num_processes': num_processes,
        'unique_workers_used': len(unique_workers),
        'results': results
    }
    
    if verbose:
        print(f"Summary")
        print(f"-------")
        print(f"Successful: {successful}/{total_images}")
        print(f"Failed: {failed}/{total_images}")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time per image: {avg_time:.4f} seconds")
        print(f"Unique workers used: {len(unique_workers)}")
        
        # Print worker log table showing PID and CPU Core
        print()
        print("Worker Execution Log (Multiprocessing - different PIDs, true parallelism):")
        print("-" * 70)
        print(f"{'Image #':<10} {'PID':<12} {'CPU Core':<10} {'Time (s)':<10}")
        print("-" * 70)
        
        # Show first 10 and last 5 results for brevity
        display_results = results[:10] + (results[-5:] if len(results) > 15 else [])
        shown_indices = list(range(min(10, len(results)))) + (list(range(len(results)-5, len(results))) if len(results) > 15 else [])
        
        for idx, r in zip(shown_indices, display_results):
            pid = r.get('worker_pid', 'N/A')
            core = r.get('cpu_core', -1)
            core_str = str(core) if core >= 0 else 'N/A'
            print(f"{idx+1:<10} {pid:<12} {core_str:<10} {r['processing_time']:.4f}")
            if idx == 9 and len(results) > 15:
                print(f"{'...':<10} {'...':<12} {'...':<10} ...")
        
        print("-" * 70)
        
        # Analyze PID and Core distribution
        unique_pids = set(r.get('worker_pid') for r in results if r.get('worker_pid'))
        unique_cores = set(r.get('cpu_core') for r in results if r.get('cpu_core', -1) >= 0)
        print(f"PID Analysis: {len(unique_pids)} distinct PIDs = {sorted(unique_pids)}")
        print(f"CPU Cores used: {sorted(unique_cores) if unique_cores else 'N/A'}")
        print(f"Note: Each process has its own PID and Python interpreter - true parallel execution!")
        
        # Report any errors
        errors = [r for r in results if not r['success']]
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors[:5]:  # Show first 5 errors
                print(f"  - {e['input_path']}: {e['error']}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
    
    return summary


def run_with_varying_processes(input_dir: str, output_dir: str,
                                process_counts: List[int],
                                limit: Optional[int] = None,
                                brightness_value: int = 30,
                                verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run the pipeline multiple times with different process counts.
    
    Useful for benchmarking scalability.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Base directory for outputs (will add process count suffix)
        process_counts: List of process counts to test (e.g., [1, 2, 4, 8])
        limit: Maximum number of images to process
        brightness_value: Brightness adjustment value
        verbose: Print progress information
    
    Returns:
        List of result dictionaries, one per process count
    """
    all_results = []
    
    for num_procs in process_counts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running with {num_procs} processes")
            print(f"{'='*60}\n")
        
        # Create unique output directory for this run
        output_subdir = f"{output_dir}_p{num_procs}"
        
        result = run_multiprocessing_pipeline(
            input_dir=input_dir,
            output_dir=output_subdir,
            num_processes=num_procs,
            limit=limit,
            brightness_value=brightness_value,
            verbose=verbose
        )
        
        all_results.append(result)
    
    return all_results


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    import os
    import numpy as np
    from utils import save_image, ensure_directory
    import shutil
    
    print("Testing multiprocessing pipeline...")
    print(f"Available CPUs: {mp.cpu_count()}")
    
    # Create test directory structure
    test_input_dir = "test_data_mp_temp"
    test_output_dir = "test_output_mp_temp"
    
    ensure_directory(test_input_dir)
    
    # Create some test images
    num_test_images = 20
    for i in range(num_test_images):
        test_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        save_image(test_img, os.path.join(test_input_dir, f"test_{i}.png"))
    
    print(f"Created {num_test_images} test images")
    
    # Run with different process counts
    for num_procs in [1, 2, 4]:
        print(f"\n--- Testing with {num_procs} processes ---")
        results = run_multiprocessing_pipeline(
            test_input_dir,
            f"{test_output_dir}_p{num_procs}",
            num_processes=num_procs,
            verbose=True
        )
        
        assert results['total_images'] == num_test_images
        assert results['successful'] == num_test_images
    
    print("\nMultiprocessing pipeline tests passed!")
    
    # Cleanup
    shutil.rmtree(test_input_dir)
    for p in [1, 2, 4]:
        output_dir = f"{test_output_dir}_p{p}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    print("Cleanup completed")
