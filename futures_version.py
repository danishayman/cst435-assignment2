"""
futures_version.py - concurrent.futures Implementation

This module implements the parallel image processing pipeline using
Python's concurrent.futures module with ThreadPoolExecutor.

Key Features:
    - Uses ThreadPoolExecutor for high-level abstraction
    - Future-based task management
    - Context manager pattern for resource cleanup
    - Thread-based parallelism (subject to GIL for CPU-bound tasks)

Paradigm: High-level futures-based parallelism
"""

import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import multiprocessing as mp

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
    """Get the current CPU core ID that this thread is running on."""
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

def _process_image_worker(input_path: str, output_path: str, 
                          brightness_value: int) -> Dict[str, Any]:
    """
    Worker function that processes a single image.
    
    This function is designed to be called via ThreadPoolExecutor.submit().
    
    Args:
        input_path: Path to input image
        output_path: Path for output image
        brightness_value: Brightness adjustment value
    
    Returns:
        Dictionary containing:
            - input_path: Original image path
            - output_path: Processed image path
            - success: Boolean indicating success
            - error: Error message if failed
            - processing_time: Time taken in seconds
            - worker_tid: Thread ID of the worker
            - worker_pid: Process ID (same for all threads)
            - cpu_core: CPU core ID where the thread executed
    """
    # Capture worker info at start
    pid = os.getpid()
    tid = threading.current_thread().ident
    cpu_core = _get_cpu_core_id()
    
    result = {
        'input_path': input_path,
        'output_path': output_path,
        'success': False,
        'error': None,
        'processing_time': 0.0,
        'worker_tid': tid,
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

def run_futures_pipeline(input_dir: str, output_dir: str,
                         max_workers: Optional[int] = None,
                         limit: Optional[int] = None,
                         brightness_value: int = 30,
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Run the image processing pipeline using concurrent.futures.ThreadPoolExecutor.
    
    This implementation uses Python's concurrent.futures module, which provides
    a higher-level interface than raw multiprocessing. It uses Future objects
    for task management and result retrieval.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for processed images
        max_workers: Maximum number of worker threads (default: CPU count)
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
            - max_workers: Number of worker threads used
            - results: List of individual result dictionaries
    
    Example:
        >>> results = run_futures_pipeline(
        ...     "data/food-101-subset",
        ...     "output/futures",
        ...     max_workers=4,
        ...     limit=100
        ... )
        >>> print(f"Processed {results['successful']} images successfully")
    """
    # Determine number of workers
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Get list of images to process
    image_paths = get_image_paths(input_dir, limit=limit)
    total_images = len(image_paths)
    
    if verbose:
        print(f"Concurrent.Futures Pipeline (ThreadPoolExecutor)")
        print(f"=================================================")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Images to process: {total_images}")
        print(f"Max workers: {max_workers}")
        print()
    
    # Prepare tasks
    tasks = []
    for input_path in image_paths:
        output_path = get_output_path(input_path, input_dir, output_dir)
        tasks.append((input_path, output_path))
    
    # Start timing
    start_time = time.perf_counter()
    
    results = []
    
    # Use ThreadPoolExecutor with context manager
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get Future objects
        # Using submit() allows for more control than map()
        future_to_task = {
            executor.submit(_process_image_worker, task[0], task[1], brightness_value): task
            for task in tasks
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            
            try:
                result = future.result()  # Get the result (may raise exception)
                results.append(result)
                
                if not result['success'] and verbose:
                    print(f"  ERROR: {result['input_path']}: {result['error']}")
                    
            except Exception as e:
                # Handle any exception that wasn't caught in the worker
                results.append({
                    'input_path': task[0],
                    'output_path': task[1],
                    'success': False,
                    'error': f"Future exception: {str(e)}",
                    'processing_time': 0.0,
                    'worker_tid': None
                })
            
            # Print processing info in the desired format
            if verbose and result['success']:
                pid = result.get('worker_pid', 'N/A')
                core = result.get('cpu_core', -1)
                core_str = str(core) if core >= 0 else '1'  # Default to 1 if unavailable
                filename = os.path.basename(result['input_path'])
                print(f"[PID: {pid}] [Core: {core_str}] Processing: {filename}")
            
            # Progress bar update
            if verbose and (completed % 10 == 0 or completed == total_images):
                percentage = int((completed / total_images) * 100)
                bar_length = 50
                filled_length = int(bar_length * completed / total_images)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"Futures ThreadPool ({max_workers} threads): {percentage}%|{bar}|")
    
    total_time = time.perf_counter() - start_time
    
    # Count successes and failures
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    avg_time = total_time / total_images if total_images > 0 else 0
    
    # Collect unique worker thread IDs
    unique_workers = set(r['worker_tid'] for r in results if r['worker_tid'] is not None)
    
    summary = {
        'total_images': total_images,
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'max_workers': max_workers,
        'unique_workers_used': len(unique_workers),
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
        print(f"Unique workers used: {len(unique_workers)}")
        
        # Analyze PID and Core distribution
        print()
        unique_pids = set(r.get('worker_pid') for r in results if r.get('worker_pid'))
        unique_cores = set(r.get('cpu_core') for r in results if r.get('cpu_core', -1) >= 0)
        print(f"PID Analysis: All threads share the SAME PID = {list(unique_pids)[0] if unique_pids else 'N/A'}")
        print(f"CPU Cores used: {sorted(unique_cores) if unique_cores else 'N/A'}")
        
        # Report any errors
        errors = [r for r in results if not r['success']]
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors[:5]:
                print(f"  - {e['input_path']}: {e['error']}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
    
    return summary


def run_futures_pipeline_with_map(input_dir: str, output_dir: str,
                                   max_workers: Optional[int] = None,
                                   limit: Optional[int] = None,
                                   brightness_value: int = 30,
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Alternative implementation using executor.map() instead of submit().
    
    This version is simpler but provides less control over task management.
    Results are returned in the order tasks were submitted.
    
    Args:
        Same as run_futures_pipeline()
    
    Returns:
        Same as run_futures_pipeline()
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    ensure_directory(output_dir)
    image_paths = get_image_paths(input_dir, limit=limit)
    total_images = len(image_paths)
    
    if verbose:
        print(f"Concurrent.Futures Pipeline (using map())")
        print(f"==========================================")
        print(f"Images to process: {total_images}")
        print(f"Max workers: {max_workers}")
        print()
    
    # Prepare argument lists for map()
    input_paths = []
    output_paths = []
    brightness_values = []
    
    for input_path in image_paths:
        output_path = get_output_path(input_path, input_dir, output_dir)
        input_paths.append(input_path)
        output_paths.append(output_path)
        brightness_values.append(brightness_value)
    
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map() returns results in order, as an iterator
        results = list(executor.map(
            _process_image_worker,
            input_paths,
            output_paths,
            brightness_values
        ))
    
    total_time = time.perf_counter() - start_time
    
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])
    avg_time = total_time / total_images if total_images > 0 else 0
    unique_workers = set(r['worker_tid'] for r in results if r.get('worker_tid'))
    
    summary = {
        'total_images': total_images,
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'avg_time_per_image': avg_time,
        'max_workers': max_workers,
        'unique_workers_used': len(unique_workers),
        'results': results
    }
    
    if verbose:
        print(f"Summary")
        print(f"-------")
        print(f"Successful: {successful}/{total_images}")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time per image: {avg_time:.4f} seconds")
    
    return summary


def run_with_varying_workers(input_dir: str, output_dir: str,
                              worker_counts: List[int],
                              limit: Optional[int] = None,
                              brightness_value: int = 30,
                              verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run the pipeline multiple times with different worker counts.
    
    Useful for benchmarking scalability.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Base directory for outputs
        worker_counts: List of worker counts to test (e.g., [1, 2, 4, 8])
        limit: Maximum number of images to process
        brightness_value: Brightness adjustment value
        verbose: Print progress information
    
    Returns:
        List of result dictionaries, one per worker count
    """
    all_results = []
    
    for num_workers in worker_counts:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running with {num_workers} workers")
            print(f"{'='*60}\n")
        
        output_subdir = f"{output_dir}_w{num_workers}"
        
        result = run_futures_pipeline(
            input_dir=input_dir,
            output_dir=output_subdir,
            max_workers=num_workers,
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
    
    print("Testing concurrent.futures pipeline (ThreadPoolExecutor)...")
    print(f"Available CPUs: {mp.cpu_count()}")
    
    # Create test directory structure
    test_input_dir = "test_data_futures_temp"
    test_output_dir = "test_output_futures_temp"
    
    ensure_directory(test_input_dir)
    
    # Create some test images
    num_test_images = 20
    for i in range(num_test_images):
        test_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        save_image(test_img, os.path.join(test_input_dir, f"test_{i}.png"))
    
    print(f"Created {num_test_images} test images")
    
    # Test with submit() version
    print("\n--- Testing with submit() ---")
    results = run_futures_pipeline(
        test_input_dir,
        f"{test_output_dir}_submit",
        max_workers=2,
        verbose=True
    )
    assert results['successful'] == num_test_images
    
    # Test with map() version
    print("\n--- Testing with map() ---")
    results = run_futures_pipeline_with_map(
        test_input_dir,
        f"{test_output_dir}_map",
        max_workers=2,
        verbose=True
    )
    assert results['successful'] == num_test_images
    
    print("\nConcurrent.futures pipeline tests passed!")
    
    # Cleanup
    shutil.rmtree(test_input_dir)
    for suffix in ['_submit', '_map']:
        output_dir = f"{test_output_dir}{suffix}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    print("Cleanup completed")
