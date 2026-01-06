"""
benchmark.py - Benchmarking and Performance Measurement

This module provides comprehensive benchmarking for the parallel
image processing pipeline implementations.

Features:
    - Execution time measurement
    - Speedup calculation (S = T_seq / T_parallel)
    - Efficiency calculation (E = S / num_processes)
    - Parallel overhead measurement
    - CSV output for later plotting
    - Support for multiple process counts

Key Metrics:
    - Speedup: How much faster parallel is vs sequential
    - Efficiency: How well we utilize available processors
    - Parallel Overhead: Extra cost of parallelization
"""

import os
import csv
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import multiprocessing as mp

from pipeline import run_sequential_pipeline
from mp_version import run_multiprocessing_pipeline
from futures_version import run_futures_pipeline
from utils import ensure_directory


def run_benchmark(input_dir: str, output_base_dir: str,
                  process_counts: Optional[List[int]] = None,
                  image_limit: Optional[int] = None,
                  brightness_value: int = 30,
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Run a complete benchmark comparing all implementations.
    
    This function:
        1. Runs the sequential pipeline (baseline)
        2. Runs multiprocessing pipeline with varying process counts
        3. Runs concurrent.futures pipeline with varying process counts
        4. Calculates speedup and efficiency for each configuration
    
    Args:
        input_dir: Directory containing input images
        output_base_dir: Base directory for outputs
        process_counts: List of process counts to test (default: [1, 2, 4, 8])
        image_limit: Maximum number of images to process
        brightness_value: Brightness adjustment value
        verbose: Print progress information
    
    Returns:
        Dictionary containing:
            - sequential: Results from sequential run
            - multiprocessing: List of results for each process count
            - futures: List of results for each process count
            - metrics: Calculated speedup and efficiency values
            - timestamp: When the benchmark was run
            - system_info: Information about the system
    """
    # Default process counts
    if process_counts is None:
        cpu_count = mp.cpu_count()
        process_counts = [1, 2, 4, min(8, cpu_count)]
        # Remove duplicates and sort
        process_counts = sorted(list(set(process_counts)))
    
    # System information
    system_info = {
        'cpu_count': mp.cpu_count(),
        'process_counts_tested': process_counts,
        'image_limit': image_limit
    }
    
    if verbose:
        print("="*70)
        print("PARALLEL IMAGE PROCESSING BENCHMARK")
        print("="*70)
        print(f"\nSystem Information:")
        print(f"  Available CPUs: {system_info['cpu_count']}")
        print(f"  Process counts to test: {process_counts}")
        print(f"  Image limit: {image_limit or 'All images'}")
        print(f"  Input directory: {input_dir}")
        print(f"  Output base directory: {output_base_dir}")
        print()
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'sequential': None,
        'multiprocessing': [],
        'futures': [],
        'metrics': {
            'multiprocessing': [],
            'futures': []
        }
    }
    
    # =========================================================================
    # Step 1: Run Sequential Baseline
    # =========================================================================
    if verbose:
        print("-"*70)
        print("STEP 1: Sequential Baseline")
        print("-"*70)
    
    seq_output_dir = os.path.join(output_base_dir, "sequential")
    seq_result = run_sequential_pipeline(
        input_dir=input_dir,
        output_dir=seq_output_dir,
        limit=image_limit,
        brightness_value=brightness_value,
        verbose=verbose
    )
    
    results['sequential'] = {
        'total_images': seq_result['total_images'],
        'successful': seq_result['successful'],
        'failed': seq_result['failed'],
        'total_time': seq_result['total_time'],
        'avg_time_per_image': seq_result['avg_time_per_image']
    }
    
    baseline_time = seq_result['total_time']
    
    if verbose:
        print(f"\nBaseline time: {baseline_time:.4f} seconds")
    
    # =========================================================================
    # Step 2: Run Multiprocessing with Varying Process Counts
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("STEP 2: Multiprocessing Pipeline")
        print("-"*70)
    
    for num_procs in process_counts:
        if verbose:
            print(f"\n>>> Testing with {num_procs} processes...")
        
        mp_output_dir = os.path.join(output_base_dir, f"multiprocessing_p{num_procs}")
        
        mp_result = run_multiprocessing_pipeline(
            input_dir=input_dir,
            output_dir=mp_output_dir,
            num_processes=num_procs,
            limit=image_limit,
            brightness_value=brightness_value,
            verbose=verbose
        )
        
        # Store raw results
        results['multiprocessing'].append({
            'num_processes': num_procs,
            'total_images': mp_result['total_images'],
            'successful': mp_result['successful'],
            'failed': mp_result['failed'],
            'total_time': mp_result['total_time'],
            'avg_time_per_image': mp_result['avg_time_per_image']
        })
        
        # Calculate metrics
        speedup = baseline_time / mp_result['total_time'] if mp_result['total_time'] > 0 else 0
        efficiency = speedup / num_procs if num_procs > 0 else 0
        
        results['metrics']['multiprocessing'].append({
            'num_processes': num_procs,
            'speedup': speedup,
            'efficiency': efficiency,
            'time': mp_result['total_time']
        })
        
        if verbose:
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Efficiency: {efficiency:.2%}")
    
    # =========================================================================
    # Step 3: Run Concurrent.Futures with Varying Process Counts
    # =========================================================================
    if verbose:
        print("\n" + "-"*70)
        print("STEP 3: Concurrent.Futures Pipeline")
        print("-"*70)
    
    for num_workers in process_counts:
        if verbose:
            print(f"\n>>> Testing with {num_workers} workers...")
        
        fut_output_dir = os.path.join(output_base_dir, f"futures_w{num_workers}")
        
        fut_result = run_futures_pipeline(
            input_dir=input_dir,
            output_dir=fut_output_dir,
            max_workers=num_workers,
            limit=image_limit,
            brightness_value=brightness_value,
            verbose=verbose
        )
        
        # Store raw results
        results['futures'].append({
            'max_workers': num_workers,
            'total_images': fut_result['total_images'],
            'successful': fut_result['successful'],
            'failed': fut_result['failed'],
            'total_time': fut_result['total_time'],
            'avg_time_per_image': fut_result['avg_time_per_image']
        })
        
        # Calculate metrics
        speedup = baseline_time / fut_result['total_time'] if fut_result['total_time'] > 0 else 0
        efficiency = speedup / num_workers if num_workers > 0 else 0
        
        results['metrics']['futures'].append({
            'num_workers': num_workers,
            'speedup': speedup,
            'efficiency': efficiency,
            'time': fut_result['total_time']
        })
        
        if verbose:
            print(f"    Speedup: {speedup:.2f}x")
            print(f"    Efficiency: {efficiency:.2%}")
    
    return results


def calculate_parallel_overhead(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate parallel overhead from benchmark results.
    
    Parallel overhead is the extra time spent on parallelization
    when using only 1 process, compared to sequential execution.
    
    Overhead = T_parallel(1) - T_sequential
    
    Args:
        results: Benchmark results from run_benchmark()
    
    Returns:
        Dictionary with overhead values for each implementation
    """
    seq_time = results['sequential']['total_time']
    
    # Find single-process results
    mp_single = next((m for m in results['multiprocessing'] if m['num_processes'] == 1), None)
    fut_single = next((f for f in results['futures'] if f['max_workers'] == 1), None)
    
    overhead = {
        'sequential_time': seq_time,
        'multiprocessing_overhead': None,
        'futures_overhead': None
    }
    
    if mp_single:
        overhead['multiprocessing_overhead'] = mp_single['total_time'] - seq_time
        overhead['multiprocessing_overhead_percent'] = (overhead['multiprocessing_overhead'] / seq_time) * 100
    
    if fut_single:
        overhead['futures_overhead'] = fut_single['total_time'] - seq_time
        overhead['futures_overhead_percent'] = (overhead['futures_overhead'] / seq_time) * 100
    
    return overhead


def export_results_to_csv(results: Dict[str, Any], output_path: str) -> None:
    """
    Export benchmark results to a CSV file.
    
    Creates a CSV with columns suitable for plotting:
        - implementation, processes, time, speedup, efficiency
    
    Args:
        results: Benchmark results from run_benchmark()
        output_path: Path for the CSV file
    """
    ensure_directory(os.path.dirname(output_path) if os.path.dirname(output_path) else '.')
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'implementation', 
            'num_processes', 
            'total_time_sec', 
            'speedup', 
            'efficiency',
            'total_images',
            'successful'
        ])
        
        # Write sequential baseline
        seq = results['sequential']
        writer.writerow([
            'sequential',
            1,
            f"{seq['total_time']:.6f}",
            1.0,
            1.0,
            seq['total_images'],
            seq['successful']
        ])
        
        # Write multiprocessing results
        for i, mp_data in enumerate(results['multiprocessing']):
            metrics = results['metrics']['multiprocessing'][i]
            writer.writerow([
                'multiprocessing',
                mp_data['num_processes'],
                f"{mp_data['total_time']:.6f}",
                f"{metrics['speedup']:.4f}",
                f"{metrics['efficiency']:.4f}",
                mp_data['total_images'],
                mp_data['successful']
            ])
        
        # Write futures results
        for i, fut_data in enumerate(results['futures']):
            metrics = results['metrics']['futures'][i]
            writer.writerow([
                'concurrent.futures',
                fut_data['max_workers'],
                f"{fut_data['total_time']:.6f}",
                f"{metrics['speedup']:.4f}",
                f"{metrics['efficiency']:.4f}",
                fut_data['total_images'],
                fut_data['successful']
            ])
    
    print(f"Results exported to: {output_path}")


def print_summary_table(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary table of benchmark results with nice borders.
    
    Args:
        results: Benchmark results from run_benchmark()
    """
    # Box drawing characters
    TL = "╔"  # Top left
    TR = "╗"  # Top right
    BL = "╚"  # Bottom left
    BR = "╝"  # Bottom right
    H = "═"   # Horizontal
    V = "║"   # Vertical
    LT = "╠"  # Left T
    RT = "╣"  # Right T
    TT = "╦"  # Top T
    BT = "╩"  # Bottom T
    CR = "╬"  # Cross
    
    # Column widths
    col1, col2, col3, col4 = 15, 15, 15, 15
    total_width = col1 + col2 + col3 + col4 + 5  # 5 for borders
    
    print()
    print(TL + H*total_width + TR)
    title = "BENCHMARK SUMMARY"
    print(V + title.center(total_width) + V)
    print(LT + H*total_width + RT)
    
    # Info section
    print(V + f"  Timestamp: {results['timestamp'][:19]}".ljust(total_width) + V)
    print(V + f"  Total images processed: {results['sequential']['total_images']}".ljust(total_width) + V)
    print(V + f"  Sequential baseline time: {results['sequential']['total_time']:.4f} seconds".ljust(total_width) + V)
    print(LT + H*total_width + RT)
    
    # Multiprocessing header
    mp_title = "MULTIPROCESSING RESULTS"
    print(V + mp_title.center(total_width) + V)
    print(LT + H*col1 + TT + H*col2 + TT + H*col3 + TT + H*col4 + RT)
    print(V + "Processes".center(col1) + V + "Time (s)".center(col2) + V + "Speedup".center(col3) + V + "Efficiency".center(col4) + V)
    print(LT + H*col1 + CR + H*col2 + CR + H*col3 + CR + H*col4 + RT)
    
    for i, mp_data in enumerate(results['multiprocessing']):
        metrics = results['metrics']['multiprocessing'][i]
        print(V + str(mp_data['num_processes']).center(col1) + V + 
              f"{mp_data['total_time']:.4f}".center(col2) + V + 
              f"{metrics['speedup']:.2f}x".center(col3) + V + 
              f"{metrics['efficiency']*100:.2f}%".center(col4) + V)
    
    print(LT + H*col1 + BT + H*col2 + BT + H*col3 + BT + H*col4 + RT)
    
    # Futures header
    fut_title = "CONCURRENT.FUTURES RESULTS (ThreadPoolExecutor)"
    print(V + fut_title.center(total_width) + V)
    print(LT + H*col1 + TT + H*col2 + TT + H*col3 + TT + H*col4 + RT)
    print(V + "Workers".center(col1) + V + "Time (s)".center(col2) + V + "Speedup".center(col3) + V + "Efficiency".center(col4) + V)
    print(LT + H*col1 + CR + H*col2 + CR + H*col3 + CR + H*col4 + RT)
    
    for i, fut_data in enumerate(results['futures']):
        metrics = results['metrics']['futures'][i]
        print(V + str(fut_data['max_workers']).center(col1) + V + 
              f"{fut_data['total_time']:.4f}".center(col2) + V + 
              f"{metrics['speedup']:.2f}x".center(col3) + V + 
              f"{metrics['efficiency']*100:.2f}%".center(col4) + V)
    
    print(LT + H*col1 + BT + H*col2 + BT + H*col3 + BT + H*col4 + RT)
    
    # Parallel overhead section
    overhead = calculate_parallel_overhead(results)
    oh_title = "PARALLEL OVERHEAD (1 process vs sequential)"
    print(V + oh_title.center(total_width) + V)
    print(LT + H*total_width + RT)
    
    if overhead['multiprocessing_overhead'] is not None:
        mp_oh = f"  Multiprocessing: {overhead['multiprocessing_overhead']:>8.4f}s ({overhead['multiprocessing_overhead_percent']:>5.1f}%)"
        print(V + mp_oh.ljust(total_width) + V)
    if overhead['futures_overhead'] is not None:
        fut_oh = f"  Futures:         {overhead['futures_overhead']:>8.4f}s ({overhead['futures_overhead_percent']:>5.1f}%)"
        print(V + fut_oh.ljust(total_width) + V)
    
    print(BL + H*total_width + BR)


def run_quick_benchmark(input_dir: str, output_dir: str = "benchmark_output",
                        image_limit: int = 50) -> Dict[str, Any]:
    """
    Run a quick benchmark with sensible defaults.
    
    Convenience function for rapid testing.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Output directory
        image_limit: Number of images to process
    
    Returns:
        Benchmark results dictionary
    """
    print("Running quick benchmark...")
    print(f"Using {image_limit} images")
    
    results = run_benchmark(
        input_dir=input_dir,
        output_base_dir=output_dir,
        process_counts=[1, 2, 4],
        image_limit=image_limit,
        verbose=True
    )
    
    print_summary_table(results)
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    export_results_to_csv(results, csv_path)
    
    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Benchmark parallel image processing pipeline'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output', '-o',
        default='benchmark_output',
        help='Base output directory (default: benchmark_output)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of images to process'
    )
    parser.add_argument(
        '--processes', '-p',
        type=int,
        nargs='+',
        default=None,
        help='Process counts to test (default: 1 2 4 8)'
    )
    parser.add_argument(
        '--csv',
        default=None,
        help='Path for CSV output file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run benchmark
    results = run_benchmark(
        input_dir=args.input_dir,
        output_base_dir=args.output,
        process_counts=args.processes,
        image_limit=args.limit,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary_table(results)
    
    # Export to CSV if requested
    csv_path = args.csv or os.path.join(args.output, "benchmark_results.csv")
    export_results_to_csv(results, csv_path)
    
    print("\nBenchmark complete!")
