# Parallel Image Processing Pipeline

**CST435 Assignment 2 - Parallel Computing**

A parallel image processing pipeline using the Food-101 dataset, implemented with two different Python parallel paradigms: `multiprocessing` and `concurrent.futures`.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Filter Pipeline](#filter-pipeline)
- [Quick Start (Local)](#quick-start-local)
- [Quick Start (GCP VM)](#quick-start-gcp-vm)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)

## Overview

This project demonstrates parallel computing concepts by implementing an image processing pipeline that applies 5 sequential filters to images. The same pipeline is implemented using two different parallelization paradigms:

1. **`multiprocessing`** - Low-level process management with `Pool.map()`
2. **`concurrent.futures`** - High-level abstraction with `ProcessPoolExecutor`

### Key Features

- ✅ Process-based parallelism (bypasses Python GIL)
- ✅ Image-level parallelism (not pixel-level)
- ✅ Modular, testable code
- ✅ Comprehensive benchmarking
- ✅ CSV output for analysis
- ✅ GCP-deployment ready

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT (Food-101 Images)                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5-STAGE FILTER PIPELINE                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Grayscale │→│ Gaussian │→│  Sobel   │→│ Sharpen  │→│Brightness│  │
│  │          │ │   Blur   │ │  Edge    │ │          │ │  Adjust  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │Sequential│      │  Multi-  │      │ Futures  │
     │(baseline)│      │processing│      │ProcessPool│
     └──────────┘      └──────────┘      └──────────┘
```

### Parallelization Strategy

- **By Image**: Each worker processes one complete image through all 5 filters
- **Embarrassingly Parallel**: No inter-image dependencies
- **Process-Based**: Uses OS processes (not threads) for true parallelism

## Filter Pipeline

Each image passes through 5 filters in sequence:

| # | Filter | Description | Formula/Kernel |
|---|--------|-------------|----------------|
| 1 | **Grayscale** | Convert RGB to luminance | `Y = 0.299R + 0.587G + 0.114B` |
| 2 | **Gaussian Blur** | 3×3 smoothing | `[1,2,1; 2,4,2; 1,2,1] / 16` |
| 3 | **Edge Detection** | Sobel operator | `G = sqrt(Gx² + Gy²)` |
| 4 | **Sharpening** | Edge enhancement | `[0,-1,0; -1,5,-1; 0,-1,0]` |
| 5 | **Brightness** | Scalar adjustment | `Output = clip(Input + value, 0, 255)` |

---

## Quick Start (Local)

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional)

### Step 1: Setup Environment

**Windows (PowerShell/CMD):**
```powershell
# Navigate to project directory
cd cst435-assignment2

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Navigate to project directory
cd cst435-assignment2

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenGL system libraries (required for OpenCV)
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

> **Note**: On Linux systems, OpenCV requires OpenGL libraries. If you encounter `ImportError: libGL.so.1: cannot open shared object file`, run:
> ```bash
> sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
> ```

### Step 2: Run Demo or Benchmark

```bash
# Option 1: Run demo (quick test with 100 images)
python run_demo.py

# Option 2: Run demo with custom image count
python run_demo.py --images 1000

# Option 3: Run full benchmark with custom options
python benchmark.py data/food-101-subset --limit 1000 -p 1 2 4

# Results will be saved to output/benchmark_results.csv
```

### Step 3: View Results

- Check console output for speedup and efficiency metrics
- Open `benchmark_output/benchmark_results.csv` for raw data
- Processed images are in `benchmark_output/` subdirectories

---

## Quick Start (GCP VM)

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed locally

### Step 1: Create a VM Instance

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine** → **VM instances**
3. Click **Create Instance**
4. Configure:
   - **Name**: `image-processor`
   - **Region**: Choose closest to you
   - **Machine type**: `e2-standard-4` (4 vCPUs) or `e2-standard-8` (8 vCPUs)
   - **Boot disk**: Ubuntu 22.04 LTS, 20GB
5. Click **Create**

### Step 2: SSH into VM

Click the **SSH** button next to your VM instance in the console.

### Step 3: Setup Environment on VM

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv git -y

# Clone your repository (or upload files)
git clone https://github.com/danishayman/cst435-assignment2.git
cd cst435-assignment2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenGL system libraries (required for OpenCV)
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Step 4: Run Benchmark on VM

```bash
# Run demo with all 1000 images
python run_demo.py

# Or run benchmark with custom options
python benchmark.py data/food-101-subset --limit 1000 -p 1 2 4 8
```

### Step 5: Download Results

```bash
# View results on VM
cat output/benchmark_results.csv
```

## Usage

### Running Individual Pipelines

```bash
# Run sequential pipeline only
python -c "
from pipeline import run_sequential_pipeline
run_sequential_pipeline('data/food-101-subset', 'output/sequential', limit=50)
"

# Run multiprocessing pipeline only
python -c "
from mp_version import run_multiprocessing_pipeline
run_multiprocessing_pipeline('data/food-101-subset', 'output/mp', num_processes=4, limit=50)
"

# Run concurrent.futures pipeline only
python -c "
from futures_version import run_futures_pipeline
run_futures_pipeline('data/food-101-subset', 'output/futures', max_workers=4, limit=50)
"
```

### Running the Full Benchmark

```bash
# Basic benchmark
python benchmark.py data/food-101-subset

# With options
python benchmark.py data/food-101-subset \
    --output benchmark_output \
    --limit 100 \
    --processes 1 2 4 8 \
    --csv results.csv
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output directory | `benchmark_output` |
| `--limit` | `-l` | Max images to process | All |
| `--processes` | `-p` | Process counts to test | `1 2 4 8` |
| `--csv` | | CSV output path | `<output>/benchmark_results.csv` |
| `--quiet` | `-q` | Suppress verbose output | False |

---

## Benchmarking

### What Gets Measured

| Metric | Formula | Description |
|--------|---------|-------------|
| **Execution Time** | Wall-clock time | Total time from start to finish |
| **Speedup** | `S = T_seq / T_parallel` | How much faster than sequential |
| **Efficiency** | `E = S / num_processes` | Utilization of processors |
| **Parallel Overhead** | `T_parallel(1) - T_seq` | Cost of parallelization |

### Sample Output

```
======================================================================
BENCHMARK SUMMARY
======================================================================

Timestamp: 2026-01-04T10:30:00
Total images processed: 100
Sequential baseline time: 45.2340 seconds

----------------------------------------------------------------------
MULTIPROCESSING RESULTS
----------------------------------------------------------------------
Processes    Time (s)     Speedup      Efficiency  
------------------------------------------------
1            46.1234      0.98         98.00%      
2            24.5678      1.84         92.00%      
4            13.2345      3.42         85.50%      
8            8.9012       5.08         63.50%      

----------------------------------------------------------------------
CONCURRENT.FUTURES RESULTS
----------------------------------------------------------------------
Workers      Time (s)     Speedup      Efficiency  
------------------------------------------------
1            46.3456      0.98         98.00%      
2            24.8901      1.82         91.00%      
4            13.4567      3.36         84.00%      
8            9.1234       4.96         62.00%      
```

### CSV Output Format

```csv
implementation,num_processes,total_time_sec,speedup,efficiency,total_images,successful
sequential,1,45.234000,1.0,1.0,100,100
multiprocessing,1,46.123400,0.9807,0.9807,100,100
multiprocessing,2,24.567800,1.8411,0.9206,100,100
multiprocessing,4,13.234500,3.4179,0.8545,100,100
concurrent.futures,1,46.345600,0.9760,0.9760,100,100
concurrent.futures,2,24.890100,1.8173,0.9086,100,100
concurrent.futures,4,13.456700,3.3609,0.8402,100,100
```

---

## Performance Metrics

### Understanding Speedup

- **Ideal speedup**: `S = n` (linear with process count)
- **Actual speedup**: Usually `S < n` due to overhead
- **Speedup plateau**: Explained by Amdahl's Law

### Amdahl's Law

```
S_max = 1 / (f + (1-f)/n)

Where:
- f = sequential fraction (cannot be parallelized)
- n = number of processors
- S_max = maximum theoretical speedup
```

### Why Speedup Plateaus

1. **Sequential Overhead**: Process creation, task distribution
2. **I/O Bottleneck**: Disk read/write becomes the limiting factor
3. **Memory Bandwidth**: Multiple processes competing for memory
4. **Load Imbalance**: Some images take longer than others

---

## Project Structure

```
cst435-assignment2/
│
├── filters/                    # Filter package (5 filters)
│   ├── __init__.py            # Package exports
│   ├── grayscale.py           # RGB to luminance
│   ├── gaussian_blur.py       # 3×3 Gaussian smoothing
│   ├── edge_detection.py      # Sobel edge detection
│   ├── sharpen.py             # Sharpening kernel
│   ├── brightness.py          # Brightness adjustment
│   ├── convolution.py         # Shared convolution function
│   └── pipeline.py            # apply_all_filters()
│
├── pipeline.py                # Sequential pipeline
├── mp_version.py              # multiprocessing implementation
├── futures_version.py         # concurrent.futures implementation
├── benchmark.py               # Benchmarking and metrics
├── utils.py                   # I/O utility functions
├── run_demo.py                # Demo using food-101-subset images
├── select_random_images.py    # Script to select random images from Food-101
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/                      # Input images
│   └── food-101-subset/
│
└── output/                    # Processed images and results
    └── benchmark_results.csv
```

---

## Paradigm Comparison

| Aspect | `multiprocessing` | `concurrent.futures` |
|--------|-------------------|----------------------|
| **Level** | Low-level | High-level |
| **Pool** | `Pool(n)` | `ProcessPoolExecutor(n)` |
| **Dispatch** | `pool.map()` | `executor.submit()` / `executor.map()` |
| **Futures** | Manual `AsyncResult` | Built-in `Future` |
| **Context Manager** | Optional | Recommended |
| **Error Handling** | Manual | Via `Future.result()` |
| **Use Case** | Fine control needed | Simpler interface |


