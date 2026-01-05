"""
download_dataset.py - Download and Prepare Food-101 Dataset

Downloads the Food-101 dataset from Kaggle and copies a random subset
of images to the local data folder for benchmarking.

Prerequisites:
    pip install kagglehub

Usage:
    python download_dataset.py

"""

import os
import random
import shutil
from pathlib import Path

import kagglehub


# =============================================================================
# CONFIGURATION - Set how many random images you want
# =============================================================================

NUM_IMAGES = 100  # <-- Change this to set how many random images to use

# =============================================================================


def download_and_prepare_dataset(num_images: int = NUM_IMAGES, 
                                  output_dir: str = "data/food-101-subset"):
    """
    Download Food-101 dataset from Kaggle and copy a random subset.
    
    Args:
        num_images: Number of random images to copy
        output_dir: Directory to save the subset
    
    Returns:
        Path to the output directory
    """
    print("="*60)
    print("FOOD-101 DATASET DOWNLOADER")
    print("="*60)
    print(f"\nTarget: {num_images} random images")
    print(f"Output: {output_dir}")
    print()
    
    # Step 1: Download dataset from Kaggle
    print("Step 1: Downloading from Kaggle...")
    print("       (This may take a while on first run)")
    
    dataset_path = kagglehub.dataset_download("dansbecker/food-101")
    print(f"       Dataset downloaded to: {dataset_path}")
    
    # Step 2: Find all images in the dataset
    print("\nStep 2: Scanning for images...")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                all_images.append(os.path.join(root, file))
    
    print(f"       Found {len(all_images)} total images")
    
    if len(all_images) == 0:
        print("ERROR: No images found in dataset!")
        return None
    
    # Step 3: Select random subset
    print(f"\nStep 3: Selecting {num_images} random images...")
    
    if num_images > len(all_images):
        print(f"       Warning: Requested {num_images} but only {len(all_images)} available")
        num_images = len(all_images)
    
    selected_images = random.sample(all_images, num_images)
    print(f"       Selected {len(selected_images)} images")
    
    # Step 4: Copy to output directory
    print(f"\nStep 4: Copying to {output_dir}...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    copied = 0
    for i, src_path in enumerate(selected_images, 1):
        # Get the category folder name (parent directory)
        src_path_obj = Path(src_path)
        category = src_path_obj.parent.name
        filename = src_path_obj.name
        
        # Create category subdirectory
        category_dir = output_path / category
        category_dir.mkdir(exist_ok=True)
        
        # Copy file
        dst_path = category_dir / filename
        shutil.copy2(src_path, dst_path)
        copied += 1
        
        if i % 20 == 0 or i == len(selected_images):
            print(f"       Progress: {i}/{len(selected_images)} images")
    
    print(f"\n       Copied {copied} images to {output_dir}")
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    
    # Count categories
    categories = [d for d in output_path.iterdir() if d.is_dir()]
    images_per_category = {}
    for cat in categories:
        count = len(list(cat.glob('*')))
        images_per_category[cat.name] = count
    
    print(f"\nSummary:")
    print(f"  Total images: {copied}")
    print(f"  Categories: {len(categories)}")
    print(f"  Output path: {output_path.absolute()}")
    
    print(f"\nImages per category:")
    for cat, count in sorted(images_per_category.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    if len(categories) > 10:
        print(f"  ... and {len(categories) - 10} more categories")
    
    print(f"\nYou can now run:")
    print(f"  python benchmark.py {output_dir} --limit {num_images}")
    
    return str(output_path)


def clear_existing_data(output_dir: str = "data/food-101-subset"):
    """
    Clear existing data in the output directory.
    
    Args:
        output_dir: Directory to clear
    """
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Clearing existing data in {output_dir}...")
        shutil.rmtree(output_path)
        print("Done.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Food-101 dataset')
    parser.add_argument('--num-images', '-n', type=int, default=NUM_IMAGES,
                        help=f'Number of random images to download (default: {NUM_IMAGES})')
    parser.add_argument('--output', '-o', type=str, default='data/food-101-subset',
                        help='Output directory (default: data/food-101-subset)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear existing data before downloading')
    
    args = parser.parse_args()
    
    if args.clear:
        clear_existing_data(args.output)
    
    download_and_prepare_dataset(
        num_images=args.num_images,
        output_dir=args.output
    )
