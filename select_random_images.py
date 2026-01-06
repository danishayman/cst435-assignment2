"""
Script to select random 1000 images from food-101 folder and copy to data folder.
"""

import os
import random
import shutil
from pathlib import Path

def get_all_images(source_dir):
    """Get all image file paths from the source directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    images = []
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(root, file))
    
    return images

def select_and_copy_images(source_dir, dest_dir, num_images=1000, seed=42):
    """
    Select random images from source directory and copy to destination.
    
    Args:
        source_dir: Path to the food-101/images folder
        dest_dir: Path to the destination data folder
        num_images: Number of random images to select (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all images
    print(f"Scanning for images in: {source_dir}")
    all_images = get_all_images(source_dir)
    print(f"Found {len(all_images)} total images")
    
    if len(all_images) < num_images:
        print(f"Warning: Only {len(all_images)} images available, selecting all of them")
        num_images = len(all_images)
    
    # Randomly select images
    selected_images = random.sample(all_images, num_images)
    print(f"Selected {len(selected_images)} random images")
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy images to destination
    print(f"Copying images to: {dest_dir}")
    copied_count = 0
    
    for img_path in selected_images:
        # Get the category folder name and filename
        parts = Path(img_path).parts
        # Find the index of 'images' folder to get category
        try:
            img_idx = parts.index('images')
            category = parts[img_idx + 1]
            filename = parts[-1]
        except (ValueError, IndexError):
            # Fallback: just use parent folder name
            category = Path(img_path).parent.name
            filename = Path(img_path).name
        
        # Create unique filename with category prefix to avoid duplicates
        new_filename = f"{category}_{filename}"
        dest_path = os.path.join(dest_dir, new_filename)
        
        # Handle duplicate filenames by adding a suffix
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(new_filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                counter += 1
        
        shutil.copy2(img_path, dest_path)
        copied_count += 1
        
        if copied_count % 200 == 0:
            print(f"Progress: {copied_count}/{num_images} images copied")
    
    print(f"\nDone! Copied {copied_count} images to {dest_dir}")
    
    # Print summary by category
    print("\nSummary by category:")
    categories = {}
    for img_path in selected_images:
        parts = Path(img_path).parts
        try:
            img_idx = parts.index('images')
            category = parts[img_idx + 1]
        except (ValueError, IndexError):
            category = Path(img_path).parent.name
        categories[category] = categories.get(category, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} images")

def main():
    # Define paths
    script_dir = Path(__file__).parent
    source_dir = script_dir / "food-101" / "images"
    dest_dir = script_dir / "data" / "food-101-subset"
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    # Run the selection and copy process
    select_and_copy_images(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        num_images=1000,
        seed=42  # Set seed for reproducibility
    )

if __name__ == "__main__":
    main()
