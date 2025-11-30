#!/usr/bin/env python3
"""
Create train/val/test splits for synthetic SkyScenes data only.
This script scans the raw_data/SkyScenes directory and creates paths.txt files
organized by camera angle, splitting by town to ensure no data leakage.
"""
import os
import re
from pathlib import Path
from collections import defaultdict

# Configuration
RAW_DATA_DIR = "/home/p24s09/image_angle_classifier/raw_data/SkyScenes/Images"
OUTPUT_BASE_DIR = "/home/p24s09/image_angle_classifier/data/synthetic_only"

# Town-based splits (no overlap to prevent data leakage)
TRAIN_TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']
VAL_TOWNS = ['Town07']
TEST_TOWNS = ['Town10', "Town06"]

# Angle mapping based on pitch (P) value
# P_0 -> 0-30°, P_45 -> 30-60°, P_90 -> 60-90° (only using P_90, excluding P_60)
ANGLE_MAPPING = {
    'P_0': '0-30',
    'P_45': '30-60',
    'P_90': '60-90'
}

def get_town_from_path(path):
    """Extract town name from image path"""
    # Match Town followed by digits (e.g., Town01, Town10)
    match = re.search(r'Town\d+', path)
    return match.group(0) if match else None

def collect_skyscenes_images():
    """Collect all SkyScenes images and organize by angle category and town"""
    print("Scanning SkyScenes directory...")
    print(f"Train towns: {', '.join(TRAIN_TOWNS)}")
    print(f"Val towns: {', '.join(VAL_TOWNS)}")
    print(f"Test towns: {', '.join(TEST_TOWNS)}")
    
    # Structure: {split: {angle: [paths]}}
    split_data = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    # Track statistics
    town_stats = defaultdict(lambda: defaultdict(int))
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Directory {RAW_DATA_DIR} does not exist!")
        return None
    
    # Walk through the SkyScenes directory
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img_path = os.path.abspath(os.path.join(root, file))
                
                # Extract town from path
                town = get_town_from_path(img_path)
                if not town:
                    continue
                
                # Determine which split this town belongs to
                split = None
                if town in TRAIN_TOWNS:
                    split = 'train'
                elif town in VAL_TOWNS:
                    split = 'val'
                elif town in TEST_TOWNS:
                    split = 'test'
                else:
                    continue  # Skip towns not in any split
                
                # Extract angle from path (e.g., H_60_P_45)
                # Path format: .../H_XX_P_YY/...
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('H_') and '_P_' in part:
                        # Extract pitch value (e.g., 'P_45' from 'H_60_P_45')
                        pitch = '_'.join(part.split('_')[2:4])  # Gets 'P_45'
                        if pitch in ANGLE_MAPPING:
                            angle_class = ANGLE_MAPPING[pitch]
                            split_data[split][angle_class].append(img_path)
                            town_stats[town][angle_class] += 1
                        break
    
    # Print statistics
    print("\n" + "="*60)
    print("COLLECTION STATISTICS BY TOWN")
    print("="*60)
    for town in sorted(town_stats.keys()):
        total = sum(town_stats[town].values())
        print(f"\n{town} (Total: {total}):")
        for angle in sorted(town_stats[town].keys()):
            print(f"  {angle}: {town_stats[town][angle]} images")
    
    print("\n" + "="*60)
    print("COLLECTION STATISTICS BY SPLIT")
    print("="*60)
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()}:")
        for angle in sorted(split_data[split].keys()):
            print(f"  {angle}: {len(split_data[split][angle])} images")
    
    return split_data

def create_split_files(split_data):
    """Write paths.txt files for each split and angle"""
    print("\n" + "="*60)
    print("WRITING SPLIT FILES")
    print("="*60)
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        for angle in sorted(split_data[split_name].keys()):
            paths = split_data[split_name][angle]
            
            # Create directory structure
            split_dir = os.path.join(OUTPUT_BASE_DIR, split_name, angle)
            os.makedirs(split_dir, exist_ok=True)
            
            # Write paths.txt
            output_file = os.path.join(split_dir, 'paths.txt')
            with open(output_file, 'w') as f:
                for path in sorted(paths):  # Sort for reproducibility
                    f.write(f"{path}\n")
            
            print(f"  {angle}: {len(paths)} images -> {output_file}")

def verify_splits():
    """Verify that all splits were created correctly"""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    angles = ['0-30', '30-60', '60-90']
    
    for split in splits:
        print(f"\n{split.upper()}:")
        for angle in angles:
            path_file = os.path.join(OUTPUT_BASE_DIR, split, angle, 'paths.txt')
            if os.path.exists(path_file):
                with open(path_file, 'r') as f:
                    count = sum(1 for line in f if line.strip())
                print(f"  {angle}: {count} images")
            else:
                print(f"  {angle}: MISSING!")

def main():
    print("="*60)
    print("Creating Synthetic-Only Dataset Splits (By Town)")
    print("="*60)
    
    # Collect all SkyScenes images organized by split and angle
    split_data = collect_skyscenes_images()
    
    if split_data is None:
        print("No images found! Exiting.")
        return
    
    # Check if we have data for all splits
    if not all(split_data[split] for split in ['train', 'val', 'test']):
        print("Warning: Some splits have no data!")
    
    # Create the split files
    create_split_files(split_data)
    
    # Verify the results
    verify_splits()
    
    print("\n" + "="*60)
    print("DONE! Town-based splits created successfully.")
    print(f"Train: Towns {', '.join(TRAIN_TOWNS)}")
    print(f"Val: Towns {', '.join(VAL_TOWNS)}")
    print(f"Test: Towns {', '.join(TEST_TOWNS)}")
    print("\nYou can now use this dataset with:")
    print(f"  train_dir: {OUTPUT_BASE_DIR}/train")
    print(f"  val_dir: {OUTPUT_BASE_DIR}/val")
    print(f"  test_dir: {OUTPUT_BASE_DIR}/test")
    print("="*60)

if __name__ == "__main__":
    main()
