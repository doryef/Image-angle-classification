#!/usr/bin/env python3
"""
Create LIGHT train/val/test splits for synthetic SkyScenes data.
Uses a small subset for quick training/testing - approximately 100 images per class.
Includes images from all towns to maintain diversity.
"""
import os
import re
from pathlib import Path
from collections import defaultdict
import random

# Configuration
RAW_DATA_DIR = "/home/p24s09/image_angle_classifier/raw_data/SkyScenes/Images"
OUTPUT_BASE_DIR = "/home/p24s09/image_angle_classifier/data/synthetic_only_light"

# Town-based splits (include ALL towns in training for diversity)
TRAIN_TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town10' ]
VAL_TOWNS = ['Town07']
TEST_TOWNS = ['Town05', 'Town06',]  # Will reuse Town10 but with different samples

# Target images per class for training (will sample from all train towns)
TARGET_TRAIN_IMAGES_PER_CLASS = 100
TARGET_VAL_IMAGES_PER_CLASS = 20
TARGET_TEST_IMAGES_PER_CLASS = 20

# Angle mapping based on pitch (P) value
# P_0 -> 0-30°, P_45 -> 30-60°, P_90 -> 60-90° (only using P_90, excluding P_60)
ANGLE_MAPPING = {
    'P_0': '0-30',
    'P_45': '30-60',
    'P_90': '60-90'
}

# Random seed for reproducibility
RANDOM_SEED = 42

def get_town_from_path(path):
    """Extract town name from image path"""
    match = re.search(r'Town\d+', path)
    return match.group(0) if match else None

def collect_skyscenes_images():
    """Collect all SkyScenes images and organize by angle category and town"""
    print("Scanning SkyScenes directory...")
    print(f"Train towns: {', '.join(TRAIN_TOWNS)}")
    print(f"Val towns: {', '.join(VAL_TOWNS)}")
    print(f"Test towns: {', '.join(TEST_TOWNS)}")
    
    # Structure: {split: {angle: [paths]}}
    all_data = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list),
    }
    
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
                if town in TRAIN_TOWNS:
                    split = 'train'
                elif town in VAL_TOWNS:
                    split = 'val'
                elif town in TEST_TOWNS:
                    split = 'test'
                else:
                    continue
                
                # Extract angle from path
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('H_') and '_P_' in part:
                        pitch = '_'.join(part.split('_')[2:4])
                        if pitch in ANGLE_MAPPING:
                            angle_class = ANGLE_MAPPING[pitch]
                            all_data[split][angle_class].append(img_path)
                        break
    
    return all_data

def sample_diverse_images(image_paths, target_count, seed=RANDOM_SEED):
    """
    Sample images ensuring diversity across towns.
    Tries to get roughly equal representation from each town.
    """
    random.seed(seed)
    
    # Group by town
    town_images = defaultdict(list)
    for path in image_paths:
        town = get_town_from_path(path)
        if town:
            town_images[town].append(path)
    
    num_towns = len(town_images)
    if num_towns == 0:
        return []
    
    # Calculate images per town (roughly equal)
    images_per_town = max(1, target_count // num_towns)
    remainder = target_count % num_towns
    
    sampled = []
    for i, (town, paths) in enumerate(sorted(town_images.items())):
        # Add one extra image to first 'remainder' towns
        town_target = images_per_town + (1 if i < remainder else 0)
        town_sample = random.sample(paths, min(town_target, len(paths)))
        sampled.extend(town_sample)
    
    # If we still need more images, sample randomly from all
    if len(sampled) < target_count:
        remaining = [p for p in image_paths if p not in sampled]
        additional = random.sample(remaining, min(target_count - len(sampled), len(remaining)))
        sampled.extend(additional)
    
    return sampled

def create_light_splits(all_data):
    """Create light (small) splits by sampling from collected data"""
    print("\n" + "="*60)
    print("CREATING LIGHT SPLITS")
    print("="*60)
    
    split_data = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }
    
    # Sample training data (diverse across all train towns)
    print(f"\nTRAIN (target: {TARGET_TRAIN_IMAGES_PER_CLASS} per class):")
    for angle in sorted(all_data['train'].keys()):
        sampled = sample_diverse_images(
            all_data['train'][angle], 
            TARGET_TRAIN_IMAGES_PER_CLASS
        )
        split_data['train'][angle] = sampled
        
        # Count towns represented
        towns = set(get_town_from_path(p) for p in sampled)
        print(f"  {angle}: {len(sampled)} images from {len(towns)} towns {sorted(towns)}")
    
    # Sample validation data
    print(f"\nVAL (target: {TARGET_VAL_IMAGES_PER_CLASS} per class):")
    for angle in sorted(all_data['val'].keys()):
        available = all_data['val'][angle]
        sampled = random.sample(available, min(TARGET_VAL_IMAGES_PER_CLASS, len(available)))
        split_data['val'][angle] = sampled
        
        towns = set(get_town_from_path(p) for p in sampled)
        print(f"  {angle}: {len(sampled)} images from towns {sorted(towns)}")
    
    # Sample test data
    print(f"\nTEST (target: {TARGET_TEST_IMAGES_PER_CLASS} per class):")
    for angle in sorted(all_data['test'].keys()):
        available = all_data['test'][angle]
        sampled = random.sample(available, min(TARGET_TEST_IMAGES_PER_CLASS, len(available)))
        split_data['test'][angle] = sampled
        
        towns = set(get_town_from_path(p) for p in sampled)
        print(f"  {angle}: {len(sampled)} images from towns {sorted(towns)}")
    
    return split_data

def write_split_files(split_data):
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
                for path in sorted(paths):
                    f.write(f"{path}\n")
            
            print(f"  {angle}: {len(paths)} images -> {output_file}")

def verify_splits():
    """Verify that all splits were created correctly"""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    angles = ['0-30', '30-60', '60-90']
    
    total_images = 0
    for split in splits:
        print(f"\n{split.upper()}:")
        split_total = 0
        for angle in angles:
            path_file = os.path.join(OUTPUT_BASE_DIR, split, angle, 'paths.txt')
            if os.path.exists(path_file):
                with open(path_file, 'r') as f:
                    count = sum(1 for line in f if line.strip())
                print(f"  {angle}: {count} images")
                split_total += count
            else:
                print(f"  {angle}: MISSING!")
        print(f"  Total: {split_total} images")
        total_images += split_total
    
    print(f"\nGrand Total: {total_images} images")

def main():
    print("="*60)
    print("Creating LIGHT Synthetic Dataset Splits")
    print("="*60)
    
    # Collect all available images
    all_data = collect_skyscenes_images()
    
    if all_data is None:
        print("No images found! Exiting.")
        return
    
    # Create light splits by sampling
    split_data = create_light_splits(all_data)
    
    # Write the files
    write_split_files(split_data)
    
    # Verify the results
    verify_splits()
    
    print("\n" + "="*60)
    print("DONE! Light dataset created successfully.")
    print(f"\nThis is a small dataset (~{TARGET_TRAIN_IMAGES_PER_CLASS} train images per class)")
    print("Perfect for quick training tests and experimentation!")
    print("\nYou can use this dataset with:")
    print(f"  train_dir: {OUTPUT_BASE_DIR}/train")
    print(f"  val_dir: {OUTPUT_BASE_DIR}/val")
    print(f"  test_dir: {OUTPUT_BASE_DIR}/test")
    print("="*60)

if __name__ == "__main__":
    main()
