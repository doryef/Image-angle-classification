#!/usr/bin/env python3
"""
Script to sort VisDrone images based on existing labels.
Maps the labels to angle categories and creates text files with image paths.
"""

import csv
from pathlib import Path
from collections import defaultdict

# Define label mapping
LABEL_MAPPING = {
    'horizontal': '0-30',
    'diagonal': '30-60',
    'vertical': '60-90'
}

def main():
    # Paths
    csv_path = Path('/home/p24s09/image_angle_classifier/raw_data/VisDrone/image_labels.csv')
    images_dir = Path('/home/p24s09/image_angle_classifier/raw_data/VisDrone/VisDrone2019-DET-train/VisDrone2019-DET-train/images')
    output_dir = Path('/home/p24s09/image_angle_classifier/automated_sorted_images')
    
    # Dictionary to hold image paths for each category
    categorized_images = defaultdict(list)
    
    # Read CSV and categorize images
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            label = row['label']
            
            # Map label to angle category
            if label in LABEL_MAPPING:
                category = LABEL_MAPPING[label]
                image_path = images_dir / filename
                
                # Verify image exists
                if image_path.exists():
                    categorized_images[category].append(str(image_path))
                else:
                    print(f"Warning: Image not found: {image_path}")
    
    # Write text files for each category
    for category, image_paths in sorted(categorized_images.items()):
        output_file = output_dir / category / 'image_paths.txt'
        
        with open(output_file, 'w') as f:
            for path in sorted(image_paths):
                f.write(path + '\n')
        
        print(f"Category {category}: {len(image_paths)} images written to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    total = sum(len(paths) for paths in categorized_images.values())
    print(f"Total images categorized: {total}")
    for category in sorted(categorized_images.keys()):
        count = len(categorized_images[category])
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {category}°: {count} images ({percentage:.1f}%)")

if __name__ == '__main__':
    main()
