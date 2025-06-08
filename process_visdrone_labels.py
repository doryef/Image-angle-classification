#!/usr/bin/env python3
"""
Script to process VisDrone CSV file and organize image paths by label.

This script reads the CSV file containing image filenames and labels,
and organizes the image paths by their corresponding labels (diagonal, horizontal, vertical).
"""

import os
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json


def process_visdrone_labels(csv_path, images_dir, output_format='dict'):
    """
    Process the VisDrone labels CSV and organize image paths by label.
    
    Args:
        csv_path (str): Path to the CSV file containing filenames and labels
        images_dir (str): Path to the directory containing the images
        output_format (str): Output format - 'dict', 'json', or 'summary'
    
    Returns:
        dict: Dictionary with labels as keys and lists of image paths as values
    """
    # Read the CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV file must contain 'filename' and 'label' columns")
    
    # Get unique labels
    unique_labels = df['label'].unique()
    print(f"Found labels: {unique_labels}")
    
    # Initialize result dictionary
    organized_paths = defaultdict(list)
    missing_files = []
    
    # Process each row
    print("Processing images...")
    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label']
        
        # Construct full image path
        image_path = os.path.join(images_dir, filename)
        
        # Check if image file exists
        if os.path.exists(image_path):
            organized_paths[label].append(image_path)
        else:
            missing_files.append(filename)
    
    # Convert defaultdict to regular dict
    result = dict(organized_paths)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    total_images = len(df)
    found_images = sum(len(paths) for paths in result.values())
    
    print(f"Total images in CSV: {total_images}")
    print(f"Images found: {found_images}")
    print(f"Missing images: {len(missing_files)}")
    
    print("\nImages per label:")
    for label, paths in result.items():
        print(f"  {label}: {len(paths)} images")
    
    if missing_files:
        print(f"\nFirst 10 missing files:")
        for filename in missing_files[:10]:
            print(f"  {filename}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Handle different output formats
    if output_format == 'json':
        return json.dumps(result, indent=2)
    elif output_format == 'summary':
        return {
            'total_images': total_images,
            'found_images': found_images,
            'missing_images': len(missing_files),
            'labels_summary': {label: len(paths) for label, paths in result.items()},
            'missing_files': missing_files[:10]  # First 10 missing files
        }
    else:
        return result


def save_organized_paths(organized_paths, output_dir="organized_data"):
    """
    Save organized paths to separate files for each label.
    
    Args:
        organized_paths (dict): Dictionary with labels as keys and image paths as values
        output_dir (str): Directory to save the organized files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\nSaving organized paths to: {output_dir}")
    
    for label, paths in organized_paths.items():
        output_file = os.path.join(output_dir, f"{label}_images.txt")
        
        with open(output_file, 'w') as f:
            for path in paths:
                f.write(f"{path}\n")
        
        print(f"  {label}: {len(paths)} paths saved to {output_file}")
    
    # Save summary JSON
    summary_file = os.path.join(output_dir, "summary.json")
    summary = {
        'total_labels': len(organized_paths),
        'labels': list(organized_paths.keys()),
        'counts': {label: len(paths) for label, paths in organized_paths.items()}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary saved to {summary_file}")


def main():
    """Main function to process VisDrone labels."""
    # Define paths
    csv_path = "/home/p24s09/image_angle_classifier/raw_data/VisDrone/image_labels.csv"
    images_dir = "/home/p24s09/image_angle_classifier/raw_data/VisDrone/VisDrone2019-DET-train/VisDrone2019-DET-train/images"
    
    # Check if paths exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    try:
        # Process the labels
        organized_paths = process_visdrone_labels(csv_path, images_dir)
        
        # Save organized paths to files
        save_organized_paths(organized_paths)
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE!")
        print("="*50)
        
        # Return the organized paths for potential further use
        return organized_paths
        
    except Exception as e:
        print(f"Error processing labels: {str(e)}")
        return None


if __name__ == "__main__":
    organized_data = main()
