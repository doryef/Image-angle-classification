#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def generate_path_file(source_dir, output_file):
    """Generate a text file containing paths to all images in source_dir"""
    # Get all image files recursively
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Use absolute path
                image_paths.append(os.path.abspath(os.path.join(root, file)))
    
    # Write paths to output file
    with open(output_file, 'a') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    
    print(f"Found {len(image_paths)} images")
    print(f"Paths written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a file containing image paths from a directory')
    parser.add_argument('source_dir', help='Directory containing images (will be searched recursively)')
    parser.add_argument('output_file', help='Output text file where paths will be written')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory {args.source_dir} does not exist")
        return
    
    # Create parent directories of output file if they don't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_path_file(args.source_dir, args.output_file)

if __name__ == "__main__":
    main()
