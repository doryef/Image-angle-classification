import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import json
from pathlib import Path

class DatasetDownloader:
    def __init__(self, save_dir="raw_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def download_file(self, url, filename, headers=None):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            return False

    def download_visdrone(self, year="2019"):
        """
        Downloads VisDrone dataset.
        Args:
            year (str): Dataset year version (default: "2019")
        """
        print(f"\nDownloading VisDrone-{year} dataset...")
        
        # VisDrone dataset links
        visdrone_urls = {
            "2019": {
                "DET": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.6.2/VisDrone2019-DET-val.zip",
                "VID": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v0.6.2/VisDrone2019-VID-val.zip"
            }
        }
        
        if year not in visdrone_urls:
            print(f"Error: VisDrone-{year} is not available. Available years: {list(visdrone_urls.keys())}")
            return False
        
        # Create VisDrone directory
        visdrone_dir = os.path.join(self.save_dir, f"visdrone{year}")
        os.makedirs(visdrone_dir, exist_ok=True)
        
        # Download and extract each subset
        for subset, url in visdrone_urls[year].items():
            print(f"\nDownloading VisDrone-{year} {subset} subset...")
            zip_path = os.path.join(visdrone_dir, f"visdrone{year}_{subset.lower()}.zip")
            
            if self.download_file(url, zip_path):
                try:
                    print(f"Extracting VisDrone-{year} {subset} subset...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(visdrone_dir, subset.lower()))
                    os.remove(zip_path)
                    print(f"VisDrone-{year} {subset} subset downloaded and extracted successfully!")
                except zipfile.BadZipFile:
                    print(f"Error: Downloaded file for {subset} is not a valid zip file")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    return False
            else:
                print(f"Error downloading {subset} subset")
                return False
        
        return True

def print_dataset_instructions():
    """Print instructions for dataset access"""
    print("\nDataset Access Instructions:")
    
    print("\n1. VisDrone Dataset:")
    print("   - Automatic download available through this script")
    print("   - Full dataset available at: http://VisDrone.ia.ac.cn/")
    print("   - Contains aerial imagery from various angles")
    print("   - Perfect for camera angle classification tasks")
    
    print("\nOnce you have the images:")
    print("1. Place them in a directory")
    print("2. Use our labeling tool to categorize them:")
    print("   python src/data/label_tool.py")

def main():
    print("Dataset Downloader")
    print("-----------------")
    
    # Print instructions
    print_dataset_instructions()
    
    # Download VisDrone dataset
    print("\nDownloading VisDrone dataset...")
    downloader = DatasetDownloader()
    if downloader.download_visdrone():
        print("\nVisDrone dataset downloaded successfully!")
        print("You can now use the labeling tool to categorize the images by camera angle.")
    else:
        print("\nNote: Could not download VisDrone dataset.")
        print("Please check your internet connection or visit http://VisDrone.ia.ac.cn/ for manual download.")

if __name__ == "__main__":
    main()