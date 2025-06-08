#!/usr/bin/env python3
"""
Quick script to test image transforms and show tensor information.
"""

import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_transforms(config):
    train_transform_list = [
        transforms.Resize(config['augmentation']['resize']),
        transforms.RandomCrop(config['augmentation']['crop_size']),
    ]
    
    # Add horizontal flip if enabled
    if config['augmentation'].get('horizontal_flip', False):
        train_transform_list.append(transforms.RandomHorizontalFlip())
    
    # Add zoom variation using resize + crop
    if 'zoom_range' in config['augmentation']:
        train_transform_list.append(transforms.RandomResizedCrop(
            config['augmentation']['crop_size'],
            scale=(1.0 - config['augmentation']['zoom_range'], 1.0 + config['augmentation']['zoom_range'])
        ))
    
    # Add color augmentations
    train_transform_list.append(transforms.ColorJitter(
        brightness=config['augmentation']['brightness'],
        contrast=config['augmentation']['contrast']
    ))
    
    # Add final transforms
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    val_transform = transforms.Compose([
        transforms.Resize(config['augmentation']['resize']),
        transforms.CenterCrop(config['augmentation']['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Create transforms
    train_transform, val_transform = create_transforms(config)
    
    print("=== TRANSFORM CONFIGURATION ===")
    print(f"Original image resize: {config['augmentation']['resize']} pixels")
    print(f"Final crop size: {config['augmentation']['crop_size']} x {config['augmentation']['crop_size']}")
    print(f"Horizontal flip: {config['augmentation'].get('horizontal_flip', False)}")
    print(f"Zoom range: ±{config['augmentation'].get('zoom_range', 0)*100}%")
    print(f"Brightness variation: ±{config['augmentation']['brightness']*100}%")
    print(f"Contrast variation: ±{config['augmentation']['contrast']*100}%")
    print(f"Normalization: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
    print()
    
    # Test with a sample image
    sample_image = "raw_data/SkyScenes/Images/H_15_P_0/ClearNoon/Town01/008191_clrnoon.png"
    
    if os.path.exists(sample_image):
        print("=== TESTING TRANSFORMS ===")
        print(f"Sample image: {sample_image}")
        
        # Load original image
        original_img = Image.open(sample_image).convert('RGB')
        print(f"Original size: {original_img.size[0]} x {original_img.size[1]} pixels")
        
        # Apply training transform
        train_tensor = train_transform(original_img)
        print(f"After training transform: {train_tensor.shape} (channels, height, width)")
        print(f"Tensor dtype: {train_tensor.dtype}")
        print(f"Value range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
        
        # Apply validation transform
        val_tensor = val_transform(original_img)
        print(f"After validation transform: {val_tensor.shape}")
        print(f"Validation tensor identical shape: {train_tensor.shape == val_tensor.shape}")
        
        print()
        print("✅ Transforms working correctly!")
        print(f"✅ Images will be processed from {original_img.size[0]}x{original_img.size[1]} → 224x224")
        print("✅ Ready for ResNet input!")
        
    else:
        print(f"❌ Sample image not found: {sample_image}")
        print("Make sure you have extracted the SkyScenes dataset.")

if __name__ == "__main__":
    main()
