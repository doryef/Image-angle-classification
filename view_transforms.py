#!/usr/bin/env python3
"""
Script to visualize images before and after applying transforms.
This helps understand how the data preprocessing affects the input images.
"""

import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_transforms(config):
    train_transform_list = [
        transforms.RandomCrop(config['augmentation']['crop_size']),
        transforms.Resize(config['augmentation']['resize']),
    ]
    
    # Add horizontal flip if enabled
    if config['augmentation'].get('horizontal_flip', False):
        train_transform_list.append(transforms.RandomHorizontalFlip())
    
    # Add zoom variation using resize + crop
    if 'zoom_range' in config['augmentation']:
        base_size = int(config['augmentation']['crop_size'] * (1 + config['augmentation']['zoom_range']))
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
        transforms.CenterCrop(config['augmentation']['crop_size']),
        transforms.Resize(config['augmentation']['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Convert normalized tensor back to displayable image"""
    denorm = tensor.clone()
    for t, m, s in zip(denorm, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(denorm, 0, 1)

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image for display"""
    denorm_tensor = denormalize_tensor(tensor)
    return transforms.ToPILImage()(denorm_tensor)

def visualize_transforms(image_path, train_transform, val_transform, num_variations=3):
    """Visualize original image and transformed versions"""
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, axes = plt.subplots(2, num_variations + 1, figsize=(15, 8))
    fig.suptitle(f'Image Transforms Visualization\nOriginal: {os.path.basename(image_path)}', fontsize=14)
    
    # Show original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'Original\n{original_img.size[0]}x{original_img.size[1]}')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title(f'Original\n{original_img.size[0]}x{original_img.size[1]}')
    axes[1, 0].axis('off')
    
    # Show training transforms (random variations)
    for i in range(num_variations):
        train_tensor = train_transform(original_img)
        train_img = tensor_to_pil(train_tensor)
        
        axes[0, i + 1].imshow(train_img)
        axes[0, i + 1].set_title(f'Train Transform #{i+1}\n224x224')
        axes[0, i + 1].axis('off')
    
    # Show validation transform (deterministic)
    for i in range(num_variations):
        val_tensor = val_transform(original_img)
        val_img = tensor_to_pil(val_tensor)
        
        axes[1, i + 1].imshow(val_img)
        axes[1, i + 1].set_title(f'Val Transform\n224x224')
        axes[1, i + 1].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Training\n(Random)', rotation=90, 
                    verticalalignment='center', horizontalalignment='center',
                    transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
    
    axes[1, 0].text(-0.1, 0.5, 'Validation\n(Deterministic)', rotation=90, 
                    verticalalignment='center', horizontalalignment='center',
                    transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure instead of showing it (useful for remote environments)
    output_filename = f"transform_visualization_{os.path.basename(image_path).split('.')[0]}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_filename}")
    plt.close()

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    train_transform, val_transform = create_transforms(config)
    
    print("Transform Configuration:")
    print(f"- Resize to: {config['augmentation']['resize']}")
    print(f"- Crop to: {config['augmentation']['crop_size']}")
    print(f"- Horizontal flip: {config['augmentation'].get('horizontal_flip', False)}")
    print(f"- Zoom range: {config['augmentation'].get('zoom_range', 'None')}")
    print(f"- Brightness: {config['augmentation']['brightness']}")
    print(f"- Contrast: {config['augmentation']['contrast']}")
    print()
    
    # Find some sample images to visualize
    sample_images = []
    
    # Look for images in the SkyScenes dataset
    skyscenes_path = "raw_data/SkyScenes/Images"
    if os.path.exists(skyscenes_path):
        for root, dirs, files in os.walk(skyscenes_path):
            for file in files:
                if file.endswith('.png') and len(sample_images) < 3:
                    sample_images.append(os.path.join(root, file))
    
    # If no SkyScenes images found, look for any images in the project
    if not sample_images:
        for file in os.listdir('.'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and len(sample_images) < 3:
                sample_images.append(file)
    
    if not sample_images:
        print("No images found! Please ensure you have images in the SkyScenes dataset or project root.")
        return
    
    print(f"Found {len(sample_images)} sample images:")
    for img in sample_images:
        print(f"  - {img}")
    print()
    
    # Visualize transforms for each sample image
    for image_path in sample_images:
        if os.path.exists(image_path):
            print(f"Visualizing transforms for: {image_path}")
            visualize_transforms(image_path, train_transform, val_transform)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()
