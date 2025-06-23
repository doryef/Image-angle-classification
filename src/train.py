#!/home/p24s09/image_angle_classifier/venv/bin/python3.12
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

###
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import ImageDataset
from src.models.angle_classifier import AngleClassifier
from src.trainers.trainer import ModelTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_transforms(config):
    # Determine scale range based on zoom_range
    scale_min = 0.8
    scale_max = 1.0
    if 'zoom_range' in config['augmentation']:
        scale_min = max(0.6, 1.0 - config['augmentation']['zoom_range'])
        scale_max = min(1.2, 1.0 + config['augmentation']['zoom_range'])
    
    train_transform_list = [
        # Use RandomResizedCrop to handle varying image sizes and zoom
        transforms.RandomResizedCrop(
            config['augmentation']['crop_size'],
            scale=(scale_min, scale_max),
            ratio=(0.75, 1.33)  # Maintain reasonable aspect ratios
        ),
        transforms.Resize(config['augmentation']['resize']),
    ]
    
    # Add horizontal flip if enabled
    if config['augmentation'].get('horizontal_flip', False):
        train_transform_list.append(transforms.RandomHorizontalFlip())
    
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
        # Use Resize first to ensure consistent size, then CenterCrop
        transforms.Resize((config['augmentation']['crop_size'], config['augmentation']['crop_size'])),
        transforms.Resize(config['augmentation']['resize']),
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
    
    # Create transforms
    train_transform, val_transform = create_transforms(config)
    
    # Create datasets
    train_dataset = ImageDataset(
        config['data']['train_dir'],
        transform=train_transform
    )
    val_dataset = ImageDataset(
        config['data']['val_dir'],
        transform=val_transform
    )
    synthetic_dataset = ImageDataset(
        config['data']['synthetic_dir'],
        transform=train_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    # Initialize model
    model = AngleClassifier(
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        synthetic_loader=synthetic_loader,
        config=config['training'],
        device=device
    )
    
    # Start training
    trainer.train(config['training']['num_epochs'])

if __name__ == "__main__":
    main()