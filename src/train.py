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
    train_transform_list = [
        transforms.Resize(config['augmentation']['resize']),
        transforms.RandomCrop(config['augmentation']['crop_size']),
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
        config=config['training'],
        device=device
    )
    
    # Start training
    trainer.train(config['training']['num_epochs'])

if __name__ == "__main__":
    main()