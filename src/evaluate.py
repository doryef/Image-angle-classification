
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.dataset import ImageDataset
from src.models.angle_classifier import AngleClassifier
from src.inference import AnglePredictor
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_transform(config):
    return transforms.Compose([
        transforms.CenterCrop(config['augmentation']['crop_size']),
        transforms.Resize(config['augmentation']['resize']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['val_acc']

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/confusion_matrix_{timestamp}.png')
    plt.close()
    print(f"\nConfusion matrix saved to 'plots/confusion_matrix_{timestamp}.png'")

def plot_misclassified_samples(test_dataset, all_preds, all_labels, classes, checkpoint_path, num_samples=10):
    """Plot misclassified samples with their predictions and probability distributions"""
    # Find misclassified indices
    misclassified_indices = [i for i in range(len(all_preds)) 
                            if all_preds[i] != all_labels[i]]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    # Initialize predictor to get probability distributions
    predictor = AnglePredictor(checkpoint_path)
    
    # Limit to requested number
    num_to_plot = min(num_samples, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, num_to_plot, replace=False)
    
    # Create figure with subplots for images and probability bars
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 5, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Misclassified Samples with Probability Distributions', fontsize=16, fontweight='bold')
    
    for idx, sample_idx in enumerate(sample_indices):
        # Get the image path from dataset
        img_path, _ = test_dataset.samples[sample_idx]
        
        # Load and display the original image (without transforms)
        img = Image.open(img_path).convert('RGB')
        
        # Get predictions and labels
        true_label = classes[all_labels[sample_idx]]
        pred_label = classes[all_preds[sample_idx]]
        
        # Get probability distribution using inference
        result = predictor.predict_image(img_path)
        probs = result['probabilities']
        
        # Create subplot for image (top half)
        row = idx // 5
        col = idx % 5
        ax_img = fig.add_subplot(gs[row, col])
        
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(
            f'True: {true_label} | Pred: {pred_label}',
            fontsize=9,
            color='red',
            fontweight='bold',
            pad=2
        )
        
        # Add small bar chart below image showing probabilities
        ax_bar = ax_img.inset_axes([0.1, -0.35, 0.8, 0.25])
        
        class_labels = list(probs.keys())
        prob_values = list(probs.values())
        colors = ['green' if cls == true_label else 'red' if cls == pred_label else 'gray' 
                 for cls in class_labels]
        
        # Use numeric positions for y-axis
        y_pos = np.arange(len(class_labels))
        bars = ax_bar.barh(y_pos, prob_values, color=colors, alpha=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(class_labels)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_xlabel('Probability (%)', fontsize=7)
        ax_bar.tick_params(labelsize=6)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, prob_values):
            ax_bar.text(prob + 2, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.1f}%', va='center', fontsize=6)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/misclassified_samples_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nMisclassified samples visualization saved to 'plots/misclassified_samples_{timestamp}.png'")
    print(f"Total misclassified: {len(misclassified_indices)} out of {len(all_preds)} samples")
    
    # Print detailed probability info for misclassified samples
    print("\nDetailed Probability Distributions for Misclassified Samples:")
    print("=" * 80)
    for idx, sample_idx in enumerate(sample_indices[:5]):  # Show first 5 in detail
        img_path, _ = test_dataset.samples[sample_idx]
        result = predictor.predict_image(img_path)
        true_label = classes[all_labels[sample_idx]]
        pred_label = classes[all_preds[sample_idx]]
        
        print(f"\nSample {idx+1}: {os.path.basename(img_path)}")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {pred_label} (confidence: {result['confidence']:.2f}%)")
        print(f"  Probabilities:")
        for cls, prob in result['probabilities'].items():
            marker = "✓" if cls == true_label else "✗" if cls == pred_label else " "
            print(f"    {marker} {cls}: {prob:.2f}%")

def plot_training_history(test_accuracy=None):
    """Plot training and validation loss/accuracy over epochs"""
    # Find the most recent training history file
    history_files = glob.glob('logs/training_history_*.csv')
    
    if not history_files:
        print("\nNo training history found. Train the model first to generate history.")
        return
    
    # Use the most recent history file
    latest_history = max(history_files, key=os.path.getmtime)
    print(f"\nLoading training history from: {latest_history}")
    
    # Load the data
    df = pd.read_csv(latest_history)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(df['epoch'], df['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(df['epoch'], df['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(df['epoch'], df['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    
    # Add test accuracy as a horizontal line if provided
    if test_accuracy is not None:
        ax2.axhline(y=test_accuracy, color='g', linestyle='--', linewidth=2, label=f'Test Accuracy ({test_accuracy:.1f}%)')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'plots/training_history_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to 'plots/training_history_{timestamp}.png'")
    print(f"\nTraining Summary:")
    print(f"  Final Training Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Final Training Accuracy: {df['train_acc'].iloc[-1]:.2f}%")
    print(f"  Best Validation Loss: {df['val_loss'].min():.4f}")
    print(f"  Best Validation Accuracy: {df['val_acc'].max():.2f}%")
    if test_accuracy is not None:
        print(f"  Test Accuracy: {test_accuracy:.2f}%")

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset and loader
    test_transform = create_test_transform(config)
    test_dataset = ImageDataset(config['data']['test_dir'], transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model
    model = AngleClassifier(
        pretrained=False,  # We'll load weights from checkpoint
        freeze_backbone=True
    ).to(device)
    
    # Get the latest checkpoint
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoints found! Please train the model first.")
        return
    
    latest_checkpoint = max([os.path.join('checkpoints', cp) for cp in checkpoints], 
                          key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    
    # Load the model weights
    model, val_acc = load_checkpoint(latest_checkpoint, model)
    print(f"Model validation accuracy from checkpoint: {val_acc:.2f}%")
    
    # Evaluate on test set
    test_accuracy, all_preds, all_labels = evaluate(model, test_loader, device)
    print(f"\nTest Set Performance:")
    print(f"Overall Accuracy: {test_accuracy:.2f}%")
    
    # Generate detailed classification report
    class_names = ['0-30°', '30-60°', '60-90°']
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    # Plot misclassified samples with probability distributions
    plot_misclassified_samples(test_dataset, all_preds, all_labels, class_names, 
                               latest_checkpoint, num_samples=10)
    
    # Plot training history with test accuracy
    plot_training_history(test_accuracy=test_accuracy)

if __name__ == "__main__":
    main()