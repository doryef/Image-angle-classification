
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.dataset import ImageDataset
from src.models.angle_classifier import AngleClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_test_transform(config):
    return transforms.Compose([
        transforms.Resize(config['augmentation']['resize']),
        transforms.CenterCrop(config['augmentation']['crop_size']),
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
    plt.savefig('plots/confusion_matrix.png')
    plt.close()

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset and loader
    test_transform = create_test_transform(config)
    test_dataset = ImageDataset('data/test', transform=test_transform)
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
    print("\nConfusion matrix has been saved to 'plots/confusion_matrix.png'")

if __name__ == "__main__":
    main()