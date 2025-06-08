from .base_monitor import BaseMonitor
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision

class MetricsMonitor(BaseMonitor):
    """Monitor training metrics, confusion matrix, and predictions"""
    
    def __init__(self, classes, enabled=True):
        super().__init__(enabled)
        self.classes = classes
    
    def log(self, *args, **kwargs):
        """Implementation of abstract log method from BaseMonitor"""
        # This is a wrapper around our specialized logging methods
        # Each specific logging task has its own method for better organization
        if not self.enabled:
            return
            
        if 'metrics' in kwargs and 'step' in kwargs:
            self.log_metrics(kwargs['metrics'], kwargs['step'], 
                           kwargs.get('phase', 'train'))
        elif 'optimizer' in kwargs and 'step' in kwargs:
            self.log_learning_rates(kwargs['optimizer'], kwargs['step'])
        elif all(k in kwargs for k in ['images', 'labels', 'outputs', 'step']):
            self.log_sample_predictions(kwargs['images'], kwargs['labels'], 
                                     kwargs['outputs'], kwargs['step'])
        elif all(k in kwargs for k in ['labels', 'predicted', 'step']):
            self.log_confusion_matrix(kwargs['labels'], kwargs['predicted'], 
                                   kwargs['step'])
    
    def log_metrics(self, metrics, step, phase='train'):
        """Log basic metrics like loss and accuracy"""
        if not self.enabled:
            return
            
        for name, value in metrics.items():
            self.writer.add_scalar(f'{name}/{phase}', value, step)
    
    def log_learning_rates(self, optimizer, step):
        """Log learning rates for different parameter groups"""
        if not self.enabled:
            return
            
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = "backbone" if i == 0 else "head"
            self.writer.add_scalar(f'LearningRate/{group_name}', 
                                 param_group['lr'], step)
    
    def log_sample_predictions(self, images, labels, outputs, step):
        """Log sample predictions with their images"""
        if not self.enabled:
            return
            
        _, predicted = torch.max(outputs, 1)
        
        # Select up to 8 random samples
        num_samples = min(8, images.size(0))
        indices = np.random.choice(images.size(0), num_samples, replace=False)
        
        # Create image grid
        img_grid = torchvision.utils.make_grid(images[indices])
        
        # Add predictions as text
        text = ""
        for idx in indices:
            pred_class = self.classes[predicted[idx]]
            true_class = self.classes[labels[idx]]
            text += f"True: {true_class}, Pred: {pred_class}\n"
        
        self.writer.add_image('Predictions/samples', img_grid, step)
        self.writer.add_text('Predictions/labels', text, step)
    
    def log_confusion_matrix(self, labels, predicted, step):
        """Log confusion matrix and per-class metrics"""
        if not self.enabled:
            return
            
        # Create confusion matrix
        matrix = torch.zeros(len(self.classes), len(self.classes))
        for t, p in zip(labels, predicted):
            matrix[t, p] += 1
            
        # Log per-class accuracy
        for i, class_name in enumerate(self.classes):
            class_correct = matrix[i][i]
            class_total = matrix[i].sum()
            if class_total > 0:
                class_acc = class_correct / class_total
                self.writer.add_scalar(f'Accuracy/class_{class_name}', 
                                     class_acc, step)
        
        # Convert to probabilities for visualization
        for i in range(len(self.classes)):
            if matrix[i].sum() > 0:
                matrix[i] = matrix[i] / matrix[i].sum()
        
        # Create confusion matrix plot
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(matrix.numpy(), annot=True, fmt='.2f', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        self.writer.add_figure('Metrics/confusion_matrix', fig, step)
        plt.close()