from .model_monitor import ModelMonitor
from .system_monitor import SystemMonitor
from .metrics_monitor import MetricsMonitor
import torch

class TrainingMonitor:
    """Unified interface for all training monitoring"""
    
    def __init__(self, model, classes, enable_monitoring=True):
        """Initialize all monitoring components
        
        Args:
            model: The model being trained
            classes: List of class names
            enable_monitoring: Whether to enable TensorBoard monitoring
        """
        self.model_monitor = ModelMonitor(enabled=enable_monitoring)
        self.system_monitor = SystemMonitor(enabled=enable_monitoring)
        self.metrics_monitor = MetricsMonitor(classes, enabled=enable_monitoring)
        self.enabled = enable_monitoring
        self.global_step = 0
        
        # Initialize model visualization if monitoring is enabled
        if enable_monitoring and hasattr(model, 'train_loader'):
            sample_batch = next(iter(model.train_loader))
            if isinstance(sample_batch, (tuple, list)):
                sample_input = sample_batch[0].to(next(model.parameters()).device)
                self.model_monitor.log(model, sample_input)
    
    def log_training_step(self, model, optimizer, images, labels, outputs, loss):
        """Log all relevant information for a training step"""
        if not self.enabled:
            return
            
        # Log system metrics periodically
        if self.global_step % 10 == 0:  # Every 10 steps
            self.system_monitor.log(self.global_step)
        
        # Log learning rates
        self.metrics_monitor.log_learning_rates(optimizer, self.global_step)
        
        # Log basic metrics
        metrics = {
            'Loss': loss.item(),
            'Accuracy': self._compute_accuracy(outputs, labels)
        }
        self.metrics_monitor.log_metrics(metrics, self.global_step, phase='train')
        
        # Log sample predictions periodically
        if self.global_step % 50 == 0:  # Every 50 steps
            self.metrics_monitor.log_sample_predictions(
                images, labels, outputs, self.global_step
            )
        
        # Log parameter distributions periodically
        if self.global_step % 100 == 0:  # Every 100 steps
            self.model_monitor.log_parameters(model, self.global_step)
        
        self.global_step += 1
    
    def log_validation_step(self, model, val_loss, val_acc, all_labels, all_predictions):
        """Log validation metrics and confusion matrix"""
        if not self.enabled:
            return
            
        # Log validation metrics
        metrics = {
            'Loss': val_loss,
            'Accuracy': val_acc
        }
        self.metrics_monitor.log_metrics(metrics, self.global_step, phase='validation')
        
        # Log confusion matrix
        self.metrics_monitor.log_confusion_matrix(
            all_labels, all_predictions, self.global_step
        )
    
    def _compute_accuracy(self, outputs, labels):
        """Helper to compute accuracy for a batch"""
        _, predicted = torch.max(outputs, 1)
        return 100. * (predicted == labels).sum().item() / labels.size(0)
    
    def close(self):
        """Clean up all monitors"""
        if self.enabled:
            self.model_monitor.close()
            self.system_monitor.close()
            self.metrics_monitor.close()