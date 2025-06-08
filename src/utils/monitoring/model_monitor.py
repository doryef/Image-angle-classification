from .base_monitor import BaseMonitor
import os
from torchviz import make_dot

class ModelMonitor(BaseMonitor):
    """Monitor for model architecture and parameters"""
    
    def __init__(self, enabled=True):
        super().__init__(enabled)
        
    def log(self, model, sample_input):
        """Log model architecture and create visualization"""
        if not self.enabled:
            return
            
        # Add model graph (basic visualization)
        self.writer.add_graph(model, sample_input)
        
        # Advanced visualization
        output = model(sample_input)
        graph = make_dot(output, params=dict(model.named_parameters()))
        graph.render(os.path.join(self.log_dir, "model_architecture"), format="png")
        
    def log_parameters(self, model, epoch):
        """Log parameter distributions and gradients"""
        if not self.enabled:
            return
            
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"parameters/{name}", param.data, epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)