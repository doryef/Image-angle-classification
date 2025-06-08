from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from abc import ABC, abstractmethod

class BaseMonitor(ABC):
    """Base class for all monitoring implementations"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            return
            
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join('logs', current_time)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def __call__(self, *args, **kwargs):
        """Make the monitor callable - if disabled, this is a no-op"""
        if not self.enabled:
            return
        return self.log(*args, **kwargs)
    
    @abstractmethod
    def log(self, *args, **kwargs):
        """Implementation specific logging logic"""
        pass
        
    def close(self):
        """Clean up resources"""
        if self.enabled:
            self.writer.close()