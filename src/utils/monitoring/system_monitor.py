from .base_monitor import BaseMonitor
import torch
import psutil
import GPUtil

class SystemMonitor(BaseMonitor):
    """Monitor system resources (CPU, Memory, GPU)"""
    
    def __init__(self, enabled=True):
        super().__init__(enabled)
    
    def log(self, step):
        """Log current system resource usage"""
        if not self.enabled:
            return
            
        # Log CPU and memory usage
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.writer.add_scalar('System/CPU_Usage', cpu_percent, step)
        self.writer.add_scalar('System/Memory_Usage', memory_percent, step)
        
        # Log GPU info if CUDA is available
        if torch.cuda.is_available():
            # Detailed GPU monitoring
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                self.writer.add_scalar(f'System/GPU_{i}_Memory', gpu.memoryUsed, step)
                self.writer.add_scalar(f'System/GPU_{i}_Utilization', gpu.load * 100, step)
                
            # Also log CUDA memory for completeness
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                self.writer.add_scalar(f'System/GPU_{i}_Memory_Allocated', memory_allocated, step)