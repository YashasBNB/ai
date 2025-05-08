import logging
import time
import psutil
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import json
import os
from pathlib import Path

class PerformanceMonitor:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, list] = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_memory': [],
            'execution_time': [],
            'batch_processing_time': [],
            'learning_rate': [],
            'loss_values': [],
            'reconstruction_error': [],
            'latent_stats': []
        }
        
        self.start_time = time.time()
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def log_system_info(self):
        """Log system information"""
        cpu_info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total
        }
        
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory
                })
                
        system_info = {
            'cpu': cpu_info,
            'gpu': gpu_info,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        
        self.logger.info(f"System Information: {json.dumps(system_info, indent=2)}")
        
    def update_metrics(self, **kwargs):
        """Update monitoring metrics"""
        # System metrics
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.memory_allocated(i) / 1024**2  # MB
                gpu_memory.append(memory)
            self.metrics['gpu_memory'].append(gpu_memory)
            
        # Custom metrics
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def log_batch_metrics(self, batch_idx: int, **kwargs):
        """Log metrics for current batch"""
        metrics_str = [f"Batch {batch_idx}"]
        
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics_str.append(f"{key}: {value:.4f}")
            else:
                metrics_str.append(f"{key}: {value}")
                
        self.logger.info(" - ".join(metrics_str))
        
    def log_epoch_metrics(self, epoch: int, **kwargs):
        """Log metrics for current epoch"""
        metrics_str = [f"Epoch {epoch}"]
        
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics_str.append(f"{key}: {value:.4f}")
            else:
                metrics_str.append(f"{key}: {value}")
                
        self.logger.info(" - ".join(metrics_str))
        
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.log_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [arr.tolist() for arr in value]
            else:
                serializable_metrics[key] = value
                
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
            
    def plot_metrics(self, save_dir: Optional[str] = None):
        """Plot monitoring metrics"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        save_dir = Path(save_dir) if save_dir else self.log_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Plot system metrics
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU and Memory Usage
        axes[0].plot(self.metrics['cpu_usage'], label='CPU')
        axes[0].plot(self.metrics['memory_usage'], label='Memory')
        axes[0].set_title('System Resource Usage')
        axes[0].set_ylabel('Percentage')
        axes[0].legend()
        
        # GPU Memory if available
        if self.metrics['gpu_memory']:
            gpu_memory = np.array(self.metrics['gpu_memory'])
            for i in range(gpu_memory.shape[1]):
                axes[1].plot(gpu_memory[:, i], label=f'GPU {i}')
            axes[1].set_title('GPU Memory Usage')
            axes[1].set_ylabel('MB')
            axes[1].legend()
            
        plt.tight_layout()
        plt.savefig(save_dir / 'system_metrics.png')
        plt.close()
        
        # Plot training metrics
        if self.metrics['loss_values']:
            plt.figure(figsize=(12, 4))
            plt.plot(self.metrics['loss_values'])
            plt.title('Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.savefig(save_dir / 'training_loss.png')
            plt.close()

def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logging.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        return result
    return wrapper

class MemoryTracker:
    """Track memory usage during execution"""
    def __init__(self):
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss': memory_info.rss / 1024**2,  # MB
            'vms': memory_info.vms / 1024**2,  # MB
            'percent': self.process.memory_percent()
        }
        
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        memory = self.get_memory_usage()
        logging.info(
            f"Memory Usage {context} - "
            f"RSS: {memory['rss']:.2f}MB, "
            f"VMS: {memory['vms']:.2f}MB, "
            f"Percent: {memory['percent']:.2f}%"
        )

class ErrorHandler:
    """Handle and log errors during execution"""
    @staticmethod
    def handle_error(error: Exception, context: str = ""):
        """Handle and log error"""
        error_msg = f"Error in {context}: {str(error)}"
        logging.error(error_msg, exc_info=True)
        
        # Save error details
        error_file = Path("logs") / f"errors_{datetime.now():%Y%m%d}.log"
        with open(error_file, 'a') as f:
            f.write(f"{datetime.now()}: {error_msg}\n")
            
    @staticmethod
    def wrap_execution(func: Callable) -> Callable:
        """Decorator to handle errors in function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, func.__name__)
                raise
        return wrapper

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    monitor.log_system_info()
    
    # Example function with monitoring
    @timer
    @ErrorHandler.wrap_execution
    def example_function():
        memory_tracker = MemoryTracker()
        memory_tracker.log_memory_usage("Start")
        
        # Simulate some work
        time.sleep(1)
        
        memory_tracker.log_memory_usage("End")
        return "Success"
    
    # Run example
    result = example_function()
    print(result)
