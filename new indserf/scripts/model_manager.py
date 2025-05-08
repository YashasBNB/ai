import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ModelManager:
    def __init__(self, 
                 model_dir: str = "models",
                 max_checkpoints: int = 5,
                 min_improvement: float = 0.01):
        """
        Initialize ModelManager
        
        Args:
            model_dir: Directory for model storage
            max_checkpoints: Maximum number of checkpoints to keep
            min_improvement: Minimum improvement required to save new checkpoint
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.min_improvement = min_improvement
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       metrics: Dict,
                       epoch: int,
                       model_name: str):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp
        }
        
        # Generate checkpoint path
        checkpoint_path = self.model_dir / f"{model_name}_checkpoint_{timestamp}.pth"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metrics separately for easy access
        metrics_path = self.model_dir / f"{model_name}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Manage checkpoints
        self._manage_checkpoints(model_name)
        
        return checkpoint_path
        
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[optim.Optimizer] = None,
                       model_name: str = None,
                       checkpoint_path: str = None) -> Tuple[int, Dict]:
        """
        Load model checkpoint
        
        Returns:
            Tuple of (epoch, metrics)
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint(model_name)
            
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.logger.warning("No checkpoint found")
            return 0, {}
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint['epoch'], checkpoint['metrics']
        
    def _manage_checkpoints(self, model_name: str):
        """Manage number of checkpoints"""
        checkpoints = self._get_all_checkpoints(model_name)
        
        if len(checkpoints) > self.max_checkpoints:
            # Sort by metrics (assuming lower is better)
            checkpoints_with_metrics = []
            for checkpoint in checkpoints:
                metrics_path = str(checkpoint).replace('.pth', '_metrics.json')
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    checkpoints_with_metrics.append((checkpoint, metrics))
                except Exception as e:
                    self.logger.warning(f"Could not load metrics for {checkpoint}: {e}")
                    
            # Sort by primary metric (e.g., reconstruction error)
            sorted_checkpoints = sorted(
                checkpoints_with_metrics,
                key=lambda x: x[1].get('reconstruction_error', float('inf'))
            )
            
            # Remove worst checkpoints
            for checkpoint, _ in sorted_checkpoints[self.max_checkpoints:]:
                metrics_path = str(checkpoint).replace('.pth', '_metrics.json')
                try:
                    os.remove(checkpoint)
                    os.remove(metrics_path)
                    self.logger.info(f"Removed checkpoint: {checkpoint}")
                except Exception as e:
                    self.logger.error(f"Error removing checkpoint: {e}")
                    
    def _get_latest_checkpoint(self, model_name: str) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoints = self._get_all_checkpoints(model_name)
        if not checkpoints:
            return None
        return str(sorted(checkpoints, key=os.path.getctime)[-1])
        
    def _get_all_checkpoints(self, model_name: str) -> List[Path]:
        """Get all checkpoints for given model"""
        return list(self.model_dir.glob(f"{model_name}_checkpoint_*.pth"))
        
    def should_save_checkpoint(self,
                             current_metrics: Dict,
                             model_name: str) -> bool:
        """Determine if new checkpoint should be saved"""
        latest_checkpoint = self._get_latest_checkpoint(model_name)
        if not latest_checkpoint:
            return True
            
        # Load previous metrics
        metrics_path = str(latest_checkpoint).replace('.pth', '_metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                previous_metrics = json.load(f)
        except Exception:
            return True
            
        # Compare primary metric (e.g., reconstruction error)
        current_error = current_metrics.get('reconstruction_error', float('inf'))
        previous_error = previous_metrics.get('reconstruction_error', float('inf'))
        
        improvement = (previous_error - current_error) / previous_error
        return improvement > self.min_improvement
        
    def export_model(self,
                    model: nn.Module,
                    model_name: str,
                    format: str = 'torchscript'):
        """Export model for production"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'torchscript':
            # Export to TorchScript
            model.eval()
            example_input = torch.randn(1, model.input_dim)
            traced_model = torch.jit.trace(model, example_input)
            
            export_path = self.model_dir / f"{model_name}_production_{timestamp}.pt"
            torch.jit.save(traced_model, export_path)
            
        elif format == 'onnx':
            # Export to ONNX
            model.eval()
            example_input = torch.randn(1, model.input_dim)
            export_path = self.model_dir / f"{model_name}_production_{timestamp}.onnx"
            
            torch.onnx.export(
                model,
                example_input,
                export_path,
                export_params=True,
                opset_version=11,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        self.logger.info(f"Exported model to {export_path}")
        return export_path
        
    def cleanup_old_models(self, days_threshold: int = 30):
        """Clean up old model files"""
        current_time = datetime.now().timestamp()
        
        with ThreadPoolExecutor() as executor:
            for file in self.model_dir.glob("*.*"):
                if file.is_file():
                    file_time = os.path.getctime(file)
                    if (current_time - file_time) > (days_threshold * 24 * 3600):
                        try:
                            os.remove(file)
                            self.logger.info(f"Removed old file: {file}")
                        except Exception as e:
                            self.logger.error(f"Error removing {file}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Management")
    parser.add_argument('--model_dir', type=str, required=True,
                       help="Directory for model storage")
    parser.add_argument('--cleanup', action='store_true',
                       help="Clean up old model files")
    parser.add_argument('--days', type=int, default=30,
                       help="Days threshold for cleanup")
    
    args = parser.parse_args()
    
    manager = ModelManager(args.model_dir)
    
    if args.cleanup:
        manager.cleanup_old_models(args.days)
