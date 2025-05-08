import torch
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path

from scripts.train_model import UnsupervisedModel
from scripts.model_manager import ModelManager
from logging_config import setup_logging

class ContinuousLearner:
    def __init__(self,
                 model_dir: str,
                 timeframe: str,
                 buffer_size: int = 1000,
                 update_interval: int = 60,  # minutes
                 min_samples: int = 100,
                 learning_rate: float = 0.0001):
        """
        Initialize continuous learner
        
        Args:
            model_dir: Directory containing trained models
            timeframe: Trading timeframe
            buffer_size: Size of experience buffer
            update_interval: Minutes between model updates
            min_samples: Minimum samples before update
            learning_rate: Fine-tuning learning rate
        """
        self.model_dir = Path(model_dir)
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.min_samples = min_samples
        self.learning_rate = learning_rate
        
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize components
        self.model_manager = ModelManager(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Experience buffer
        self.buffer = deque(maxlen=buffer_size)
        
        # Performance tracking
        self.performance_metrics = {
            'updates': 0,
            'avg_loss': 0.0,
            'last_update': None
        }
        
        # Load model
        self.model = None
        self.scaler = None
        self._load_model()
        
    def _load_model(self):
        """Load trained model"""
        try:
            # Load model
            self.model = UnsupervisedModel(input_dim=None)  # Set appropriate input_dim
            epoch, metrics = self.model_manager.load_checkpoint(
                self.model,
                f"pattern_learner_{self.timeframe}"
            )
            self.model.to(self.device)
            self.model.train()  # Set to training mode
            
            # Load scaler
            scaler_path = self.model_dir / f"scaler_{self.timeframe}.pkl"
            self.scaler = torch.load(scaler_path)
            
            self.logger.info(f"Loaded model from epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    async def update_model(self, force: bool = False) -> Optional[Dict]:
        """Update model with new data"""
        try:
            # Check if update is needed
            if not force and not self._should_update():
                return None
                
            if len(self.buffer) < self.min_samples:
                return None
                
            # Prepare batch
            batch = self._prepare_batch()
            if batch is None:
                return None
                
            # Fine-tune model
            loss = self._fine_tune(batch)
            
            # Update metrics
            self.performance_metrics['updates'] += 1
            self.performance_metrics['avg_loss'] = (
                0.95 * self.performance_metrics['avg_loss'] +
                0.05 * loss
            )
            self.performance_metrics['last_update'] = datetime.now()
            
            # Save updated model
            self.model_manager.save_checkpoint(
                self.model,
                self.performance_metrics['updates'],
                {
                    "loss": loss,
                    "avg_loss": self.performance_metrics['avg_loss']
                },
                f"pattern_learner_{self.timeframe}"
            )
            
            update_info = {
                "timestamp": datetime.now().isoformat(),
                "samples": len(self.buffer),
                "loss": loss,
                "avg_loss": self.performance_metrics['avg_loss']
            }
            
            self.logger.info(f"Model updated: {update_info}")
            return update_info
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return None
            
    def add_experience(self, features: np.ndarray):
        """Add new experience to buffer"""
        try:
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))
            self.buffer.append(scaled_features)
            
        except Exception as e:
            self.logger.error(f"Error adding experience: {str(e)}")
            
    def _should_update(self) -> bool:
        """Check if model should be updated"""
        if self.performance_metrics['last_update'] is None:
            return True
            
        time_since_update = datetime.now() - self.performance_metrics['last_update']
        return time_since_update.total_seconds() / 60 >= self.update_interval
        
    def _prepare_batch(self) -> Optional[torch.Tensor]:
        """Prepare batch for training"""
        try:
            # Convert buffer to tensor
            batch = np.vstack(self.buffer)
            return torch.FloatTensor(batch).to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error preparing batch: {str(e)}")
            return None
            
    def _fine_tune(self, batch: torch.Tensor) -> float:
        """Fine-tune model on batch"""
        try:
            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=0.01
            )
            
            # Forward pass
            batch_flat = batch.view(batch.size(0), -1)
            recon_batch, mu, log_var = self.model(batch_flat)
            
            # Calculate loss
            recon_loss = torch.nn.MSELoss()(recon_batch, batch_flat)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.1 * kl_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Error fine-tuning model: {str(e)}")
            return float('inf')
            
    def get_reconstruction_error(self, features: np.ndarray) -> float:
        """Get reconstruction error for features"""
        try:
            with torch.no_grad():
                # Scale features
                scaled_features = self.scaler.transform(features.reshape(1, -1))
                features_tensor = torch.FloatTensor(scaled_features).to(self.device)
                
                # Get reconstruction
                features_flat = features_tensor.view(features_tensor.size(0), -1)
                recon, _, _ = self.model(features_flat)
                
                # Calculate error
                error = torch.mean((features_flat - recon) ** 2).item()
                return error
                
        except Exception as e:
            self.logger.error(f"Error calculating reconstruction error: {str(e)}")
            return float('inf')
            
    async def start_continuous_learning(self):
        """Start continuous learning loop"""
        self.logger.info("Starting continuous learning")
        
        try:
            while True:
                await self.update_model()
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Continuous learning error: {str(e)}")
            
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            "updates": self.performance_metrics['updates'],
            "avg_loss": self.performance_metrics['avg_loss'],
            "last_update": (self.performance_metrics['last_update'].isoformat()
                          if self.performance_metrics['last_update']
                          else None),
            "buffer_size": len(self.buffer),
            "device": str(self.device)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Learner")
    parser.add_argument('--model-dir', required=True,
                       help="Model directory")
    parser.add_argument('--timeframe', required=True,
                       help="Trading timeframe")
    parser.add_argument('--update-interval', type=int, default=60,
                       help="Update interval in minutes")
    
    args = parser.parse_args()
    
    # Initialize learner
    learner = ContinuousLearner(
        model_dir=args.model_dir,
        timeframe=args.timeframe,
        update_interval=args.update_interval
    )
    
    # Example usage
    features = np.random.randn(10)  # Example features
    learner.add_experience(features)
    
    # Run continuous learning
    asyncio.run(learner.start_continuous_learning())
