import torch
import torch.nn as nn
import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

class PersistentLearner(nn.Module):
    def __init__(self, input_size: int = 60, hidden_size: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, input_size),
        )
        
        # Experience replay buffer
        self.replay_buffer = []
        self.max_buffer_size = 10000
        self.min_buffer_size = 1000
        
        # Performance tracking
        self.performance_history = []
        self.model_version = 0
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
        
    def update_buffer(self, new_data: torch.Tensor):
        """Add new data to replay buffer"""
        self.replay_buffer.extend(new_data.tolist())
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
            
    def train_step(self, batch_size: int = 64):
        """Perform a training step using replay buffer"""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
            
        # Sample from buffer
        indices = np.random.choice(len(self.replay_buffer), batch_size)
        batch = torch.tensor([self.replay_buffer[i] for i in indices], dtype=torch.float32)
        
        # Train on batch
        self.train()
        reconstructed, _ = self(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        
        return loss.item()
        
    def save_state(self, model_dir: str):
        """Save model state and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(model_dir) / f"persistent_model_v{self.model_version}_{timestamp}.pt"
        
        # Save model state
        state_dict = {
            'model_state': self.state_dict(),
            'replay_buffer': self.replay_buffer,
            'performance_history': self.performance_history,
            'model_version': self.model_version,
            'timestamp': timestamp
        }
        torch.save(state_dict, model_path)
        
        # Save metadata
        meta_path = Path(model_dir) / "model_metadata.json"
        metadata = {
            'last_version': self.model_version,
            'last_update': timestamp,
            'performance_summary': {
                'recent_loss': np.mean(self.performance_history[-100:]) if self.performance_history else None,
                'total_updates': len(self.performance_history)
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Saved model state v{self.model_version} to {model_path}")
        
    def load_state(self, model_dir: str):
        """Load latest model state"""
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logging.warning(f"Model directory {model_dir} not found. Starting fresh.")
            return
            
        # Find latest model file
        model_files = list(model_dir.glob("persistent_model_v*.pt"))
        if not model_files:
            logging.warning("No existing model found. Starting fresh.")
            return
            
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Load state
        state_dict = torch.load(latest_model)
        self.load_state_dict(state_dict['model_state'])
        self.replay_buffer = state_dict['replay_buffer']
        self.performance_history = state_dict['performance_history']
        self.model_version = state_dict['model_version']
        
        logging.info(f"Loaded model state v{self.model_version} from {latest_model}")
        
    def predict(self, data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Make prediction and return reconstruction error"""
        self.eval()
        with torch.no_grad():
            reconstructed, encoded = self(data)
            error = nn.MSELoss()(reconstructed, data)
        return encoded, error.item()

class ModelManager:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = PersistentLearner()
        self.load_or_initialize_model()
        
    def load_or_initialize_model(self):
        """Load existing model or initialize new one"""
        self.model.load_state(self.model_dir)
        
    def update_model(self, new_data: torch.Tensor, batch_size: int = 64) -> Dict:
        """Update model with new data"""
        # Add new data to buffer
        self.model.update_buffer(new_data)
        
        # Perform training step
        loss = self.model.train_step(batch_size)
        if loss is not None:
            self.model.performance_history.append(loss)
            self.model.model_version += 1
            
            # Save periodically
            if self.model.model_version % 100 == 0:
                self.model.save_state(self.model_dir)
                
        return {
            'loss': loss,
            'version': self.model.model_version,
            'buffer_size': len(self.model.replay_buffer)
        }
        
    def get_predictions(self, data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Get model predictions"""
        return self.model.predict(data)
        
    def save_checkpoint(self):
        """Force save current model state"""
        self.model.save_state(self.model_dir)
