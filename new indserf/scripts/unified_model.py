import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class UnifiedModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], latent_dim: int = 16):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent projection
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Experience replay
        self.replay_buffer = []
        self.max_buffer_size = 50000
        self.min_buffer_size = 1000
        
        # Training history
        self.performance_history = []
        self.model_version = 0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        encoded = self.encoder(x)
        latent = self.fc_latent(encoded)
        decoded = self.decoder(latent)
        return decoded, latent
        
    def update_buffer(self, new_data: torch.Tensor):
        """Update experience replay buffer"""
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.detach().cpu().numpy()
        self.replay_buffer.extend(new_data.tolist())
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]
            
    def get_training_batch(self, batch_size: int) -> Optional[torch.Tensor]:
        """Get batch from replay buffer"""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
            
        indices = np.random.choice(len(self.replay_buffer), batch_size)
        batch = torch.tensor([self.replay_buffer[i] for i in indices], dtype=torch.float32)
        return batch

class UnifiedTrainer:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.load_or_initialize_model()
        
    def load_or_initialize_model(self, input_dim: int = 60):
        """Load existing model or create new one"""
        model_files = list(self.model_dir.glob("unified_model_v*.pt"))
        
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            state_dict = torch.load(latest_model, map_location=self.device)
            self.model = UnifiedModel(input_dim=state_dict['input_dim'])
            self.model.load_state_dict(state_dict['model_state'])
            self.model.replay_buffer = state_dict['replay_buffer']
            self.model.performance_history = state_dict['performance_history']
            self.model.model_version = state_dict['model_version']
            self.scaler = state_dict['scaler']
            logging.info(f"Loaded model v{self.model.model_version} from {latest_model}")
        else:
            self.model = UnifiedModel(input_dim=input_dim)
            logging.info("Initialized new model")
            
        self.model = self.model.to(self.device)
        
    def save_checkpoint(self):
        """Save model state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"unified_model_v{self.model.model_version}_{timestamp}.pt"
        
        state_dict = {
            'model_state': self.model.state_dict(),
            'input_dim': self.model.encoder[0].in_features,
            'replay_buffer': self.model.replay_buffer,
            'performance_history': self.model.performance_history,
            'model_version': self.model.model_version,
            'scaler': self.scaler,
            'timestamp': timestamp
        }
        
        torch.save(state_dict, model_path)
        logging.info(f"Saved model v{self.model.model_version} to {model_path}")
        
    def train_step(self, data: torch.Tensor, batch_size: int = 64) -> Dict:
        """Perform training step with new data"""
        # Update replay buffer
        self.model.update_buffer(data)
        
        # Get training batch
        batch = self.model.get_training_batch(batch_size)
        if batch is None:
            return {'status': 'buffer_filling', 'buffer_size': len(self.model.replay_buffer)}
            
        # Train
        batch = batch.to(self.device)
        self.model.train()
        
        reconstructed, _ = self.model(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        loss_value = loss.item()
        self.model.performance_history.append(loss_value)
        self.model.model_version += 1
        
        # Save periodically
        if self.model.model_version % 100 == 0:
            self.save_checkpoint()
            
        return {
            'status': 'training',
            'loss': loss_value,
            'version': self.model.model_version,
            'buffer_size': len(self.model.replay_buffer)
        }
        
    def predict(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Generate predictions"""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            reconstructed, latent = self.model(data)
            error = nn.MSELoss()(reconstructed, data).item()
        return reconstructed.cpu(), latent.cpu(), error

def create_trainer(model_dir: str) -> UnifiedTrainer:
    """Factory function to create trainer"""
    return UnifiedTrainer(model_dir)
