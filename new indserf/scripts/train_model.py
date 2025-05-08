import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from logging_config import setup_logging
from scripts.model_manager import ModelManager

class CandleDataset(Dataset):
    def __init__(self, features: np.ndarray, window_size: int = 30):
        """
        Initialize dataset
        
        Args:
            features: Feature array [n_samples, n_features]
            window_size: Number of candles per window
        """
        self.features = torch.FloatTensor(features)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.features) - self.window_size + 1
        
    def __getitem__(self, idx):
        window = self.features[idx:idx + self.window_size]
        return window

class UnsupervisedModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 latent_dim: int = 16,
                 dropout: float = 0.1):
        """
        Initialize model
        
        Args:
            input_dim: Input dimension (features per candle * window size)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space"""
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_var(hidden)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class ModelTrainer:
    def __init__(self,
                 data_dir: str,
                 model_dir: str = "models",
                 window_size: int = 30,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 n_epochs: int = 100):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing historical data
            model_dir: Directory to save models
            window_size: Number of candles per window
            batch_size: Training batch size
            learning_rate: Learning rate
            n_epochs: Number of training epochs
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.window_size = window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize model manager
        self.model_manager = ModelManager(model_dir)
        
    def prepare_data(self) -> Tuple[np.ndarray, StandardScaler]:
        """Prepare data for training"""
        all_data = []
        
        # Load all CSV files
        for file in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(file)
                
                # Extract features
                features = self._extract_features(df)
                all_data.append(features)
                
            except Exception as e:
                self.logger.error(f"Error loading {file}: {str(e)}")
                continue
                
        if not all_data:
            raise ValueError("No data loaded")
            
        # Combine all features
        features = np.vstack(all_data)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        return scaled_features, scaler
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from candle data"""
        # Basic candle features
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        
        # Technical indicators
        df['ma_10'] = df['close'].rolling(10).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Volatility
        df['volatility'] = df['std_20'] / df['ma_20']
        
        # Select features
        feature_columns = [
            'body', 'upper_shadow', 'lower_shadow', 'range',
            'ma_10', 'ma_20', 'std_20', 'rsi',
            'momentum', 'volatility'
        ]
        
        # Drop rows with NaN
        features = df[feature_columns].dropna().values
        
        return features
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def train_model(self,
                   features: np.ndarray,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Train model"""
        self.logger.info(f"Training on device: {device}")
        
        # Create dataset and loader
        dataset = CandleDataset(features, self.window_size)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Initialize model
        input_dim = self.window_size * features.shape[1]
        model = UnsupervisedModel(input_dim=input_dim).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(self.n_epochs):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                batch_flat = batch.view(batch.size(0), -1)
                
                # Forward pass
                recon_batch, mu, log_var = model(batch_flat)
                
                # Calculate loss
                recon_loss = nn.MSELoss()(recon_batch, batch_flat)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + 0.1 * kl_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.model_manager.save_checkpoint(
                    model,
                    epoch + 1,
                    {"loss": avg_loss},
                    "pattern_learner"
                )
                
        self.logger.info("Training complete")
        return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Pattern Model")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="Directory containing historical data")
    parser.add_argument('--model_dir', type=str, default="models",
                       help="Directory to save models")
    parser.add_argument('--window_size', type=int, default=30,
                       help="Window size for patterns")
    parser.add_argument('--batch_size', type=int, default=64,
                       help="Training batch size")
    parser.add_argument('--epochs', type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = ModelTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_epochs=args.epochs
    )
    
    # Train model
    features, scaler = trainer.prepare_data()
    model = trainer.train_model(features)
