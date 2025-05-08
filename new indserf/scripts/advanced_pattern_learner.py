import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import logging

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE specific layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class PatternLearner:
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VariationalAutoencoder(input_dim, latent_dim, hidden_dims).to(self.device)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        
    def compute_loss(self, x, recon_x, mu, log_var):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss with beta-VAE weighting
        beta = 0.1  # Adjust this to control disentanglement
        return recon_loss + beta * kl_loss
        
    def train(self, data, epochs=50, batch_size=256, learning_rate=1e-3):
        logging.info(f"Training on device: {self.device}")
        
        # Preprocess data
        data_scaled = self.scaler.fit_transform(data)
        data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Shuffle data
            perm = torch.randperm(len(data_tensor))
            
            for i in range(0, len(data_tensor), batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = data_tensor[batch_idx]
                
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(batch)
                loss = self.compute_loss(batch, recon_batch, mu, log_var)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logging.info("Early stopping triggered!")
                    break
                    
    def detect_patterns(self, data, threshold=3.0):
        """Detect patterns and anomalies in the data"""
        self.model.eval()
        with torch.no_grad():
            # Scale data
            data_scaled = self.scaler.transform(data)
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
            
            # Get reconstructions and latent representations
            recon, mu, _ = self.model(data_tensor)
            
            # Compute reconstruction errors
            errors = torch.mean((data_tensor - recon) ** 2, dim=1).cpu().numpy()
            
            # Get latent space representations
            latent_repr = mu.cpu().numpy()
            
            # PCA on latent space
            latent_pca = self.pca.fit_transform(latent_repr)
            
            # Cluster patterns
            clusters = self.clusterer.fit_predict(latent_pca)
            
            # Detect anomalies (points with high reconstruction error)
            anomalies = errors > (np.mean(errors) + threshold * np.std(errors))
            
            return {
                'reconstruction_errors': errors,
                'latent_representations': latent_repr,
                'latent_pca': latent_pca,
                'clusters': clusters,
                'anomalies': anomalies
            }
            
    def save(self, path):
        """Save the model and preprocessing components"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'pca': self.pca,
        }, path)
        
    def load(self, path):
        """Load the model and preprocessing components"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.pca = checkpoint['pca']

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Advanced Pattern Learner")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to historical data directory")
    parser.add_argument('--model_path', type=str, default="models/advanced_pattern_autoencoder.pth", help="Path to save/load the model")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Training batch size")
    parser.add_argument('--latent_dim', type=int, default=8, help="Dimension of latent space")
    args = parser.parse_args()
    
    # Load and prepare your data here
    # data = ...
    
    input_dim = data.shape[1]  # Update with your actual input dimension
    learner = PatternLearner(input_dim, latent_dim=args.latent_dim)
    
    if os.path.exists(args.model_path):
        logging.info(f"Loading existing model from {args.model_path}")
        learner.load(args.model_path)
    
    learner.train(data, epochs=args.epochs, batch_size=args.batch_size)
    learner.save(args.model_path)
    
    # Analyze patterns
    patterns = learner.detect_patterns(data)
    logging.info(f"Found {len(np.unique(patterns['clusters']))} distinct patterns")
    logging.info(f"Detected {np.sum(patterns['anomalies'])} anomalies")
