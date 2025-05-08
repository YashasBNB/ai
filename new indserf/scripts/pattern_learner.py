import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_all_assets, prepare_data

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_autoencoder(data, input_dim, epochs=10, batch_size=256, model_path=None):
    """
    Trains the autoencoder. If model_path is given and exists, loads model weights and continues training (continual learning).
    """
    model = Autoencoder(input_dim)
    if model_path and os.path.exists(model_path):
        print(f"Loading existing model weights from {model_path} for continual learning...")
        model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    data_tensor = torch.tensor(data)
    for epoch in range(epochs):
        perm = torch.randperm(len(data_tensor))
        for i in range(0, len(data_tensor), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = data_tensor[batch_idx]
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    # --- Confusion Matrix for anomaly detection ---
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        model.eval()
        with torch.no_grad():
            recon = model(data_tensor)
            errors = ((data_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        # Heuristic: top 1% largest errors are anomalies
        threshold = np.percentile(errors, 99)
        y_pred = (errors > threshold).astype(int)
        # For synthetic evaluation, assume all data is 'normal' (0)
        y_true = np.zeros_like(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix (Normal=0, Anomaly=1):")
        print(cm)
        print(classification_report(y_true, y_pred, target_names=["Normal","Anomaly"]))
        print(f"Anomaly threshold (99th percentile): {threshold:.6f}")
        print(f"Anomalies detected: {y_pred.sum()} / {len(y_pred)}")
    except Exception as e:
        print(f"[WARNING] Could not compute confusion matrix: {e}")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, input_dim):
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pattern Learner Trainer")
    parser.add_argument('--data_dir', type=str, default="/Users/yashasnaidu/AI/historical_data", help="Path to historical data directory")
    parser.add_argument('--model_path', type=str, default="models/pattern_autoencoder.pth", help="Path to save/load the model")
    parser.add_argument('--file_pattern', type=str, default="*.csv", help="File pattern to select candles (e.g., *.csv for 1h, *_M15.csv for 15min)")
    args = parser.parse_args()
    data_dict = load_all_assets(args.data_dir, file_pattern=args.file_pattern)
    from data_loader import get_total_rows
    logging.info(f"Loaded {len(data_dict)} assets, {get_total_rows(data_dict)} total rows.")
    data = prepare_data(data_dict)
    input_dim = data.shape[1]
    logging.info(f"Training autoencoder on {data.shape[0]} samples, input_dim={input_dim}")
    model = train_autoencoder(data, input_dim, model_path=args.model_path)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    save_model(model, args.model_path)
    logging.info(f"Model trained and saved at {args.model_path}.")
