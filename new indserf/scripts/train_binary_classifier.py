import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_all_assets, prepare_data
from pattern_learner import load_model as load_autoencoder
# --- Advanced Activations ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

# --- Binary Classifier with selectable activation ---
# --- Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation, dropout=0.2):
        super().__init__()
        # Activation is passed as a nn.Module
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.activation = activation
    def forward(self, x):
        out = self.block(x)
        return self.activation(out + x)

# --- Deep Residual Binary Classifier ---
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=4, activation='gelu', dropout=0.2):
        super().__init__()
        # Choose activation
        if activation == 'gelu':
            act = nn.GELU()
        elif activation == 'swish':
            act = Swish()
        elif activation == 'mish':
            act = Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, act, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 3 outputs, no softmax (handled by loss)
        )
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.head(x)  # logits
        return x

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_classifier(X, y, input_dim, model_path, epochs=30, batch_size=512, patience=5, replay_ratio=0.1):
    model = BinaryClassifier(input_dim)
    loaded_weights = False
    if model_path and os.path.exists(model_path):
        print(f"Loading existing classifier weights from {model_path} for continual learning...")
        try:
            model.load_state_dict(torch.load(model_path))
            loaded_weights = True
        except Exception as e:
            print(f"[WARNING] Could not load previous weights due to: {e}. Starting from scratch.")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # Validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # --- REPLAY BUFFER: If fine-tuning, mix old data with new ---
    if loaded_weights and len(X_train) > 0:
        n_replay = int(len(X_train) * replay_ratio)
        if n_replay > 0:
            replay_idx = np.random.choice(len(X_train), n_replay, replace=False)
            print(f"Using replay buffer: mixing {n_replay} old samples into each epoch for continual learning.")
            X_replay = X_train[replay_idx]
            y_replay = y_train[replay_idx]
            # Concatenate replay buffer to training set
            X_train = np.concatenate([X_train, X_replay], axis=0)
            y_train = np.concatenate([y_train, y_replay], axis=0)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # integer class labels
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_tensor))
        for i in range(0, len(X_train_tensor), batch_size):
            idx = perm[i:i+batch_size]
            batch_X = X_train_tensor[idx]
            batch_y = y_train_tensor[idx]
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_tensor)
            val_loss = criterion(val_out, y_val_tensor).item()
            val_probs = torch.softmax(val_out, dim=1).cpu().numpy()
            val_pred = np.argmax(val_probs, axis=1)
            val_true = y_val_tensor.cpu().numpy()
            acc = accuracy_score(val_true, val_pred)
            prec = precision_score(val_true, val_pred, average='macro', zero_division=0)
            rec = recall_score(val_true, val_pred, average='macro', zero_division=0)
            f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
            cm = confusion_matrix(val_true, val_pred)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), model_path)
        print(f"Best classifier model saved at {model_path}")
    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_tensor)
        val_probs = torch.softmax(val_out, dim=1).cpu().numpy()
        val_pred = np.argmax(val_probs, axis=1)
        val_true = y_val_tensor.cpu().numpy()
        acc = accuracy_score(val_true, val_pred)
        prec = precision_score(val_true, val_pred, average='macro', zero_division=0)
        rec = recall_score(val_true, val_pred, average='macro', zero_division=0)
        f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
        cm = confusion_matrix(val_true, val_pred)
        class_names = ['CALL', 'PUT', 'NEUTRAL']
        print(f"\nValidation Results:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix (rows: true, cols: pred):\n{cm}")
        print(f"Class mapping: 0=CALL, 1=PUT, 2=NEUTRAL")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train binary classifier for 1H, 15min, or both.")
    parser.add_argument('--data_dir', type=str, default='/Users/yashasnaidu/AI/historical_data')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print("What do you want to train?")
    print("1  - 1 hour candles")
    print("15 - 15 min candles")
    print("b  - both")
    choice = input("Enter your choice (1/15/b): ").strip()
    tasks = []
    if choice == '1' or choice.lower() == 'b':
        tasks.append(('1H', '*.csv', 'models/binary_classifier_1h.pth'))
    if choice == '15' or choice.lower() == 'b':
        tasks.append(('15min', '*_M15.csv', 'models/binary_classifier_15m.pth'))
    for label, file_pattern, model_path in tasks:
        print(f"\nTraining classifier for {label} candles...")
        data_dict = load_all_assets(args.data_dir, file_pattern=file_pattern)
        # Prepare features and labels
        # --- Prepare 10-candle windowed features ---
        features = prepare_data(data_dict)  # shape: (N, 9)
        window = 10
        X_win = []
        for i in range(window-1, len(features)):
            windowed = features[i-window+1:i+1].flatten()  # shape: (9*10,)
            X_win.append(windowed)
        X = np.array(X_win)

        # --- Autoencoder integration: compute anomaly scores ---
        try:
            autoencoder_path = 'models/pattern_autoencoder.pth'
            autoencoder = load_autoencoder(autoencoder_path, input_dim=9*window)
            autoencoder.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                recon = autoencoder(X_tensor)
                anomaly_scores = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
        except Exception as e:
            print(f"[WARNING] Could not compute anomaly scores: {e}")
            anomaly_scores = np.zeros(len(X))

        # Append anomaly score to each feature vector
        X_with_anomaly = np.concatenate([X, anomaly_scores[:, None]], axis=1)

        # Get binary outcome for each row in all assets, align to last candle in window
        y_full = np.concatenate([df['binary_outcome'].values[:len(df)] for df in data_dict.values()])
        y = y_full[window-1:]
        # Convert -1/1 to 0/1 if needed, and handle NaN
        y = ((y > 0)*1).astype(np.float32)
        y = np.nan_to_num(y, nan=0)
        input_dim = X_with_anomaly.shape[1]
        print(f"Loaded {len(X_with_anomaly)} samples, input_dim={input_dim} (with anomaly score)")
        train_classifier(X_with_anomaly, y, input_dim, model_path, epochs=args.epochs)
        print(f"Classifier for {label} candles training complete.\n")

if __name__ == "__main__":
    import os
    main()
