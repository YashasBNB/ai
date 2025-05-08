import argparse
import numpy as np
import pandas as pd
import torch
from data_loader import load_all_assets, prepare_data
from train_binary_classifier import BinaryClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def prepare_windowed_features(features, window=10):
    X_win = []
    for i in range(window-1, len(features)):
        windowed = features[i-window+1:i+1].flatten()
        X_win.append(windowed)
    return np.array(X_win)

def fine_tune_classifier(model, X, y, replay_X=None, replay_y=None, epochs=1, lr=1e-4):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    if replay_X is not None and replay_y is not None and len(replay_X) > 0:
        # Mix replay buffer with mistakes
        X_tensor = torch.cat([X_tensor, torch.tensor(replay_X, dtype=torch.float32)], dim=0)
        y_tensor = torch.cat([y_tensor, torch.tensor(replay_y, dtype=torch.float32).unsqueeze(1)], dim=0)
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()  # Set model to eval mode after fine-tuning
    return loss.item()

def backtest_classifier(model, features, y_true, times, close_prices, batch_size=1024, mistake_batch_size=8, replay_size=32, model_path=None):
    model.eval()  # Set model to eval mode before inference
    features = np.array(features)
    y_true = np.array(y_true)
    preds = []  # will store class probabilities
    pred_labels = []
    mistake_buffer_X = []
    mistake_buffer_y = []
    replay_X = []
    replay_y = []
    for i in range(len(features)):
        with torch.no_grad():
            x_tensor = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
        preds.append(probs)
        pred_labels.append(pred)
        # Collect replay buffer
        if len(replay_X) < replay_size:
            replay_X.append(features[i])
            replay_y.append(y_true[i])
        # If mistake, add to buffer
        if pred != int(y_true[i]):
            mistake_buffer_X.append(features[i])
            mistake_buffer_y.append(y_true[i])
        # Fine-tune when buffer full
        if len(mistake_buffer_X) >= mistake_batch_size:
            # Convert to np.array for efficient tensor creation
            mb_X = np.array(mistake_buffer_X)
            mb_y = np.array(mistake_buffer_y)
            rX = np.array(replay_X) if len(replay_X) > 0 else None
            rY = np.array(replay_y) if len(replay_y) > 0 else None
            loss = fine_tune_classifier(model, mb_X, mb_y, rX, rY)
            print(f"[LEARNING] Fine-tuned on {len(mistake_buffer_X)} mistakes + replay. Loss: {loss:.6f} at idx {i} (time {times[i]})")
            if model_path:
                torch.save(model.state_dict(), model_path)
            mistake_buffer_X = []
            mistake_buffer_y = []
    # Final fine-tune on leftovers
    if len(mistake_buffer_X) > 0:
        mb_X = np.array(mistake_buffer_X)
        mb_y = np.array(mistake_buffer_y)
        rX = np.array(replay_X) if len(replay_X) > 0 else None
        rY = np.array(replay_y) if len(replay_y) > 0 else None
        loss = fine_tune_classifier(model, mb_X, mb_y, rX, rY)
        print(f"[LEARNING] Final fine-tune on {len(mistake_buffer_X)} mistakes + replay. Loss: {loss:.6f}")
        if model_path:
            torch.save(model.state_dict(), model_path)
    preds = np.array(preds)
    pred_labels = np.array(pred_labels)
    # Metrics
    acc = accuracy_score(y_true, pred_labels)
    prec = precision_score(y_true, pred_labels, average='macro', zero_division=0)
    rec = recall_score(y_true, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, pred_labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, pred_labels)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix (rows: true, cols: pred):\n{cm}")
    print(f"Class mapping: 0=CALL, 1=PUT, 2=NEUTRAL")
    # Save results
    df = pd.DataFrame({
        'time': times,
        'prediction': pred_labels,
        'prob_call': preds[:,0],
        'prob_put': preds[:,1],
        'prob_neutral': preds[:,2],
        'true_label': y_true,
        'close': close_prices
    })
    return df, acc, prec, rec, f1, cm

def main():
    parser = argparse.ArgumentParser(description="Backtest binary classifier for 1H, 15min, or both.")
    parser.add_argument('--data_dir', type=str, default='/Users/yashasnaidu/AI/historical_data')
    args = parser.parse_args()
    print("What do you want to backtest?")
    print("1  - 1 hour candles")
    print("15 - 15 min candles")
    print("b  - both")
    choice = input("Enter your choice (1/15/b): ").strip()
    tasks = []
    if choice == '1' or choice.lower() == 'b':
        tasks.append(('1H', '*.csv', 'models/binary_classifier_1h.pth', 'backtest_classifier_1h_{}.csv'))
    if choice == '15' or choice.lower() == 'b':
        tasks.append(('15min', '*_M15.csv', 'models/binary_classifier_15m.pth', 'backtest_classifier_15m_{}.csv'))
    window = 10
    for label, file_pattern, model_path, out_pattern in tasks:
        print(f"\nBacktesting classifier for {label} candles...")
        data_dict = load_all_assets(args.data_dir, file_pattern=file_pattern)
        model = BinaryClassifier(input_dim=window*9)
        if not os.path.exists(model_path):
            print(f"Model weights not found at {model_path}. Skipping {label}.")
            continue
        model.load_state_dict(torch.load(model_path))
        for asset, df in data_dict.items():
            features = prepare_data({asset: df})
            if len(features) < window:
                print(f"Not enough data for {asset}, skipping.")
                continue
            X = prepare_windowed_features(features, window=window)
            y_full = df['binary_outcome'].values[:len(df)]
            y = y_full[window-1:]
            # Ensure labels are integer class indices (0=CALL, 1=PUT, 2=NEUTRAL)
            y = y.astype(int)
            y = np.nan_to_num(y, nan=2)  # Default NEUTRAL for NaN
            times = df['time'].values[window-1:]
            close_prices = df['close'].values[window-1:]
            print(f"Backtesting on asset: {asset}")
            results_df, acc, prec, rec, f1, cm = backtest_classifier(model, X, y, times, close_prices)
            out_csv = out_pattern.format(asset)
            results_df.to_csv(out_csv, index=False)
            print(f"Saved results to {out_csv}\n")

if __name__ == "__main__":
    main()
