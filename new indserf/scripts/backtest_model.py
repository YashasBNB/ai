import argparse
import numpy as np
import pandas as pd
from data_loader import load_all_assets, prepare_data
import torch
import torch.nn as nn
import torch.optim as optim
from pattern_learner import Autoencoder, load_model

# --- CONFIG ---
THRESHOLD = 0.01  # anomaly threshold for autoencoder loss (tune as needed)


def backtest_autoencoder(model, data, original_df, window=1, replay_data=None, model_path=None):
    """
    For each sample, compute reconstruction error. If error > THRESHOLD, flag as anomaly.
    Returns DataFrame with columns: time, error, is_anomaly
    Also: model is updated (fine-tuned) on mistakes + replay buffer.
    """
    import random
    mistake_buffer = []
    BATCH_SIZE = 100
    FINE_TUNE_EPOCHS = 3
    REPLAY_SIZE = 200

    model.eval()
    with torch.no_grad():
        X = torch.tensor(data, dtype=torch.float32)
        recon = model(X)
        errors = ((X - recon) ** 2).mean(dim=1).numpy()
    is_anomaly = errors > THRESHOLD
    times = original_df['time'].iloc[:len(errors)].values
    closes = original_df['close'].iloc[:len(errors)].values
    results = pd.DataFrame({'time': times, 'recon_error': errors, 'is_anomaly': is_anomaly, 'close': closes})

    # --- Continual learning: learn from mistakes (immediate online) ---
    trade_indices = results.index[results['is_anomaly']].tolist()
    running_mistakes = 0
    running_wins = 0
    running_losses = 0
    for idx in trade_indices:
        if idx + 1 >= len(results):
            continue
        entry_close = results.at[idx, 'close']
        exit_close = results.at[idx + 1, 'close']
        sample = data[idx]
        # If loss, fine-tune immediately on this mistake + replay
        if exit_close <= entry_close:
            running_mistakes += 1
            # Compute loss before fine-tune
            model.eval()
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                recon = model(sample_tensor)
                loss_before = ((sample_tensor - recon) ** 2).mean().item()
            fine_tune_data = [sample]
            # Add replay buffer (random sample from replay_data)
            if replay_data is not None and len(replay_data) > 0:
                import random
                fine_tune_data += random.sample(list(replay_data), min(REPLAY_SIZE, len(replay_data)))
            fine_tune_data = np.array(fine_tune_data)
            _fine_tune_autoencoder(model, fine_tune_data, epochs=FINE_TUNE_EPOCHS)
            # Compute loss after fine-tune
            model.eval()
            with torch.no_grad():
                recon = model(sample_tensor)
                loss_after = ((sample_tensor - recon) ** 2).mean().item()
            # Save model
            if model_path:
                torch.save(model.state_dict(), model_path)
            # Print learning log
            print(f"[LEARNING] Fine-tuned on mistake at idx {idx}, time {results.at[idx, 'time']}. Loss before: {loss_before:.6f}, after: {loss_after:.6f}. Total mistakes so far: {running_mistakes}")
        # Track running win/loss
        if exit_close > entry_close:
            running_wins += 1
        else:
            running_losses += 1
        if (running_wins + running_losses) % 50 == 0:
            win_rate = (running_wins / (running_wins + running_losses)) * 100 if (running_wins + running_losses) > 0 else 0.0
            print(f"[PROGRESS] Trades evaluated: {running_wins + running_losses}, Current win rate: {win_rate:.2f}%")
    return results


def _fine_tune_autoencoder(model, data, epochs=3):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    for epoch in range(epochs):
        perm = torch.randperm(len(data_tensor))
        for i in range(0, len(data_tensor), 256):
            batch_idx = perm[i:i+256]
            batch = data_tensor[batch_idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="Generic Model Backtester")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--file_pattern', type=str, default='*.csv')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--asset', type=str, default=None, help="(Optional) Only backtest this asset")
    args = parser.parse_args()

    print(f"Loading data for backtest: {args.file_pattern}")
    data_dict = load_all_assets(args.data_dir, file_pattern=args.file_pattern)
    if args.asset:
        data_dict = {args.asset: data_dict[args.asset]}

    for asset, df in data_dict.items():
        print(f"\nBacktesting on asset: {asset}")
        # Prepare features as in training
        data = df[['open','high','low','close','body','upper_wick','lower_wick','direction','range']].values.astype(np.float32)
        input_dim = data.shape[1]
        model = Autoencoder(input_dim)
        model = load_model(args.model_path, input_dim)
        # Prepare replay buffer (random sample from data)
        replay_data = None
        if len(data) > 0:
            import random
            replay_data = random.sample(list(data), min(200, len(data)))
        results = backtest_autoencoder(model, data, df, replay_data=replay_data, model_path=args.model_path)
        print(results.head())
        n_anom = results['is_anomaly'].sum()
        print(f"Total trades placed (anomalies): {n_anom} out of {len(results)} candles")
        
        # Simulate trade results
        wins = 0
        losses = 0
        trade_indices = results.index[results['is_anomaly']].tolist()
        for idx in trade_indices:
            # Check if there is a next candle
            if idx + 1 < len(results):
                entry_close = results.at[idx, 'close']
                exit_close = results.at[idx + 1, 'close']
                if exit_close > entry_close:
                    wins += 1
                else:
                    losses += 1
        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
        print(f"Winning trades (in profit): {wins}")
        print(f"Losing trades (in loss): {losses}")
        print(f"Win rate: {win_rate:.2f}%")
        results.to_csv(f"backtest_{asset}.csv", index=False)
        print(f"Saved results to backtest_{asset}.csv")

if __name__ == "__main__":
    main()
