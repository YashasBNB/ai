import os
import numpy as np
import pandas as pd
from data_loader import load_all_assets
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest

WINDOW_SIZE = 10
N_CLUSTERS = 20  # You can adjust this


def extract_windows(df, window_size=10, feature_cols=None):
    """
    Extract rolling windows of size window_size from DataFrame, flattening features.
    Returns a 2D numpy array: [num_windows, window_size * num_features]
    """
    if feature_cols is None:
        feature_cols = ['open', 'high', 'low', 'close', 'body', 'upper_wick', 'lower_wick', 'direction', 'range']
    arr = df[feature_cols].values
    num_windows = arr.shape[0] - window_size + 1
    if num_windows <= 0:
        return np.empty((0, window_size * len(feature_cols)))
    windows = np.lib.stride_tricks.sliding_window_view(arr, (window_size, arr.shape[1]))
    windows = windows.squeeze(axis=1)
    flat_windows = windows.reshape(num_windows, -1)
    return flat_windows

def main():
    print("Loading data and extracting features...")
    data = load_all_assets("/Users/yashasnaidu/AI/historical_data")
    window_features = []
    asset_window_indices = []  # (asset, start_idx) for each window
    for asset, df in data.items():
        feats = extract_windows(df, window_size=WINDOW_SIZE)
        window_features.append(feats)
        asset_window_indices.extend([(asset, i) for i in range(feats.shape[0])])
    X = np.vstack(window_features)
    print(f"Total windows: {X.shape[0]}")

    print("Clustering windows with MiniBatchKMeans...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=10000, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    print("Detecting anomalies with IsolationForest...")
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    anomaly_scores = iso.fit_predict(X)  # -1 = anomaly, 1 = normal

    # Save results
    results = pd.DataFrame(asset_window_indices, columns=['asset', 'start_idx'])
    results['cluster'] = cluster_labels
    results['anomaly'] = anomaly_scores
    results.to_csv("pattern_discovery_results.csv", index=False)
    print("Results saved to pattern_discovery_results.csv")

    # Optionally: show cluster sizes and a few example patterns
    print("\nCluster sizes:")
    print(results['cluster'].value_counts())
    print("\nExample patterns from each cluster:")
    for c in range(N_CLUSTERS):
        ex = results[results['cluster'] == c].iloc[0]
        asset, idx = ex['asset'], ex['start_idx']
        print(f"\nAsset: {asset}, Start Candle: {idx}, Cluster: {c}")
        print(data[asset].iloc[idx:idx+WINDOW_SIZE][['open','high','low','close','body','upper_wick','lower_wick','direction','range']])

if __name__ == "__main__":
    main()
