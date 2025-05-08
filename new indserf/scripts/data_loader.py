import os
import pandas as pd
from typing import Dict

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds detailed candle features to the DataFrame:
    - body: close - open
    - upper_wick: high - max(open, close)
    - lower_wick: min(open, close) - low
    - direction: 1 if bullish, -1 if bearish, 0 if doji
    - range: high - low
    """
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
    df['range'] = df['high'] - df['low']
    return df

def add_binary_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary_outcome column:
    - 1 if next close > current open (Call win)
    - -1 if next close < current open (Put win)
    - 0 if next close == current open (draw)
    """
    df = df.copy()
    next_close = df['close'].shift(-1)
    curr_open = df['open']
    outcome = (next_close > curr_open).astype(int) - (next_close < curr_open).astype(int)
    df['binary_outcome'] = outcome
    return df

import glob

def load_all_assets(data_dir: str, file_pattern: str = '*.csv') -> Dict[str, pd.DataFrame]:
    """
    Loads all CSVs from the given directory matching file_pattern (e.g., *_M15.csv for 15min candles), returns a dict of asset_name -> DataFrame.
    Adds detailed candle features and binary outcome labels to each DataFrame.
    Logs total assets and rows loaded.
    """
    import logging
    data = {}
    total_rows = 0
    search_path = os.path.join(data_dir, file_pattern)
    for fpath in glob.glob(search_path):
        fname = os.path.basename(fpath)
        asset = fname.rsplit('.', 1)[0]
        try:
            df = pd.read_csv(fpath, parse_dates=['time'])
            df = df.sort_values('time')
            df = add_candle_features(df)
            df = add_binary_outcome_label(df)
            data[asset] = df
            total_rows += len(df)
        except Exception as e:
            logging.warning(f"Failed to load {fname}: {e}")
    logging.info(f"Loaded {len(data)} assets, {total_rows} total rows.")
    return data

def get_total_rows(data_dict):
    return sum(len(df) for df in data_dict.values())

def prepare_data(data_dict):
    """
    Concatenate all assets' engineered features (excluding time, binary_outcome, and any non-feature columns)
    """
    import numpy as np
    all_data = []
    for df in data_dict.values():
        # Use engineered features for model input
        feats = df[['open','high','low','close','body','upper_wick','lower_wick','direction','range']].values.astype(np.float32)
        all_data.append(feats)
    return np.vstack(all_data) if all_data else np.empty((0, 9))

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    data = load_all_assets("/Users/yashasnaidu/AI/historical_data")
    for asset, df in data.items():
        print(f"{asset}: {df.shape}")
    print(f"Total assets: {len(data)} | Total rows: {get_total_rows(data)}")
