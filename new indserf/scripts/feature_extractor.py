import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import pandas as pd
import talib
import logging

class AdvancedFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_candle_features(self, df):
        """Extract advanced features from OHLCV data"""
        features = {}
        
        # Basic candle features
        features['body'] = df['close'] - df['open']
        features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        features['range'] = df['high'] - df['low']
        
        # Candle patterns (using TA-Lib)
        for pattern in [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDLENGULFING', 'CDLHARAMI', 'CDLMARUBOZU', 'CDLMORNINGSTAR',
            'CDLSHOOTINGSTAR', 'CDLSPINNINGTOP'
        ]:
            try:
                pattern_func = getattr(talib, pattern)
                features[f'pattern_{pattern}'] = pattern_func(
                    df['open'].values, df['high'].values,
                    df['low'].values, df['close'].values
                )
            except Exception as e:
                logging.warning(f"Could not compute pattern {pattern}: {e}")
        
        # Technical indicators
        features['rsi'] = talib.RSI(df['close'].values)
        features['macd'], features['macd_signal'], features['macd_hist'] = talib.MACD(df['close'].values)
        features['slowk'], features['slowd'] = talib.STOCH(
            df['high'].values, df['low'].values, df['close'].values
        )
        
        # Volatility indicators
        features['atr'] = talib.ATR(
            df['high'].values, df['low'].values, df['close'].values
        )
        features['natr'] = talib.NATR(
            df['high'].values, df['low'].values, df['close'].values
        )
        
        # Volume-based features
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['obv'] = talib.OBV(df['close'].values, df['volume'].values)
            features['ad'] = talib.AD(
                df['high'].values, df['low'].values,
                df['close'].values, df['volume'].values
            )
        
        # Convert all features to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Handle NaN values
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        return feature_df
    
    def extract_window_features(self, feature_df, window_size=10):
        """Extract features from rolling windows"""
        window_features = []
        
        for i in range(len(feature_df) - window_size + 1):
            window = feature_df.iloc[i:i+window_size]
            
            # Statistical features for each column
            stats = {}
            for col in feature_df.columns:
                values = window[col].values
                stats.update({
                    f'{col}_mean': np.mean(values),
                    f'{col}_std': np.std(values),
                    f'{col}_skew': skew(values),
                    f'{col}_kurt': kurtosis(values),
                    f'{col}_min': np.min(values),
                    f'{col}_max': np.max(values),
                    f'{col}_range': np.ptp(values),
                })
            
            window_features.append(stats)
        
        return pd.DataFrame(window_features)
    
    def fit_transform(self, df, window_size=10):
        """Extract and scale all features"""
        # Extract basic features
        feature_df = self.extract_candle_features(df)
        
        # Extract window features
        window_features = self.extract_window_features(feature_df, window_size)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(window_features)
        
        return scaled_features, window_features.columns.tolist()
    
    def transform(self, df, window_size=10):
        """Transform new data using fitted scaler"""
        feature_df = self.extract_candle_features(df)
        window_features = self.extract_window_features(feature_df, window_size)
        return self.scaler.transform(window_features)

def prepare_data_for_training(data_dict, window_size=10):
    """Prepare data from multiple assets for training"""
    extractor = AdvancedFeatureExtractor()
    all_features = []
    feature_names = None
    
    for symbol, df in data_dict.items():
        try:
            # Extract features for each asset
            features, names = extractor.fit_transform(df, window_size)
            if feature_names is None:
                feature_names = names
            all_features.append(features)
            logging.info(f"Processed {symbol}: extracted {features.shape[1]} features")
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if all_features:
        # Combine features from all assets
        combined_features = np.vstack(all_features)
        return combined_features, feature_names, extractor
    else:
        raise ValueError("No features could be extracted from the data")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    args = parser.parse_args()
    
    # Example usage (implement data loading according to your needs)
    # data_dict = load_data(args.data_dir)
    # features, names, extractor = prepare_data_for_training(data_dict)
    # logging.info(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
