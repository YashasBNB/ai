import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import glob
import pytz
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataProcessor:
    def __init__(self, base_dir: str):
        """
        Initialize DataProcessor with data directory path
        
        Args:
            base_dir (str): Base directory containing data files
        """
        self.base_dir = base_dir
        self.timeframes = {
            'M1': timedelta(minutes=1),
            'M5': timedelta(minutes=5),
            'M15': timedelta(minutes=15),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D1': timedelta(days=1)
        }
    
    def load_data(self, 
                 timeframe: str = 'M15',
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess data for multiple symbols
        
        Args:
            timeframe (str): Timeframe code (M1, M5, M15, etc.)
            start_date (datetime): Start date for filtering data
            end_date (datetime): End date for filtering data
            symbols (List[str]): List of symbols to load, None for all available
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of preprocessed DataFrames by symbol
        """
        logging.info(f"Loading data for timeframe {timeframe}")
        
        # Find all data files
        pattern = f"*_{timeframe}.csv" if timeframe else "*.csv"
        data_files = glob.glob(os.path.join(self.base_dir, pattern))
        
        if symbols:
            data_files = [f for f in data_files if any(s in f for s in symbols)]
            
        if not data_files:
            raise ValueError(f"No data files found matching pattern {pattern}")
            
        # Load data in parallel
        with ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self._load_single_file, f): f 
                for f in data_files
            }
            
            data_dict = {}
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    symbol, df = future.result()
                    if df is not None and not df.empty:
                        data_dict[symbol] = df
                except Exception as e:
                    logging.error(f"Error loading {file}: {str(e)}")
                    
        # Filter by date range if specified
        if start_date or end_date:
            data_dict = self._filter_date_range(data_dict, start_date, end_date)
            
        logging.info(f"Loaded data for {len(data_dict)} symbols")
        return data_dict
    
    def _load_single_file(self, filepath: str) -> Tuple[str, pd.DataFrame]:
        """Load and preprocess a single data file"""
        try:
            # Extract symbol from filename
            symbol = os.path.basename(filepath).split('_')[0]
            
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {filepath}")
                
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
            
            # Basic data validation
            df = self._validate_data(df)
            
            return symbol, df
            
        except Exception as e:
            logging.error(f"Error processing {filepath}: {str(e)}")
            return None, None
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data"""
        # Remove rows with NaN values
        df = df.dropna()
        
        # Ensure OHLC values are numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Validate OHLC relationships
        valid_candles = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        df = df[valid_candles]
        
        # Remove extreme outliers
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            df = df[df[col].between(mean - 4*std, mean + 4*std)]
            
        return df
    
    def _filter_date_range(self, 
                          data_dict: Dict[str, pd.DataFrame],
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Filter data by date range"""
        filtered_dict = {}
        for symbol, df in data_dict.items():
            mask = True
            if start_date:
                mask &= df.index >= start_date
            if end_date:
                mask &= df.index <= end_date
            filtered_dict[symbol] = df[mask]
        return filtered_dict
    
    def resample_timeframe(self,
                          data_dict: Dict[str, pd.DataFrame],
                          target_timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Resample data to a different timeframe
        
        Args:
            data_dict (Dict[str, pd.DataFrame]): Input data dictionary
            target_timeframe (str): Target timeframe code (M1, M5, M15, etc.)
            
        Returns:
            Dict[str, pd.DataFrame]: Resampled data dictionary
        """
        if target_timeframe not in self.timeframes:
            raise ValueError(f"Invalid timeframe: {target_timeframe}")
            
        resampled_dict = {}
        for symbol, df in data_dict.items():
            # Define resampling rule
            rule = self.timeframes[target_timeframe]
            
            # Resample OHLCV data
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' if 'volume' in df.columns else None
            }).dropna()
            
            resampled_dict[symbol] = resampled
            
        return resampled_dict
    
    def align_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align data from multiple symbols to ensure consistent timestamps
        
        Args:
            data_dict (Dict[str, pd.DataFrame]): Input data dictionary
            
        Returns:
            Dict[str, pd.DataFrame]: Aligned data dictionary
        """
        # Find common date range
        start_dates = [df.index.min() for df in data_dict.values()]
        end_dates = [df.index.max() for df in data_dict.values()]
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Align all dataframes
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned = df[common_start:common_end]
            if not aligned.empty:
                aligned_dict[symbol] = aligned
                
        return aligned_dict

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument('--data_dir', type=str, required=True, help="Data directory path")
    parser.add_argument('--timeframe', type=str, default='M15', help="Timeframe to load")
    args = parser.parse_args()
    
    processor = DataProcessor(args.data_dir)
    data = processor.load_data(timeframe=args.timeframe)
    logging.info(f"Loaded {len(data)} symbols")
    
    # Example: Print first few rows of first symbol
    first_symbol = list(data.keys())[0]
    logging.info(f"\nSample data for {first_symbol}:")
    logging.info(data[first_symbol].head())
