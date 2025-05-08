import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

class SyntheticDataGenerator:
    def __init__(self, 
                 num_symbols: int = 10,
                 timeframe: str = 'M15',
                 start_date: datetime = None,
                 num_days: int = 30):
        """
        Initialize synthetic data generator
        
        Args:
            num_symbols: Number of symbols to generate
            timeframe: Trading timeframe (M1, M5, M15, M30, H1, H4, D1)
            start_date: Start date for data generation
            num_days: Number of days of data to generate
        """
        self.num_symbols = num_symbols
        self.timeframe = timeframe
        self.start_date = start_date or datetime.now() - timedelta(days=num_days)
        self.num_days = num_days
        
        # Timeframe to minutes mapping
        self.timeframe_minutes = {
            'M1': 1,
            'M5': 5,
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }
        
        # Generate symbol names
        self.symbols = [f"SYM{i+1}" for i in range(num_symbols)]
        
        # Initialize random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)
        
    def generate_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic trading data for all symbols"""
        data_dict = {}
        
        for symbol in self.symbols:
            df = self._generate_symbol_data(symbol)
            data_dict[symbol] = df
            
        return data_dict
        
    def _generate_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic data for a single symbol"""
        # Calculate number of candles
        minutes_per_day = 24 * 60
        candles_per_day = minutes_per_day // self.timeframe_minutes[self.timeframe]
        total_candles = candles_per_day * self.num_days
        
        # Generate timestamps
        timestamps = [
            self.start_date + timedelta(minutes=i*self.timeframe_minutes[self.timeframe])
            for i in range(total_candles)
        ]
        
        # Generate price data with trends, patterns, and noise
        data = self._generate_price_series(total_candles)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })
        
        df.set_index('timestamp', inplace=True)
        return df
        
    def _generate_price_series(self, n_candles: int) -> Dict[str, np.ndarray]:
        """Generate synthetic price series with realistic patterns"""
        # Base parameters
        volatility = random.uniform(0.001, 0.003)
        trend = random.uniform(-0.0001, 0.0001)
        
        # Generate log returns with trends and patterns
        returns = np.random.normal(trend, volatility, n_candles)
        
        # Add some patterns
        returns = self._add_patterns(returns)
        
        # Convert returns to prices
        starting_price = random.uniform(10, 100)
        prices = starting_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = {
            'close': prices,
            'open': np.zeros_like(prices),
            'high': np.zeros_like(prices),
            'low': np.zeros_like(prices),
            'volume': np.zeros_like(prices)
        }
        
        # Generate realistic OHLC relationships
        for i in range(n_candles):
            if i == 0:
                data['open'][i] = starting_price
            else:
                data['open'][i] = data['close'][i-1]
            
            # Random high/low around open/close
            price_range = abs(data['close'][i] - data['open'][i])
            high_wick = random.uniform(0.001, 0.003) * data['open'][i]
            low_wick = random.uniform(0.001, 0.003) * data['open'][i]
            
            data['high'][i] = max(data['open'][i], data['close'][i]) + high_wick
            data['low'][i] = min(data['open'][i], data['close'][i]) - low_wick
            
            # Generate volume with some correlation to price movement
            price_change = abs(data['close'][i] - data['open'][i])
            base_volume = np.random.lognormal(mean=10, sigma=1)
            volume_factor = 1 + (price_change / data['open'][i]) * 10
            data['volume'][i] = base_volume * volume_factor
            
        return data
        
    def _add_patterns(self, returns: np.ndarray) -> np.ndarray:
        """Add synthetic patterns to returns series"""
        n_candles = len(returns)
        
        # Add trends
        trend_points = np.random.choice(n_candles, size=5)
        for point in trend_points:
            trend_length = random.randint(20, 50)
            trend_strength = random.uniform(-0.001, 0.001)
            end_point = min(point + trend_length, n_candles)
            returns[point:end_point] += trend_strength
            
        # Add volatility clusters
        vol_points = np.random.choice(n_candles, size=3)
        for point in vol_points:
            cluster_length = random.randint(10, 30)
            vol_multiplier = random.uniform(1.5, 3.0)
            end_point = min(point + cluster_length, n_candles)
            returns[point:end_point] *= vol_multiplier
            
        # Add mean reversion patterns
        reversion_points = np.random.choice(n_candles, size=4)
        for point in reversion_points:
            pattern_length = random.randint(5, 15)
            strength = random.uniform(0.001, 0.003)
            end_point = min(point + pattern_length, n_candles)
            returns[point:end_point] = -strength * np.sum(returns[max(0, point-pattern_length):point])
            
        return returns
        
    def save_data(self, data_dict: Dict[str, pd.DataFrame], output_dir: str):
        """Save synthetic data to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filename = f"{symbol}_{self.timeframe}.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)
            print(f"Saved {filepath}")

def generate_test_data(output_dir: str,
                      num_symbols: int = 10,
                      timeframe: str = 'M15',
                      num_days: int = 30):
    """Convenience function to generate test data"""
    generator = SyntheticDataGenerator(
        num_symbols=num_symbols,
        timeframe=timeframe,
        num_days=num_days
    )
    
    data_dict = generator.generate_data()
    generator.save_data(data_dict, output_dir)
    return data_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic trading data")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for data files")
    parser.add_argument('--num_symbols', type=int, default=10, help="Number of symbols to generate")
    parser.add_argument('--timeframe', type=str, default='M15', help="Trading timeframe")
    parser.add_argument('--num_days', type=int, default=30, help="Number of days of data")
    
    args = parser.parse_args()
    
    # Generate test data
    data = generate_test_data(
        output_dir=args.output_dir,
        num_symbols=args.num_symbols,
        timeframe=args.timeframe,
        num_days=args.num_days
    )
    
    print(f"Generated synthetic data for {len(data)} symbols")
