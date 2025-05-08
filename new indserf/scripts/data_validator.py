import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from logging_config import setup_logging

class DataValidator:
    """Validate and prepare historical data for training"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = setup_logging().get_logger(__name__)
        
        # Required columns
        self.required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        # Data quality thresholds
        self.max_missing_pct = 0.01  # Maximum 1% missing values
        self.min_data_points = 1000  # Minimum number of candles
        self.max_price_change = 10.0  # Maximum 10% price change between candles
        
    def validate_files(self) -> Dict[str, Dict]:
        """Validate all CSV files in directory"""
        validation_results = {}
        
        try:
            files = list(self.data_dir.glob("*.csv"))
            if not files:
                raise ValueError(f"No CSV files found in {self.data_dir}")
                
            self.logger.info(f"Found {len(files)} CSV files")
            
            for file in files:
                try:
                    result = self._validate_file(file)
                    validation_results[file.name] = result
                except Exception as e:
                    self.logger.error(f"Error validating {file}: {str(e)}")
                    validation_results[file.name] = {"valid": False, "errors": [str(e)]}
                    
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return {}
            
    def _validate_file(self, file_path: Path) -> Dict:
        """Validate individual CSV file"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Read data
            df = pd.read_csv(file_path)
            
            # Check columns
            missing_cols = set(self.required_columns) - set(df.columns)
            if missing_cols:
                result["valid"] = False
                result["errors"].append(f"Missing columns: {missing_cols}")
                return result
                
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Basic validations
            validations = [
                self._validate_data_size(df),
                self._validate_missing_values(df),
                self._validate_price_sequence(df),
                self._validate_timestamps(df),
                self._validate_price_changes(df),
                self._validate_hlc_consistency(df)
            ]
            
            for validation in validations:
                if not validation["valid"]:
                    result["valid"] = False
                    result["errors"].extend(validation["errors"])
                result["warnings"].extend(validation.get("warnings", []))
                
            # Calculate statistics
            result["stats"] = self._calculate_statistics(df)
            
            return result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "stats": {}
            }
            
    def _validate_data_size(self, df: pd.DataFrame) -> Dict:
        """Validate data size"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        if len(df) < self.min_data_points:
            result["valid"] = False
            result["errors"].append(
                f"Insufficient data points: {len(df)} < {self.min_data_points}"
            )
            
        return result
        
    def _validate_missing_values(self, df: pd.DataFrame) -> Dict:
        """Validate missing values"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check each required column
        for col in self.required_columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0:
                if missing_pct > self.max_missing_pct:
                    result["valid"] = False
                    result["errors"].append(
                        f"Too many missing values in {col}: {missing_pct:.2%}"
                    )
                else:
                    result["warnings"].append(
                        f"Missing values in {col}: {missing_pct:.2%}"
                    )
                    
        return result
        
    def _validate_price_sequence(self, df: pd.DataFrame) -> Dict:
        """Validate price sequence"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            if (df[col] <= 0).any():
                result["valid"] = False
                result["errors"].append(f"Negative or zero values found in {col}")
                
        # Check high/low relationship
        if (df['high'] < df['low']).any():
            result["valid"] = False
            result["errors"].append("High price less than low price")
            
        return result
        
    def _validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """Validate timestamp sequence"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            result["warnings"].append(f"Found {duplicates.sum()} duplicate timestamps")
            
        # Check for gaps
        time_diff = df['timestamp'].diff()
        median_diff = time_diff.median()
        large_gaps = time_diff > (median_diff * 2)
        
        if large_gaps.any():
            result["warnings"].append(
                f"Found {large_gaps.sum()} large gaps in timestamps"
            )
            
        return result
        
    def _validate_price_changes(self, df: pd.DataFrame) -> Dict:
        """Validate price changes between candles"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Calculate percentage changes
        for col in ['open', 'high', 'low', 'close']:
            pct_change = df[col].pct_change().abs()
            large_changes = pct_change > (self.max_price_change / 100)
            
            if large_changes.any():
                result["warnings"].append(
                    f"Large price changes in {col}: {large_changes.sum()} instances"
                )
                
        return result
        
    def _validate_hlc_consistency(self, df: pd.DataFrame) -> Dict:
        """Validate high/low/close price consistency"""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check if close is between high and low
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            result["valid"] = False
            result["errors"].append(
                f"Found {invalid_close.sum()} candles where close is outside high-low range"
            )
            
        # Check if open is between high and low
        invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
        if invalid_open.any():
            result["valid"] = False
            result["errors"].append(
                f"Found {invalid_open.sum()} candles where open is outside high-low range"
            )
            
        return result
        
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate dataset statistics"""
        stats = {
            "total_candles": len(df),
            "date_range": {
                "start": df['timestamp'].min().strftime("%Y-%m-%d"),
                "end": df['timestamp'].max().strftime("%Y-%m-%d")
            },
            "price_stats": {
                "min_price": df['low'].min(),
                "max_price": df['high'].max(),
                "avg_price": df['close'].mean()
            },
            "volume_stats": {
                "min_volume": df['volume'].min(),
                "max_volume": df['volume'].max(),
                "avg_volume": df['volume'].mean()
            }
        }
        
        return stats
        
    def prepare_data(self) -> List[pd.DataFrame]:
        """Prepare validated data for training"""
        prepared_data = []
        
        # Validate files first
        validation_results = self.validate_files()
        
        for file_name, result in validation_results.items():
            if result["valid"]:
                try:
                    df = pd.read_csv(self.data_dir / file_name)
                    
                    # Basic preprocessing
                    df = self._preprocess_dataframe(df)
                    
                    prepared_data.append(df)
                    self.logger.info(f"Prepared {file_name} successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error preparing {file_name}: {str(e)}")
                    
        return prepared_data
        
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess individual dataframe"""
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'])
        
        # Forward fill missing values
        df = df.ffill()
        
        # Add basic features
        df['body'] = df['close'] - df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return df
        
    def save_validation_report(self, output_path: str):
        """Save validation results to file"""
        validation_results = self.validate_files()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(validation_results),
            "valid_files": sum(1 for r in validation_results.values() if r["valid"]),
            "results": validation_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info(f"Validation report saved to {output_path}")
        
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validator")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="Directory containing historical data")
    parser.add_argument('--report', type=str, default="validation_report.json",
                       help="Path to save validation report")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = DataValidator(args.data_dir)
    
    # Generate and save report
    report = validator.save_validation_report(args.report)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Total files: {report['total_files']}")
    print(f"Valid files: {report['valid_files']}")
    
    if report['valid_files'] > 0:
        print("\nPreparing data...")
        prepared_data = validator.prepare_data()
        print(f"Successfully prepared {len(prepared_data)} datasets")
