"""
Unsupervised Trading Pattern Analysis - Usage Examples
---------------------------------------------------

This script demonstrates how to use the unsupervised learning pipeline
for trading pattern analysis with different configurations and scenarios.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import Config, get_default_config
from scripts.data_processor import DataProcessor
from scripts.feature_extractor import AdvancedFeatureExtractor
from scripts.advanced_pattern_learner import PatternLearner
from scripts.pattern_analyzer import PatternAnalyzer
from utils.synthetic_data import generate_test_data

def basic_usage_example():
    """Basic usage with default configuration"""
    print("\n=== Basic Usage Example ===")
    
    # Generate sample data
    data_dir = "example_data"
    data = generate_test_data(
        output_dir=data_dir,
        num_symbols=5,
        timeframe='M15',
        num_days=10
    )
    
    # Get default configuration
    config = get_default_config()
    config.data.data_dir = data_dir
    
    # Initialize components
    processor = DataProcessor(config.data.data_dir)
    extractor = AdvancedFeatureExtractor()
    
    # Process data
    data_dict = processor.load_data(timeframe='M15')
    print(f"Loaded data for {len(data_dict)} symbols")
    
    # Extract features for first symbol
    symbol = list(data_dict.keys())[0]
    features, names = extractor.fit_transform(
        data_dict[symbol],
        window_size=config.data.window_size
    )
    print(f"Extracted {len(names)} features")
    
    # Train pattern learner
    learner = PatternLearner(
        input_dim=features.shape[1],
        latent_dim=8,
        hidden_dims=[64, 32]
    )
    
    learner.train(features, epochs=5)
    print("Model trained successfully")
    
    # Detect patterns
    patterns = learner.detect_patterns(features)
    print(f"Detected {len(np.unique(patterns['clusters']))} distinct patterns")

def advanced_configuration_example():
    """Example with custom configuration"""
    print("\n=== Advanced Configuration Example ===")
    
    config = Config(
        data=Config.DataConfig(
            timeframe='M15',
            window_size=20,
            train_test_split=0.8
        ),
        features=Config.FeatureConfig(
            use_wavelet=True,
            use_fourier=True,
            correlation_window=30
        ),
        model=Config.ModelConfig(
            latent_dim=16,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2,
            learning_rate=0.001
        ),
        cluster=Config.ClusterConfig(
            eps=0.3,
            min_samples=10,
            anomaly_threshold=2.5
        )
    )
    
    print("Custom configuration created with:")
    print(f"- Window size: {config.data.window_size}")
    print(f"- Feature engineering: Wavelet and Fourier transforms enabled")
    print(f"- Model architecture: {config.model.hidden_dims}")

def pattern_analysis_example():
    """Example of detailed pattern analysis"""
    print("\n=== Pattern Analysis Example ===")
    
    # Generate and process data
    data_dir = "example_data"
    data = generate_test_data(
        output_dir=data_dir,
        num_symbols=3,
        timeframe='M15',
        num_days=5
    )
    
    # Initialize analyzer
    analyzer = PatternAnalyzer("analysis_results")
    
    # Generate sample patterns
    features = np.random.randn(100, 20)
    patterns = {
        'clusters': np.random.randint(0, 3, 100),
        'latent_pca': np.random.randn(100, 2),
        'reconstruction_errors': np.random.rand(100),
        'anomalies': np.random.choice([True, False], 100)
    }
    
    # Analyze patterns
    feature_names = [f"feature_{i}" for i in range(20)]
    timestamps = pd.date_range('2023-01-01', periods=100, freq='15T')
    
    results = analyzer.analyze_patterns(
        features=features,
        patterns=patterns,
        feature_names=feature_names,
        timestamps=timestamps
    )
    
    print("\nPattern Analysis Results:")
    print(f"- Number of clusters: {len(np.unique(patterns['clusters']))}")
    print(f"- Anomalies detected: {np.sum(patterns['anomalies'])}")
    print("- Visualizations saved in 'analysis_results' directory")

def real_time_simulation_example():
    """Example of simulated real-time pattern detection"""
    print("\n=== Real-time Simulation Example ===")
    
    # Generate data
    data_dir = "example_data"
    data = generate_test_data(
        output_dir=data_dir,
        num_symbols=1,
        timeframe='M15',
        num_days=5
    )
    
    # Setup components
    processor = DataProcessor(data_dir)
    extractor = AdvancedFeatureExtractor()
    
    # Get data and initial training
    data_dict = processor.load_data(timeframe='M15')
    symbol = list(data_dict.keys())[0]
    df = data_dict[symbol]
    
    # Split data for simulation
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size]
    stream_data = df.iloc[train_size:]
    
    # Initial training
    features, names = extractor.fit_transform(train_data, window_size=10)
    learner = PatternLearner(
        input_dim=features.shape[1],
        latent_dim=8,
        hidden_dims=[32, 16]
    )
    learner.train(features, epochs=5)
    
    print("\nSimulating real-time pattern detection:")
    # Simulate streaming data
    window_size = 10
    for i in range(len(stream_data) - window_size):
        window = stream_data.iloc[i:i+window_size]
        features = extractor.transform(window)
        
        # Detect patterns
        patterns = learner.detect_patterns(features)
        
        # Print updates every 10 steps
        if i % 10 == 0:
            print(f"Processed window {i}: "
                  f"Pattern {patterns['clusters'][-1]}, "
                  f"Anomaly: {patterns['anomalies'][-1]}")

def main():
    """Run all examples"""
    logging.basicConfig(level=logging.INFO)
    
    print("Unsupervised Trading Pattern Analysis - Usage Examples")
    print("=" * 50)
    
    try:
        basic_usage_example()
        advanced_configuration_example()
        pattern_analysis_example()
        real_time_simulation_example()
        
    except Exception as e:
        logging.error(f"Example failed: {str(e)}", exc_info=True)
        
    print("\nExamples completed. Check the output directories for results.")

if __name__ == "__main__":
    main()
