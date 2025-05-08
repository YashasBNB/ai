import argparse
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime
import torch

from config import Config, get_default_config
from scripts.data_processor import DataProcessor
from scripts.feature_extractor import AdvancedFeatureExtractor
from scripts.advanced_pattern_learner import PatternLearner
from scripts.pattern_analyzer import PatternAnalyzer
from utils.monitor import PerformanceMonitor, ErrorHandler, timer

class UnsupervisedTradingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.monitor = PerformanceMonitor(config.log_dir)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = Path(self.config.log_dir) / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    @timer
    @ErrorHandler.wrap_execution
    def run(self):
        """Run the complete unsupervised learning pipeline"""
        self.monitor.log_system_info()
        
        # 1. Load and Process Data
        logging.info("Loading and processing data...")
        data_processor = DataProcessor(self.config.data.data_dir)
        data_dict = data_processor.load_data(
            timeframe=self.config.data.timeframe,
            symbols=None  # Load all available symbols
        )
        
        # Align data across symbols
        data_dict = data_processor.align_data(data_dict)
        
        # 2. Extract Features
        logging.info("Extracting features...")
        feature_extractor = AdvancedFeatureExtractor()
        features = []
        timestamps = None
        
        for symbol, df in data_dict.items():
            # Extract features for current symbol
            symbol_features, feature_names = feature_extractor.fit_transform(
                df,
                window_size=self.config.data.window_size
            )
            features.append(symbol_features)
            
            # Keep timestamps from first symbol
            if timestamps is None:
                timestamps = df.index[self.config.data.window_size-1:]
                
            logging.info(f"Processed {symbol}: {symbol_features.shape[0]} samples")
            
        # Combine features from all symbols
        import numpy as np
        combined_features = np.vstack(features)
        logging.info(f"Combined features shape: {combined_features.shape}")
        
        # 3. Initialize and Train Pattern Learner
        logging.info("Training pattern learner...")
        pattern_learner = PatternLearner(
            input_dim=combined_features.shape[1],
            latent_dim=self.config.model.latent_dim,
            hidden_dims=self.config.model.hidden_dims
        )
        
        # Train the model
        pattern_learner.train(
            data=combined_features,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            learning_rate=self.config.model.learning_rate
        )
        
        # Save the trained model
        model_path = Path(self.config.model_dir) / "pattern_learner.pth"
        pattern_learner.save(str(model_path))
        logging.info(f"Model saved to {model_path}")
        
        # 4. Detect and Analyze Patterns
        logging.info("Analyzing patterns...")
        patterns = pattern_learner.detect_patterns(
            combined_features,
            threshold=self.config.cluster.anomaly_threshold
        )
        
        # 5. Analyze Results
        logging.info("Generating analysis and visualizations...")
        analyzer = PatternAnalyzer(self.config.results_dir)
        analysis_results = analyzer.analyze_patterns(
            features=combined_features,
            patterns=patterns,
            feature_names=feature_names,
            timestamps=timestamps
        )
        
        # 6. Save Results
        self._save_results(patterns, analysis_results)
        
        # 7. Save Performance Metrics
        self.monitor.save_metrics()
        self.monitor.plot_metrics()
        
        logging.info("Pipeline completed successfully!")
        return analysis_results
        
    def _save_results(self, patterns, analysis_results):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_dir)
        
        # Save patterns
        patterns_file = results_dir / f"patterns_{timestamp}.npz"
        np.savez(
            patterns_file,
            clusters=patterns['clusters'],
            latent_pca=patterns['latent_pca'],
            reconstruction_errors=patterns['reconstruction_errors'],
            anomalies=patterns['anomalies']
        )
        
        # Save analysis results
        analysis_file = results_dir / f"analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
            
        logging.info(f"Results saved to {results_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised Trading Pattern Analysis")
    parser.add_argument('--config', type=str, help="Path to config file")
    parser.add_argument('--data_dir', type=str, help="Path to data directory")
    parser.add_argument('--timeframe', type=str, help="Trading timeframe")
    parser.add_argument('--window_size', type=int, help="Window size for feature extraction")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Training batch size")
    parser.add_argument('--latent_dim', type=int, help="Latent space dimension")
    return parser.parse_args()

def load_config(args) -> Config:
    """Load and update configuration"""
    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = get_default_config()
        
    # Update config with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.timeframe:
        config.data.timeframe = args.timeframe
    if args.window_size:
        config.data.window_size = args.window_size
    if args.epochs:
        config.model.epochs = args.epochs
    if args.batch_size:
        config.model.batch_size = args.batch_size
    if args.latent_dim:
        config.model.latent_dim = args.latent_dim
        
    return config

if __name__ == "__main__":
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args)
    
    # Run pipeline
    pipeline = UnsupervisedTradingPipeline(config)
    try:
        results = pipeline.run()
        logging.info("Pipeline completed successfully!")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
