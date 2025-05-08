import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import prepare_data_for_training
from advanced_pattern_learner import PatternLearner

class UnsupervisedPipeline:
    def __init__(self, data_dir, model_dir="models", results_dir="results"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.feature_extractor = None
        self.pattern_learner = None
        self.feature_names = None
        
    def prepare_data(self, window_size=10):
        """Prepare and extract features from raw data"""
        logging.info("Preparing data and extracting features...")
        
        # Load your data here according to your data structure
        data_dict = self._load_data()
        
        # Extract features
        features, names, extractor = prepare_data_for_training(data_dict, window_size)
        self.feature_names = names
        self.feature_extractor = extractor
        
        return features
        
    def train_model(self, features, latent_dim=8, epochs=50, batch_size=256):
        """Train the pattern learner model"""
        logging.info("Training pattern learner...")
        
        input_dim = features.shape[1]
        self.pattern_learner = PatternLearner(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[128, 64, 32]
        )
        
        # Train the model
        self.pattern_learner.train(
            data=features,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save the model
        model_path = os.path.join(self.model_dir, "pattern_learner.pth")
        self.pattern_learner.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
    def analyze_patterns(self, features):
        """Analyze and evaluate discovered patterns"""
        logging.info("Analyzing patterns...")
        
        # Detect patterns
        patterns = self.pattern_learner.detect_patterns(features)
        
        # Evaluate clustering quality
        silhouette_avg = silhouette_score(
            patterns['latent_pca'],
            patterns['clusters'],
            metric='euclidean'
        )
        
        # Generate visualizations
        self._plot_patterns(patterns)
        
        # Calculate pattern statistics
        pattern_stats = self._calculate_pattern_stats(patterns, features)
        
        # Save results
        self._save_results(patterns, pattern_stats, silhouette_avg)
        
        return patterns, pattern_stats
        
    def _plot_patterns(self, patterns):
        """Generate visualizations of discovered patterns"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot latent space visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            patterns['latent_pca'][:, 0],
            patterns['latent_pca'][:, 1],
            c=patterns['clusters'],
            cmap='tab20',
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title("Latent Space Pattern Clusters")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.savefig(os.path.join(self.results_dir, f"patterns_latent_space_{timestamp}.png"))
        plt.close()
        
        # Plot reconstruction errors
        plt.figure(figsize=(12, 4))
        plt.plot(patterns['reconstruction_errors'], alpha=0.6)
        plt.axhline(y=np.mean(patterns['reconstruction_errors']) + 
                   3 * np.std(patterns['reconstruction_errors']),
                   color='r', linestyle='--', label='Anomaly Threshold')
        plt.title("Reconstruction Errors Over Time")
        plt.xlabel("Time")
        plt.ylabel("Reconstruction Error")
        plt.legend()
        plt.savefig(os.path.join(self.results_dir, f"reconstruction_errors_{timestamp}.png"))
        plt.close()
        
    def _calculate_pattern_stats(self, patterns, features):
        """Calculate statistics for each discovered pattern"""
        unique_clusters = np.unique(patterns['clusters'])
        stats = []
        
        for cluster in unique_clusters:
            if cluster == -1:  # DBSCAN noise points
                continue
                
            cluster_mask = patterns['clusters'] == cluster
            cluster_features = features[cluster_mask]
            
            # Calculate basic statistics
            cluster_stats = {
                'cluster_id': cluster,
                'size': np.sum(cluster_mask),
                'avg_recon_error': np.mean(patterns['reconstruction_errors'][cluster_mask]),
                'std_recon_error': np.std(patterns['reconstruction_errors'][cluster_mask]),
                'anomaly_ratio': np.mean(patterns['anomalies'][cluster_mask])
            }
            
            # Add feature statistics
            for i, name in enumerate(self.feature_names):
                cluster_stats[f'{name}_mean'] = np.mean(cluster_features[:, i])
                cluster_stats[f'{name}_std'] = np.std(cluster_features[:, i])
            
            stats.append(cluster_stats)
        
        return pd.DataFrame(stats)
        
    def _save_results(self, patterns, pattern_stats, silhouette_score):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save pattern statistics
        stats_path = os.path.join(self.results_dir, f"pattern_stats_{timestamp}.csv")
        pattern_stats.to_csv(stats_path, index=False)
        
        # Save summary report
        report = {
            'timestamp': timestamp,
            'num_patterns': len(np.unique(patterns['clusters'])),
            'num_anomalies': np.sum(patterns['anomalies']),
            'silhouette_score': silhouette_score,
            'avg_recon_error': np.mean(patterns['reconstruction_errors']),
            'std_recon_error': np.std(patterns['reconstruction_errors'])
        }
        
        report_path = os.path.join(self.results_dir, f"analysis_report_{timestamp}.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logging.info(f"Results saved to {self.results_dir}")
        
    def _load_data(self):
        """Load data from the data directory"""
        # Implement your data loading logic here
        # Return a dictionary of DataFrames: {symbol: pd.DataFrame}
        raise NotImplementedError("Implement data loading according to your data structure")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Unsupervised Trading Pattern Analysis")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--model_dir', type=str, default="models", help="Directory for saving models")
    parser.add_argument('--results_dir', type=str, default="results", help="Directory for saving results")
    parser.add_argument('--window_size', type=int, default=10, help="Window size for feature extraction")
    parser.add_argument('--latent_dim', type=int, default=8, help="Latent space dimension")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="Training batch size")
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = UnsupervisedPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir
    )
    
    # Prepare data
    features = pipeline.prepare_data(window_size=args.window_size)
    
    # Train model
    pipeline.train_model(
        features,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Analyze patterns
    patterns, stats = pipeline.analyze_patterns(features)
    
    logging.info("Pipeline completed successfully!")
