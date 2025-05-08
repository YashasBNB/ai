import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.train_model import UnsupervisedModel
from scripts.model_manager import ModelManager
from logging_config import setup_logging

class ModelAnalyzer:
    def __init__(self,
                 model_dir: str,
                 results_dir: str = "results/analysis",
                 n_clusters: int = 10):
        """
        Initialize analyzer
        
        Args:
            model_dir: Directory containing trained models
            results_dir: Directory to save analysis results
            n_clusters: Number of clusters for pattern analysis
        """
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_clusters = n_clusters
        self.logger = setup_logging().get_logger(__name__)
        self.model_manager = ModelManager(model_dir)
        
    def analyze_model(self,
                     timeframe: str,
                     test_data: np.ndarray) -> Dict:
        """Analyze model performance"""
        try:
            # Load model
            model = self._load_model(timeframe)
            if not model:
                raise ValueError(f"No model found for timeframe {timeframe}")
                
            # Get model predictions
            reconstructions, embeddings = self._get_predictions(model, test_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_data, reconstructions, embeddings)
            
            # Analyze patterns
            pattern_analysis = self._analyze_patterns(embeddings)
            
            # Generate visualizations
            self._generate_visualizations(
                timeframe,
                test_data,
                reconstructions,
                embeddings,
                pattern_analysis
            )
            
            # Combine results
            results = {
                "timeframe": timeframe,
                "metrics": metrics,
                "pattern_analysis": pattern_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            self._save_results(results, timeframe)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            return {}
            
    def _load_model(self, timeframe: str) -> Optional[UnsupervisedModel]:
        """Load trained model"""
        try:
            # Get latest checkpoint
            model = UnsupervisedModel(input_dim=None)  # Set appropriate input_dim
            epoch, metrics = self.model_manager.load_checkpoint(
                model,
                f"pattern_learner_{timeframe}"
            )
            
            self.logger.info(f"Loaded model from epoch {epoch}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None
            
    def _get_predictions(self,
                        model: UnsupervisedModel,
                        data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions and latent representations"""
        model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            reconstructions, mu, _ = model(data_tensor)
            
            return (
                reconstructions.numpy(),
                mu.numpy()  # Using mean of latent distribution
            )
            
    def _calculate_metrics(self,
                          original: np.ndarray,
                          reconstructed: np.ndarray,
                          embeddings: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # Reconstruction error
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        
        metrics["reconstruction"] = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(np.sqrt(mse))
        }
        
        # Latent space metrics
        metrics["latent_space"] = {
            "variance_explained": float(np.var(embeddings) / np.var(original)),
            "compactness": float(np.mean(np.linalg.norm(embeddings, axis=1)))
        }
        
        # Clustering metrics
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        metrics["clustering"] = {
            "silhouette_score": float(silhouette_score(embeddings, clusters)),
            "inertia": float(kmeans.inertia_),
            "cluster_sizes": [int(size) for size in np.bincount(clusters)]
        }
        
        return metrics
        
    def _analyze_patterns(self, embeddings: np.ndarray) -> Dict:
        """Analyze discovered patterns"""
        # Cluster patterns
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze cluster characteristics
        cluster_centers = kmeans.cluster_centers_
        cluster_distances = kmeans.transform(embeddings)
        
        analysis = {
            "n_clusters": self.n_clusters,
            "clusters": {
                "sizes": [int(size) for size in np.bincount(clusters)],
                "centers": cluster_centers.tolist(),
                "avg_distance": float(np.mean(cluster_distances.min(axis=1)))
            },
            "pattern_stats": {
                "most_common": int(np.bincount(clusters).argmax()),
                "least_common": int(np.bincount(clusters).argmin()),
                "diversity": float(np.std(np.bincount(clusters)))
            }
        }
        
        return analysis
        
    def _generate_visualizations(self,
                               timeframe: str,
                               original: np.ndarray,
                               reconstructed: np.ndarray,
                               embeddings: np.ndarray,
                               pattern_analysis: Dict):
        """Generate analysis visualizations"""
        try:
            # Create figure directory
            fig_dir = self.results_dir / "figures" / timeframe
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Reconstruction Quality
            plt.figure(figsize=(10, 6))
            plt.scatter(
                original.flatten(),
                reconstructed.flatten(),
                alpha=0.1,
                color='blue'
            )
            plt.plot([original.min(), original.max()],
                    [original.min(), original.max()],
                    'r--', label='Perfect reconstruction')
            plt.xlabel('Original Values')
            plt.ylabel('Reconstructed Values')
            plt.title('Reconstruction Quality')
            plt.legend()
            plt.savefig(fig_dir / 'reconstruction_quality.png')
            plt.close()
            
            # 2. Latent Space Visualization
            if embeddings.shape[1] >= 2:
                plt.figure(figsize=(10, 6))
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                clusters = kmeans.fit_predict(embeddings)
                
                scatter = plt.scatter(
                    embeddings[:, 0],
                    embeddings[:, 1],
                    c=clusters,
                    cmap='tab10',
                    alpha=0.6
                )
                plt.colorbar(scatter)
                plt.xlabel('First Latent Dimension')
                plt.ylabel('Second Latent Dimension')
                plt.title('Latent Space Clustering')
                plt.savefig(fig_dir / 'latent_space.png')
                plt.close()
                
            # 3. Pattern Distribution
            plt.figure(figsize=(10, 6))
            cluster_sizes = pattern_analysis['clusters']['sizes']
            plt.bar(range(len(cluster_sizes)), cluster_sizes)
            plt.xlabel('Pattern Cluster')
            plt.ylabel('Number of Occurrences')
            plt.title('Pattern Distribution')
            plt.savefig(fig_dir / 'pattern_distribution.png')
            plt.close()
            
            # 4. Reconstruction Error Distribution
            plt.figure(figsize=(10, 6))
            errors = np.mean((original - reconstructed) ** 2, axis=1)
            sns.histplot(errors, bins=50)
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Count')
            plt.title('Reconstruction Error Distribution')
            plt.savefig(fig_dir / 'error_distribution.png')
            plt.close()
            
            self.logger.info(f"Generated visualizations for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            
    def _save_results(self, results: Dict, timeframe: str):
        """Save analysis results"""
        try:
            # Save results as JSON
            results_file = self.results_dir / f"analysis_{timeframe}_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            self.logger.info(f"Saved analysis results to {results_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            
    def generate_report(self, timeframes: List[str]) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "timeframes": {}
        }
        
        for timeframe in timeframes:
            # Load latest analysis
            analysis_files = list(self.results_dir.glob(f"analysis_{timeframe}_*.json"))
            if not analysis_files:
                continue
                
            latest_analysis = max(analysis_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_analysis, 'r') as f:
                analysis = json.load(f)
                
            report["timeframes"][timeframe] = {
                "metrics": analysis["metrics"],
                "pattern_summary": {
                    "total_patterns": sum(analysis["pattern_analysis"]["clusters"]["sizes"]),
                    "dominant_pattern": analysis["pattern_analysis"]["pattern_stats"]["most_common"],
                    "pattern_diversity": analysis["pattern_analysis"]["pattern_stats"]["diversity"]
                }
            }
            
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Analyzer")
    parser.add_argument('--model_dir', type=str, required=True,
                       help="Directory containing trained models")
    parser.add_argument('--results_dir', type=str, default="results/analysis",
                       help="Directory to save analysis results")
    parser.add_argument('--timeframes', nargs='+', required=True,
                       help="Timeframes to analyze")
    parser.add_argument('--report', action='store_true',
                       help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer(args.model_dir, args.results_dir)
    
    if args.report:
        report = analyzer.generate_report(args.timeframes)
        print("\nAnalysis Report:")
        print(json.dumps(report, indent=2))
    else:
        # Example: Load some test data and analyze
        # In practice, you would load your actual test data here
        test_data = np.random.randn(1000, 10)  # Example dimensions
        
        for timeframe in args.timeframes:
            results = analyzer.analyze_model(timeframe, test_data)
            print(f"\nResults for {timeframe}:")
            print(json.dumps(results, indent=2))
