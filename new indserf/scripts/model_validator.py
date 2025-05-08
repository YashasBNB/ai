import numpy as np
import torch
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from logging_config import setup_logging

class ModelValidator:
    def __init__(self, results_dir: str = "results"):
        """
        Initialize ModelValidator
        
        Args:
            results_dir: Directory to save validation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging().get_logger(__name__)
        
    def compute_reconstruction_metrics(self,
                                    original: torch.Tensor,
                                    reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute reconstruction quality metrics"""
        # Convert to numpy if needed
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
            
        # Compute various metrics
        mse = np.mean((original - reconstructed) ** 2)
        mae = np.mean(np.abs(original - reconstructed))
        rmse = np.sqrt(mse)
        
        # Compute relative errors
        relative_error = np.mean(np.abs(original - reconstructed) / (np.abs(original) + 1e-8))
        
        # Compute correlation
        correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'relative_error': float(relative_error),
            'correlation': float(correlation)
        }
        
    def compute_clustering_metrics(self,
                                 latent_vectors: np.ndarray,
                                 cluster_labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering quality metrics"""
        try:
            silhouette = silhouette_score(latent_vectors, cluster_labels)
            calinski = calinski_harabasz_score(latent_vectors, cluster_labels)
            
            # Compute cluster statistics
            unique_clusters = np.unique(cluster_labels)
            cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]
            
            return {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski),
                'num_clusters': len(unique_clusters),
                'avg_cluster_size': float(np.mean(cluster_sizes)),
                'std_cluster_size': float(np.std(cluster_sizes)),
                'min_cluster_size': float(np.min(cluster_sizes)),
                'max_cluster_size': float(np.max(cluster_sizes))
            }
        except Exception as e:
            self.logger.error(f"Error computing clustering metrics: {e}")
            return {}
            
    def compute_anomaly_metrics(self,
                              reconstruction_errors: np.ndarray,
                              threshold: float) -> Dict[str, float]:
        """Compute anomaly detection metrics"""
        # Identify anomalies
        anomalies = reconstruction_errors > threshold
        
        # Compute statistics
        anomaly_ratio = np.mean(anomalies)
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        return {
            'anomaly_ratio': float(anomaly_ratio),
            'mean_reconstruction_error': float(mean_error),
            'std_reconstruction_error': float(std_error),
            'error_threshold': float(threshold),
            'num_anomalies': int(np.sum(anomalies))
        }
        
    def validate_model(self,
                      model: torch.nn.Module,
                      validation_data: torch.Tensor,
                      clustering_results: Optional[Dict] = None) -> Dict:
        """Perform comprehensive model validation"""
        model.eval()
        results = {}
        
        with torch.no_grad():
            # Get model outputs
            reconstructed, mu, log_var = model(validation_data)
            
            # Compute reconstruction metrics
            results['reconstruction'] = self.compute_reconstruction_metrics(
                validation_data, reconstructed
            )
            
            # Compute KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            results['kl_divergence'] = float(kl_div)
            
            # Compute clustering metrics if available
            if clustering_results is not None:
                results['clustering'] = self.compute_clustering_metrics(
                    clustering_results['latent_vectors'],
                    clustering_results['labels']
                )
            
            # Compute anomaly metrics
            reconstruction_errors = torch.mean((validation_data - reconstructed) ** 2, dim=1)
            threshold = torch.mean(reconstruction_errors) + 3 * torch.std(reconstruction_errors)
            
            results['anomaly'] = self.compute_anomaly_metrics(
                reconstruction_errors.cpu().numpy(),
                threshold.cpu().numpy()
            )
            
        return results
        
    def plot_validation_results(self,
                              results: Dict,
                              save_prefix: str):
        """Generate validation plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn')
        
        # 1. Reconstruction Error Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(
            results['anomaly']['reconstruction_errors'],
            kde=True
        )
        plt.axvline(
            results['anomaly']['error_threshold'],
            color='r',
            linestyle='--',
            label='Anomaly Threshold'
        )
        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(
            self.results_dir / f"{save_prefix}_error_dist_{timestamp}.png"
        )
        plt.close()
        
        # 2. Clustering Results (if available)
        if 'clustering' in results:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.scatterplot(
                data=results['clustering']['latent_vectors'][:, :2],
                hue=results['clustering']['labels'],
                alpha=0.6
            )
            plt.title('Latent Space Clustering')
            
            plt.subplot(1, 2, 2)
            sns.histplot(
                results['clustering']['labels'],
                discrete=True
            )
            plt.title('Cluster Size Distribution')
            
            plt.tight_layout()
            plt.savefig(
                self.results_dir / f"{save_prefix}_clustering_{timestamp}.png"
            )
            plt.close()
            
    def save_validation_results(self,
                              results: Dict,
                              save_prefix: str):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_path = self.results_dir / f"{save_prefix}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Generate summary report
        report = self._generate_report(results)
        report_path = self.results_dir / f"{save_prefix}_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Saved validation results to {self.results_dir}")
        
    def _generate_report(self, results: Dict) -> str:
        """Generate human-readable report"""
        lines = ["Model Validation Report", "=" * 22, ""]
        
        # Reconstruction metrics
        lines.extend([
            "Reconstruction Metrics:",
            "-" * 21,
            f"MSE: {results['reconstruction']['mse']:.6f}",
            f"MAE: {results['reconstruction']['mae']:.6f}",
            f"RMSE: {results['reconstruction']['rmse']:.6f}",
            f"Correlation: {results['reconstruction']['correlation']:.6f}",
            ""
        ])
        
        # Clustering metrics (if available)
        if 'clustering' in results:
            lines.extend([
                "Clustering Metrics:",
                "-" * 18,
                f"Number of Clusters: {results['clustering']['num_clusters']}",
                f"Silhouette Score: {results['clustering']['silhouette_score']:.6f}",
                f"Average Cluster Size: {results['clustering']['avg_cluster_size']:.2f}",
                ""
            ])
            
        # Anomaly metrics
        lines.extend([
            "Anomaly Detection Metrics:",
            "-" * 24,
            f"Anomaly Ratio: {results['anomaly']['anomaly_ratio']:.4f}",
            f"Number of Anomalies: {results['anomaly']['num_anomalies']}",
            f"Error Threshold: {results['anomaly']['error_threshold']:.6f}",
            ""
        ])
        
        return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Validation")
    parser.add_argument('--results_dir', type=str, required=True,
                       help="Directory to save results")
    parser.add_argument('--model_path', type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, required=True,
                       help="Path to validation data")
    
    args = parser.parse_args()
    
    validator = ModelValidator(args.results_dir)
    
    # Example usage (implement data and model loading according to your needs)
    # model = load_model(args.model_path)
    # data = load_data(args.data_path)
    # results = validator.validate_model(model, data)
    # validator.save_validation_results(results, "validation")
