import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from typing import Dict, List, Tuple
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PatternAnalyzer:
    def __init__(self, results_dir: str = "results"):
        """
        Initialize PatternAnalyzer
        
        Args:
            results_dir (str): Directory to save analysis results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def analyze_patterns(self,
                        features: np.ndarray,
                        patterns: Dict,
                        feature_names: List[str],
                        timestamps: pd.DatetimeIndex = None) -> Dict:
        """
        Comprehensive pattern analysis
        
        Args:
            features: Original feature array
            patterns: Dictionary containing pattern detection results
            feature_names: List of feature names
            timestamps: Optional timestamp index for time-based analysis
            
        Returns:
            Dictionary containing analysis results
        """
        logging.info("Starting pattern analysis...")
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Cluster quality metrics
        results['metrics'] = self._compute_cluster_metrics(
            patterns['latent_pca'],
            patterns['clusters']
        )
        
        # Pattern characteristics
        results['characteristics'] = self._analyze_pattern_characteristics(
            features, patterns['clusters'], feature_names
        )
        
        # Temporal analysis if timestamps provided
        if timestamps is not None:
            results['temporal'] = self._analyze_temporal_patterns(
                patterns, timestamps
            )
        
        # Generate visualizations
        self._generate_visualizations(
            features, patterns, feature_names, timestamps, timestamp
        )
        
        # Save results
        self._save_results(results, timestamp)
        
        return results
    
    def _compute_cluster_metrics(self,
                               data: np.ndarray,
                               clusters: np.ndarray) -> Dict:
        """Compute clustering quality metrics"""
        metrics = {
            'silhouette_score': silhouette_score(data, clusters),
            'calinski_harabasz_score': calinski_harabasz_score(data, clusters),
            'num_clusters': len(np.unique(clusters[clusters != -1])),
            'noise_ratio': np.mean(clusters == -1)
        }
        return metrics
    
    def _analyze_pattern_characteristics(self,
                                      features: np.ndarray,
                                      clusters: np.ndarray,
                                      feature_names: List[str]) -> pd.DataFrame:
        """Analyze characteristics of each pattern cluster"""
        unique_clusters = np.unique(clusters)
        characteristics = []
        
        for cluster in unique_clusters:
            if cluster == -1:  # Skip noise points
                continue
                
            mask = clusters == cluster
            cluster_features = features[mask]
            
            # Basic statistics
            stats = {
                'cluster_id': cluster,
                'size': np.sum(mask),
                'density': np.sum(mask) / len(clusters)
            }
            
            # Feature-wise statistics
            for i, name in enumerate(feature_names):
                stats.update({
                    f'{name}_mean': np.mean(cluster_features[:, i]),
                    f'{name}_std': np.std(cluster_features[:, i]),
                    f'{name}_min': np.min(cluster_features[:, i]),
                    f'{name}_max': np.max(cluster_features[:, i])
                })
            
            characteristics.append(stats)
            
        return pd.DataFrame(characteristics)
    
    def _analyze_temporal_patterns(self,
                                 patterns: Dict,
                                 timestamps: pd.DatetimeIndex) -> Dict:
        """Analyze temporal aspects of patterns"""
        clusters = patterns['clusters']
        temporal = {
            'cluster_transitions': self._analyze_transitions(clusters),
            'temporal_distribution': self._analyze_distribution(clusters, timestamps)
        }
        return temporal
    
    def _analyze_transitions(self, clusters: np.ndarray) -> pd.DataFrame:
        """Analyze cluster transitions"""
        transitions = pd.DataFrame(0,
                                 index=np.unique(clusters),
                                 columns=np.unique(clusters))
        
        for i in range(len(clusters)-1):
            transitions.loc[clusters[i], clusters[i+1]] += 1
            
        return transitions
    
    def _analyze_distribution(self,
                            clusters: np.ndarray,
                            timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Analyze temporal distribution of clusters"""
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cluster': clusters
        })
        return df.groupby([
            pd.Grouper(key='timestamp', freq='D'),
            'cluster'
        ]).size().unstack(fill_value=0)
    
    def _generate_visualizations(self,
                               features: np.ndarray,
                               patterns: Dict,
                               feature_names: List[str],
                               timestamps: pd.DatetimeIndex,
                               timestamp: str):
        """Generate comprehensive visualizations"""
        # 1. Latent Space Visualization
        self._plot_latent_space(patterns, timestamp)
        
        # 2. Pattern Timeline
        if timestamps is not None:
            self._plot_pattern_timeline(patterns, timestamps, timestamp)
        
        # 3. Feature Importance
        self._plot_feature_importance(features, patterns, feature_names, timestamp)
        
        # 4. Cluster Characteristics
        self._plot_cluster_characteristics(patterns, timestamp)
        
        # 5. Interactive 3D Visualization
        self._create_interactive_viz(patterns, timestamp)
    
    def _plot_latent_space(self, patterns: Dict, timestamp: str):
        """Create latent space visualization"""
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
        plt.savefig(
            os.path.join(self.results_dir, f"latent_space_{timestamp}.png")
        )
        plt.close()
    
    def _plot_pattern_timeline(self,
                             patterns: Dict,
                             timestamps: pd.DatetimeIndex,
                             timestamp: str):
        """Create pattern timeline visualization"""
        fig = go.Figure()
        
        # Add cluster timeline
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=patterns['clusters'],
            mode='markers',
            marker=dict(
                size=8,
                color=patterns['clusters'],
                colorscale='Viridis',
                showscale=True
            ),
            name='Cluster'
        ))
        
        # Add reconstruction errors
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=patterns['reconstruction_errors'],
            name='Reconstruction Error',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Pattern Timeline with Reconstruction Errors',
            xaxis_title='Time',
            yaxis_title='Cluster',
            yaxis2=dict(
                title='Reconstruction Error',
                overlaying='y',
                side='right'
            )
        )
        
        fig.write_html(
            os.path.join(self.results_dir, f"pattern_timeline_{timestamp}.html")
        )
    
    def _plot_feature_importance(self,
                               features: np.ndarray,
                               patterns: Dict,
                               feature_names: List[str],
                               timestamp: str):
        """Create feature importance visualization"""
        pca = PCA()
        pca.fit(features)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        
        plt.subplot(1, 2, 2)
        importance = abs(pca.components_[0])
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, importance[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Feature Importance (First Component)')
        plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, f"feature_importance_{timestamp}.png")
        )
        plt.close()
    
    def _plot_cluster_characteristics(self, patterns: Dict, timestamp: str):
        """Create cluster characteristics visualization"""
        unique_clusters = np.unique(patterns['clusters'])
        cluster_sizes = [
            np.sum(patterns['clusters'] == c)
            for c in unique_clusters if c != -1
        ]
        
        fig = make_subplots(rows=2, cols=2)
        
        # Cluster size distribution
        fig.add_trace(
            go.Bar(x=unique_clusters[unique_clusters != -1],
                  y=cluster_sizes,
                  name='Cluster Sizes'),
            row=1, col=1
        )
        
        # Reconstruction error distribution
        fig.add_trace(
            go.Box(y=patterns['reconstruction_errors'],
                  x=patterns['clusters'],
                  name='Reconstruction Errors'),
            row=1, col=2
        )
        
        fig.update_layout(title='Cluster Characteristics')
        fig.write_html(
            os.path.join(self.results_dir,
                        f"cluster_characteristics_{timestamp}.html")
        )
    
    def _create_interactive_viz(self, patterns: Dict, timestamp: str):
        """Create interactive 3D visualization"""
        # Perform t-SNE for 3D visualization
        tsne = TSNE(n_components=3, random_state=42)
        tsne_results = tsne.fit_transform(patterns['latent_pca'])
        
        fig = go.Figure(data=[go.Scatter3d(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            z=tsne_results[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=patterns['clusters'],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title='3D Pattern Visualization',
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3'
            )
        )
        
        fig.write_html(
            os.path.join(self.results_dir, f"3d_visualization_{timestamp}.html")
        )
    
    def _save_results(self, results: Dict, timestamp: str):
        """Save analysis results"""
        # Save metrics
        pd.DataFrame([results['metrics']]).to_csv(
            os.path.join(self.results_dir, f"metrics_{timestamp}.csv")
        )
        
        # Save characteristics
        results['characteristics'].to_csv(
            os.path.join(self.results_dir, f"characteristics_{timestamp}.csv")
        )
        
        # Save temporal analysis if available
        if 'temporal' in results:
            results['temporal']['cluster_transitions'].to_csv(
                os.path.join(self.results_dir,
                            f"transitions_{timestamp}.csv")
            )
            results['temporal']['temporal_distribution'].to_csv(
                os.path.join(self.results_dir,
                            f"distribution_{timestamp}.csv")
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage will be implemented as part of the pipeline
