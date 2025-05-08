import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsExporter:
    def __init__(self, results_dir: str = "results"):
        """Initialize ResultsExporter with output directory"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def export_patterns(self, 
                       patterns: Dict,
                       timestamps: pd.DatetimeIndex,
                       symbol: str,
                       timeframe: str):
        """Export pattern detection results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results dictionary
        results = {
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': timestamp,
                'num_patterns': len(np.unique(patterns['clusters'])),
                'num_anomalies': np.sum(patterns['anomalies'])
            },
            'patterns': {
                'clusters': patterns['clusters'].tolist(),
                'reconstruction_errors': patterns['reconstruction_errors'].tolist(),
                'anomalies': patterns['anomalies'].tolist()
            }
        }
        
        # Save JSON results
        json_path = self.results_dir / f"patterns_{symbol}_{timeframe}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        # Save CSV results
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cluster': patterns['clusters'],
            'reconstruction_error': patterns['reconstruction_errors'],
            'is_anomaly': patterns['anomalies']
        })
        csv_path = self.results_dir / f"patterns_{symbol}_{timeframe}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Exported pattern results to {json_path} and {csv_path}")
        
        # Generate visualizations
        self._create_pattern_visualizations(patterns, timestamps, symbol, timeframe, timestamp)
        
    def export_analysis(self,
                       analysis_results: Dict,
                       symbol: str,
                       timeframe: str):
        """Export analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis results
        json_path = self.results_dir / f"analysis_{symbol}_{timeframe}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=4)
            
        self.logger.info(f"Exported analysis results to {json_path}")
        
    def export_metrics(self,
                      metrics: Dict[str, Union[float, List[float]]],
                      symbol: str,
                      timeframe: str):
        """Export performance metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        json_path = self.results_dir / f"metrics_{symbol}_{timeframe}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Generate metrics report
        report = self._generate_metrics_report(metrics)
        report_path = self.results_dir / f"report_{symbol}_{timeframe}_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Exported metrics to {json_path} and {report_path}")
        
    def _create_pattern_visualizations(self,
                                     patterns: Dict,
                                     timestamps: pd.DatetimeIndex,
                                     symbol: str,
                                     timeframe: str,
                                     timestamp: str):
        """Create visualizations for pattern analysis"""
        # Set style
        plt.style.use('seaborn')
        
        # 1. Pattern Timeline
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.scatter(timestamps, patterns['clusters'], c=patterns['clusters'], 
                   cmap='tab20', alpha=0.6)
        plt.title(f'Pattern Timeline - {symbol} {timeframe}')
        plt.xlabel('Time')
        plt.ylabel('Cluster')
        
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, patterns['reconstruction_errors'], alpha=0.6)
        plt.axhline(y=np.mean(patterns['reconstruction_errors']) + 
                   3 * np.std(patterns['reconstruction_errors']),
                   color='r', linestyle='--', label='Anomaly Threshold')
        plt.title('Reconstruction Errors')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"patterns_timeline_{symbol}_{timeframe}_{timestamp}.png")
        plt.close()
        
        # 2. Pattern Distribution
        plt.figure(figsize=(12, 6))
        unique_clusters = np.unique(patterns['clusters'])
        cluster_sizes = [np.sum(patterns['clusters'] == c) for c in unique_clusters]
        
        plt.bar(unique_clusters, cluster_sizes)
        plt.title(f'Pattern Distribution - {symbol} {timeframe}')
        plt.xlabel('Pattern ID')
        plt.ylabel('Count')
        
        plt.savefig(self.results_dir / f"pattern_distribution_{symbol}_{timeframe}_{timestamp}.png")
        plt.close()
        
        # 3. Anomaly Detection
        plt.figure(figsize=(12, 6))
        sns.kdeplot(data=patterns['reconstruction_errors'], fill=True)
        plt.axvline(x=np.mean(patterns['reconstruction_errors']) + 
                   3 * np.std(patterns['reconstruction_errors']),
                   color='r', linestyle='--', label='Anomaly Threshold')
        plt.title(f'Reconstruction Error Distribution - {symbol} {timeframe}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.legend()
        
        plt.savefig(self.results_dir / f"anomaly_distribution_{symbol}_{timeframe}_{timestamp}.png")
        plt.close()
        
    def _generate_metrics_report(self, metrics: Dict) -> str:
        """Generate a formatted metrics report"""
        report = ["Performance Metrics Report", "=" * 25, ""]
        
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                report.append(f"{metric_name}:")
                report.append(f"  Mean: {np.mean(value):.4f}")
                report.append(f"  Std:  {np.std(value):.4f}")
                report.append(f"  Min:  {np.min(value):.4f}")
                report.append(f"  Max:  {np.max(value):.4f}")
            else:
                report.append(f"{metric_name}: {value:.4f}")
            report.append("")
            
        return "\n".join(report)

def export_all_results(results_dir: str,
                      patterns: Dict,
                      analysis: Dict,
                      metrics: Dict,
                      timestamps: pd.DatetimeIndex,
                      symbol: str,
                      timeframe: str):
    """Convenience function to export all results at once"""
    exporter = ResultsExporter(results_dir)
    
    exporter.export_patterns(patterns, timestamps, symbol, timeframe)
    exporter.export_analysis(analysis, symbol, timeframe)
    exporter.export_metrics(metrics, symbol, timeframe)
    
    return exporter

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Analysis Results")
    parser.add_argument('--results_dir', type=str, required=True,
                       help="Directory to save results")
    parser.add_argument('--patterns_file', type=str, required=True,
                       help="Path to patterns JSON file")
    parser.add_argument('--symbol', type=str, required=True,
                       help="Trading symbol")
    parser.add_argument('--timeframe', type=str, required=True,
                       help="Trading timeframe")
    
    args = parser.parse_args()
    
    # Load patterns
    with open(args.patterns_file, 'r') as f:
        patterns = json.load(f)
        
    # Create example timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=len(patterns['clusters']),
                             freq=args.timeframe)
        
    # Export results
    exporter = ResultsExporter(args.results_dir)
    exporter.export_patterns(patterns, timestamps, args.symbol, args.timeframe)
