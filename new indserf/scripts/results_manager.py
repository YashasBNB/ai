import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import logging
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from logging_config import setup_logging

class ResultsManager:
    def __init__(self, results_dir: str = "results"):
        """Initialize ResultsManager"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging().get_logger(__name__)
        
    def list_results(self) -> Dict[str, List[str]]:
        """List all available results files by type"""
        results = {
            'patterns': [],
            'analysis': [],
            'metrics': [],
            'reports': []
        }
        
        for file in self.results_dir.glob("*"):
            if file.is_file():
                for category in results.keys():
                    if category in file.name:
                        results[category].append(str(file))
                        
        return results
        
    def load_results(self,
                    symbol: str,
                    timeframe: str,
                    result_type: str = 'patterns') -> Optional[Dict]:
        """Load specific results file"""
        try:
            # Find latest matching file
            pattern = f"{result_type}_{symbol}_{timeframe}_*.json"
            matching_files = list(self.results_dir.glob(pattern))
            
            if not matching_files:
                self.logger.warning(f"No matching results found for {pattern}")
                return None
                
            latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return None
            
    def analyze_patterns(self,
                        symbol: str,
                        timeframe: str) -> Dict:
        """Analyze pattern detection results"""
        results = self.load_results(symbol, timeframe, 'patterns')
        if not results:
            return {}
            
        analysis = {
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Pattern statistics
        clusters = np.array(results['patterns']['clusters'])
        unique_clusters = np.unique(clusters)
        
        analysis['pattern_stats'] = {
            'num_patterns': len(unique_clusters),
            'pattern_distribution': {
                str(c): int(np.sum(clusters == c))
                for c in unique_clusters
            },
            'entropy': float(stats.entropy(
                [np.sum(clusters == c) for c in unique_clusters]
            ))
        }
        
        # Anomaly statistics
        anomalies = np.array(results['patterns']['anomalies'])
        reconstruction_errors = np.array(results['patterns']['reconstruction_errors'])
        
        analysis['anomaly_stats'] = {
            'num_anomalies': int(np.sum(anomalies)),
            'anomaly_ratio': float(np.mean(anomalies)),
            'mean_error': float(np.mean(reconstruction_errors)),
            'std_error': float(np.std(reconstruction_errors)),
            'max_error': float(np.max(reconstruction_errors))
        }
        
        # Save analysis results
        self.save_analysis(analysis, symbol, timeframe)
        
        return analysis
        
    def save_analysis(self,
                     analysis: Dict,
                     symbol: str,
                     timeframe: str):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{symbol}_{timeframe}_{timestamp}.json"
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(analysis, f, indent=4)
            
    def generate_report(self,
                       symbol: str,
                       timeframe: str,
                       output_format: str = 'text') -> str:
        """Generate analysis report"""
        analysis = self.load_results(symbol, timeframe, 'analysis')
        if not analysis:
            return "No analysis results available"
            
        if output_format == 'text':
            return self._generate_text_report(analysis)
        elif output_format == 'html':
            return self._generate_html_report(analysis)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
    def plot_results(self,
                    symbol: str,
                    timeframe: str,
                    plot_type: str = 'all'):
        """Generate result visualizations"""
        results = self.load_results(symbol, timeframe, 'patterns')
        if not results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if plot_type in ['all', 'patterns']:
            self._plot_pattern_distribution(results, symbol, timeframe, timestamp)
            
        if plot_type in ['all', 'anomalies']:
            self._plot_anomaly_detection(results, symbol, timeframe, timestamp)
            
        if plot_type in ['all', 'errors']:
            self._plot_error_distribution(results, symbol, timeframe, timestamp)
            
    def _plot_pattern_distribution(self,
                                 results: Dict,
                                 symbol: str,
                                 timeframe: str,
                                 timestamp: str):
        """Plot pattern distribution"""
        plt.figure(figsize=(12, 6))
        
        clusters = np.array(results['patterns']['clusters'])
        unique_clusters = np.unique(clusters)
        counts = [np.sum(clusters == c) for c in unique_clusters]
        
        plt.bar(unique_clusters, counts)
        plt.title(f'Pattern Distribution - {symbol} {timeframe}')
        plt.xlabel('Pattern ID')
        plt.ylabel('Count')
        
        plt.savefig(
            self.results_dir / f"pattern_dist_{symbol}_{timeframe}_{timestamp}.png"
        )
        plt.close()
        
    def _plot_anomaly_detection(self,
                              results: Dict,
                              symbol: str,
                              timeframe: str,
                              timestamp: str):
        """Plot anomaly detection results"""
        plt.figure(figsize=(12, 6))
        
        errors = np.array(results['patterns']['reconstruction_errors'])
        anomalies = np.array(results['patterns']['anomalies'])
        
        plt.plot(errors, label='Reconstruction Error', alpha=0.6)
        plt.scatter(
            np.where(anomalies)[0],
            errors[anomalies],
            color='red',
            label='Anomalies',
            alpha=0.6
        )
        
        plt.title(f'Anomaly Detection - {symbol} {timeframe}')
        plt.xlabel('Time')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        
        plt.savefig(
            self.results_dir / f"anomalies_{symbol}_{timeframe}_{timestamp}.png"
        )
        plt.close()
        
    def _plot_error_distribution(self,
                               results: Dict,
                               symbol: str,
                               timeframe: str,
                               timestamp: str):
        """Plot error distribution"""
        plt.figure(figsize=(12, 6))
        
        errors = np.array(results['patterns']['reconstruction_errors'])
        
        sns.histplot(errors, kde=True)
        plt.axvline(
            np.mean(errors) + 3 * np.std(errors),
            color='r',
            linestyle='--',
            label='Anomaly Threshold'
        )
        
        plt.title(f'Error Distribution - {symbol} {timeframe}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.legend()
        
        plt.savefig(
            self.results_dir / f"error_dist_{symbol}_{timeframe}_{timestamp}.png"
        )
        plt.close()
        
    def _generate_text_report(self, analysis: Dict) -> str:
        """Generate text format report"""
        lines = [
            "Pattern Analysis Report",
            "=" * 21,
            "",
            f"Symbol: {analysis['metadata']['symbol']}",
            f"Timeframe: {analysis['metadata']['timeframe']}",
            f"Analysis Time: {analysis['metadata']['analysis_time']}",
            "",
            "Pattern Statistics",
            "-" * 17,
            f"Number of Patterns: {analysis['pattern_stats']['num_patterns']}",
            f"Pattern Entropy: {analysis['pattern_stats']['entropy']:.4f}",
            "",
            "Anomaly Statistics",
            "-" * 17,
            f"Number of Anomalies: {analysis['anomaly_stats']['num_anomalies']}",
            f"Anomaly Ratio: {analysis['anomaly_stats']['anomaly_ratio']:.4f}",
            f"Mean Error: {analysis['anomaly_stats']['mean_error']:.6f}",
            f"Max Error: {analysis['anomaly_stats']['max_error']:.6f}"
        ]
        
        return "\n".join(lines)
        
    def _generate_html_report(self, analysis: Dict) -> str:
        """Generate HTML format report"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .section {{ margin: 20px 0; }}
                .stat {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Pattern Analysis Report</h1>
            
            <div class="section">
                <h2>Metadata</h2>
                <div class="stat">Symbol: {analysis['metadata']['symbol']}</div>
                <div class="stat">Timeframe: {analysis['metadata']['timeframe']}</div>
                <div class="stat">Analysis Time: {analysis['metadata']['analysis_time']}</div>
            </div>
            
            <div class="section">
                <h2>Pattern Statistics</h2>
                <div class="stat">Number of Patterns: {analysis['pattern_stats']['num_patterns']}</div>
                <div class="stat">Pattern Entropy: {analysis['pattern_stats']['entropy']:.4f}</div>
            </div>
            
            <div class="section">
                <h2>Anomaly Statistics</h2>
                <div class="stat">Number of Anomalies: {analysis['anomaly_stats']['num_anomalies']}</div>
                <div class="stat">Anomaly Ratio: {analysis['anomaly_stats']['anomaly_ratio']:.4f}</div>
                <div class="stat">Mean Error: {analysis['anomaly_stats']['mean_error']:.6f}</div>
                <div class="stat">Max Error: {analysis['anomaly_stats']['max_error']:.6f}</div>
            </div>
        </body>
        </html>
        """
        return html

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Results Management")
    parser.add_argument('--results_dir', type=str, default="results",
                       help="Results directory")
    parser.add_argument('--symbol', type=str, required=True,
                       help="Trading symbol")
    parser.add_argument('--timeframe', type=str, required=True,
                       help="Trading timeframe")
    parser.add_argument('--action', choices=['analyze', 'report', 'plot'],
                       required=True, help="Action to perform")
    parser.add_argument('--format', choices=['text', 'html'],
                       default='text', help="Output format for report")
    
    args = parser.parse_args()
    
    manager = ResultsManager(args.results_dir)
    
    if args.action == 'analyze':
        manager.analyze_patterns(args.symbol, args.timeframe)
    elif args.action == 'report':
        report = manager.generate_report(args.symbol, args.timeframe, args.format)
        print(report)
    elif args.action == 'plot':
        manager.plot_results(args.symbol, args.timeframe)
