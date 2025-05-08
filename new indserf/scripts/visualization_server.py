import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List

from logging_config import setup_logging

class VisualizationDashboard:
    def __init__(self, results_dir: str = "results"):
        """Initialize dashboard with results directory"""
        self.results_dir = Path(results_dir)
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.H1("Unsupervised Trading Pattern Analysis Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control Panel
            html.Div([
                html.H3("Controls"),
                
                # Date Range Selector
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=(datetime.now() - timedelta(days=30)).date(),
                    end_date=datetime.now().date()
                ),
                
                # Symbol Selector
                html.Label("Symbol:"),
                dcc.Dropdown(
                    id='symbol-selector',
                    options=[],  # Will be populated dynamically
                    value=None
                ),
                
                # Pattern Type Selector
                html.Label("Pattern Type:"),
                dcc.Dropdown(
                    id='pattern-type',
                    options=[
                        {'label': 'All Patterns', 'value': 'all'},
                        {'label': 'Anomalies', 'value': 'anomalies'},
                        {'label': 'Clusters', 'value': 'clusters'}
                    ],
                    value='all'
                ),
                
                # Update Button
                html.Button('Update', id='update-button', n_clicks=0)
            ], style={'padding': 20, 'backgroundColor': '#f9f9f9'}),
            
            # Main Content
            html.Div([
                # Pattern Timeline
                html.Div([
                    html.H3("Pattern Timeline"),
                    dcc.Graph(id='pattern-timeline')
                ]),
                
                # Pattern Distribution
                html.Div([
                    html.H3("Pattern Distribution"),
                    dcc.Graph(id='pattern-distribution')
                ]),
                
                # Anomaly Detection
                html.Div([
                    html.H3("Anomaly Detection"),
                    dcc.Graph(id='anomaly-detection')
                ]),
                
                # Metrics Summary
                html.Div([
                    html.H3("Metrics Summary"),
                    html.Div(id='metrics-summary')
                ], style={'padding': 20})
            ])
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('symbol-selector', 'options'),
             Output('symbol-selector', 'value')],
            [Input('update-button', 'n_clicks')]
        )
        def update_symbols(n_clicks):
            """Update available symbols"""
            try:
                symbols = self._get_available_symbols()
                options = [{'label': s, 'value': s} for s in symbols]
                default_value = symbols[0] if symbols else None
                return options, default_value
            except Exception as e:
                self.logger.error(f"Error updating symbols: {e}")
                return [], None
        
        @self.app.callback(
            [Output('pattern-timeline', 'figure'),
             Output('pattern-distribution', 'figure'),
             Output('anomaly-detection', 'figure'),
             Output('metrics-summary', 'children')],
            [Input('update-button', 'n_clicks')],
            [State('date-range', 'start_date'),
             State('date-range', 'end_date'),
             State('symbol-selector', 'value'),
             State('pattern-type', 'value')]
        )
        def update_visualizations(n_clicks, start_date, end_date, symbol, pattern_type):
            """Update all visualizations"""
            try:
                # Load data
                data = self._load_results(symbol, start_date, end_date)
                if data is None:
                    return self._empty_figures()
                
                # Create visualizations
                timeline_fig = self._create_timeline(data, pattern_type)
                dist_fig = self._create_distribution(data, pattern_type)
                anomaly_fig = self._create_anomaly_plot(data)
                metrics = self._create_metrics_summary(data)
                
                return timeline_fig, dist_fig, anomaly_fig, metrics
                
            except Exception as e:
                self.logger.error(f"Error updating visualizations: {e}")
                return self._empty_figures()
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols from results"""
        pattern_files = list(self.results_dir.glob("patterns_*.json"))
        symbols = set()
        for f in pattern_files:
            # Extract symbol from filename
            parts = f.stem.split('_')
            if len(parts) > 1:
                symbols.add(parts[1])
        return sorted(list(symbols))
    
    def _load_results(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load and process results data"""
        try:
            # Find latest results file for symbol
            pattern_files = list(self.results_dir.glob(f"patterns_{symbol}_*.json"))
            if not pattern_files:
                return None
                
            latest_file = max(pattern_files, key=lambda x: x.stat().st_mtime)
            
            # Load data
            with open(latest_file, 'r') as f:
                data = json.load(f)
                
            # Convert to DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['metadata']['timestamps']),
                'cluster': data['patterns']['clusters'],
                'reconstruction_error': data['patterns']['reconstruction_errors'],
                'is_anomaly': data['patterns']['anomalies']
            })
            
            # Filter by date range
            mask = (df['timestamp'] >= pd.to_datetime(start_date)) & \
                   (df['timestamp'] <= pd.to_datetime(end_date))
            return df[mask]
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return None
    
    def _create_timeline(self, data: pd.DataFrame, pattern_type: str) -> go.Figure:
        """Create pattern timeline visualization"""
        fig = go.Figure()
        
        if pattern_type in ['all', 'clusters']:
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['cluster'],
                mode='markers',
                name='Clusters',
                marker=dict(
                    size=8,
                    color=data['cluster'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
        if pattern_type in ['all', 'anomalies']:
            anomalies = data[data['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['reconstruction_error'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
            
        fig.update_layout(
            title='Pattern Timeline',
            xaxis_title='Time',
            yaxis_title='Pattern ID',
            showlegend=True
        )
        
        return fig
    
    def _create_distribution(self, data: pd.DataFrame, pattern_type: str) -> go.Figure:
        """Create pattern distribution visualization"""
        if pattern_type in ['all', 'clusters']:
            fig = px.histogram(
                data,
                x='cluster',
                title='Pattern Distribution',
                labels={'cluster': 'Pattern ID', 'count': 'Frequency'},
                nbins=30
            )
        else:
            fig = px.histogram(
                data,
                x='reconstruction_error',
                title='Reconstruction Error Distribution',
                labels={'reconstruction_error': 'Error', 'count': 'Frequency'},
                nbins=30
            )
            
        return fig
    
    def _create_anomaly_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create anomaly detection visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['reconstruction_error'],
            mode='lines',
            name='Reconstruction Error'
        ))
        
        threshold = np.mean(data['reconstruction_error']) + \
                   3 * np.std(data['reconstruction_error'])
        
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Anomaly Threshold"
        )
        
        fig.update_layout(
            title='Anomaly Detection',
            xaxis_title='Time',
            yaxis_title='Reconstruction Error'
        )
        
        return fig
    
    def _create_metrics_summary(self, data: pd.DataFrame) -> html.Div:
        """Create metrics summary"""
        metrics = [
            f"Total Patterns: {len(np.unique(data['cluster']))}",
            f"Total Anomalies: {sum(data['is_anomaly'])}",
            f"Average Reconstruction Error: {data['reconstruction_error'].mean():.4f}",
            f"Pattern Distribution Entropy: {self._calculate_entropy(data['cluster']):.4f}"
        ]
        
        return html.Div([html.P(m) for m in metrics])
    
    def _empty_figures(self):
        """Return empty figures when data is not available"""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title='No Data Available',
            annotations=[{
                'text': 'No data available for selected parameters',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
        
        empty_metrics = html.Div([html.P("No metrics available")])
        
        return empty_fig, empty_fig, empty_fig, empty_metrics
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of pattern distribution"""
        counts = series.value_counts(normalize=True)
        return -sum(counts * np.log2(counts))
    
    def run_server(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = False):
        """Run the dashboard server"""
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualization Dashboard")
    parser.add_argument('--results_dir', type=str, default="results",
                       help="Directory containing results")
    parser.add_argument('--port', type=int, default=8050,
                       help="Port to run the server on")
    parser.add_argument('--debug', action='store_true',
                       help="Run in debug mode")
    
    args = parser.parse_args()
    
    dashboard = VisualizationDashboard(args.results_dir)
    dashboard.run_server(port=args.port, debug=args.debug)
