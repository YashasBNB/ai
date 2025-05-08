import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional

from logging_config import setup_logging
from trading_config import TradingConfig, get_default_config

class TradingDashboard:
    def __init__(self, results_dir: str = "results"):
        """Initialize dashboard"""
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
            html.Div([
                html.H1("Trading System Dashboard",
                        style={'textAlign': 'center'}),
                html.Div([
                    html.Span("Last Updated: ",
                             style={'fontWeight': 'bold'}),
                    html.Span(id='last-update-time')
                ], style={'textAlign': 'right'})
            ], style={'marginBottom': 20}),
            
            # Trading Summary
            html.Div([
                html.H3("Trading Summary"),
                html.Div([
                    html.Div([
                        html.H4("Daily Performance"),
                        html.Div(id='daily-stats')
                    ], className='summary-box'),
                    html.Div([
                        html.H4("Overall Metrics"),
                        html.Div(id='overall-metrics')
                    ], className='summary-box')
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'marginBottom': 30}),
            
            # Active Trades
            html.Div([
                html.H3("Active Trades"),
                html.Div(id='active-trades-table')
            ], style={'marginBottom': 30}),
            
            # Trading Charts
            html.Div([
                html.H3("Performance Charts"),
                dcc.Tabs([
                    dcc.Tab(label='Profit/Loss', children=[
                        dcc.Graph(id='pnl-chart')
                    ]),
                    dcc.Tab(label='Win Rate', children=[
                        dcc.Graph(id='winrate-chart')
                    ]),
                    dcc.Tab(label='Risk Analysis', children=[
                        dcc.Graph(id='risk-chart')
                    ])
                ])
            ], style={'marginBottom': 30}),
            
            # Pattern Analysis
            html.Div([
                html.H3("Pattern Analysis"),
                dcc.Dropdown(
                    id='pattern-timeframe',
                    options=[
                        {'label': '15 Minutes', 'value': 'M15'},
                        {'label': '1 Hour', 'value': 'H1'}
                    ],
                    value='M15'
                ),
                dcc.Graph(id='pattern-distribution')
            ], style={'marginBottom': 30}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ])
        
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('last-update-time', 'children'),
             Output('daily-stats', 'children'),
             Output('overall-metrics', 'children'),
             Output('active-trades-table', 'children'),
             Output('pnl-chart', 'figure'),
             Output('winrate-chart', 'figure'),
             Output('risk-chart', 'figure'),
             Output('pattern-distribution', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('pattern-timeframe', 'value')]
        )
        def update_dashboard(n_intervals, timeframe):
            try:
                # Load latest trading data
                trading_data = self._load_trading_data()
                
                # Update components
                update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                daily_stats = self._create_daily_stats(trading_data)
                overall_metrics = self._create_overall_metrics(trading_data)
                active_trades = self._create_trades_table(trading_data)
                pnl_fig = self._create_pnl_chart(trading_data)
                winrate_fig = self._create_winrate_chart(trading_data)
                risk_fig = self._create_risk_chart(trading_data)
                pattern_fig = self._create_pattern_chart(timeframe)
                
                return (
                    update_time,
                    daily_stats,
                    overall_metrics,
                    active_trades,
                    pnl_fig,
                    winrate_fig,
                    risk_fig,
                    pattern_fig
                )
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {str(e)}")
                return self._create_error_components()
                
    def _load_trading_data(self) -> Dict:
        """Load latest trading data"""
        try:
            # Find latest report
            reports = list(self.results_dir.glob("report_*.json"))
            if not reports:
                return {}
                
            latest_report = max(reports, key=lambda x: x.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading trading data: {str(e)}")
            return {}
            
    def _create_daily_stats(self, data: Dict) -> html.Div:
        """Create daily statistics component"""
        if not data or 'daily_stats' not in data:
            return html.Div("No data available")
            
        stats = data['daily_stats']
        return html.Div([
            html.P(f"Trades Taken: {stats['trades_taken']}"),
            html.P(f"Winning Trades: {stats['winning_trades']}"),
            html.P(f"Total Profit: ${stats['total_profit']:.2f}"),
            html.P(f"Risk Taken: {stats['risk_taken']:.1f}%")
        ])
        
    def _create_overall_metrics(self, data: Dict) -> html.Div:
        """Create overall metrics component"""
        if not data or 'performance_metrics' not in data:
            return html.Div("No data available")
            
        metrics = data['performance_metrics']
        return html.Div([
            html.P(f"Win Rate: {metrics['win_rate']*100:.1f}%"),
            html.P(f"Profit Factor: {metrics['profit_factor']:.2f}"),
            html.P(f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}"),
            html.P(f"Max Drawdown: ${abs(metrics['max_drawdown']):.2f}")
        ])
        
    def _create_trades_table(self, data: Dict) -> html.Table:
        """Create active trades table"""
        if not data or 'active_trades' not in data:
            return html.Div("No active trades")
            
        trades = data['active_trades']
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Symbol"),
                    html.Th("Direction"),
                    html.Th("Entry Price"),
                    html.Th("Current Price"),
                    html.Th("Profit/Loss"),
                    html.Th("Duration")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(trade['symbol']),
                    html.Td(trade['direction']),
                    html.Td(f"${trade['entry_price']:.5f}"),
                    html.Td(f"${trade['current_price']:.5f}"),
                    html.Td(
                        f"${trade['current_profit']:.2f}",
                        style={'color': 'green' if trade['current_profit'] > 0 else 'red'}
                    ),
                    html.Td(self._format_duration(trade['entry_time']))
                ]) for trade in trades
            ])
        ])
        
    def _create_pnl_chart(self, data: Dict) -> go.Figure:
        """Create profit/loss chart"""
        if not data or 'trade_history' not in data:
            return go.Figure()
            
        df = pd.DataFrame(data['trade_history'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_profit'],
            mode='lines',
            name='Cumulative P/L'
        ))
        
        fig.update_layout(
            title='Cumulative Profit/Loss',
            xaxis_title='Time',
            yaxis_title='Profit/Loss ($)'
        )
        
        return fig
        
    def _create_winrate_chart(self, data: Dict) -> go.Figure:
        """Create win rate chart"""
        if not data or 'trade_history' not in data:
            return go.Figure()
            
        df = pd.DataFrame(data['trade_history'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['is_win'] = df['profit'] > 0
        
        # Calculate rolling win rate
        window = 20
        df['win_rate'] = df['is_win'].rolling(window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['win_rate'],
            mode='lines',
            name=f'{window}-Trade Win Rate'
        ))
        
        fig.update_layout(
            title='Rolling Win Rate',
            xaxis_title='Time',
            yaxis_title='Win Rate',
            yaxis_tickformat=',.0%'
        )
        
        return fig
        
    def _create_risk_chart(self, data: Dict) -> go.Figure:
        """Create risk analysis chart"""
        if not data or 'daily_stats' not in data:
            return go.Figure()
            
        risk_data = data['daily_stats']
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=risk_data['risk_taken'],
            title={'text': "Daily Risk Usage"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        return fig
        
    def _create_pattern_chart(self, timeframe: str) -> go.Figure:
        """Create pattern distribution chart"""
        try:
            # Load pattern data
            pattern_file = self.results_dir / f"patterns_{timeframe}_latest.json"
            if not pattern_file.exists():
                return go.Figure()
                
            with open(pattern_file, 'r') as f:
                patterns = json.load(f)
                
            # Create distribution plot
            df = pd.DataFrame(patterns['clusters'])
            
            fig = px.histogram(
                df,
                title=f'Pattern Distribution ({timeframe})',
                labels={'value': 'Pattern ID', 'count': 'Frequency'},
                nbins=30
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating pattern chart: {str(e)}")
            return go.Figure()
            
    def _format_duration(self, entry_time: str) -> str:
        """Format trade duration"""
        entry = datetime.fromisoformat(entry_time)
        duration = datetime.now() - entry
        minutes = duration.total_seconds() / 60
        
        if minutes < 60:
            return f"{int(minutes)}m"
        else:
            hours = minutes / 60
            return f"{int(hours)}h {int(minutes%60)}m"
            
    def _create_error_components(self):
        """Create error components when data loading fails"""
        error_div = html.Div("Error loading data")
        error_fig = go.Figure()
        
        return (
            "Error",
            error_div,
            error_div,
            error_div,
            error_fig,
            error_fig,
            error_fig,
            error_fig
        )
        
    def run_server(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = False):
        """Run the dashboard server"""
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Dashboard")
    parser.add_argument('--results-dir', type=str, default="results",
                       help="Results directory")
    parser.add_argument('--port', type=int, default=8050,
                       help="Port to run the server on")
    parser.add_argument('--debug', action='store_true',
                       help="Run in debug mode")
    
    args = parser.parse_args()
    
    dashboard = TradingDashboard(args.results_dir)
    dashboard.run_server(port=args.port, debug=args.debug)
