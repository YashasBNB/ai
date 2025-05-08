#!/usr/bin/env python3
import asyncio
import click
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import signal
import sys
from dotenv import load_dotenv

from scripts.deriv_trader import DerivTrader
from scripts.trade_monitor import TradeMonitor
from scripts.advanced_pattern_learner import PatternLearner
from scripts.model_manager import ModelManager
from trading_config import TradingConfig, get_default_config
from logging_config import setup_logging

class TradingSystem:
    def __init__(self,
                 config_path: Optional[str] = None,
                 model_dir: str = "models",
                 results_dir: str = "results"):
        """Initialize trading system"""
        # Load configuration
        self.config = (TradingConfig.load(config_path) if config_path 
                      else get_default_config())
        
        # Setup directories
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        for directory in [self.model_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize components
        self.model_manager = ModelManager(model_dir)
        self.traders: Dict[str, DerivTrader] = {}
        self.monitor = TradeMonitor(config_path, str(results_dir / "trades"))
        
        # Trading state
        self.is_running = False
        self.tasks = []
        
    async def start_trading(self):
        """Start trading system"""
        try:
            self.logger.info("Starting trading system...")
            self.is_running = True
            
            # Setup signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._signal_handler)
                
            # Initialize traders for each timeframe
            api_token = os.getenv('DERIV_API_TOKEN')
            if not api_token:
                raise ValueError("DERIV_API_TOKEN not found in environment")
                
            for timeframe in self.config.time.enabled_timeframes:
                trader = DerivTrader(
                    api_token=api_token,
                    model_dir=str(self.model_dir),
                    timeframe=timeframe,
                    risk_percent=self.config.risk.max_risk_percent,
                    max_concurrent_trades=self.config.risk.max_concurrent_trades
                )
                self.traders[timeframe] = trader
                
                # Start trading task
                task = asyncio.create_task(
                    trader.run(self.config.symbols)
                )
                self.tasks.append(task)
                
            # Start monitoring task
            monitor_task = asyncio.create_task(
                self._monitor_trading()
            )
            self.tasks.append(monitor_task)
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            self.logger.error(f"Error starting trading system: {str(e)}")
            await self.stop_trading()
            
    async def stop_trading(self):
        """Stop trading system"""
        self.logger.info("Stopping trading system...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Close all traders
        for trader in self.traders.values():
            if trader.websocket:
                await trader.websocket.close()
                
        # Generate final report
        report = self.monitor.generate_report()
        self.logger.info("Final trading report generated")
        
        self.logger.info("Trading system stopped")
        
    async def _monitor_trading(self):
        """Monitor trading activities"""
        while self.is_running:
            try:
                # Check daily limits
                daily_stats = self.monitor.daily_stats
                if daily_stats['risk_taken'] >= self.config.risk.max_daily_risk:
                    self.logger.warning("Daily risk limit reached")
                    await self.stop_trading()
                    break
                    
                if daily_stats['trades_taken'] >= self.config.risk.max_daily_trades:
                    self.logger.warning("Daily trade limit reached")
                    await self.stop_trading()
                    break
                    
                # Generate periodic report
                report = self.monitor.generate_report()
                self._log_performance(report)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in trading monitor: {str(e)}")
                
    def _log_performance(self, report: Dict):
        """Log trading performance metrics"""
        metrics = report['performance_metrics']
        self.logger.info(
            f"Trading Performance - "
            f"Win Rate: {metrics['win_rate']:.2%}, "
            f"Profit Factor: {metrics['profit_factor']:.2f}, "
            f"Total Profit: {metrics['total_profit']:.2f}"
        )
        
    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        asyncio.create_task(self.stop_trading())

@click.group()
def cli():
    """Deriv Trading System CLI"""
    pass

@cli.command()
@click.option('--config', type=str, help='Path to configuration file')
@click.option('--model-dir', type=str, default='models', help='Model directory')
@click.option('--results-dir', type=str, default='results', help='Results directory')
def start(config, model_dir, results_dir):
    """Start trading system"""
    # Load environment variables
    load_dotenv()
    
    # Initialize and run system
    system = TradingSystem(config, model_dir, results_dir)
    
    try:
        asyncio.run(system.start_trading())
    except KeyboardInterrupt:
        print("\nShutting down...")
        asyncio.run(system.stop_trading())

@cli.command()
@click.option('--results-dir', type=str, default='results', help='Results directory')
def report(results_dir):
    """Generate trading report"""
    monitor = TradeMonitor(results_dir=results_dir)
    report = monitor.generate_report()
    click.echo(json.dumps(report, indent=2))

@cli.command()
@click.option('--timeframe', type=str, required=True, help='Trading timeframe (M15/H1)')
@click.option('--model-dir', type=str, default='models', help='Model directory')
def validate(timeframe, model_dir):
    """Validate model for timeframe"""
    model_manager = ModelManager(model_dir)
    
    try:
        # Load model
        model = PatternLearner(input_dim=None)  # Set appropriate input_dim
        epoch, metrics = model_manager.load_checkpoint(
            model,
            model_name=f"pattern_learner_{timeframe}"
        )
        
        click.echo(f"Model loaded from epoch {epoch}")
        click.echo("Recent metrics:")
        click.echo(json.dumps(metrics, indent=2))
        
    except Exception as e:
        click.echo(f"Error validating model: {str(e)}")

@cli.command()
def setup():
    """Setup trading environment"""
    try:
        # Create default config
        config = get_default_config()
        config.save("trading_config.json")
        click.echo("Created default configuration")
        
        # Create directories
        for directory in ['models', 'results', 'logs', 'data']:
            os.makedirs(directory, exist_ok=True)
            click.echo(f"Created directory: {directory}")
            
        # Create .env template
        env_template = """
# Deriv API Configuration
DERIV_API_TOKEN=your_api_token_here

# Trading Configuration
DEMO_MODE=true
MAX_RISK_PERCENT=1.0
        """.strip()
        
        with open(".env.example", "w") as f:
            f.write(env_template)
            
        click.echo("Created .env.example template")
        click.echo("\nSetup complete! Please:")
        click.echo("1. Copy .env.example to .env")
        click.echo("2. Update .env with your Deriv API token")
        click.echo("3. Review trading_config.json")
        
    except Exception as e:
        click.echo(f"Error during setup: {str(e)}")

@cli.command()
@click.option('--days', type=int, default=30, help='Days of history to clean')
def cleanup(days):
    """Clean up old files"""
    try:
        # Clean up models
        model_manager = ModelManager("models")
        model_manager.cleanup_old_models(days)
        click.echo("Cleaned up old models")
        
        # Clean up results
        results_dir = Path("results")
        current_time = datetime.now().timestamp()
        
        for file in results_dir.glob("**/*"):
            if file.is_file():
                file_time = os.path.getctime(file)
                if (current_time - file_time) > (days * 24 * 3600):
                    os.remove(file)
                    click.echo(f"Removed old file: {file}")
                    
    except Exception as e:
        click.echo(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    cli()
