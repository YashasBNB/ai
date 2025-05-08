#!/usr/bin/env python3
import asyncio
import click
import logging
import json
from pathlib import Path
from datetime import datetime
import sys
import torch
import numpy as np
from typing import Dict, List, Optional

from scripts.train_pipeline import TrainingPipeline
from scripts.model_analyzer import ModelAnalyzer
from scripts.data_validator import DataValidator
from scripts.deriv_trader import DerivTrader
from trading_config import TradingConfig, get_default_config
from logging_config import setup_logging

logger = setup_logging().get_logger(__name__)

def validate_environment():
    """Validate execution environment"""
    requirements = {
        "cuda": torch.cuda.is_available(),
        "python_version": sys.version_info >= (3, 7),
        "directories": ["models", "results", "data"]
    }
    
    for directory in requirements["directories"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
    return requirements

@click.group()
def cli():
    """Unsupervised Trading Model Training CLI"""
    pass

@cli.command()
@click.option('--data-dir', required=True, help='Historical data directory')
@click.option('--model-dir', default='models', help='Model directory')
@click.option('--results-dir', default='results', help='Results directory')
@click.option('--config', help='Trading configuration file')
@click.option('--force/--no-force', default=False, help='Force retrain existing models')
def train(data_dir: str,
          model_dir: str,
          results_dir: str,
          config: Optional[str],
          force: bool):
    """Train models on historical data"""
    try:
        # Validate environment
        requirements = validate_environment()
        device = "cuda" if requirements["cuda"] else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize pipeline
        pipeline = TrainingPipeline(
            data_dir=data_dir,
            model_dir=model_dir,
            results_dir=results_dir,
            config_path=config
        )
        
        # Check existing models
        if not force:
            existing = pipeline.validate_models()
            if any(info.get("model_files", 0) > 0 for info in existing.values()):
                if not click.confirm("Models already exist. Do you want to retrain?"):
                    logger.info("Training cancelled")
                    return
                    
        # Run training pipeline
        success = asyncio.run(pipeline.run_pipeline())
        
        if success:
            click.echo("Training completed successfully!")
        else:
            click.echo("Training failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--model-dir', required=True, help='Model directory')
@click.option('--results-dir', default='results/analysis', help='Analysis results directory')
@click.option('--timeframes', multiple=True, required=True, help='Timeframes to analyze')
@click.option('--report/--no-report', default=True, help='Generate analysis report')
def analyze(model_dir: str,
           results_dir: str,
           timeframes: List[str],
           report: bool):
    """Analyze trained models"""
    try:
        analyzer = ModelAnalyzer(model_dir, results_dir)
        
        if report:
            # Generate comprehensive report
            analysis_report = analyzer.generate_report(timeframes)
            
            # Save report
            report_path = Path(results_dir) / f"analysis_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(report_path, 'w') as f:
                json.dump(analysis_report, f, indent=4)
                
            click.echo(f"Analysis report saved to {report_path}")
            
        else:
            # Analyze each timeframe
            for timeframe in timeframes:
                # Load test data (you should modify this to load your actual test data)
                test_data = np.random.randn(1000, 10)
                results = analyzer.analyze_model(timeframe, test_data)
                
                click.echo(f"\nResults for {timeframe}:")
                click.echo(json.dumps(results, indent=2))
                
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--data-dir', required=True, help='Data directory to validate')
@click.option('--report-file', default='validation_report.json', help='Output report file')
def validate(data_dir: str, report_file: str):
    """Validate historical data"""
    try:
        validator = DataValidator(data_dir)
        
        # Run validation
        report = validator.save_validation_report(report_file)
        
        # Print summary
        click.echo("\nValidation Summary:")
        click.echo(f"Total files: {report['total_files']}")
        click.echo(f"Valid files: {report['valid_files']}")
        
        if report['valid_files'] > 0:
            click.echo("\nPreparing data...")
            prepared_data = validator.prepare_data()
            click.echo(f"Successfully prepared {len(prepared_data)} datasets")
            
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--model-dir', required=True, help='Model directory')
@click.option('--timeframe', required=True, help='Trading timeframe')
@click.option('--symbols', multiple=True, required=True, help='Trading symbols')
@click.option('--demo/--live', default=True, help='Use demo or live account')
def deploy(model_dir: str,
          timeframe: str,
          symbols: List[str],
          demo: bool):
    """Deploy model for trading"""
    try:
        # Load configuration
        config = get_default_config()
        
        # Validate model
        pipeline = TrainingPipeline(model_dir=model_dir)
        model_status = pipeline.validate_models()
        
        if timeframe not in model_status or not model_status[timeframe].get("model_files"):
            raise ValueError(f"No valid model found for timeframe {timeframe}")
            
        # Initialize trader
        trader = DerivTrader(
            api_token=os.getenv('DERIV_API_TOKEN'),
            model_dir=model_dir,
            timeframe=timeframe,
            risk_percent=config.risk.max_risk_percent,
            max_concurrent_trades=config.risk.max_concurrent_trades
        )
        
        click.echo(f"Starting trading system for {timeframe}")
        click.echo(f"Trading symbols: {', '.join(symbols)}")
        click.echo(f"Mode: {'Demo' if demo else 'Live'}")
        
        if click.confirm("Do you want to proceed?"):
            # Run trading system
            asyncio.run(trader.run(symbols))
            
    except Exception as e:
        logger.error(f"Deployment error: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--config-template', is_flag=True, help='Create configuration template')
@click.option('--check-environment', is_flag=True, help='Check execution environment')
def setup(config_template: bool, check_environment: bool):
    """Setup trading environment"""
    try:
        if config_template:
            # Create default configuration
            config = get_default_config()
            config_path = "trading_config.json"
            
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=4)
                
            click.echo(f"Created configuration template: {config_path}")
            
        if check_environment:
            # Check environment
            requirements = validate_environment()
            
            click.echo("\nEnvironment Check:")
            click.echo(f"CUDA available: {requirements['cuda']}")
            click.echo(f"Python version: {sys.version}")
            click.echo("\nDirectories:")
            for directory in requirements["directories"]:
                click.echo(f"- {directory}: {'✓' if Path(directory).exists() else '✗'}")
                
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    cli()
