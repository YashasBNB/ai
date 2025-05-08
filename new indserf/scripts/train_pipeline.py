import asyncio
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json

from logging_config import setup_logging
from scripts.data_validator import DataValidator
from scripts.train_model import ModelTrainer
from scripts.model_manager import ModelManager
from trading_config import TradingConfig, get_default_config

class TrainingPipeline:
    def __init__(self,
                 data_dir: str,
                 model_dir: str = "models",
                 results_dir: str = "results",
                 config_path: Optional[str] = None):
        """
        Initialize training pipeline
        
        Args:
            data_dir: Directory containing historical data
            model_dir: Directory to save models
            results_dir: Directory to save results
            config_path: Path to trading configuration
        """
        # Setup directories
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        
        for directory in [self.model_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Load configuration
        self.config = (TradingConfig.load(config_path) if config_path 
                      else get_default_config())
        
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize components
        self.validator = DataValidator(data_dir)
        self.model_manager = ModelManager(str(model_dir))
        
        # Training parameters
        self.training_params = {
            'M15': {
                'window_size': 30,
                'batch_size': 64,
                'learning_rate': 0.001,
                'n_epochs': 100
            },
            'H1': {
                'window_size': 24,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'n_epochs': 150
            }
        }
        
    async def run_pipeline(self):
        """Run complete training pipeline"""
        try:
            self.logger.info("Starting training pipeline")
            
            # Step 1: Validate data
            validation_report = await self._validate_data()
            if not validation_report['valid_files']:
                raise ValueError("No valid data files found")
                
            # Step 2: Train models for each timeframe
            for timeframe in self.config.time.enabled_timeframes:
                await self._train_timeframe(timeframe)
                
            # Step 3: Generate training report
            await self._generate_report()
            
            self.logger.info("Training pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return False
            
    async def _validate_data(self) -> Dict:
        """Validate historical data"""
        self.logger.info("Validating historical data")
        
        try:
            # Run validation
            validation_report = self.validator.validate_files()
            
            # Save report
            report_path = self.results_dir / f"validation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=4)
                
            # Log summary
            total_files = len(validation_report)
            valid_files = sum(1 for r in validation_report.values() if r["valid"])
            self.logger.info(f"Data validation complete - {valid_files}/{total_files} files valid")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_files": total_files,
                "valid_files": valid_files,
                "results": validation_report
            }
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            raise
            
    async def _train_timeframe(self, timeframe: str):
        """Train model for specific timeframe"""
        self.logger.info(f"Training model for {timeframe} timeframe")
        
        try:
            # Get training parameters
            params = self.training_params[timeframe]
            
            # Initialize trainer
            trainer = ModelTrainer(
                data_dir=str(self.data_dir),
                model_dir=str(self.model_dir),
                window_size=params['window_size'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                n_epochs=params['n_epochs']
            )
            
            # Prepare data
            features, scaler = trainer.prepare_data()
            
            # Save scaler
            scaler_path = self.model_dir / f"scaler_{timeframe}.pkl"
            torch.save(scaler, scaler_path)
            
            # Train model
            model = trainer.train_model(features)
            
            # Save final model
            self.model_manager.save_checkpoint(
                model,
                params['n_epochs'],
                {"timeframe": timeframe},
                f"pattern_learner_{timeframe}"
            )
            
            self.logger.info(f"Completed training for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"Error training {timeframe} model: {str(e)}")
            raise
            
    async def _generate_report(self):
        """Generate training report"""
        self.logger.info("Generating training report")
        
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "timeframes": {}
            }
            
            for timeframe in self.config.time.enabled_timeframes:
                # Load latest model checkpoint
                checkpoint_info = self.model_manager.get_latest_checkpoint(
                    f"pattern_learner_{timeframe}"
                )
                
                if checkpoint_info:
                    report["timeframes"][timeframe] = {
                        "model_path": str(checkpoint_info["path"]),
                        "epochs_trained": checkpoint_info["epoch"],
                        "metrics": checkpoint_info["metrics"]
                    }
                    
            # Save report
            report_path = self.results_dir / f"training_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            self.logger.info("Training report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
            
    def validate_models(self) -> Dict:
        """Validate trained models"""
        validation_results = {}
        
        try:
            for timeframe in self.config.time.enabled_timeframes:
                # Check model files
                model_files = list(self.model_dir.glob(f"pattern_learner_{timeframe}*.pth"))
                scaler_file = self.model_dir / f"scaler_{timeframe}.pkl"
                
                validation_results[timeframe] = {
                    "model_files": len(model_files),
                    "has_scaler": scaler_file.exists(),
                    "latest_checkpoint": None
                }
                
                # Get latest checkpoint info
                checkpoint_info = self.model_manager.get_latest_checkpoint(
                    f"pattern_learner_{timeframe}"
                )
                
                if checkpoint_info:
                    validation_results[timeframe]["latest_checkpoint"] = {
                        "epoch": checkpoint_info["epoch"],
                        "path": str(checkpoint_info["path"])
                    }
                    
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Model validation error: {str(e)}")
            return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument('--data_dir', type=str, required=True,
                       help="Directory containing historical data")
    parser.add_argument('--model_dir', type=str, default="models",
                       help="Directory to save models")
    parser.add_argument('--results_dir', type=str, default="results",
                       help="Directory to save results")
    parser.add_argument('--config', type=str, help="Path to trading configuration")
    parser.add_argument('--validate', action='store_true',
                       help="Only validate existing models")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        config_path=args.config
    )
    
    if args.validate:
        # Validate existing models
        results = pipeline.validate_models()
        print("\nModel Validation Results:")
        print(json.dumps(results, indent=2))
    else:
        # Run complete pipeline
        asyncio.run(pipeline.run_pipeline())
