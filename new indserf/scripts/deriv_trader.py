import json
import asyncio
import websockets
import logging
import time
import signal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from logging_config import setup_logging
from scripts.advanced_pattern_learner import PatternLearner
from scripts.model_manager import ModelManager
from scripts.data_processor import DataProcessor
from scripts.continuous_learner import ContinuousLearner

class DerivTrader:
    def __init__(self,
                 api_token: str,
                 model_dir: str = "models",
                 timeframe: str = "M15",
                 risk_percent: float = 1.0,
                 max_concurrent_trades: int = 3):
        """
        Initialize DerivTrader
        
        Args:
            api_token: Deriv API token
            model_dir: Directory containing trained models
            timeframe: Trading timeframe (M15 or H1)
            risk_percent: Risk per trade (% of account balance)
            max_concurrent_trades: Maximum number of concurrent trades
        """
        self.api_token = api_token
        self.model_dir = Path(model_dir)
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.max_concurrent_trades = max_concurrent_trades
        
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize components
        self.model_manager = ModelManager(model_dir)
        self.data_processor = DataProcessor("data")
        self.continuous_learner = ContinuousLearner(
            model_dir=model_dir,
            timeframe=timeframe,
            update_interval=15  # Update model every 15 minutes
        )
        
        # Trading state
        self.active_trades = {}
        self.websocket = None
        self.account_info = None
        
        # Performance tracking
        self.model_updates = []
        
    async def connect(self):
        """Establish WebSocket connection to Deriv"""
        uri = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
        
        try:
            self.websocket = await websockets.connect(uri)
            
            # Authenticate
            auth_request = {
                "authorize": self.api_token
            }
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if "error" in auth_response:
                raise Exception(f"Authentication failed: {auth_response['error']['message']}")
                
            self.account_info = auth_response["authorize"]
            self.logger.info(f"Connected to Deriv - Balance: {self.account_info['balance']}")
            
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
            
    async def load_model(self):
        """Load trained model"""
        try:
            # Load latest model checkpoint
            self.model = PatternLearner(input_dim=None)  # Set input_dim based on your features
            epoch, metrics = self.model_manager.load_checkpoint(
                self.model,
                model_name=f"pattern_learner_{self.timeframe}"
            )
            self.logger.info(f"Loaded model from epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch recent market data from Deriv"""
        candles_request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 100,  # Adjust based on your needs
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": 900 if self.timeframe == "M15" else 3600  # 15min or 1hour
        }
        
        await self.websocket.send(json.dumps(candles_request))
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if "error" in data:
            raise Exception(f"Error fetching market data: {data['error']['message']}")
            
        # Convert to DataFrame
        candles = data["candles"]
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
        df.set_index("timestamp", inplace=True)
        
        return df
        
    async def analyze_pattern(self, data: pd.DataFrame) -> Dict:
        """Analyze market patterns using trained model"""
        try:
            # Process data
            features = self.data_processor.prepare_data(data)
            
            # Add to continuous learner
            self.continuous_learner.add_experience(features[-1])
            
            # Get model predictions
            with torch.no_grad():
                reconstructed, mu, log_var = self.model(features)
                reconstruction_error = torch.mean((features - reconstructed) ** 2, dim=1)
                
            # Detect patterns and anomalies
            patterns = self.model.detect_patterns(features)
            
            # Update model if needed
            update_info = await self.continuous_learner.update_model()
            if update_info:
                self.model_updates.append(update_info)
                self.logger.info(f"Model updated: Loss = {update_info['loss']:.6f}")
            
            return {
                'cluster': patterns['clusters'][-1],
                'is_anomaly': patterns['anomalies'][-1],
                'reconstruction_error': reconstruction_error[-1].item(),
                'model_updated': update_info is not None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern: {str(e)}")
            return None
            
    def generate_signal(self, analysis: Dict) -> Optional[str]:
        """Generate trading signal based on pattern analysis"""
        if analysis['is_anomaly']:
            self.logger.info("Anomaly detected - No trade")
            return None
            
        # Define your trading logic based on patterns
        # This is a simplified example - customize based on your research
        if analysis['reconstruction_error'] < 0.1:
            cluster = analysis['cluster']
            
            # Example logic - replace with your own
            if cluster in [0, 2, 5]:  # Bullish patterns
                return "CALL"
            elif cluster in [1, 3, 4]:  # Bearish patterns
                return "PUT"
                
        return None
        
    async def place_trade(self, symbol: str, direction: str):
        """Place trade on Deriv"""
        if len(self.active_trades) >= self.max_concurrent_trades:
            self.logger.info("Maximum concurrent trades reached")
            return
            
        try:
            # Calculate position size
            account_balance = float(self.account_info['balance'])
            risk_amount = account_balance * (self.risk_percent / 100)
            
            # Prepare trade request
            contract_request = {
                "proposal": 1,
                "amount": risk_amount,
                "basis": "stake",
                "contract_type": direction,  # "CALL" or "PUT"
                "currency": self.account_info['currency'],
                "duration": 5,  # Number of time units
                "duration_unit": "m" if self.timeframe == "M15" else "h",
                "symbol": symbol
            }
            
            await self.websocket.send(json.dumps(contract_request))
            response = await self.websocket.recv()
            proposal = json.loads(response)
            
            if "error" in proposal:
                raise Exception(f"Proposal error: {proposal['error']['message']}")
                
            # Buy contract
            buy_request = {
                "buy": proposal["proposal"]["id"],
                "price": proposal["proposal"]["ask_price"]
            }
            
            await self.websocket.send(json.dumps(buy_request))
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if "error" in result:
                raise Exception(f"Buy error: {result['error']['message']}")
                
            # Track trade
            contract_id = result["buy"]["contract_id"]
            self.active_trades[contract_id] = {
                "entry_time": datetime.now(),
                "symbol": symbol,
                "direction": direction,
                "amount": risk_amount
            }
            
            self.logger.info(f"Placed {direction} trade on {symbol} - Amount: {risk_amount}")
            
        except Exception as e:
            self.logger.error(f"Error placing trade: {str(e)}")
            
    async def monitor_trades(self):
        """Monitor and update active trades"""
        while True:
            try:
                for contract_id in list(self.active_trades.keys()):
                    # Check contract status
                    status_request = {
                        "proposal_open_contract": 1,
                        "contract_id": contract_id
                    }
                    
                    await self.websocket.send(json.dumps(status_request))
                    response = await self.websocket.recv()
                    status = json.loads(response)
                    
                    if "error" in status:
                        self.logger.error(f"Error checking trade status: {status['error']['message']}")
                        continue
                        
                    contract = status["proposal_open_contract"]
                    
                    if contract["status"] == "closed":
                        # Trade completed
                        profit_loss = float(contract["profit"])
                        self.logger.info(
                            f"Trade closed - Contract ID: {contract_id}, "
                            f"Profit/Loss: {profit_loss}"
                        )
                        del self.active_trades[contract_id]
                        
            except Exception as e:
                self.logger.error(f"Error monitoring trades: {str(e)}")
                
            await asyncio.sleep(1)  # Check every second
            
    async def run(self, symbols: List[str]):
        """Main trading loop"""
        try:
            await self.connect()
            await self.load_model()
            
            # Start trade monitoring and continuous learning
            monitor_task = asyncio.create_task(self.monitor_trades())
            learning_task = asyncio.create_task(self.continuous_learner.start_continuous_learning())
            
            # Setup signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            # Available trading pairs
            self.trading_pairs = [
                "frxAUDCAD", "frxAUDCHF", "frxAUDJPY", "frxAUDNZD", "frxAUDUSD",
                "frxEURAUD", "frxEURCAD", "frxEURCHF", "frxEURGBP", "frxEURJPY",
                "frxEURNZD", "frxEURUSD", "frxGBPAUD", "frxGBPCAD", "frxGBPCHF",
                "frxGBPJPY", "frxGBPNZD", "frxGBPUSD", "frxNZDUSD", "frxUSDCAD",
                "frxUSDCHF", "frxUSDJPY"
            ]
            
            # Validate symbols
            invalid_symbols = [s for s in symbols if s not in self.trading_pairs]
            if invalid_symbols:
                raise ValueError(f"Invalid symbols: {invalid_symbols}. Must be one of {self.trading_pairs}")
            
            for signal in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(signal, lambda: asyncio.create_task(self.stop_trading()))
            
            self.is_running = True
            while self.is_running:
                # Process all requested symbols
                for symbol in symbols:
                    try:
                        # Get market data
                        data = await self.get_market_data(symbol)
                        
                        # Analyze patterns
                        analysis = await self.analyze_pattern(data)
                        if analysis:
                            # Generate trading signal
                            signal = self.generate_signal(analysis)
                            
                            if signal:
                                await self.place_trade(symbol, signal)
                                
                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {str(e)}")
                        continue
                        
                # Wait for next candle
                wait_time = 15 if self.timeframe == "M15" else 60
                await asyncio.sleep(wait_time * 60)
                
        except Exception as e:
            self.logger.error(f"Trading error: {str(e)}")
            await self.stop_trading()
            
        except asyncio.CancelledError:
            self.logger.info("Trading system received shutdown signal")
            await self.stop_trading()
            
        finally:
            # Cancel monitoring and learning tasks
            monitor_task.cancel()
            learning_task.cancel()
            try:
                await monitor_task
                await learning_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Deriv Trading Bot")
    parser.add_argument('--timeframe', choices=['M15', 'H1'], required=True,
                       help="Trading timeframe")
    parser.add_argument('--symbols', nargs='+', required=True,
                       help="Trading symbols")
    parser.add_argument('--risk', type=float, default=1.0,
                       help="Risk per trade (%)")
    parser.add_argument('--max_trades', type=int, default=3,
                       help="Maximum concurrent trades")
    
    args = parser.parse_args()
    
    # Get API token from environment
    api_token = os.getenv('DERIV_API_TOKEN')
    if not api_token:
        raise ValueError("DERIV_API_TOKEN not found in environment variables")
    
    # Initialize and run trader
    trader = DerivTrader(
        api_token=api_token,
        timeframe=args.timeframe,
        risk_percent=args.risk,
        max_concurrent_trades=args.max_trades
    )
    
    asyncio.run(trader.run(args.symbols))

    async def stop_trading(self):
        """Stop trading system"""
        self.logger.info("Stopping trading system...")
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            
        # Save model performance metrics
        try:
            metrics_path = Path(self.model_dir) / f"model_metrics_{self.timeframe}_{datetime.now():%Y%m%d_%H%M%S}.json"
            metrics = {
                "model_updates": self.model_updates,
                "final_performance": self.continuous_learner.get_performance_metrics(),
                "active_trades": len(self.active_trades),
                "timestamp": datetime.now().isoformat()
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Model metrics saved to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error saving model metrics: {str(e)}")
        
        self.logger.info("Trading system stopped")
