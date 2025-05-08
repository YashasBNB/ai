import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from pathlib import Path
import json
import websockets
from dataclasses import dataclass, asdict

from logging_config import setup_logging
from trading_config import TradingConfig, get_default_config

@dataclass
class TradeStatus:
    """Track individual trade status"""
    contract_id: str
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    stake_amount: float
    take_profit: float
    stop_loss: float
    current_price: float
    current_profit: float
    status: str  # 'open', 'closed', 'pending'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    partial_exits: List[Dict] = None

    def __post_init__(self):
        if self.partial_exits is None:
            self.partial_exits = []

class TradeMonitor:
    def __init__(self,
                 config_path: Optional[str] = None,
                 results_dir: str = "results/trades"):
        """
        Initialize TradeMonitor
        
        Args:
            config_path: Path to trading configuration
            results_dir: Directory to save trade results
        """
        # Load configuration
        self.config = (TradingConfig.load(config_path) if config_path 
                      else get_default_config())
        
        # Setup directories and logging
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize trade tracking
        self.active_trades: Dict[str, TradeStatus] = {}
        self.daily_stats = self._init_daily_stats()
        self.trade_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'risk_reward_ratio': 0.0,
            'profit_factor': 0.0
        }
        
    def _init_daily_stats(self) -> Dict:
        """Initialize daily trading statistics"""
        return {
            'trades_taken': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'risk_taken': 0.0
        }
        
    async def monitor_trade(self, 
                          trade: TradeStatus,
                          websocket) -> None:
        """Monitor individual trade"""
        try:
            while trade.status == 'open':
                # Get current price
                price_request = {
                    "ticks": trade.symbol,
                    "subscribe": 1
                }
                await websocket.send(json.dumps(price_request))
                response = await websocket.recv()
                tick_data = json.loads(response)
                
                if "error" in tick_data:
                    self.logger.error(f"Error getting price: {tick_data['error']['message']}")
                    continue
                
                current_price = tick_data["tick"]["quote"]
                trade.current_price = current_price
                
                # Calculate current profit/loss
                trade.current_profit = self._calculate_profit(trade)
                
                # Check exit conditions
                await self._check_exit_conditions(trade, websocket)
                
                # Update metrics
                self._update_metrics(trade)
                
                # Save trade status
                self._save_trade_status(trade)
                
                await asyncio.sleep(1)  # Update every second
                
        except Exception as e:
            self.logger.error(f"Error monitoring trade {trade.contract_id}: {str(e)}")
            
    async def _check_exit_conditions(self,
                                   trade: TradeStatus,
                                   websocket) -> None:
        """Check if trade should be closed"""
        try:
            # Stop loss hit
            if trade.current_price <= trade.stop_loss:
                await self._close_trade(trade, websocket, "stop_loss")
                return
                
            # Take profit hit
            if trade.current_price >= trade.take_profit:
                await self._close_trade(trade, websocket, "take_profit")
                return
                
            # Check partial exits
            if self.config.exit.partial_exits:
                for level, percentage in self.config.exit.exit_levels:
                    profit_target = trade.entry_price * (1 + level)
                    if (trade.current_price >= profit_target and
                        not any(exit['level'] == level for exit in trade.partial_exits)):
                        await self._partial_exit(trade, websocket, level, percentage)
                        
            # Time-based exit
            trade_duration = datetime.now() - trade.entry_time
            if trade_duration.total_seconds() / 60 >= self.config.exit.max_trade_duration:
                await self._close_trade(trade, websocket, "time_exit")
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {str(e)}")
            
    async def _close_trade(self,
                          trade: TradeStatus,
                          websocket,
                          reason: str) -> None:
        """Close trade and update records"""
        try:
            # Send close request
            close_request = {
                "sell": trade.contract_id,
                "price": trade.current_price
            }
            await websocket.send(json.dumps(close_request))
            response = await websocket.recv()
            result = json.loads(response)
            
            if "error" in result:
                raise Exception(f"Close trade error: {result['error']['message']}")
                
            # Update trade status
            trade.status = 'closed'
            trade.exit_time = datetime.now()
            trade.exit_price = trade.current_price
            trade.exit_reason = reason
            
            # Update statistics
            self._update_trade_statistics(trade)
            
            # Remove from active trades
            del self.active_trades[trade.contract_id]
            
            # Save to history
            self.trade_history.append(asdict(trade))
            
            self.logger.info(
                f"Closed trade {trade.contract_id} - "
                f"Profit: {trade.current_profit:.2f}, "
                f"Reason: {reason}"
            )
            
        except Exception as e:
            self.logger.error(f"Error closing trade: {str(e)}")
            
    async def _partial_exit(self,
                           trade: TradeStatus,
                           websocket,
                           level: float,
                           percentage: float) -> None:
        """Execute partial exit"""
        try:
            # Calculate exit amount
            exit_amount = trade.stake_amount * percentage
            
            # Send partial close request
            close_request = {
                "sell": trade.contract_id,
                "price": trade.current_price,
                "amount": exit_amount
            }
            await websocket.send(json.dumps(close_request))
            response = await websocket.recv()
            result = json.loads(response)
            
            if "error" in result:
                raise Exception(f"Partial exit error: {result['error']['message']}")
                
            # Record partial exit
            trade.partial_exits.append({
                'time': datetime.now(),
                'price': trade.current_price,
                'amount': exit_amount,
                'level': level
            })
            
            # Update stake amount
            trade.stake_amount -= exit_amount
            
            self.logger.info(
                f"Partial exit on {trade.contract_id} - "
                f"Level: {level}, Amount: {exit_amount}"
            )
            
        except Exception as e:
            self.logger.error(f"Error executing partial exit: {str(e)}")
            
    def _calculate_profit(self, trade: TradeStatus) -> float:
        """Calculate current profit/loss"""
        if trade.direction == "CALL":
            profit = (trade.current_price - trade.entry_price) * trade.stake_amount
        else:  # PUT
            profit = (trade.entry_price - trade.current_price) * trade.stake_amount
            
        return profit
        
    def _update_metrics(self, trade: TradeStatus):
        """Update performance metrics"""
        self.performance_metrics['total_trades'] = len(self.trade_history) + len(self.active_trades)
        
        # Calculate win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
            
        # Update max drawdown
        current_drawdown = min(0, sum(t.current_profit for t in self.active_trades.values()))
        self.performance_metrics['max_drawdown'] = min(
            self.performance_metrics['max_drawdown'],
            current_drawdown
        )
        
    def _update_trade_statistics(self, trade: TradeStatus):
        """Update trading statistics"""
        # Update daily stats
        self.daily_stats['trades_taken'] += 1
        if trade.current_profit > 0:
            self.daily_stats['winning_trades'] += 1
        else:
            self.daily_stats['losing_trades'] += 1
            
        self.daily_stats['total_profit'] += trade.current_profit
        
        # Update overall performance
        if trade.current_profit > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['avg_win'] = (
                (self.performance_metrics['avg_win'] * (self.performance_metrics['winning_trades'] - 1) +
                 trade.current_profit) / self.performance_metrics['winning_trades']
            )
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['avg_loss'] = (
                (self.performance_metrics['avg_loss'] * (self.performance_metrics['losing_trades'] - 1) +
                 abs(trade.current_profit)) / self.performance_metrics['losing_trades']
            )
            
        # Update risk-reward and profit factor
        if self.performance_metrics['avg_loss'] > 0:
            self.performance_metrics['risk_reward_ratio'] = (
                self.performance_metrics['avg_win'] / 
                self.performance_metrics['avg_loss']
            )
            
        total_profit = self.performance_metrics['avg_win'] * self.performance_metrics['winning_trades']
        total_loss = self.performance_metrics['avg_loss'] * self.performance_metrics['losing_trades']
        
        if total_loss > 0:
            self.performance_metrics['profit_factor'] = total_profit / total_loss
            
    def _save_trade_status(self, trade: TradeStatus):
        """Save trade status to file"""
        timestamp = datetime.now().strftime("%Y%m%d")
        filepath = self.results_dir / f"trades_{timestamp}.json"
        
        try:
            # Load existing trades
            if filepath.exists():
                with open(filepath, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
                
            # Update or append trade
            trade_dict = asdict(trade)
            trade_dict['entry_time'] = trade_dict['entry_time'].isoformat()
            if trade_dict['exit_time']:
                trade_dict['exit_time'] = trade_dict['exit_time'].isoformat()
                
            # Add or update trade
            updated = False
            for i, t in enumerate(trades):
                if t['contract_id'] == trade.contract_id:
                    trades[i] = trade_dict
                    updated = True
                    break
                    
            if not updated:
                trades.append(trade_dict)
                
            # Save updated trades
            with open(filepath, 'w') as f:
                json.dump(trades, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving trade status: {str(e)}")
            
    def generate_report(self) -> Dict:
        """Generate trading performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'daily_stats': self.daily_stats,
            'performance_metrics': self.performance_metrics,
            'active_trades': len(self.active_trades),
            'trade_history_length': len(self.trade_history)
        }
        
        # Save report
        report_path = self.results_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Trade Monitor")
    parser.add_argument('--config', type=str, help="Path to trading configuration")
    parser.add_argument('--results_dir', type=str, default="results/trades",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TradeMonitor(args.config, args.results_dir)
    
    # Example: Generate report
    report = monitor.generate_report()
    print(json.dumps(report, indent=2))
