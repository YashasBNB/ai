from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, time
import os
import json
import logging
from pathlib import Path

@dataclass
class DerivAPIConfig:
    """Deriv API Configuration"""
    app_id: str = "1089"  # Default app_id
    api_url: str = "wss://ws.binaryws.com/websockets/v3"
    demo_url: str = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    
    # API endpoints
    endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = {
                "authorize": "authorize",
                "ticks": "ticks",
                "ticks_history": "ticks_history",
                "proposal": "proposal",
                "buy": "buy",
                "sell": "sell",
                "balance": "balance",
                "portfolio": "portfolio",
                "proposal_open_contract": "proposal_open_contract"
            }

@dataclass
class AccountConfig:
    """Trading Account Configuration"""
    account_type: str = "demo"  # 'demo' or 'real'
    currency: str = "USD"
    leverage: int = 100
    
    # Account limits
    max_leverage: int = 500
    max_positions: int = 10
    min_duration: int = 5        # Minimum trade duration in minutes
    max_duration: int = 1440     # Maximum trade duration in minutes (24 hours)
    
    # Balance thresholds
    min_balance: float = 100.0   # Minimum balance to continue trading
    max_balance_risk: float = 0.2  # Maximum % of balance at risk

@dataclass
class SymbolConfig:
    """Symbol Trading Configuration"""
    active_symbols: List[str] = None
    symbol_limits: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.active_symbols is None:
            self.active_symbols = [
                "EURUSD", "GBPUSD", "AUDUSD",
                "BTCUSD", "ETHUSD"
            ]
        
        if self.symbol_limits is None:
            self.symbol_limits = {
                "EURUSD": {
                    "min_stake": 1.0,
                    "max_stake": 100.0,
                    "spread": 0.0002,
                    "pip_size": 0.0001,
                    "trading_hours": [(time(0, 0), time(23, 59))]
                },
                "BTCUSD": {
                    "min_stake": 1.0,
                    "max_stake": 50.0,
                    "spread": 2.0,
                    "pip_size": 0.01,
                    "trading_hours": [(time(0, 0), time(23, 59))]
                }
            }

@dataclass
class DurationConfig:
    """Trade Duration Configuration"""
    timeframes: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = {
                "M15": {
                    "duration": 15,
                    "duration_unit": "m",
                    "min_duration": 5,
                    "max_duration": 60
                },
                "H1": {
                    "duration": 60,
                    "duration_unit": "m",
                    "min_duration": 15,
                    "max_duration": 240
                }
            }

@dataclass
class ContractConfig:
    """Contract Type Configuration"""
    contract_types: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.contract_types is None:
            self.contract_types = {
                "CALL": {
                    "name": "Higher",
                    "description": "Asset price will rise",
                    "payout_type": "stake"  # or 'payout'
                },
                "PUT": {
                    "name": "Lower",
                    "description": "Asset price will fall",
                    "payout_type": "stake"
                }
            }

@dataclass
class DerivConfig:
    """Main Deriv Configuration"""
    api: DerivAPIConfig = DerivAPIConfig()
    account: AccountConfig = AccountConfig()
    symbols: SymbolConfig = SymbolConfig()
    durations: DurationConfig = DurationConfig()
    contracts: ContractConfig = ContractConfig()
    
    # Token management
    token_file: str = ".deriv_token"
    token_refresh_days: int = 30
    
    def save_token(self, token: str):
        """Save API token securely"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump({
                    'token': token,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logging.error(f"Error saving token: {str(e)}")
    
    def load_token(self) -> Optional[str]:
        """Load API token"""
        try:
            if not os.path.exists(self.token_file):
                return None
                
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                
            # Check token age
            saved_time = datetime.fromisoformat(data['timestamp'])
            age_days = (datetime.now() - saved_time).days
            
            if age_days > self.token_refresh_days:
                logging.warning("Token has expired")
                return None
                
            return data['token']
            
        except Exception as e:
            logging.error(f"Error loading token: {str(e)}")
            return None
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        return symbol in self.symbols.active_symbols
    
    def get_symbol_limits(self, symbol: str) -> Dict:
        """Get trading limits for symbol"""
        return self.symbols.symbol_limits.get(symbol, {})
    
    def get_timeframe_config(self, timeframe: str) -> Dict:
        """Get configuration for timeframe"""
        return self.durations.timeframes.get(timeframe, {})
    
    def get_contract_config(self, contract_type: str) -> Dict:
        """Get configuration for contract type"""
        return self.contracts.contract_types.get(contract_type, {})
    
    def can_trade_now(self, symbol: str) -> bool:
        """Check if symbol can be traded at current time"""
        limits = self.get_symbol_limits(symbol)
        if not limits or 'trading_hours' not in limits:
            return False
            
        current_time = datetime.now().time()
        
        for start, end in limits['trading_hours']:
            if start <= current_time <= end:
                return True
                
        return False
    
    def validate_stake(self, symbol: str, stake_amount: float) -> bool:
        """Validate stake amount for symbol"""
        limits = self.get_symbol_limits(symbol)
        if not limits:
            return False
            
        return (limits['min_stake'] <= stake_amount <= limits['max_stake'])

def get_default_config() -> DerivConfig:
    """Get default Deriv configuration"""
    return DerivConfig()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deriv Configuration")
    parser.add_argument('--token', type=str, help="API token to save")
    parser.add_argument('--validate', type=str, help="Symbol to validate")
    
    args = parser.parse_args()
    
    config = get_default_config()
    
    if args.token:
        config.save_token(args.token)
        print("Token saved successfully")
        
    if args.validate:
        if config.validate_symbol(args.validate):
            print(f"Symbol {args.validate} is valid")
            limits = config.get_symbol_limits(args.validate)
            print("Trading limits:", json.dumps(limits, indent=2))
        else:
            print(f"Symbol {args.validate} is not supported")
