from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import time

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_risk_percent: float = 1.0  # Maximum risk per trade (% of account)
    max_daily_risk: float = 5.0    # Maximum daily risk (% of account)
    max_concurrent_trades: int = 3  # Maximum number of open trades
    max_daily_trades: int = 10     # Maximum trades per day
    max_losing_streak: int = 3     # Maximum consecutive losing trades
    recovery_wait_time: int = 60   # Minutes to wait after losing streak
    
    # Position sizing
    base_position_size: float = 1.0  # Base trade size in account currency
    position_scaling: bool = True    # Enable dynamic position sizing
    scaling_factor: float = 1.2      # Multiply position size after wins
    max_position_size: float = 5.0   # Maximum position size multiplier

@dataclass
class PatternConfig:
    """Pattern-based trading rules"""
    # Anomaly thresholds
    anomaly_threshold: float = 3.0     # Standard deviations for anomaly detection
    min_confidence: float = 0.7        # Minimum confidence for pattern recognition
    pattern_memory: int = 50           # Number of patterns to keep in memory
    
    # Pattern classifications
    bullish_patterns: List[int] = None  # Cluster IDs for bullish patterns
    bearish_patterns: List[int] = None  # Cluster IDs for bearish patterns
    neutral_patterns: List[int] = None  # Cluster IDs for neutral patterns
    
    def __post_init__(self):
        if self.bullish_patterns is None:
            self.bullish_patterns = [0, 2, 5]  # Example pattern IDs
        if self.bearish_patterns is None:
            self.bearish_patterns = [1, 3, 4]  # Example pattern IDs
        if self.neutral_patterns is None:
            self.neutral_patterns = [6, 7]     # Example pattern IDs

@dataclass
class TimeConfig:
    """Trading time configuration"""
    enabled_timeframes: List[str] = None  # Active timeframes
    trading_hours: Dict[str, List[tuple]] = None  # Trading hours per market
    wait_for_confirmation: int = 2  # Candles to wait for confirmation
    
    # Market hours (UTC)
    forex_open: time = time(22, 0)   # 22:00 UTC
    forex_close: time = time(21, 0)  # 21:00 UTC
    
    def __post_init__(self):
        if self.enabled_timeframes is None:
            self.enabled_timeframes = ["M15", "H1"]
        if self.trading_hours is None:
            self.trading_hours = {
                "forex": [(self.forex_open, self.forex_close)],
                "crypto": [(time(0, 0), time(23, 59))]  # 24/7 trading
            }

@dataclass
class FilterConfig:
    """Trade filter configuration"""
    min_volatility: float = 0.1      # Minimum price volatility
    max_volatility: float = 2.0      # Maximum price volatility
    trend_threshold: float = 0.3     # Minimum trend strength
    volume_threshold: float = 1.0    # Minimum volume multiplier
    spread_threshold: float = 0.5    # Maximum allowable spread
    news_impact_delay: int = 30      # Minutes to wait after high-impact news

@dataclass
class ExitConfig:
    """Trade exit configuration"""
    take_profit_mult: float = 2.0    # Take profit as multiplier of risk
    stop_loss_mult: float = 1.0      # Stop loss as multiplier of risk
    trailing_stop: bool = True       # Enable trailing stop
    trailing_distance: float = 0.5    # Trailing stop distance
    
    # Time-based exits
    max_trade_duration: int = 240    # Maximum minutes in trade
    min_trade_duration: int = 5      # Minimum minutes in trade
    
    # Partial exits
    partial_exits: bool = True       # Enable partial exits
    exit_levels: List[tuple] = None  # [(profit_level, exit_percentage)]
    
    def __post_init__(self):
        if self.exit_levels is None:
            self.exit_levels = [
                (1.0, 0.3),  # Exit 30% at 1x risk
                (1.5, 0.3),  # Exit 30% at 1.5x risk
                (2.0, 0.4)   # Exit 40% at 2x risk
            ]

@dataclass
class TradingConfig:
    """Main trading configuration"""
    risk: RiskConfig = RiskConfig()
    pattern: PatternConfig = PatternConfig()
    time: TimeConfig = TimeConfig()
    filter: FilterConfig = FilterConfig()
    exit: ExitConfig = ExitConfig()
    
    # Symbol configuration
    symbols: List[str] = None          # Trading symbols
    symbol_risk_multipliers: Dict[str, float] = None  # Risk multipliers per symbol
    
    # Account configuration
    account_currency: str = "USD"
    demo_mode: bool = True             # Use demo account
    hedge_mode: bool = False           # Allow hedging
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EURUSD", "GBPUSD", "BTCUSD"]
        if self.symbol_risk_multipliers is None:
            self.symbol_risk_multipliers = {
                "EURUSD": 1.0,
                "GBPUSD": 1.0,
                "BTCUSD": 0.5  # Reduce risk for crypto
            }
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Risk validation
            assert 0 < self.risk.max_risk_percent <= 5, "Invalid risk percentage"
            assert self.risk.max_daily_risk >= self.risk.max_risk_percent, "Daily risk too low"
            
            # Pattern validation
            all_patterns = (
                self.pattern.bullish_patterns +
                self.pattern.bearish_patterns +
                self.pattern.neutral_patterns
            )
            assert len(set(all_patterns)) == len(all_patterns), "Duplicate pattern IDs"
            
            # Symbol validation
            assert all(m > 0 for m in self.symbol_risk_multipliers.values()), "Invalid risk multipliers"
            assert all(s in self.symbol_risk_multipliers for s in self.symbols), "Missing risk multipliers"
            
            # Exit validation
            if self.exit.partial_exits:
                total_exit = sum(pct for _, pct in self.exit.exit_levels)
                assert abs(total_exit - 1.0) < 0.0001, "Partial exits must sum to 100%"
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {str(e)}")
            return False
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        from dataclasses import asdict
        
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                d = asdict(obj)
                # Convert time objects to string
                for k, v in d.items():
                    if isinstance(v, time):
                        d[k] = v.strftime("%H:%M")
                return d
            return obj
        
        config_dict = convert_to_dict(self)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from file"""
        import json
        from datetime import datetime
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        def convert_time(time_str: str) -> time:
            return datetime.strptime(time_str, "%H:%M").time()
        
        # Convert time strings back to time objects
        if 'time' in config_dict:
            if 'forex_open' in config_dict['time']:
                config_dict['time']['forex_open'] = convert_time(config_dict['time']['forex_open'])
            if 'forex_close' in config_dict['time']:
                config_dict['time']['forex_close'] = convert_time(config_dict['time']['forex_close'])
        
        return cls(**config_dict)

def get_default_config() -> TradingConfig:
    """Get default trading configuration"""
    return TradingConfig()

if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    
    # Validate configuration
    if config.validate():
        print("Configuration is valid")
        
        # Save configuration
        config.save("trading_config.json")
        
        # Load configuration
        loaded_config = TradingConfig.load("trading_config.json")
        
        # Access configuration
        print(f"Max risk per trade: {loaded_config.risk.max_risk_percent}%")
        print(f"Trading symbols: {loaded_config.symbols}")
        print(f"Trading hours (Forex): {loaded_config.time.trading_hours['forex']}")
