import asyncio
import websockets
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
import threading
from queue import Queue

from logging_config import setup_logging
from deriv_config import DerivConfig, get_default_config

class DerivDataStreamer:
    def __init__(self,
                 api_token: str,
                 symbols: List[str],
                 timeframes: List[str],
                 buffer_size: int = 1000,
                 config: Optional[DerivConfig] = None):
        """
        Initialize data streamer
        
        Args:
            api_token: Deriv API token
            symbols: List of symbols to stream
            timeframes: List of timeframes to stream
            buffer_size: Size of data buffer per symbol/timeframe
            config: DerivConfig instance
        """
        self.api_token = api_token
        self.symbols = symbols
        self.timeframes = timeframes
        self.buffer_size = buffer_size
        self.config = config or get_default_config()
        
        # Setup logging
        self.logger = setup_logging().get_logger(__name__)
        
        # Initialize data buffers
        self.price_buffers = {}
        self.candle_buffers = {}
        self.latest_prices = {}
        
        # Initialize callbacks
        self.price_callbacks = []
        self.candle_callbacks = []
        
        # Message queue for processing
        self.message_queue = Queue()
        
        # Connection status
        self.is_connected = False
        self.websocket = None
        
        # Initialize buffers
        self._init_buffers()
        
    def _init_buffers(self):
        """Initialize data buffers for each symbol and timeframe"""
        for symbol in self.symbols:
            # Price buffer
            self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.latest_prices[symbol] = None
            
            # Candle buffers for each timeframe
            self.candle_buffers[symbol] = {}
            for timeframe in self.timeframes:
                self.candle_buffers[symbol][timeframe] = deque(maxlen=self.buffer_size)
                
    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.config.api.api_url)
            
            # Authenticate
            auth_request = {
                "authorize": self.api_token
            }
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if "error" in auth_response:
                raise Exception(f"Authentication failed: {auth_response['error']['message']}")
                
            self.is_connected = True
            self.logger.info("Connected to Deriv WebSocket API")
            
            # Start message processing thread
            threading.Thread(target=self._process_messages, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            raise
            
    async def subscribe_prices(self, symbol: str):
        """Subscribe to price ticks for symbol"""
        if not self.is_connected:
            raise Exception("Not connected")
            
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        await self.websocket.send(json.dumps(request))
        
    async def subscribe_candles(self, symbol: str, timeframe: str):
        """Subscribe to candle data for symbol and timeframe"""
        if not self.is_connected:
            raise Exception("Not connected")
            
        # Get granularity in seconds
        granularity = self._get_granularity(timeframe)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 1000,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": granularity,
            "subscribe": 1
        }
        await self.websocket.send(json.dumps(request))
        
    def add_price_callback(self, callback: Callable):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
        
    def add_candle_callback(self, callback: Callable):
        """Add callback for candle updates"""
        self.candle_callbacks.append(callback)
        
    async def start_streaming(self):
        """Start data streaming"""
        try:
            await self.connect()
            
            # Subscribe to all symbols and timeframes
            for symbol in self.symbols:
                await self.subscribe_prices(symbol)
                
                for timeframe in self.timeframes:
                    await self.subscribe_candles(symbol, timeframe)
                    
            # Start processing messages
            while self.is_connected:
                try:
                    message = await self.websocket.recv()
                    self.message_queue.put(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.error("WebSocket connection closed")
                    break
                except Exception as e:
                    self.logger.error(f"Error receiving message: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            
        finally:
            await self.stop_streaming()
            
    async def stop_streaming(self):
        """Stop data streaming"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
            
    def _process_messages(self):
        """Process incoming messages"""
        while True:
            try:
                message = self.message_queue.get()
                data = json.loads(message)
                
                if "tick" in data:
                    self._process_tick(data["tick"])
                elif "candles" in data:
                    self._process_candles(data)
                elif "error" in data:
                    self.logger.error(f"API error: {data['error']['message']}")
                    
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                
    def _process_tick(self, tick: Dict):
        """Process price tick data"""
        symbol = tick["symbol"]
        price = tick["quote"]
        timestamp = datetime.fromtimestamp(tick["epoch"])
        
        # Update price buffer
        self.price_buffers[symbol].append({
            'timestamp': timestamp,
            'price': price
        })
        
        self.latest_prices[symbol] = price
        
        # Notify callbacks
        for callback in self.price_callbacks:
            try:
                callback(symbol, timestamp, price)
            except Exception as e:
                self.logger.error(f"Error in price callback: {str(e)}")
                
    def _process_candles(self, data: Dict):
        """Process candle data"""
        symbol = data["echo_req"]["ticks_history"]
        timeframe = self._get_timeframe(data["echo_req"]["granularity"])
        
        for candle in data["candles"]:
            timestamp = datetime.fromtimestamp(candle["epoch"])
            
            candle_data = {
                'timestamp': timestamp,
                'open': float(candle["open"]),
                'high': float(candle["high"]),
                'low': float(candle["low"]),
                'close': float(candle["close"])
            }
            
            # Update candle buffer
            self.candle_buffers[symbol][timeframe].append(candle_data)
            
            # Notify callbacks
            for callback in self.candle_callbacks:
                try:
                    callback(symbol, timeframe, candle_data)
                except Exception as e:
                    self.logger.error(f"Error in candle callback: {str(e)}")
                    
    def _get_granularity(self, timeframe: str) -> int:
        """Convert timeframe to granularity in seconds"""
        timeframe_map = {
            'M1': 60,
            'M5': 300,
            'M15': 900,
            'M30': 1800,
            'H1': 3600,
            'H4': 14400,
            'D1': 86400
        }
        return timeframe_map.get(timeframe, 900)  # Default to 15 minutes
        
    def _get_timeframe(self, granularity: int) -> str:
        """Convert granularity to timeframe string"""
        granularity_map = {
            60: 'M1',
            300: 'M5',
            900: 'M15',
            1800: 'M30',
            3600: 'H1',
            14400: 'H4',
            86400: 'D1'
        }
        return granularity_map.get(granularity, 'M15')  # Default to 15 minutes
        
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
        
    def get_price_history(self, symbol: str) -> pd.DataFrame:
        """Get price history for symbol"""
        if symbol not in self.price_buffers:
            return pd.DataFrame()
            
        data = list(self.price_buffers[symbol])
        return pd.DataFrame(data).set_index('timestamp')
        
    def get_candle_history(self,
                          symbol: str,
                          timeframe: str) -> pd.DataFrame:
        """Get candle history for symbol and timeframe"""
        if (symbol not in self.candle_buffers or
            timeframe not in self.candle_buffers[symbol]):
            return pd.DataFrame()
            
        data = list(self.candle_buffers[symbol][timeframe])
        return pd.DataFrame(data).set_index('timestamp')

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Deriv Data Streamer")
    parser.add_argument('--symbols', nargs='+', required=True,
                       help="Symbols to stream")
    parser.add_argument('--timeframes', nargs='+', required=True,
                       help="Timeframes to stream")
    
    args = parser.parse_args()
    
    # Get API token from environment
    api_token = os.getenv('DERIV_API_TOKEN')
    if not api_token:
        raise ValueError("DERIV_API_TOKEN not found in environment")
    
    # Example price callback
    def print_price(symbol: str, timestamp: datetime, price: float):
        print(f"{timestamp}: {symbol} = {price}")
    
    # Example candle callback
    def print_candle(symbol: str, timeframe: str, candle: Dict):
        print(f"{candle['timestamp']}: {symbol} {timeframe} O:{candle['open']} H:{candle['high']} L:{candle['low']} C:{candle['close']}")
    
    # Initialize and run streamer
    streamer = DerivDataStreamer(api_token, args.symbols, args.timeframes)
    streamer.add_price_callback(print_price)
    streamer.add_candle_callback(print_candle)
    
    asyncio.run(streamer.start_streaming())
