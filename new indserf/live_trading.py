import json
import websocket
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import random

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')
APP_ID = os.getenv('APP_ID')
WS_URL = os.getenv('WS_URL')

class DerivTrader:
    def __init__(self):
        self.ws = None
        self.active_symbols = [
            "frxAUDCAD", "frxAUDCHF", "frxAUDJPY", "frxAUDNZD", "frxAUDUSD",
            "frxEURAUD", "frxEURCAD", "frxEURCHF", "frxEURGBP", "frxEURJPY",
            "frxEURNZD", "frxEURUSD", "frxGBPAUD", "frxGBPCAD", "frxGBPCHF",
            "frxGBPJPY", "frxGBPNZD", "frxGBPUSD", "frxNZDUSD", "frxUSDCAD",
            "frxUSDCHF", "frxUSDJPY"
        ]
        self.historical_data = pd.DataFrame()
        self.connect()

    def connect(self):
        self.ws = websocket.WebSocket()
        self.ws.connect(WS_URL)
        self.ws.send(json.dumps({"authorize": API_TOKEN}))
        auth_response = json.loads(self.ws.recv())
        if auth_response.get('error'):
            raise Exception(f"Authentication failed: {auth_response['error']['message']}")
        print("Successfully authenticated!")

    def get_historical_candles(self, symbol, timeframe_minutes=15, count=100):
        end_time = int(time.time())
        start_time = end_time - (timeframe_minutes * 60 * count)
        candles_req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": end_time,
            "start": start_time,
            "style": "candles",
            "granularity": timeframe_minutes * 60
        }
        self.ws.send(json.dumps(candles_req))
        candles_response = json.loads(self.ws.recv())
        if candles_response.get('error'):
            raise Exception(f"Historical data error: {candles_response['error']['message']}")
        return candles_response['candles']

    def prepare_historical_data(self):
        all_data = []
        for symbol in self.active_symbols:
            try:
                candles = self.get_historical_candles(symbol)
                df = pd.DataFrame([{
                    'timestamp': c['epoch'],
                    'symbol': symbol,
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close'])
                } for c in candles])
                all_data.append(df)
                print(f"Fetched data for {symbol}")
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        if all_data:
            self.historical_data = pd.concat(all_data, ignore_index=True)
            print("Historical data prepared successfully!")
        else:
            raise Exception("Failed to fetch historical data")

    def get_ai_signals(self):
        import torch
        import torch.nn as nn
        # --- Model classes (copied from train_binary_classifier.py) ---
        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)
        class Mish(nn.Module):
            def forward(self, x):
                return x * torch.tanh(torch.nn.functional.softplus(x))
        class ResidualBlock(nn.Module):
            def __init__(self, dim, activation, dropout=0.2):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(dropout)
                )
                self.activation = activation
            def forward(self, x):
                out = self.block(x)
                return self.activation(out + x)
        class BinaryClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dim=128, num_blocks=4, activation='gelu', dropout=0.2):
                super().__init__()
                if activation == 'gelu':
                    act = nn.GELU()
                elif activation == 'swish':
                    act = Swish()
                elif activation == 'mish':
                    act = Mish()
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                self.input_layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act
                )
                self.res_blocks = nn.Sequential(
                    *[ResidualBlock(hidden_dim, act, dropout) for _ in range(num_blocks)]
                )
                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                x = self.input_layer(x)
                x = self.res_blocks(x)
                x = self.head(x)
                return x
        # --- End model classes ---
        # Load models from absolute paths as requested by user
        device = torch.device('cpu')
        window = 10
        feature_cols = ['open','high','low','close','body','upper_wick','lower_wick','direction','range']
        input_dim = window * len(feature_cols)
        model_15m = BinaryClassifier(input_dim)
        model_1h = BinaryClassifier(input_dim)
        model_15m.load_state_dict(torch.load('/Users/yashasnaidu/new indserf/models/binary_classifier_15m.pth', map_location=device))
        model_1h.load_state_dict(torch.load('/Users/yashasnaidu/new indserf/models/binary_classifier_1h.pth', map_location=device))
        model_15m.eval()
        model_1h.eval()
        if self.historical_data.empty:
            self.prepare_historical_data()
        signals = []
        for symbol in self.active_symbols:
            df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
            # --- 15m signal: only if enough 15m candles ---
            if len(df) >= window:
                # Add engineered features
                df['body'] = df['close'] - df['open']
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                df['range'] = df['high'] - df['low']
                # Prepare windowed features for last 10 candles (15m)
                feats_15m = df[feature_cols].values.astype(np.float32)
                X_15m = feats_15m[-window:].flatten().reshape(1, -1)
                with torch.no_grad():
                    prob_15m = model_15m(torch.tensor(X_15m, dtype=torch.float32)).item()
                direction_15m = 'CALL' if prob_15m > 0.5 else 'PUT'
                signals.append({
                    'symbol': symbol,
                    'timestamp': df['timestamp'].iloc[-1],
                    'direction': direction_15m,
                    'confidence': float(prob_15m) if direction_15m == 'CALL' else float(1-prob_15m),
                    'current_price': df['close'].iloc[-1],
                    'timeframe': '15m'
                })
                print(f"\nFetched last 10 candles for {symbol} (15m):")
                print(df.tail(window)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                print(f"\n[AI 15m] Direction: {direction_15m}, Confidence: {prob_15m:.4f}")
                print(f"Input features (flattened): {X_15m.flatten()}")
            # --- 1h signal: only if enough 1h candles ---
            df_1h = df.copy()
            df_1h['dt'] = pd.to_datetime(df_1h['timestamp'], unit='s')
            df_1h = df_1h.set_index('dt').resample('1H').agg({
                'open':'first','high':'max','low':'min','close':'last',
                'body':'sum','upper_wick':'sum','lower_wick':'sum','direction':'sum','range':'sum','timestamp':'last'
            }).dropna().reset_index()
            if len(df_1h) >= window:
                feats_1h = df_1h[feature_cols].values.astype(np.float32)
                X_1h = feats_1h[-window:].flatten().reshape(1, -1)
                with torch.no_grad():
                    prob_1h = model_1h(torch.tensor(X_1h, dtype=torch.float32)).item()
                direction_1h = 'CALL' if prob_1h > 0.5 else 'PUT'
                signals.append({
                    'symbol': symbol,
                    'timestamp': int(df_1h['timestamp'].iloc[-1]),
                    'direction': direction_1h,
                    'confidence': float(prob_1h) if direction_1h == 'CALL' else float(1-prob_1h),
                    'current_price': df_1h['close'].iloc[-1],
                    'timeframe': '1h'
                })
                print(f"\nFetched last 10 candles for {symbol} (1h):")
                print(df_1h.tail(window)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                print(f"\n[AI 1h] Direction: {direction_1h}, Confidence: {prob_1h:.4f}")
                print(f"Input features (flattened): {X_1h.flatten()}")

        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            return signals_df.sort_values(['symbol','timeframe','confidence'], ascending=[True,True,False])
        return pd.DataFrame()

    def get_balance(self):
        self.ws.send(json.dumps({"balance": 1, "subscribe": 0}))
        resp = json.loads(self.ws.recv())
        if resp.get('error'):
            raise Exception(f"Balance fetch error: {resp['error']['message']}")
        balance_info = resp['balance']
        print(f"Account Balance: ${float(balance_info['balance']):.2f}")
        return balance_info

    def place_trade(self, symbol, contract_type, duration=15, amount=1, timeframe=None):
        trade_req = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": "m",
            "symbol": symbol
        }
        self.ws.send(json.dumps(trade_req))
        proposal_response = json.loads(self.ws.recv())
        if proposal_response.get('error'):
            raise Exception(f"Proposal error: {proposal_response['error']['message']}")
        buy_req = {
            "buy": proposal_response['proposal']['id'],
            "price": amount
        }
        self.ws.send(json.dumps(buy_req))
        buy_response = json.loads(self.ws.recv())
        if buy_response.get('error'):
            raise Exception(f"Buy error: {buy_response['error']['message']}")
        contract_id = buy_response['buy']['contract_id']
        if timeframe:
            print(f"Trade placed: {symbol} {contract_type} | Timeframe: {timeframe} | Contract ID: {contract_id}")
        else:
            print(f"Trade placed: {symbol} {contract_type} | Contract ID: {contract_id}")
        return contract_id

    def monitor_trade(self, contract_id):
        monitor_req = {"proposal_open_contract": 1, "contract_id": contract_id}
        self.ws.send(json.dumps(monitor_req))
        monitor_response = json.loads(self.ws.recv())
        if monitor_response.get('error'):
            raise Exception(f"Monitoring error: {monitor_response['error']['message']}")
        contract = monitor_response.get('proposal_open_contract', {})
        if contract.get('is_sold', 0) == 1:
            print(f"Trade {contract_id} completed. Profit: ${contract.get('profit', 0):.2f}")
            return True
        return False

    def check_trade_result(self, contract_id):
        profit_table_req = {"profit_table": 1, "contract_id": contract_id, "limit": 1}
        self.ws.send(json.dumps(profit_table_req))
        result_response = json.loads(self.ws.recv())
        if result_response.get('error'):
            raise Exception(f"Result check error: {result_response['error']['message']}")
        if 'profit_table' in result_response and result_response['profit_table']['transactions']:
            trade = result_response['profit_table']['transactions'][0]
            print(f"Trade Result: Contract ID {trade['contract_id']}, P/L: ${float(trade['sell_price']) - float(trade['buy_price']):.2f}")
            return trade
        return None

    def execute_trades(self, max_concurrent=3, stake_amount=1):
        signals = self.get_ai_signals()
        if signals.empty:
            print("No signals to trade.")
            # Print detailed reason for each asset and timeframe
            for symbol in self.active_symbols:
                for tf in ['15m', '1h']:
                    print("\n--- NO TRADE REASON ---")
                    print(f"Symbol: {symbol}")
                    print(f"Timeframe: {tf}")
                    if tf == '15m':
                        df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                        df['body'] = df['close'] - df['open']
                        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                        df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                        df['range'] = df['high'] - df['low']
                        available = len(df)
                        print(f"Reason: Not enough candles (have {available}, need 10) or did not meet criteria.")
                        print(f"Last {min(10, available)} candles (15m):")
                        print(df.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                    elif tf == '1h':
                        df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                        df['body'] = df['close'] - df['open']
                        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                        df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                        df['range'] = df['high'] - df['low']
                        df_1h = df.copy()
                        df_1h['dt'] = pd.to_datetime(df_1h['timestamp'], unit='s')
                        df_1h = df_1h.set_index('dt').resample('1H').agg({
                            'open':'first','high':'max','low':'min','close':'last',
                            'body':'sum','upper_wick':'sum','lower_wick':'sum','direction':'sum','range':'sum','timestamp':'last'
                        }).dropna().reset_index()
                        available = len(df_1h)
                        print(f"Reason: Not enough candles (have {available}, need 10) or did not meet criteria.")
                        print(f"Last {min(10, available)} candles (1h):")
                        print(df_1h.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                    print(f"---------------------\n")
            return
        active_trades = []
        trade_log = []  # List of dicts: symbol, timeframe, direction, confidence, contract_id, status
        # Place trades for ALL valid signals (all assets, both 15m and 1h)
        placed_symbols_timeframes = set(zip(signals['symbol'], signals['timeframe']))
        for idx, signal in signals.iterrows():
            try:
                # Print reason for trade
                print(f"\n--- TRADE REASON ---")
                print(f"Symbol: {signal['symbol']}")
                print(f"Timeframe: {signal['timeframe']}")
                print(f"Direction: {signal['direction']}")
                print(f"Confidence: {signal['confidence']:.4f}")
                print(f"Current Price: {signal['current_price']}")
                # Show last 10 candles for this symbol/timeframe
                if signal['timeframe'] == '15m':
                    df = self.historical_data[self.historical_data['symbol'] == signal['symbol']].copy()
                    df['body'] = df['close'] - df['open']
                    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                    df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                    df['range'] = df['high'] - df['low']
                    print("Last 10 candles (15m):")
                    print(df.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                elif signal['timeframe'] == '1h':
                    df = self.historical_data[self.historical_data['symbol'] == signal['symbol']].copy()
                    df['body'] = df['close'] - df['open']
                    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                    df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                    df['range'] = df['high'] - df['low']
                    df_1h = df.copy()
                    df_1h['dt'] = pd.to_datetime(df_1h['timestamp'], unit='s')
                    df_1h = df_1h.set_index('dt').resample('1H').agg({
                        'open':'first','high':'max','low':'min','close':'last',
                        'body':'sum','upper_wick':'sum','lower_wick':'sum','direction':'sum','range':'sum','timestamp':'last'
                    }).dropna().reset_index()
                    print("Last 10 candles (1h):")
                    print(df_1h.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                print(f"---------------------\n")
                contract_id = self.place_trade(
                    symbol=signal['symbol'],
                    contract_type=signal['direction'],
                    duration=15,
                    amount=stake_amount,
                    timeframe=signal['timeframe'] if 'timeframe' in signal else None
                )
                if contract_id:
                    active_trades.append(contract_id)
                    trade_log.append({
                        'symbol': signal['symbol'],
                        'timeframe': signal['timeframe'],
                        'direction': signal['direction'],
                        'confidence': signal['confidence'],
                        'contract_id': contract_id,
                        'status': 'pending'
                    })
            except Exception as e:
                print(f"Error executing trade: {str(e)}")
            time.sleep(2)
        print("Monitoring trades...")
        # --- TRADE SUMMARY ---
        if trade_log:
            print("\n--- TRADE SUMMARY ---")
            print(f"{'Symbol':<10} | {'Timeframe':<8} | {'Direction':<9} | {'Confidence':<10} | {'Contract ID':<14} | {'Status'}")
            print('-'*80)
            for trade in trade_log:
                print(f"{trade['symbol']:<10} | {trade['timeframe']:<8} | {trade['direction']:<9} | {trade['confidence']:<10.4f} | {trade['contract_id']:<14} | {trade.get('status','pending')}")
            print('-'*80 + '\n')
        while active_trades:
            completed = []
            for contract_id in active_trades:
                try:
                    if self.monitor_trade(contract_id):
                        result = self.check_trade_result(contract_id)
                        completed.append(contract_id)
                        # Update trade_log status
                        for trade in trade_log:
                            if trade['contract_id'] == contract_id:
                                trade['status'] = result if result else 'closed'
                    else:
                        # Still pending
                        continue
                except Exception as e:
                    print(f"Error monitoring trade {contract_id}: {str(e)}")
                    completed.append(contract_id)
                    # Update trade_log status
                    for trade in trade_log:
                        if trade['contract_id'] == contract_id:
                            trade['status'] = 'error'
            for contract_id in completed:
                active_trades.remove(contract_id)
            if active_trades:
                time.sleep(2)
        print("Monitoring trades...")
        # --- TRADE SUMMARY ---
        if trade_log:
            print("\n--- TRADE SUMMARY ---")
            print(f"{'Symbol':<10} | {'Timeframe':<8} | {'Direction':<9} | {'Confidence':<10} | {'Contract ID':<14} | {'Status'}")
            print('-'*80)
            for trade in trade_log:
                print(f"{trade['symbol']:<10} | {trade['timeframe']:<8} | {trade['direction']:<9} | {trade['confidence']:<10.4f} | {trade['contract_id']:<14} | {trade.get('status','pending')}")
            print('-'*80 + '\n')
        # Print skipped trades due to max_concurrent
        for symbol, timeframe in skipped_due_to_limit:
            print("\n--- NO TRADE REASON ---")
            print(f"Symbol: {symbol}")
            print(f"Timeframe: {timeframe}")
            print("Reason: Skipped due to max_concurrent limit.")
            # Show last 10 candles
            if timeframe == '15m':
                df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                df['body'] = df['close'] - df['open']
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                df['range'] = df['high'] - df['low']
                available = len(df)
                print(f"Last {min(10, available)} candles (15m):")
                print(df.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
            elif timeframe == '1h':
                df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                df['body'] = df['close'] - df['open']
                df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                df['range'] = df['high'] - df['low']
                df_1h = df.copy()
                df_1h['dt'] = pd.to_datetime(df_1h['timestamp'], unit='s')
                df_1h = df_1h.set_index('dt').resample('1H').agg({
                    'open':'first','high':'max','low':'min','close':'last',
                    'body':'sum','upper_wick':'sum','lower_wick':'sum','direction':'sum','range':'sum','timestamp':'last'
                }).dropna().reset_index()
                available = len(df_1h)
                print(f"Last {min(10, available)} candles (1h):")
                print(df_1h.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
            print(f"---------------------\n")
        # Print reason for assets/timeframes not in signals
        for symbol in self.active_symbols:
            for tf in ['15m','1h']:
                if (symbol, tf) not in placed_symbols_timeframes and (symbol, tf) not in skipped_due_to_limit:
                    print("\n--- NO TRADE REASON ---")
                    print(f"Symbol: {symbol}")
                    print(f"Timeframe: {tf}")
                    if tf == '15m':
                        df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                        df['body'] = df['close'] - df['open']
                        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                        df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                        df['range'] = df['high'] - df['low']
                        available = len(df)
                        print(f"Reason: Not enough candles (have {available}, need 10) or did not meet criteria.")
                        print(f"Last {min(10, available)} candles (15m):")
                        print(df.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                    elif tf == '1h':
                        df = self.historical_data[self.historical_data['symbol'] == symbol].copy()
                        df['body'] = df['close'] - df['open']
                        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
                        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
                        df['direction'] = (df['close'] > df['open']).astype(int) - (df['close'] < df['open']).astype(int)
                        df['range'] = df['high'] - df['low']
                        df_1h = df.copy()
                        df_1h['dt'] = pd.to_datetime(df_1h['timestamp'], unit='s')
                        df_1h = df_1h.set_index('dt').resample('1H').agg({
                            'open':'first','high':'max','low':'min','close':'last',
                            'body':'sum','upper_wick':'sum','lower_wick':'sum','direction':'sum','range':'sum','timestamp':'last'
                        }).dropna().reset_index()
                        available = len(df_1h)
                        print(f"Reason: Not enough candles (have {available}, need 10) or did not meet criteria.")
                        print(f"Last {min(10, available)} candles (1h):")
                        print(df_1h.tail(10)[['timestamp','open','high','low','close','body','upper_wick','lower_wick','direction','range']])
                    print(f"---------------------\n")

    def close(self):
        if self.ws:
            self.ws.close()

def main():
    trader = DerivTrader()
    try:
        balance_info = trader.get_balance()
        initial_balance = float(balance_info['balance'])
        if initial_balance < 1:
            print("Insufficient balance to place trades!")
            return
        print(f"Initial Balance: ${initial_balance:.2f}")
        while True:
            try:
                current_balance = float(trader.get_balance()['balance'])
                profit_loss = current_balance - initial_balance
                print(f"Current Balance: ${current_balance:.2f} | P/L: ${profit_loss:.2f}")
                trader.execute_trades(max_concurrent=3, stake_amount=1)
                print("Waiting for next cycle...")
                time.sleep(900)
            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
                time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            final_balance = float(trader.get_balance()['balance'])
            print(f"Final Balance: ${final_balance:.2f}")
        except:
            pass
        trader.close()

if __name__ == "__main__":
    main()