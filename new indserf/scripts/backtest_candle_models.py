import subprocess

def backtest_model(file_pattern, label):
    print(f"\nBacktesting {label} candle model...")
    result = subprocess.run([
        'python3', 'scripts/backtest_model.py',
        '--data_dir', '/Users/yashasnaidu/AI/historical_data',
        '--file_pattern', file_pattern,
        '--model_path', 'models/pattern_autoencoder.pth'
    ])
    if result.returncode == 0:
        print(f"{label} candle model backtest complete.")
    else:
        print(f"Error backtesting {label} candle model!")

def main():
    print("What do you want to backtest?")
    print("1  - 1 hour candles")
    print("15 - 15 min candles")
    print("b  - both")
    choice = input("Enter your choice (1/15/b): ").strip()
    if choice == '1':
        backtest_model('*.csv', '1H')
    elif choice == '15':
        backtest_model('*_M15.csv', '15min')
    elif choice.lower() == 'b':
        backtest_model('*.csv', '1H')
        backtest_model('*_M15.csv', '15min')
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
