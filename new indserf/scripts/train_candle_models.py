import subprocess

def train_model(file_pattern, label):
    print(f"\nTraining {label} candle model...")
    result = subprocess.run([
        'python3', 'scripts/pattern_learner.py',
        '--data_dir', '/Users/yashasnaidu/AI/historical_data',
        '--file_pattern', file_pattern
    ])
    if result.returncode == 0:
        print(f"{label} candle model training complete.")
    else:
        print(f"Error training {label} candle model!")

def main():
    print("What do you want to train?")
    print("1  - 1 hour candles")
    print("15 - 15 min candles")
    print("b  - both")
    choice = input("Enter your choice (1/15/b): ").strip()
    if choice == '1':
        train_model('*.csv', '1H')
    elif choice == '15':
        train_model('*_M15.csv', '15min')
    elif choice.lower() == 'b':
        train_model('*.csv', '1H')
        train_model('*_M15.csv', '15min')
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
