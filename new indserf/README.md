# Candle Pattern Machine Learning Pipeline

## Artificial Intelligence and Machine Learning in This Project

This project leverages advanced artificial intelligence (AI) and machine learning (ML) techniques to analyze financial candle data and support trading decisions. By combining deep learning models with unsupervised and supervised learning approaches, the system is able to:

- **Recognize complex patterns** in historical price data that are difficult for humans or traditional algorithms to detect.
- **Detect anomalies and regime shifts** in the market using an autoencoder, which identifies when current market behavior deviates from learned normal patterns.
- **Classify market conditions** and generate actionable trading signals (e.g., buy, sell, hold/neutral) using a neural network classifier.
- **Adapt to new data** through continual learning, enabling the models to improve over time and adjust to changing market environments.

AI/ML integration in this pipeline enables more robust, data-driven trading strategies by providing:
- Early warning of unusual or risky market conditions
- Improved accuracy in trade prediction and risk management
- Automation and scalability for large volumes of financial data

---

This repository implements a machine learning pipeline for discovering, learning, and backtesting patterns in financial candle data (OHLCV data) using deep learning techniques. The codebase is designed to work with historical data of various timeframes (e.g., 1-hour, 15-minute candles) and supports both supervised and unsupervised learning approaches.

## Features

- **Feature Engineering:**
  - Extracts engineered features (body, wicks, direction, range, etc.) from raw candle data.
  - Adds binary outcome labels for supervised learning tasks.

- **Unsupervised Pattern Learning:**
  - Uses an autoencoder neural network to learn typical candle patterns.
  - Detects anomalies (unusual patterns) by reconstruction error.
  - Supports continual learning: the model fine-tunes itself on mistakes during backtesting.
  - Results are saved as CSVs (e.g., `backtest_AUDCAD_M15.csv`), with columns:
    - `time`, `recon_error`, `is_anomaly`, `close`

- **Supervised Binary Classification:**
  - Implements a deep residual neural network to classify candle windows as bullish/bearish (or similar binary outcomes).
  - Supports advanced activations (GELU, Swish, Mish) and replay buffers for continual learning.
  - Backtesting script evaluates model performance and fine-tunes on mistakes during evaluation.

- **Pattern Discovery & Clustering:**
  - Clusters rolling windows of candle features to discover common patterns.
  - Detects anomalies using IsolationForest.

- **Automation Scripts:**
  - Wrapper scripts to train and backtest models for different timeframes using subprocesses.
  - Model checkpoint management and cleanup.

## File Overview

- `scripts/data_loader.py`: Loads and processes candle data, adds features and labels.
- `scripts/pattern_learner.py`: Defines and trains an autoencoder for unsupervised pattern learning.
- `scripts/backtest_model.py`: Backtests the autoencoder, flags anomalies, and supports online continual learning.
- `scripts/train_binary_classifier.py`: Defines and trains a deep binary classifier for candle windows.
- `scripts/backtest_binary_classifier.py`: Backtests the classifier, fine-tunes on mistakes, and outputs metrics.
- `scripts/pattern_discovery.py`: Clusters and analyzes rolling windows for unsupervised pattern discovery.
- `scripts/train_candle_models.py` & `scripts/backtest_candle_models.py`: Automate training/backtesting for multiple timeframes.
- `scripts/cleanup_old_model.py`: Moves model checkpoints for backup/cleanup.

## Example Output

A typical backtest output CSV (e.g., `backtest_AUDCAD_M15.csv`) contains:

```
time,recon_error,is_anomaly,close
2021-04-30 00:00:00,0.19850917,True,0.95441
2021-04-30 00:15:00,5.758217e-05,False,0.95468
...
```
- `recon_error`: Autoencoder reconstruction error for each candle window.
- `is_anomaly`: Whether the pattern is flagged as unusual (anomaly).
- `close`: Closing price for the candle.

## Usage

1. **Prepare Data:** Place historical candle data (CSV) in the specified data directory.
2. **Train Models:**
    - Run `train_candle_models.py` for unsupervised learning.
    - Run `train_binary_classifier.py` for supervised learning.
3. **Backtest:**
    - Use `backtest_candle_models.py` or `backtest_binary_classifier.py` to evaluate models and generate results.
4. **Analyze Results:**
    - Review generated CSVs for anomaly flags, predictions, and performance metrics.

## Requirements
- Python 3.x
- PyTorch
- scikit-learn
- pandas, numpy

## Notes
- The pipeline supports continual/online learning: models are fine-tuned on mistakes during backtesting.
- The codebase is modular and can be extended for other timeframes or asset types.

---

For more details, see inline comments in each script or contact the author.
