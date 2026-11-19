# StockSight Architecture

## Overview

StockSight is organised into clear layers so that data handling, modelling, persistence and the user interface remain independent.

```
┌─────────────────────────────────────────┐
│           Streamlit Dashboard           │
│  (overview, explorer, training, …)      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│              Services Layer             │
│  StockService · TrainingService ·       │
│  PredictionService · HistoryService     │
└─────────────────┬───────────────────────┘
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐ ┌──────────┐ ┌──────────┐
│  Data   │ │  Models  │ │ Database │
│ Pipeline│ │  LSTM    │ │  SQLite  │
└─────────┘ └──────────┘ └──────────┘
```

## Core Packages

### `src/data`
- **DataFetcher** – yfinance download + cache
- **CSVLoader** – user CSV import
- **DataValidator** – quality checks
- **Preprocessor** – MinMax scaling (fit on train only), sequence generation, chronological split
- **CacheManager** – parquet cache under `data/raw`

### `src/models`
- **LSTMPredictor** – PyTorch `nn.Module`
- **StockSequenceDataset** – torch Dataset
- **Trainer** – training loop, early stopping, checkpoint
- **Predictor** – load weights + scalers, produce next-day forecast
- **ModelManager** – versioned files on disk

### `src/evaluation`
- Metrics (MAE, RMSE, MAPE, directional accuracy)
- Naive baseline
- Lightweight backtest helper

### `src/services`
High-level orchestration used by the dashboard. Services never contain Streamlit code.

### `src/database`
SQLite schema and repository classes for stocks, training runs, evaluations and predictions.

### `dashboard`
Streamlit pages and reusable chart/metric components.

## Data Flow (Training)

1. User selects stock and clicks Train.
2. StockService loads / validates OHLCV.
3. Preprocessor splits chronologically (70/15/15), fits scalers on train only, builds sequences.
4. Trainer runs LSTM with early stopping.
5. Best checkpoint, scalers and metadata are saved.
6. Test-set metrics and baseline comparison are computed and stored.

## Data Flow (Prediction)

1. Latest successful model version for the symbol is located.
2. Last `sequence_length` rows are scaled with the saved feature scaler.
3. Model runs in eval mode.
4. Output is inverse-transformed to price units and stored.
