# LSTM Model Documentation

## Architecture

```
Input: (batch, 60, 5)   # 60 days × OHLCV
        │
        ▼
   LSTM (hidden=64, batch_first=True)
        │
   Dropout(0.2)
        │
   LSTM (hidden=32, batch_first=True)
        │
   Dropout(0.2)
        │
   Take last time step → (batch, 32)
        │
   Linear(32 → 1)
        │
Output: (batch, 1)      # scaled next-day Close
```

## Design Choices

- **Two LSTM layers** provide enough capacity for short-term temporal patterns while remaining trainable on CPU in a reasonable time.
- **Dropout** is applied between layers and before the final linear head to reduce overfitting on noisy financial series.
- **Batch-first** layout matches the shape produced by the sequence generator and is the common convention in recent PyTorch code.
- **MinMax scaling** to [0, 1] is applied separately to features and to the Close target. Scalers are fitted **only** on the training split to prevent look-ahead bias.
- **Early stopping** on validation loss (default patience 10) avoids unnecessary epochs and returns the best observed weights.

## Hyperparameters (defaults in `config.yaml`)

| Parameter          | Default |
|--------------------|---------|
| Sequence length    | 60      |
| Hidden size 1      | 64      |
| Hidden size 2      | 32      |
| Dropout            | 0.2     |
| Learning rate      | 0.001   |
| Batch size         | 32      |
| Max epochs         | 100     |
| Early-stop patience| 10      |
| Random seed        | 42      |

All of the above can be overridden from the Model Training page.

## Training Protocol

1. Chronological split: 70 % train / 15 % validation / 15 % test.
2. Scalers fitted on training observations only.
3. Validation and test sequences may use earlier observations purely as *context* (the first 60 rows of a split) so that the first targets still belong to the correct period; no future information enters the training set.
4. Loss: Mean Squared Error on the scaled target.
5. Optimizer: Adam.
6. Learning-rate scheduler: ReduceLROnPlateau on validation loss.
7. Best model (lowest validation loss) is restored at the end of training.

## Inference

1. Load checkpoint and associated scaler files.
2. Take the most recent 60 trading days.
3. Scale with the saved feature scaler.
4. Forward pass in `eval()` mode (no dropout).
5. Inverse-transform the single output with the target scaler.
6. Persist the prediction together with metadata.

## Baseline

A naive forecast that sets next close = last close is evaluated on the identical test window. The dashboard reports whether the LSTM improved on MAE / RMSE / MAPE / directional accuracy relative to this baseline.
