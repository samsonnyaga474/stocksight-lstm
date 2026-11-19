"""Next-day price prediction using a trained LSTM."""

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import Preprocessor
from src.models.lstm_model import LSTMPredictor
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Load model + scalers and produce a single next-day close prediction."""

    def __init__(
        self,
        model: LSTMPredictor,
        preprocessor: Preprocessor,
        device: Optional[str] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        scaler_path: Path,
        device: Optional[str] = None,
    ) -> "Predictor":
        """Factory that loads weights and scalers from disk."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {})
        model = LSTMPredictor(
            input_size=cfg.get("input_size", 5),
            hidden_size_1=cfg.get("hidden_size_1", 64),
            hidden_size_2=cfg.get("hidden_size_2", 32),
            dropout=cfg.get("dropout", 0.2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        preprocessor = Preprocessor.load(scaler_path)
        if preprocessor.feature_columns and cfg.get("input_size") not in (None, len(preprocessor.feature_columns)):
            raise ValueError(
                "Checkpoint input size does not match the saved feature scaler "
                f"({cfg.get('input_size')} vs {len(preprocessor.feature_columns)})."
            )
        return cls(model, preprocessor, device=device)

    def predict_next(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        model_version: str = "",
        currency: str = "USD",
    ) -> Dict[str, Any]:
        """
        Use the most recent sequence_length rows to predict the next close.
        """
        seq = self.preprocessor.sequence_length
        if len(df) < seq:
            raise ValueError(
                f"Need at least {seq} observations, got {len(df)}"
            )

        missing = [c for c in self.preprocessor.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Prediction data is missing required columns: {missing}")

        # Ensure a chronological DatetimeIndex
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        df = df.sort_index()
        if len(df) < seq:
            raise ValueError(
                f"Need at least {seq} dated observations, got {len(df)}"
            )
        recent = df.iloc[-seq:]

        scaled = self.preprocessor.transform_features(recent)
        # shape -> (1, seq, features)
        x = torch.from_numpy(scaled).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_pred = self.model(x).cpu().numpy().ravel()

        pred_close = float(self.preprocessor.inverse_transform_target(scaled_pred)[0])
        last_close = float(recent["Close"].iloc[-1])
        last_date = recent.index[-1]

        abs_change = pred_close - last_close
        pct_change = (abs_change / last_close * 100.0) if last_close != 0 else 0.0

        # Rough next trading day (skip weekends; holidays not handled)
        if hasattr(last_date, "to_pydatetime"):
            last_dt = last_date.to_pydatetime()
        else:
            last_dt = pd.Timestamp(last_date).to_pydatetime()
        next_dt = last_dt + timedelta(days=1)
        while next_dt.weekday() >= 5:  # Saturday=5, Sunday=6
            next_dt += timedelta(days=1)

        result = {
            "symbol": symbol,
            "model_version": model_version,
            "last_close": last_close,
            "predicted_close": pred_close,
            "absolute_change": abs_change,
            "percent_change": pct_change,
            "last_trading_date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date),
            "target_date": next_dt.strftime("%Y-%m-%d"),
            "prediction_timestamp": utc_now_iso(),
            "currency": currency,
            "notes": "Forecast for educational use only. Not financial advice.",
        }
        logger.info(
            "Prediction for %s: last=%.4f -> next=%.4f (%.2f%%)",
            symbol, last_close, pred_close, pct_change,
        )
        return result
