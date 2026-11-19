"""Generate and store next-day predictions."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from config.stock_registry import get_stock_info
from src.database.repositories import (
    PredictionRepository,
    StockRepository,
    TrainingRunRepository,
)
from src.models.model_manager import ModelManager
from src.services.stock_service import canonical_symbol
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Load the latest trained model and produce a forecast."""

    def __init__(self):
        self.manager = ModelManager()
        self.pred_repo = PredictionRepository()
        self.train_repo = TrainingRunRepository()

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        symbol = canonical_symbol(symbol)

        if model_version is None:
            run = self.train_repo.get_latest(symbol)
            if run:
                model_version = run["model_version"]
            else:
                model_version = self.manager.latest_version(symbol)

        if not model_version or not self.manager.exists(model_version):
            raise FileNotFoundError(
                f"No trained model found for {symbol}. Please train a model first."
            )

        meta = self.manager.load_metadata(model_version)
        if meta and meta.get("symbol") and meta["symbol"] != symbol:
            raise ValueError(
                f"Model {model_version} belongs to {meta['symbol']}, not {symbol}."
            )

        ckpt = self.manager.checkpoint_path(model_version)
        scaler = self.manager.scaler_path(model_version)

        from src.models.predictor import Predictor

        predictor = Predictor.from_checkpoint(ckpt, scaler)

        info = get_stock_info(symbol) or {}
        currency = info.get("currency", "USD")

        result = predictor.predict_next(
            df=df,
            symbol=symbol,
            model_version=model_version,
            currency=currency,
        )

        # Guarantee the parent stock row exists before the foreign-key insert.
        StockRepository.upsert(
            symbol=symbol,
            name=info.get("name", symbol),
            market=info.get("market", ""),
            currency=currency,
            country=info.get("country", ""),
            provider_symbol=info.get("provider_symbol", symbol),
            verified=bool(info.get("verified", False)),
            notes=info.get("notes", ""),
        )

        pred_id = self.pred_repo.insert(result)
        result["prediction_id"] = pred_id
        return result

    def history(self, symbol: Optional[str] = None, limit: int = 50):
        if symbol:
            return self.pred_repo.list_for_symbol(canonical_symbol(symbol), limit=limit)
        return self.pred_repo.list_recent(limit=limit)
