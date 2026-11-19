"""Aggregated history and overview statistics."""

from typing import Any, Dict, List, Optional

from src.database.repositories import (
    EvaluationRepository,
    PredictionRepository,
    StockRepository,
    TrainingRunRepository,
)
from src.models.model_manager import ModelManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HistoryService:
    """Provide summary statistics for the Overview page."""

    def __init__(self):
        self.stock_repo = StockRepository()
        self.train_repo = TrainingRunRepository()
        self.pred_repo = PredictionRepository()
        self.eval_repo = EvaluationRepository()
        self.manager = ModelManager()

    def overview(self) -> Dict[str, Any]:
        return {
            "n_stocks": self.stock_repo.count(),
            "n_trained_models": self.train_repo.count(),
            "n_predictions": self.pred_repo.count(),
            "recent_predictions": self.pred_repo.list_recent(limit=10),
            "stocks": self.stock_repo.list_all(),
        }

    def latest_training(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.train_repo.get_latest(symbol)

    def latest_evaluation(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.eval_repo.get_latest(symbol)

    def prediction_history(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.pred_repo.list_for_symbol(symbol, limit=limit)
