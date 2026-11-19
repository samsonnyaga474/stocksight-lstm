"""Simple chronological backtesting helper."""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def simple_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
) -> Dict:
    """
    Compute overall metrics and optionally a rolling summary.
    This is a lightweight evaluation helper, not a full trading simulator.
    """
    metrics = compute_metrics(y_true, y_pred)
    result = {
        "metrics": metrics,
        "n_predictions": len(y_true),
    }
    if dates is not None and len(dates) == len(y_true):
        result["start_date"] = str(dates[0].date())
        result["end_date"] = str(dates[-1].date())
    return result
