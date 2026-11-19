"""Evaluation metrics for price forecasts."""

from typing import Dict

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    Rows where actual is zero are excluded to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """
    Percentage of times the predicted direction (up/down vs previous close)
    matches the actual direction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    # Treat zero movement as a match only if both are zero
    correct = (true_dir == pred_dir)
    return float(np.mean(correct) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray = None,
) -> Dict[str, float]:
    """Compute the standard suite of metrics."""
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    if y_prev is not None and len(y_prev) == len(y_true):
        result["directional_accuracy"] = directional_accuracy(y_true, y_pred, y_prev)
    else:
        result["directional_accuracy"] = float("nan")
    return result
