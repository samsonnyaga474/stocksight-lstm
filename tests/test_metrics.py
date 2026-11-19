"""Tests for evaluation metrics and baseline."""

import numpy as np
import pytest

from src.evaluation.metrics import mae, rmse, mape, directional_accuracy, compute_metrics
from src.evaluation.baseline import NaiveBaseline


def test_mae():
    assert mae(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(0.5)


def test_rmse():
    assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_mape_handles_zero():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 11.0, 18.0])
    # Only non-zero actuals used
    result = mape(y_true, y_pred)
    assert not np.isnan(result)


def test_directional_accuracy():
    y_prev = np.array([10.0, 10.0, 10.0, 10.0])
    y_true = np.array([11.0, 9.0, 12.0, 8.0])
    y_pred = np.array([11.5, 9.5, 9.0, 7.0])  # correct, correct, wrong, correct
    acc = directional_accuracy(y_true, y_pred, y_prev)
    assert acc == pytest.approx(75.0)


def test_naive_baseline():
    baseline = NaiveBaseline()
    last = np.array([100.0, 101.0, 99.0])
    y_true = np.array([102.0, 100.0, 98.0])
    metrics = baseline.evaluate(y_true, last)
    assert "mae" in metrics
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(y_true - last)))


def test_naive_baseline_directional_uses_persistence():
    baseline = NaiveBaseline()
    last = np.array([10.0, 11.0, 12.0, 13.0])
    y_true = np.array([11.0, 12.0, 13.0, 12.0])
    metrics = baseline.evaluate(y_true, last)
    # Previous moves: +1, +1, +1. Actual moves after each close: +1, +1, -1.
    assert metrics["directional_accuracy"] == pytest.approx(200.0 / 3.0)


def test_naive_baseline_directional_nan_for_single_sample():
    baseline = NaiveBaseline()
    metrics = baseline.evaluate(np.array([10.0]), np.array([9.0]))
    assert np.isnan(metrics["directional_accuracy"])
