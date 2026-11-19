"""Tests for data validation and CSV loading."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.data_validator import DataValidator
from src.data.csv_loader import CSVLoader


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, size=300))
    high = close + rng.uniform(0, 2, size=300)
    low = close - rng.uniform(0, 2, size=300)
    open_ = close + rng.normal(0, 0.5, size=300)
    volume = rng.integers(1000, 10000, size=300)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_validator_accepts_clean_data(sample_ohlcv):
    v = DataValidator()
    report = v.validate(sample_ohlcv)
    assert report["is_valid"] is True
    assert len(report["critical_errors"]) == 0


def test_validator_rejects_empty():
    v = DataValidator()
    report = v.validate(pd.DataFrame())
    assert report["is_valid"] is False


def test_validator_detects_negative_price(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[5, df.columns.get_loc("Close")] = -10
    v = DataValidator()
    report = v.validate(df)
    assert report["is_valid"] is False
    assert any("negative" in e.lower() for e in report["critical_errors"])


def test_validator_detects_insufficient_rows():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 100},
        index=dates,
    )
    v = DataValidator(sequence_length=60)
    report = v.validate(df)
    assert report["is_valid"] is False
    assert any("Insufficient" in e for e in report["critical_errors"])


def test_clean_drops_nan(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[3, df.columns.get_loc("Close")] = np.nan
    v = DataValidator()
    cleaned = v.clean(df)
    assert cleaned["Close"].isnull().sum() == 0
    assert len(cleaned) == len(sample_ohlcv) - 1


def test_csv_loader(tmp_path, sample_ohlcv):
    path = tmp_path / "test.csv"
    out = sample_ohlcv.reset_index()
    out = out.rename(columns={"index": "Date"})
    out.to_csv(path, index=False)

    loader = CSVLoader()
    df, meta = loader.load(path, symbol="TEST")
    assert len(df) == 300
    assert "Close" in df.columns
    assert meta["rows"] == 300
