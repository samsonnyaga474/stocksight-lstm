"""Tests for cache manager including Parquet/CSV fallback and corruption handling."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.cache_manager import CacheManager, _is_usable_ohlcv


@pytest.fixture
def sample_df():
    dates = pd.date_range("2022-01-01", periods=50, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 50))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1000, 5000, 50),
        },
        index=dates,
    )


def test_cache_put_and_get(tmp_path, sample_df):
    cm = CacheManager(cache_dir=tmp_path)
    path = cm.put("TEST", "1y", sample_df, meta={"provider": "test"})
    assert path is not None
    assert path.exists()
    loaded = cm.get("TEST", "1y")
    assert loaded is not None
    assert len(loaded) == len(sample_df)
    pd.testing.assert_series_equal(
        loaded["Close"].reset_index(drop=True),
        sample_df["Close"].reset_index(drop=True),
        check_names=False,
    )


def test_cache_miss(tmp_path):
    cm = CacheManager(cache_dir=tmp_path)
    assert cm.get("NOSUCH", "1y") is None


def test_cache_corrupt_returns_none(tmp_path, sample_df):
    """Corrupt both parquet and csv — get must return None, not empty/partial frame."""
    cm = CacheManager(cache_dir=tmp_path)
    cm.put("TEST", "1y", sample_df)
    for p in tmp_path.glob("TEST*"):
        if p.suffix in (".parquet", ".csv"):
            p.write_text("not valid data at all")
    result = cm.get("TEST", "1y")
    assert result is None  # graceful, no raise


def test_cache_corrupt_parquet_only(tmp_path, sample_df):
    cm = CacheManager(cache_dir=tmp_path)
    # Force CSV path by disabling parquet for this instance after put
    path = cm.put("TEST", "1y", sample_df)
    assert path is not None
    # Corrupt any parquet file; leave CSV intact if present, or rewrite CSV good
    for p in tmp_path.glob("*.parquet"):
        p.write_bytes(b"\x00\x01corrupt")
    # If only parquet existed, also write a good CSV so we can test parquet rejection
    # then corrupt and ensure None when both bad
    csv_path = cm._key_path("TEST", "1y", "csv")
    if csv_path.exists():
        csv_path.write_text("not valid data at all")
    result = cm.get("TEST", "1y")
    assert result is None


def test_cache_empty_csv_returns_none(tmp_path):
    cm = CacheManager(cache_dir=tmp_path)
    csv_path = cm._key_path("EMPTY", "1y", "csv")
    # Header only — parses to empty DataFrame
    csv_path.write_text("Open,High,Low,Close,Volume\n")
    result = cm.get("EMPTY", "1y")
    assert result is None


def test_cache_missing_ohlcv_columns_returns_none(tmp_path):
    cm = CacheManager(cache_dir=tmp_path)
    csv_path = cm._key_path("BADCOLS", "1y", "csv")
    csv_path.write_text("Date,Price,Qty\n2022-01-01,10,100\n2022-01-02,11,110\n")
    result = cm.get("BADCOLS", "1y")
    assert result is None


def test_is_usable_ohlcv_helper(sample_df):
    assert _is_usable_ohlcv(sample_df) is True
    assert _is_usable_ohlcv(pd.DataFrame()) is False
    assert _is_usable_ohlcv(None) is False
    bad = sample_df.drop(columns=["Volume"])
    assert _is_usable_ohlcv(bad) is False


def test_valid_csv_fallback(tmp_path, sample_df):
    """When only a valid CSV exists (no parquet), get returns the data."""
    cm = CacheManager(cache_dir=tmp_path)
    cm._use_parquet = False
    path = cm.put("CSVONLY", "1y", sample_df)
    assert path is not None
    assert path.suffix == ".csv"
    loaded = cm.get("CSVONLY", "1y")
    assert loaded is not None
    assert len(loaded) == len(sample_df)
    assert "Close" in loaded.columns


def test_cache_put_does_not_destroy_dataframe(tmp_path, sample_df, monkeypatch):
    """Force both parquet and CSV writes to fail; original df must stay intact."""
    cm = CacheManager(cache_dir=tmp_path)
    original_len = len(sample_df)
    original_close = sample_df["Close"].copy()

    def boom(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)

    result = cm.put("FAIL", "1y", sample_df)
    assert result is None
    assert len(sample_df) == original_len
    pd.testing.assert_series_equal(sample_df["Close"], original_close)
