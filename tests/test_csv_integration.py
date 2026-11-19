"""CSV import -> store -> train/predict path integration tests."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.dataset_store import DatasetStore
from src.data.csv_loader import CSVLoader
from src.data.data_validator import DataValidator, compute_min_rows
from src.services.stock_service import StockService, canonical_symbol
from src.database.db import init_database, get_db_path
from src.database import db as db_module




@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_path)
    init_database()
    yield

@pytest.fixture
def long_ohlcv(tmp_path):
    """Synthetic but long enough OHLCV for training (not real market data)."""
    n = max(compute_min_rows(), 250)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    close = 80 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close + rng.normal(0, 0.1, n),
            "High": close + 0.8,
            "Low": close - 0.8,
            "Close": close,
            "Volume": rng.integers(5000, 20000, n),
        }
    )
    path = tmp_path / "synthetic_long.csv"
    df.to_csv(path, index=False)
    return path, n


def test_csv_import_and_register(tmp_path, long_ohlcv, monkeypatch):
    path, n = long_ohlcv
    # Point DatasetStore at tmp
    store = DatasetStore(base_dir=tmp_path / "imports")
    loader = CSVLoader()
    df, meta = loader.load(path, symbol="SYNTH.TEST")
    assert len(df) == n
    reg = store.register_import("SYNTH.TEST", df, original_filename="synthetic_long.csv")
    assert store.has_import("SYNTH.TEST")
    loaded = store.get_import("SYNTH.TEST")
    assert loaded is not None
    df2, meta2 = loaded
    assert len(df2) == n
    assert meta2["source"] == "csv_import"


def test_stock_service_prefers_import(tmp_path, long_ohlcv, monkeypatch):
    """After registering a CSV, load_data must return it without needing network."""
    path, n = long_ohlcv
    # Use isolated store
    from src.data import dataset_store as ds_mod
    monkeypatch.setattr(ds_mod, "IMPORTS_DIR", tmp_path / "imports")
    monkeypatch.setattr(ds_mod, "REGISTRY_PATH", tmp_path / "imports" / "registry.json")

    service = StockService()
    service.dataset_store = DatasetStore(base_dir=tmp_path / "imports")

    df, meta = service.load_csv(path, symbol="KE.TEST", original_filename="x.csv")
    assert meta.get("data_source") == "csv_import"
    assert service.has_imported_data("KE.TEST")

    # Second load should hit import, not yfinance
    df2, meta2 = service.load_data("KE.TEST", prefer_import=True)
    assert meta2.get("data_source") == "csv_import" or meta2.get("from_import")
    assert len(df2) == len(df)


def test_canonical_symbol_consistency():
    # Known registry entry
    assert canonical_symbol("AAPL") in ("AAPL", "AAPL")
    s = canonical_symbol("scom.nr")
    assert isinstance(s, str) and len(s) > 0


def test_min_rows_computation():
    m = compute_min_rows(sequence_length=60, train_ratio=0.7, min_train_sequences=10)
    assert m > 80
    assert m >= int(np.ceil((10 + 60) / 0.7)) + 60 - 5  # approximate lower bound


def test_validator_rejects_too_small():
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 100},
        index=dates,
    )
    v = DataValidator(sequence_length=60)
    report = v.validate(df)
    assert report["is_valid"] is False
    assert any("Insufficient" in e for e in report["critical_errors"])
