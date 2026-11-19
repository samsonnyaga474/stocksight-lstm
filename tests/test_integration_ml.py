"""
Integration tests: CSV import → TrainingService → PredictionService.

Uses deterministic synthetic OHLCV data only (not real market data).
Exercises the real Preprocessor, PyTorch LSTM, SQLite, and services.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.data.dataset_store import DatasetStore
from src.data.data_validator import compute_min_rows
from src.database import db as db_module
from src.database.db import init_database
from src.database.repositories import TrainingRunRepository, PredictionRepository
from src.services.stock_service import StockService, canonical_symbol
from src.services.training_service import TrainingService
from src.services.prediction_service import PredictionService
from src.models.model_manager import ModelManager


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Isolate DB, imports dir, and model artifact dirs under tmp_path."""
    db_path = tmp_path / "integration.db"
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_path)
    init_database()

    imports_dir = tmp_path / "imports"
    imports_dir.mkdir()

    # Point DatasetStore defaults used by StockService
    import src.data.dataset_store as ds_mod
    monkeypatch.setattr(ds_mod, "IMPORTS_DIR", imports_dir)
    monkeypatch.setattr(ds_mod, "REGISTRY_PATH", imports_dir / "registry.json")

    # Isolate model artifacts
    from config.settings import get_settings
    settings = get_settings()
    ckpt = tmp_path / "checkpoints"
    scalers = tmp_path / "scalers"
    meta = tmp_path / "metadata"
    for d in (ckpt, scalers, meta):
        d.mkdir()

    manager = ModelManager()
    monkeypatch.setattr(manager, "checkpoints_dir", ckpt)
    monkeypatch.setattr(manager, "scalers_dir", scalers)
    monkeypatch.setattr(manager, "metadata_dir", meta)

    return {
        "tmp": tmp_path,
        "imports_dir": imports_dir,
        "manager": manager,
        "db_path": db_path,
    }


@pytest.fixture
def synthetic_csv(tmp_path):
    """Long enough synthetic OHLCV CSV for seq=30 training (not real market data)."""
    n = max(compute_min_rows(sequence_length=30), 200)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(99)
    close = 50 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Open": close + rng.normal(0, 0.05, n),
            "High": close + 0.6,
            "Low": close - 0.6,
            "Close": close,
            "Volume": rng.integers(3000, 15000, n),
        }
    )
    path = tmp_path / "synthetic_integration.csv"
    df.to_csv(path, index=False)
    return path, n


def test_csv_to_training_integration(isolated_env, synthetic_csv, monkeypatch):
    """
    1. Import + register CSV
    2. load_data prefer_import
    3. Train real small CPU LSTM via TrainingService
    4. Confirm SQLite training run with correct canonical ticker
    """
    path, n = synthetic_csv
    env = isolated_env

    service = StockService()
    service.dataset_store = DatasetStore(base_dir=env["imports_dir"])

    # Register CSV
    df, meta = service.load_csv(path, symbol="INTG.TEST", original_filename="synthetic_integration.csv")
    assert meta.get("data_source") == "csv_import"
    assert len(df) == n
    assert service.has_imported_data("INTG.TEST")

    # Retrieve via load_data (must not need network)
    df2, meta2 = service.load_data("INTG.TEST", prefer_import=True)
    assert meta2.get("data_source") == "csv_import" or meta2.get("from_import")
    assert len(df2) == n

    # Train with short settings
    train_svc = TrainingService()
    train_svc.manager = env["manager"]

    result = train_svc.train(
        df=df2,
        symbol="INTG.TEST",
        sequence_length=30,
        hidden_size_1=16,
        hidden_size_2=8,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=3,
    )

    assert result.get("success") is True, result.get("error")
    version = result["model_version"]
    assert version
    assert env["manager"].exists(version)

    # SQLite persistence
    run = TrainingRunRepository.get_latest(canonical_symbol("INTG.TEST"))
    assert run is not None
    assert run["status"] == "completed"
    assert run["model_version"] == version
    assert run["symbol"] == canonical_symbol("INTG.TEST")
    assert run["epochs_trained"] >= 1
    assert run["val_loss"] is not None


def test_csv_to_prediction_integration(isolated_env, synthetic_csv, monkeypatch):
    """
    Train then predict end-to-end with imported data.
    Confirms PredictionService + SQLite prediction row.
    """
    path, n = synthetic_csv
    env = isolated_env

    service = StockService()
    service.dataset_store = DatasetStore(base_dir=env["imports_dir"])

    df, meta = service.load_csv(path, symbol="PRED.TEST", original_filename="synth.csv")
    df2, _ = service.load_data("PRED.TEST", prefer_import=True)

    train_svc = TrainingService()
    train_svc.manager = env["manager"]

    train_result = train_svc.train(
        df=df2,
        symbol="PRED.TEST",
        sequence_length=30,
        hidden_size_1=16,
        hidden_size_2=8,
        dropout=0.0,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=2,
    )
    assert train_result.get("success") is True, train_result.get("error")
    version = train_result["model_version"]

    pred_svc = PredictionService()
    pred_svc.manager = env["manager"]

    prediction = pred_svc.predict(df2, "PRED.TEST", model_version=version)

    assert "predicted_close" in prediction
    assert "last_close" in prediction
    assert prediction["symbol"] == canonical_symbol("PRED.TEST")
    assert prediction["model_version"] == version
    assert isinstance(prediction["predicted_close"], float)
    assert prediction.get("prediction_id") is not None

    # SQLite
    rows = PredictionRepository.list_for_symbol(canonical_symbol("PRED.TEST"), limit=5)
    assert len(rows) >= 1
    assert rows[0]["model_version"] == version
    assert rows[0]["symbol"] == canonical_symbol("PRED.TEST")
