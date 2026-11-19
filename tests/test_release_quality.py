"""Release-quality regression tests for artifacts, integrity and edge cases.

All market data here is deterministic synthetic data, never real prices.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache_manager import CacheManager
from src.data.dataset_store import DatasetStore
from src.data.preprocessing import Preprocessor
from src.database import db as db_module
from src.database.db import init_database
from src.models.lstm_model import LSTMPredictor
from src.models.model_manager import ModelManager
from src.models.predictor import Predictor
from src.models.trainer import Trainer


def _synthetic_df(n: int = 140, seed: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 60 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.05, n),
            "High": close + 0.7,
            "Low": close - 0.7,
            "Close": close,
            "Volume": rng.integers(2000, 9000, n),
        },
        index=dates,
    )


def test_early_stopping_triggers_and_restores_best_weights():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(70, 12, 5)).astype(np.float32)
    y = rng.normal(size=70).astype(np.float32)

    model = LSTMPredictor(input_size=5, hidden_size_1=8, hidden_size_2=4, dropout=0.0)
    trainer = Trainer(
        model=model,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=40,
        patience=2,
        device="cpu",
        seed=11,
    )
    # Constant targets make validation loss plateau quickly.
    constant = np.zeros(70, dtype=np.float32)
    result = trainer.train(X[:55], constant[:55], X[55:], constant[55:])

    assert result["epochs_trained"] < 40
    assert result["best_val_loss"] == pytest.approx(min(result["history"]["val_loss"]))
    assert trainer.model is not None


def test_checkpoint_and_scaler_round_trip_produces_prediction(tmp_path):
    df = _synthetic_df(140)
    prep = Preprocessor(sequence_length=20)
    datasets = prep.prepare_datasets(df)

    model = LSTMPredictor(input_size=5, hidden_size_1=8, hidden_size_2=4, dropout=0.0)
    trainer = Trainer(model=model, max_epochs=2, batch_size=16, device="cpu", seed=5)
    trainer.train(
        datasets["X_train"],
        datasets["y_train"],
        datasets["X_val"],
        datasets["y_val"],
    )

    ckpt = tmp_path / "model.pt"
    scaler = tmp_path / "scaler.joblib"
    trainer.save_checkpoint(ckpt)
    prep.save(scaler)
    assert ckpt.exists() and scaler.exists()

    predictor = Predictor.from_checkpoint(ckpt, scaler, device="cpu")
    result = predictor.predict_next(df, symbol="SYNTH", model_version="v-test")

    assert np.isfinite(result["predicted_close"])
    assert result["symbol"] == "SYNTH"
    assert result["last_close"] == pytest.approx(float(df["Close"].iloc[-1]))


def test_model_version_is_unique_for_same_symbol():
    manager = ModelManager()
    versions = {manager.make_version("AAPL") for _ in range(50)}
    assert len(versions) == 50


def test_predictor_rejects_missing_feature_columns():
    df = _synthetic_df(80)
    prep = Preprocessor(sequence_length=10)
    prep.fit(df)
    model = LSTMPredictor(input_size=5, hidden_size_1=8, hidden_size_2=4, dropout=0.0)
    predictor = Predictor(model, prep, device="cpu")

    with pytest.raises(ValueError, match="missing required columns"):
        predictor.predict_next(df.drop(columns=["Volume"]))


def test_cache_uses_valid_csv_when_parquet_is_corrupt(tmp_path):
    cm = CacheManager(cache_dir=tmp_path)
    cm._use_parquet = False
    df = _synthetic_df(40)
    csv_path = cm.put("FALLBACK", "1y", df)
    assert csv_path is not None and csv_path.suffix == ".csv"

    parquet_path = cm._key_path("FALLBACK", "1y", "parquet")
    parquet_path.write_bytes(b"\x00\x01definitely-not-parquet")

    loaded = cm.get("FALLBACK", "1y")
    assert loaded is not None
    assert len(loaded) == len(df)


def test_foreign_key_blocks_orphan_prediction(tmp_path, monkeypatch):
    db_path = tmp_path / "fk.db"
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_path)
    init_database()

    from src.database.repositories import PredictionRepository

    with pytest.raises(sqlite3.IntegrityError):
        PredictionRepository.insert(
            {"symbol": "MISSING", "last_close": 1.0, "predicted_close": 2.0}
        )


def test_dataset_store_rejects_csv_without_ohlcv(tmp_path):
    base = tmp_path / "imports"
    base.mkdir()
    bad = base / "BAD_ohlcv.csv"
    bad.write_text("Date,Price\n2024-01-01,10\n", encoding="utf-8")
    (base / "registry.json").write_text(
        json.dumps({"BAD": {"symbol": "BAD", "path": str(bad)}}),
        encoding="utf-8",
    )
    store = DatasetStore(base_dir=base)
    assert store.get_import("BAD") is None


def test_upload_filename_is_sanitised():
    from dashboard.pages.stock_explorer import _safe_upload_name

    for raw in ["../../evil.csv", r"C:\temp\..\prices.csv", "bad name?!.csv"]:
        safe = _safe_upload_name(raw)
        assert "/" not in safe and "\\" not in safe
        assert safe.lower().endswith(".csv")
        assert ".." not in safe
