"""Tests for SQLite schema and repositories."""

import os
from pathlib import Path

import pytest

# Use a temporary database for tests
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_stocksight.db"
    monkeypatch.setenv("STOCKSIGHT_TEST_DB", str(db_path))
    # Patch settings path via monkeypatch on get_db_path
    from src.database import db as db_module
    monkeypatch.setattr(db_module, "get_db_path", lambda: db_path)
    db_module.init_database()
    yield db_path


def test_init_and_stock_repo(temp_db):
    from src.database.repositories import StockRepository
    StockRepository.upsert("TEST", "Test Corp", market="TEST", currency="USD", verified=True)
    row = StockRepository.get("TEST")
    assert row is not None
    assert row["name"] == "Test Corp"
    assert StockRepository.count() >= 1


def test_prediction_persistence(temp_db):
    from src.database.repositories import PredictionRepository, StockRepository
    StockRepository.upsert("TEST", "Test Corp")
    pid = PredictionRepository.insert({
        "symbol": "TEST",
        "model_version": "v1",
        "last_close": 100.0,
        "predicted_close": 101.5,
        "absolute_change": 1.5,
        "percent_change": 1.5,
        "last_trading_date": "2024-01-15",
        "target_date": "2024-01-16",
        "currency": "USD",
    })
    assert pid > 0
    rows = PredictionRepository.list_for_symbol("TEST")
    assert len(rows) >= 1
    assert rows[0]["predicted_close"] == 101.5
