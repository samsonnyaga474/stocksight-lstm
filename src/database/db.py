"""SQLite database connection and schema initialization."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_db_path() -> Path:
    settings = get_settings()
    path = settings.database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection with row factory."""
    path = get_db_path()
    conn = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    market TEXT,
    currency TEXT,
    country TEXT,
    provider_symbol TEXT,
    verified INTEGER DEFAULT 0,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS data_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    provider TEXT,
    period TEXT,
    start_date TEXT,
    end_date TEXT,
    rows_count INTEGER,
    retrieved_at TEXT,
    adjusted INTEGER DEFAULT 1,
    cache_path TEXT,
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    sequence_length INTEGER,
    hidden_size_1 INTEGER,
    hidden_size_2 INTEGER,
    dropout REAL,
    learning_rate REAL,
    batch_size INTEGER,
    epochs_trained INTEGER,
    best_epoch INTEGER,
    train_loss REAL,
    val_loss REAL,
    status TEXT DEFAULT 'completed',
    checkpoint_path TEXT,
    scaler_path TEXT,
    metadata_path TEXT,
    started_at TEXT,
    finished_at TEXT,
    notes TEXT,
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER,
    symbol TEXT NOT NULL,
    model_version TEXT,
    mae REAL,
    rmse REAL,
    mape REAL,
    directional_accuracy REAL,
    baseline_mae REAL,
    baseline_rmse REAL,
    baseline_mape REAL,
    baseline_directional_accuracy REAL,
    evaluated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    model_version TEXT,
    last_close REAL,
    predicted_close REAL,
    absolute_change REAL,
    percent_change REAL,
    last_trading_date TEXT,
    target_date TEXT,
    prediction_timestamp TEXT DEFAULT (datetime('now')),
    currency TEXT,
    notes TEXT,
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_training_runs_symbol ON training_runs(symbol);
CREATE INDEX IF NOT EXISTS idx_evaluation_symbol ON evaluation_results(symbol);
"""


def init_database() -> None:
    """Create all tables if they do not exist."""
    path = get_db_path()
    logger.info("Initializing database at %s", path)
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database schema ready.")
