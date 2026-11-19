"""Repository classes for StockSight database operations."""

from typing import Any, Dict, List, Optional

from src.database.db import get_connection
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StockRepository:
    """CRUD for stocks table."""

    @staticmethod
    def upsert(symbol: str, name: str, market: str = "", currency: str = "",
               country: str = "", provider_symbol: str = "", verified: bool = False,
               notes: str = "") -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO stocks (symbol, name, market, currency, country,
                                    provider_symbol, verified, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(symbol) DO UPDATE SET
                    name=excluded.name,
                    market=excluded.market,
                    currency=excluded.currency,
                    country=excluded.country,
                    provider_symbol=excluded.provider_symbol,
                    verified=excluded.verified,
                    notes=excluded.notes,
                    updated_at=datetime('now')
                """,
                (symbol, name, market, currency, country, provider_symbol or symbol,
                 1 if verified else 0, notes),
            )

    @staticmethod
    def get(symbol: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM stocks WHERE symbol = ?", (symbol,)
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_all() -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM stocks ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def count() -> int:
        with get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]


class TrainingRunRepository:
    """CRUD for training_runs."""

    @staticmethod
    def insert(data: Dict[str, Any]) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO training_runs (
                    symbol, model_version, sequence_length, hidden_size_1,
                    hidden_size_2, dropout, learning_rate, batch_size,
                    epochs_trained, best_epoch, train_loss, val_loss,
                    status, checkpoint_path, scaler_path, metadata_path,
                    started_at, finished_at, notes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    data.get("symbol"),
                    data.get("model_version"),
                    data.get("sequence_length"),
                    data.get("hidden_size_1"),
                    data.get("hidden_size_2"),
                    data.get("dropout"),
                    data.get("learning_rate"),
                    data.get("batch_size"),
                    data.get("epochs_trained"),
                    data.get("best_epoch"),
                    data.get("train_loss"),
                    data.get("val_loss"),
                    data.get("status", "completed"),
                    data.get("checkpoint_path"),
                    data.get("scaler_path"),
                    data.get("metadata_path"),
                    data.get("started_at"),
                    data.get("finished_at"),
                    data.get("notes"),
                ),
            )
            return cur.lastrowid

    @staticmethod
    def get_latest(symbol: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM training_runs
                WHERE symbol = ? AND status = 'completed'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_for_symbol(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM training_runs
                WHERE symbol = ?
                ORDER BY finished_at DESC LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def count() -> int:
        with get_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM training_runs WHERE status = 'completed'"
            ).fetchone()[0]


class PredictionRepository:
    """CRUD for predictions."""

    @staticmethod
    def insert(data: Dict[str, Any]) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    symbol, model_version, last_close, predicted_close,
                    absolute_change, percent_change, last_trading_date,
                    target_date, prediction_timestamp, currency, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("symbol"),
                    data.get("model_version"),
                    data.get("last_close"),
                    data.get("predicted_close"),
                    data.get("absolute_change"),
                    data.get("percent_change"),
                    data.get("last_trading_date"),
                    data.get("target_date"),
                    data.get("prediction_timestamp") or utc_now_iso(),
                    data.get("currency"),
                    data.get("notes"),
                ),
            )
            return cur.lastrowid

    @staticmethod
    def list_recent(limit: int = 20) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY prediction_timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def list_for_symbol(symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE symbol = ?
                ORDER BY prediction_timestamp DESC LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def count() -> int:
        with get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


class EvaluationRepository:
    """CRUD for evaluation_results."""

    @staticmethod
    def insert(data: Dict[str, Any]) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluation_results (
                    training_run_id, symbol, model_version,
                    mae, rmse, mape, directional_accuracy,
                    baseline_mae, baseline_rmse, baseline_mape,
                    baseline_directional_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("training_run_id"),
                    data.get("symbol"),
                    data.get("model_version"),
                    data.get("mae"),
                    data.get("rmse"),
                    data.get("mape"),
                    data.get("directional_accuracy"),
                    data.get("baseline_mae"),
                    data.get("baseline_rmse"),
                    data.get("baseline_mape"),
                    data.get("baseline_directional_accuracy"),
                ),
            )
            return cur.lastrowid

    @staticmethod
    def get_latest(symbol: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM evaluation_results
                WHERE symbol = ?
                ORDER BY evaluated_at DESC LIMIT 1
                """,
                (symbol,),
            ).fetchone()
            return dict(row) if row else None
