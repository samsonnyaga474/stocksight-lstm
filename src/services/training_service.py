"""Orchestrate data prep, training and artifact saving."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from config.stock_registry import get_stock_info
from src.database.repositories import (
    EvaluationRepository,
    StockRepository,
    TrainingRunRepository,
)
from src.evaluation.baseline import NaiveBaseline
from src.evaluation.metrics import compute_metrics
from src.models.model_manager import ModelManager
from src.services.stock_service import canonical_symbol
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingService:
    """End-to-end training workflow."""

    def __init__(self):
        self.manager = ModelManager()
        self.train_repo = TrainingRunRepository()
        self.eval_repo = EvaluationRepository()
        self.settings = get_settings()
        self._is_training = False

    @property
    def is_training(self) -> bool:
        return self._is_training

    def train(
        self,
        df: pd.DataFrame,
        symbol: str,
        sequence_length: Optional[int] = None,
        hidden_size_1: Optional[int] = None,
        hidden_size_2: Optional[int] = None,
        dropout: Optional[float] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if self._is_training:
            raise RuntimeError("A training job is already running.")

        # Always use canonical application identity for DB + artifacts
        symbol = canonical_symbol(symbol)

        self._is_training = True
        started_at = utc_now_iso()
        version = self.manager.make_version(symbol)

        try:
            # Import the training stack lazily: dashboard pages stay light.
            from src.data.preprocessing import Preprocessor
            from src.models.lstm_model import LSTMPredictor
            from src.models.trainer import Trainer

            # Guarantee the parent stock row exists before FK-backed inserts.
            self._ensure_stock_row(symbol)

            # Preprocess
            preprocessor = Preprocessor(sequence_length=sequence_length)
            datasets = preprocessor.prepare_datasets(df)

            X_train = datasets["X_train"]
            y_train = datasets["y_train"]
            X_val = datasets["X_val"]
            y_val = datasets["y_val"]
            X_test = datasets["X_test"]
            y_test = datasets["y_test"]
            test_df = datasets["test_df"]

            if len(X_train) < 10:
                raise ValueError("Not enough training sequences. Need more historical data.")

            # Model & trainer
            model = LSTMPredictor(
                input_size=len(preprocessor.feature_columns),
                hidden_size_1=hidden_size_1,
                hidden_size_2=hidden_size_2,
                dropout=dropout,
            )
            trainer = Trainer(
                model=model,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_epochs=max_epochs,
            )

            train_result = trainer.train(
                X_train, y_train, X_val, y_val, progress_callback=progress_callback
            )

            # Save artifacts
            ckpt_path = self.manager.checkpoint_path(version)
            scaler_path = self.manager.scaler_path(version)
            trainer.save_checkpoint(ckpt_path)
            preprocessor.save(scaler_path)

            # Evaluate on test set
            eval_metrics = self._evaluate(model, preprocessor, X_test, y_test, test_df, trainer.device)

            meta = {
                "symbol": symbol,
                "model_version": version,
                "sequence_length": preprocessor.sequence_length,
                "hidden_size_1": model.hidden_size_1,
                "hidden_size_2": model.hidden_size_2,
                "dropout": model.dropout_rate,
                "learning_rate": trainer.learning_rate,
                "batch_size": trainer.batch_size,
                "epochs_trained": train_result["epochs_trained"],
                "best_epoch": train_result["best_epoch"],
                "train_loss": train_result["final_train_loss"],
                # Use best validation loss (matches restored checkpoint), not last epoch
                "val_loss": train_result["best_val_loss"],
                "final_val_loss": train_result["final_val_loss"],
                "test_metrics": eval_metrics,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
            }
            meta_path = self.manager.save_metadata(version, meta)

            finished_at = utc_now_iso()

            # Persist training run
            run_id = self.train_repo.insert({
                "symbol": symbol,
                "model_version": version,
                "sequence_length": preprocessor.sequence_length,
                "hidden_size_1": model.hidden_size_1,
                "hidden_size_2": model.hidden_size_2,
                "dropout": model.dropout_rate,
                "learning_rate": trainer.learning_rate,
                "batch_size": trainer.batch_size,
                "epochs_trained": train_result["epochs_trained"],
                "best_epoch": train_result["best_epoch"],
                "train_loss": train_result["final_train_loss"],
                "val_loss": train_result["best_val_loss"],
                "status": "completed",
                "checkpoint_path": str(ckpt_path),
                "scaler_path": str(scaler_path),
                "metadata_path": str(meta_path),
                "started_at": started_at,
                "finished_at": finished_at,
            })

            # Persist evaluation
            self.eval_repo.insert({
                "training_run_id": run_id,
                "symbol": symbol,
                "model_version": version,
                "mae": eval_metrics.get("lstm", {}).get("mae"),
                "rmse": eval_metrics.get("lstm", {}).get("rmse"),
                "mape": eval_metrics.get("lstm", {}).get("mape"),
                "directional_accuracy": eval_metrics.get("lstm", {}).get("directional_accuracy"),
                "baseline_mae": eval_metrics.get("baseline", {}).get("mae"),
                "baseline_rmse": eval_metrics.get("baseline", {}).get("rmse"),
                "baseline_mape": eval_metrics.get("baseline", {}).get("mape"),
                "baseline_directional_accuracy": eval_metrics.get("baseline", {}).get("directional_accuracy"),
            })

            return {
                "success": True,
                "model_version": version,
                "training_result": train_result,
                "evaluation": eval_metrics,
                "checkpoint_path": str(ckpt_path),
                "scaler_path": str(scaler_path),
                "run_id": run_id,
            }

        except Exception as e:
            logger.exception("Training failed for %s", symbol)
            return {
                "success": False,
                "error": str(e),
                "model_version": version,
            }
        finally:
            self._is_training = False

    def _ensure_stock_row(self, symbol: str) -> None:
        """Insert/update the parent stock row required by foreign keys."""
        info = get_stock_info(symbol) or {}
        StockRepository.upsert(
            symbol=symbol,
            name=info.get("name", symbol),
            market=info.get("market", ""),
            currency=info.get("currency", "USD"),
            country=info.get("country", ""),
            provider_symbol=info.get("provider_symbol", symbol),
            verified=bool(info.get("verified", False)),
            notes=info.get("notes", ""),
        )

    def _evaluate(
        self,
        model: LSTMPredictor,
        preprocessor: Preprocessor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame,
        device,
    ) -> Dict[str, Any]:
        if len(X_test) == 0:
            return {"lstm": {}, "baseline": {}, "note": "No test samples"}

        model.eval()
        import torch
        with torch.no_grad():
            X_t = torch.from_numpy(X_test).float().to(device)
            preds_scaled = model(X_t).cpu().numpy().ravel()

        y_true = preprocessor.inverse_transform_target(y_test)
        y_pred = preprocessor.inverse_transform_target(preds_scaled)

        # Previous closes for directional accuracy & baseline
        # The last close of each sequence is approximately the previous day
        # For simplicity we use the true previous value from the scaled target shifted
        # A practical approximation: use y_true shifted or the Close from test_df
        # Here we reconstruct last closes from the sequences' final Close feature
        # Feature index of Close is usually 3
        close_idx = preprocessor.feature_columns.index("Close") if "Close" in preprocessor.feature_columns else 3
        last_closes_scaled = X_test[:, -1, close_idx]
        last_closes = preprocessor.inverse_transform_target(last_closes_scaled)

        lstm_metrics = compute_metrics(y_true, y_pred, y_prev=last_closes)

        baseline = NaiveBaseline()
        baseline_metrics = baseline.evaluate(y_true, last_closes)

        return {
            "lstm": lstm_metrics,
            "baseline": baseline_metrics,
            "n_test": len(y_true),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "last_closes": last_closes.tolist(),
        }
