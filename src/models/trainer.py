"""Training loop with early stopping and checkpointing."""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import get_settings
from src.models.dataset import StockSequenceDataset
from src.models.lstm_model import LSTMPredictor
from src.utils.helpers import ensure_dir, set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Train an LSTMPredictor and return history + best state."""

    def __init__(
        self,
        model: Optional[LSTMPredictor] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        settings = get_settings()
        self.learning_rate = learning_rate if learning_rate is not None else settings.learning_rate
        self.batch_size = batch_size if batch_size is not None else settings.batch_size
        self.max_epochs = max_epochs if max_epochs is not None else settings.max_epochs
        self.patience = patience if patience is not None else settings.early_stopping_patience
        self.seed = seed if seed is not None else settings.random_seed

        set_seed(self.seed)

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model or LSTMPredictor()
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Note: verbose= was removed in recent PyTorch versions; omit it for compatibility
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        self.best_state: Optional[Dict] = None
        self.best_epoch = 0
        self.best_val_loss = float("inf")

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = StockSequenceDataset(X, y)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        progress_callback receives a dict each epoch with:
        epoch, train_loss, val_loss, best_epoch, elapsed, status
        """
        if len(X_train) == 0:
            raise ValueError("Training set is empty")

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if len(X_val) > 0 else None

        early = EarlyStopping(patience=self.patience)
        start_time = time.time()

        logger.info(
            "Training on %s | samples train=%d val=%d | epochs=%d",
            self.device,
            len(X_train),
            len(X_val),
            self.max_epochs,
        )

        for epoch in range(1, self.max_epochs + 1):
            # ---- train ----
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0

            # ---- validate ----
            avg_val = 0.0
            if val_loader is not None and len(X_val) > 0:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        pred = self.model(xb)
                        loss = self.criterion(pred, yb)
                        val_losses.append(loss.item())
                avg_val = float(np.mean(val_losses)) if val_losses else 0.0
                self.scheduler.step(avg_val)
            else:
                avg_val = avg_train  # fallback

            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)

            # Track best
            if avg_val < self.best_val_loss:
                self.best_val_loss = avg_val
                self.best_epoch = epoch
                self.best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            elapsed = time.time() - start_time
            status = {
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "elapsed": elapsed,
                "status": "running",
            }
            if progress_callback:
                progress_callback(status)

            logger.info(
                "Epoch %03d | train_loss=%.6f | val_loss=%.6f | best=%d",
                epoch, avg_train, avg_val, self.best_epoch,
            )

            if early.step(avg_val):
                logger.info("Early stopping at epoch %d", epoch)
                status["status"] = "early_stopped"
                if progress_callback:
                    progress_callback(status)
                break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        result = {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.history["train_loss"]),
            "device": str(self.device),
            "final_train_loss": self.history["train_loss"][-1] if self.history["train_loss"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
        }
        return result

    def save_checkpoint(self, path: Path) -> None:
        ensure_dir(path.parent)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "config": {
                    "input_size": self.model.input_size,
                    "hidden_size_1": self.model.hidden_size_1,
                    "hidden_size_2": self.model.hidden_size_2,
                    "dropout": self.model.dropout_rate,
                },
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)
