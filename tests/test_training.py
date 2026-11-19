"""Tests for training loop (short runs on synthetic data)."""

import numpy as np
import pytest
import torch

from src.models.lstm_model import LSTMPredictor
from src.models.trainer import Trainer
from src.utils.helpers import set_seed


@pytest.fixture
def synthetic_sequences():
    set_seed(0)
    n = 80
    seq = 20
    feat = 5
    X = np.random.randn(n, seq, feat).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    return X, y


def test_training_runs(synthetic_sequences):
    X, y = synthetic_sequences
    split = 60
    model = LSTMPredictor(input_size=5, hidden_size_1=16, hidden_size_2=8, dropout=0.1)
    trainer = Trainer(
        model=model,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=3,
        patience=5,
        device="cpu",
        seed=0,
    )
    result = trainer.train(X[:split], y[:split], X[split:], y[split:])
    assert result["epochs_trained"] >= 1
    assert "train_loss" in result["history"]
    assert len(result["history"]["train_loss"]) == result["epochs_trained"]


def test_scheduler_constructs_without_verbose():
    """Regression: ReduceLROnPlateau must not pass unsupported verbose kwarg."""
    model = LSTMPredictor(input_size=5, hidden_size_1=8, hidden_size_2=4, dropout=0.0)
    trainer = Trainer(model=model, max_epochs=1, device="cpu", seed=1)
    # Simply constructing Trainer must not raise
    assert trainer.scheduler is not None
    # Step once
    trainer.scheduler.step(0.5)


def test_best_val_loss_tracked(synthetic_sequences):
    X, y = synthetic_sequences
    split = 60
    model = LSTMPredictor(input_size=5, hidden_size_1=16, hidden_size_2=8, dropout=0.0)
    trainer = Trainer(
        model=model,
        learning_rate=0.01,
        batch_size=16,
        max_epochs=5,
        patience=10,
        device="cpu",
        seed=2,
    )
    result = trainer.train(X[:split], y[:split], X[split:], y[split:])
    assert "best_val_loss" in result
    assert result["best_val_loss"] <= min(result["history"]["val_loss"]) + 1e-9
    assert result["best_epoch"] >= 1
