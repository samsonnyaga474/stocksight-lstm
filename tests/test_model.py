"""Tests for LSTM model architecture."""

import torch
import pytest

from src.models.lstm_model import LSTMPredictor
from src.utils.helpers import set_seed


def test_lstm_forward_shape():
    set_seed(42)
    model = LSTMPredictor(input_size=5, hidden_size_1=32, hidden_size_2=16, dropout=0.1)
    x = torch.randn(8, 60, 5)  # batch=8, seq=60, features=5
    out = model(x)
    assert out.shape == (8, 1)


def test_lstm_cpu():
    model = LSTMPredictor()
    model.to("cpu")
    x = torch.randn(2, 60, 5)
    out = model(x)
    assert out.device.type == "cpu"
    assert out.shape == (2, 1)


def test_parameter_count():
    model = LSTMPredictor(input_size=5, hidden_size_1=64, hidden_size_2=32)
    n = model.count_parameters()
    assert n > 0
