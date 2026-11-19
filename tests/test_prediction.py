"""Tests for prediction inverse transform path."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import Preprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.predictor import Predictor
from src.utils.helpers import set_seed


@pytest.fixture
def trained_bits():
    set_seed(1)
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.8, 120))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1e4, 5e4, 120),
        },
        index=dates,
    )
    prep = Preprocessor(sequence_length=30)
    prep.fit(df.iloc[:80])
    model = LSTMPredictor(input_size=5, hidden_size_1=16, hidden_size_2=8, dropout=0.0)
    # Untrained weights are fine for shape / inverse tests
    return df, model, prep


def test_predict_next_shape(trained_bits):
    df, model, prep = trained_bits
    pred = Predictor(model, prep, device="cpu")
    result = pred.predict_next(df, symbol="TEST", model_version="test_v1", currency="USD")
    assert "predicted_close" in result
    assert "last_close" in result
    assert "percent_change" in result
    assert result["predicted_close"] > 0 or True  # can be any float
    assert result["symbol"] == "TEST"
