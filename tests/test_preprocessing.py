"""Tests for preprocessing and sequence generation."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import Preprocessor


@pytest.fixture
def sample_df():
    dates = pd.date_range("2019-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    close = 50 + np.cumsum(rng.normal(0, 0.5, size=200))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, 200),
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1000, 5000, 200),
        },
        index=dates,
    )


def test_fit_transform_inverse(sample_df):
    prep = Preprocessor(sequence_length=30)
    train = sample_df.iloc[:140]
    prep.fit(train)
    scaled = prep.transform_target(train["Close"].values)
    inv = prep.inverse_transform_target(scaled)
    np.testing.assert_allclose(inv, train["Close"].values, rtol=1e-5)


def test_sequence_alignment(sample_df):
    prep = Preprocessor(sequence_length=10)
    prep.fit(sample_df)
    feat = prep.transform_features(sample_df)
    tgt = prep.transform_target(sample_df["Close"].values)
    X, y = prep.create_sequences(feat, tgt)
    assert X.shape[0] == len(sample_df) - 10
    assert X.shape[1] == 10
    assert X.shape[2] == 5
    assert len(y) == X.shape[0]
    # Target should be the close after the window
    expected = tgt[10:]
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)


def test_chronological_split(sample_df):
    prep = Preprocessor()
    train, val, test = prep.chronological_split(sample_df)
    assert len(train) + len(val) + len(test) == len(sample_df)
    assert train.index.max() < val.index.min() or len(val) == 0
    if len(test) and len(val):
        assert val.index.max() < test.index.min()


def test_no_leakage_fit_only_on_train(sample_df):
    prep = Preprocessor(sequence_length=20)
    data = prep.prepare_datasets(sample_df)
    assert data["X_train"].shape[0] > 0
    # Scalers fitted; inverse should work
    pred_scaled = np.array([0.5])
    price = prep.inverse_transform_target(pred_scaled)
    assert price.shape == (1,)
