"""Feature scaling and sequence generation for LSTM."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import get_settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Fit scalers on training data only, generate sliding-window sequences,
    and support inverse transformation of predictions.
    """

    def __init__(
        self,
        sequence_length: Optional[int] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        settings = get_settings()
        self.sequence_length = sequence_length or settings.sequence_length
        self.feature_columns = feature_columns or settings.features
        self.feature_scaler: Optional[MinMaxScaler] = None
        self.target_scaler: Optional[MinMaxScaler] = None
        self.fitted = False

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        """Fit scalers exclusively on the training portion."""
        features = train_df[self.feature_columns].values.astype(np.float64)
        targets = train_df[["Close"]].values.astype(np.float64)

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        self.fitted = True
        logger.info(
            "Scalers fitted on %d training rows (features=%s)",
            len(train_df),
            self.feature_columns,
        )
        return self

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.feature_scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.feature_scaler.transform(
            df[self.feature_columns].values.astype(np.float64)
        )

    def transform_target(self, close_values: np.ndarray) -> np.ndarray:
        if not self.fitted or self.target_scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.target_scaler.transform(close_values.reshape(-1, 1)).ravel()

    def inverse_transform_target(self, scaled: np.ndarray) -> np.ndarray:
        if not self.fitted or self.target_scaler is None:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        return self.target_scaler.inverse_transform(scaled.reshape(-1, 1)).ravel()

    def create_sequences(
        self, scaled_features: np.ndarray, scaled_targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding windows.

        X[i] = features[i : i+seq_len]
        y[i] = target[i+seq_len]   (next close after the window)
        """
        X, y = [], []
        seq = self.sequence_length
        for i in range(len(scaled_features) - seq):
            X.append(scaled_features[i : i + seq])
            y.append(scaled_targets[i + seq])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train / val / test preserving time order.
        Default ratios from config.
        """
        settings = get_settings()
        n = len(df)
        train_end = int(n * settings.train_ratio)
        val_end = train_end + int(n * settings.val_ratio)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df

    def prepare_datasets(
        self, df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Full pipeline: split → fit on train → transform all → sequences.

        Returns dict with X_train, y_train, X_val, y_val, X_test, y_test
        and the raw split DataFrames for later inverse transform / evaluation.
        """
        train_df, val_df, test_df = self.chronological_split(df)

        # Fit only on train
        self.fit(train_df)

        # Transform
        train_feat = self.transform_features(train_df)
        val_feat = self.transform_features(val_df)
        test_feat = self.transform_features(test_df)

        train_tgt = self.transform_target(train_df["Close"].values)
        val_tgt = self.transform_target(val_df["Close"].values)
        test_tgt = self.transform_target(test_df["Close"].values)

        # For val and test sequences we need context from the end of previous split
        # so the first sequences can be formed without leaking future data into training.
        seq = self.sequence_length

        # Training sequences purely from train
        X_train, y_train = self.create_sequences(train_feat, train_tgt)

        # Validation: use last `seq` rows of train as context + val
        if len(val_df) > 0:
            context_feat = train_feat[-seq:] if len(train_feat) >= seq else train_feat
            context_tgt = train_tgt[-seq:] if len(train_tgt) >= seq else train_tgt
            val_feat_full = np.vstack([context_feat, val_feat])
            val_tgt_full = np.concatenate([context_tgt, val_tgt])
            X_val, y_val = self.create_sequences(val_feat_full, val_tgt_full)
            # Keep only sequences whose target falls inside the original val period
            # (i.e. drop pure-context targets)
            n_context = len(context_feat)
            # After create_sequences the first targets start at index seq of the combined array
            # We want targets that correspond to original val indices
            keep_from = max(0, n_context - seq)
            X_val = X_val[keep_from:]
            y_val = y_val[keep_from:]
        else:
            X_val, y_val = np.empty((0, seq, len(self.feature_columns))), np.empty((0,))

        # Test: context from end of train+val
        if len(test_df) > 0:
            prior_feat = np.vstack([train_feat, val_feat])
            prior_tgt = np.concatenate([train_tgt, val_tgt])
            context_feat = prior_feat[-seq:] if len(prior_feat) >= seq else prior_feat
            context_tgt = prior_tgt[-seq:] if len(prior_tgt) >= seq else prior_tgt
            test_feat_full = np.vstack([context_feat, test_feat])
            test_tgt_full = np.concatenate([context_tgt, test_tgt])
            X_test, y_test = self.create_sequences(test_feat_full, test_tgt_full)
            n_context = len(context_feat)
            keep_from = max(0, n_context - seq)
            X_test = X_test[keep_from:]
            y_test = y_test[keep_from:]
        else:
            X_test, y_test = np.empty((0, seq, len(self.feature_columns))), np.empty((0,))

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
        }

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "sequence_length": self.sequence_length,
                "feature_columns": self.feature_columns,
                "fitted": self.fitted,
            },
            path,
        )
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        data = joblib.load(path)
        obj = cls(
            sequence_length=data["sequence_length"],
            feature_columns=data["feature_columns"],
        )
        obj.feature_scaler = data["feature_scaler"]
        obj.target_scaler = data["target_scaler"]
        obj.fitted = data["fitted"]
        return obj
