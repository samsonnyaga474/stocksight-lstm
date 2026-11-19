"""Validate OHLCV DataFrames for common data quality issues."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_min_rows(
    sequence_length: Optional[int] = None,
    train_ratio: Optional[float] = None,
    min_train_sequences: int = 10,
) -> int:
    """
    Minimum observations needed so that after chronological split and
    sequence generation there are at least `min_train_sequences` training examples.

    Rough formula:
      train_rows ≈ n * train_ratio
      train_sequences ≈ train_rows - sequence_length
      require train_sequences >= min_train_sequences
      => n >= (min_train_sequences + sequence_length) / train_ratio
    """
    settings = get_settings()
    seq = sequence_length if sequence_length is not None else settings.sequence_length
    tr = train_ratio if train_ratio is not None else settings.train_ratio
    # Add a small buffer for val/test context and edge cases
    needed = int(np.ceil((min_train_sequences + seq) / max(tr, 0.5))) + seq
    return max(needed, seq + min_train_sequences + 20)


class DataValidator:
    """Perform structural and logical checks on price data."""

    def __init__(self, sequence_length: Optional[int] = None):
        settings = get_settings()
        self.sequence_length = sequence_length or settings.sequence_length
        self.min_rows = compute_min_rows(self.sequence_length)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a validation report.

        is_valid is True only when there are no critical errors that
        would prevent model training.
        """
        report: Dict[str, Any] = {
            "is_valid": True,
            "critical_errors": [],
            "warnings": [],
            "stats": {},
            "min_rows_required": self.min_rows,
        }

        if df is None or df.empty:
            report["is_valid"] = False
            report["critical_errors"].append("DataFrame is empty")
            return report

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            report["is_valid"] = False
            report["critical_errors"].append(f"Missing columns: {missing}")
            return report

        n = len(df)
        report["stats"]["rows"] = n
        report["stats"]["start"] = str(df.index.min())
        report["stats"]["end"] = str(df.index.max())

        if n < self.min_rows:
            report["is_valid"] = False
            report["critical_errors"].append(
                f"Insufficient observations: {n} rows available, "
                f"but at least {self.min_rows} are required "
                f"(sequence_length={self.sequence_length}, need enough for "
                f"train/val/test splits and sliding windows). "
                f"Provide a longer history or a longer CSV."
            )

        # Duplicate dates
        if df.index.duplicated().any():
            n_dup = int(df.index.duplicated().sum())
            report["warnings"].append(f"{n_dup} duplicate date(s) found")

        # Missing values
        null_counts = df[required].isnull().sum()
        total_nulls = int(null_counts.sum())
        if total_nulls > 0:
            report["warnings"].append(f"{total_nulls} missing value(s) across OHLCV")
            report["stats"]["null_counts"] = null_counts.to_dict()

        # Negative prices or volume
        for col in ["Open", "High", "Low", "Close"]:
            neg = (df[col] < 0).sum()
            if neg > 0:
                report["critical_errors"].append(f"{neg} negative values in {col}")
                report["is_valid"] = False

        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            report["critical_errors"].append(f"{neg_vol} negative volume values")
            report["is_valid"] = False

        # OHLC relationships
        bad_hl = (df["High"] < df["Low"]).sum()
        if bad_hl > 0:
            report["warnings"].append(f"{bad_hl} rows where High < Low")

        bad_close = ((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum()
        if bad_close > 0:
            report["warnings"].append(
                f"{bad_close} rows where Close outside High-Low range"
            )

        # Chronological order
        if not df.index.is_monotonic_increasing:
            report["warnings"].append("Index is not strictly chronological")

        # Zero prices (suspicious)
        zero_close = (df["Close"] == 0).sum()
        if zero_close > 0:
            report["warnings"].append(f"{zero_close} zero Close prices")

        if report["critical_errors"]:
            report["is_valid"] = False

        return report

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply safe cleaning:
        - Drop duplicate dates (keep last)
        - Drop rows with any NaN in OHLCV
        - Sort by date
        Never fill missing prices with zero.
        """
        df = df.copy()
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df.dropna(subset=required)
        return df
