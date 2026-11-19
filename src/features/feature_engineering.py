"""Optional additional feature construction.

Currently the model uses raw OHLCV. This module provides a clean
extension point for future technical indicators.
"""

from typing import List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Add or select features for the model."""

    def __init__(self, base_features: Optional[List[str]] = None):
        self.base_features = base_features or ["Open", "High", "Low", "Close", "Volume"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame containing the selected features.
        Currently a pass-through; extend here for RSI, MACD, etc.
        """
        missing = [c for c in self.base_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return df[self.base_features].copy()
