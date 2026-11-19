"""Load OHLCV data from user-supplied CSV files."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from src.data.data_validator import DataValidator
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CSVLoader:
    """Import and normalize CSV historical data."""

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
    DATE_CANDIDATES = ["Date", "date", "Datetime", "datetime", "timestamp", "Timestamp"]

    def __init__(self):
        self.validator = DataValidator()

    def load(
        self,
        source: Union[str, Path],
        symbol: Optional[str] = None,
        date_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a CSV file into a standardized OHLCV DataFrame.

        Expected columns (case-insensitive): Date, Open, High, Low, Close, Volume.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path)

        # Identify date column
        date_col = date_column
        if date_col is None:
            for candidate in self.DATE_CANDIDATES:
                if candidate in df.columns:
                    date_col = candidate
                    break
            if date_col is None:
                # Try first column
                date_col = df.columns[0]

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index.name = "Date"
        df = df.sort_index()

        # Normalize column names
        col_map = {}
        for col in df.columns:
            low = col.strip().lower()
            if low in ("open",):
                col_map[col] = "Open"
            elif low in ("high",):
                col_map[col] = "High"
            elif low in ("low",):
                col_map[col] = "Low"
            elif low in ("close", "adj close", "adj_close", "adjusted close"):
                col_map[col] = "Close" if low == "close" else "Adj Close"
            elif low in ("volume", "vol"):
                col_map[col] = "Volume"
        df = df.rename(columns=col_map)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        # Keep standard columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if c in df.columns]
        df = df[keep].copy()

        # Coerce numeric
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        validation = self.validator.validate(df)
        meta = {
            "source": str(path),
            "symbol": symbol,
            "provider": "csv",
            "rows": len(df),
            "start_date": str(df.index.min().date()) if len(df) else None,
            "end_date": str(df.index.max().date()) if len(df) else None,
            "validation": validation,
            "retrieved_at": utc_now_iso(),
        }
        return df, meta
