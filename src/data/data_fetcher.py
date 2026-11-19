"""Historical market data retrieval via yfinance."""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.stock_registry import get_stock_info
from src.data.cache_manager import CacheManager
from src.data.data_validator import DataValidator
from src.utils.helpers import utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataFetcher:
    """Fetch OHLCV data from Yahoo Finance with caching and validation."""

    def __init__(self, use_cache: bool = True):
        self.cache = CacheManager() if use_cache else None
        self.validator = DataValidator()

    def fetch(
        self,
        symbol: str,
        period: str = "5y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Download historical OHLCV data.

        Returns
        -------
        df : pd.DataFrame
            Columns: Open, High, Low, Close, Volume (and possibly Adj Close).
            Index: DatetimeIndex.
        meta : dict
            Retrieval metadata.
        """
        provider_symbol = self._resolve_symbol(symbol)
        meta: Dict[str, Any] = {
            "requested_symbol": symbol,
            "provider_symbol": provider_symbol,
            "period": period,
            "provider": "yfinance",
            "retrieved_at": utc_now_iso(),
            "adjusted": True,
        }

        # Cache check
        if self.cache and not force_refresh:
            cached = self.cache.get(provider_symbol, period)
            if cached is not None and not cached.empty:
                meta["from_cache"] = True
                meta["rows"] = len(cached)
                meta["start_date"] = str(pd.Timestamp(cached.index.min()).date())
                meta["end_date"] = str(pd.Timestamp(cached.index.max()).date())
                return cached, meta

        logger.info("Downloading %s (period=%s) via yfinance...", provider_symbol, period)

        try:
            import yfinance as yf

            ticker = yf.Ticker(provider_symbol)
            if start or end:
                df = ticker.history(start=start, end=end, auto_adjust=True)
            else:
                df = ticker.history(period=period, auto_adjust=True)
        except Exception as e:
            logger.error("yfinance download failed for %s: %s", provider_symbol, e)
            raise RuntimeError(f"Failed to download data for {provider_symbol}: {e}") from e

        if df is None or df.empty:
            # Try alternative symbols for Kenyan stocks
            info = get_stock_info(symbol)
            alternatives = (info or {}).get("alternative_symbols", [])
            for alt in alternatives:
                logger.info("Trying alternative symbol: %s", alt)
                try:
                    df = yf.Ticker(alt).history(period=period, auto_adjust=True)
                    if df is not None and not df.empty:
                        provider_symbol = alt
                        meta["provider_symbol"] = alt
                        break
                except Exception:
                    continue

        if df is None or df.empty:
            raise ValueError(
                f"No historical data returned for {symbol}. "
                "Yahoo Finance may not support this ticker. "
                "Please import a CSV file instead."
            )

        # Normalize columns
        df = self._normalize_columns(df)
        df = df.sort_index()

        # Validate
        validation = self.validator.validate(df)
        meta["validation"] = validation
        meta["rows"] = len(df)
        meta["start_date"] = str(df.index.min().date()) if len(df) else None
        meta["end_date"] = str(df.index.max().date()) if len(df) else None
        meta["from_cache"] = False

        if not validation["is_valid"] and validation.get("critical_errors"):
            logger.warning("Data validation issues for %s: %s", symbol, validation["critical_errors"])

        # Cache
        if self.cache:
            self.cache.put(provider_symbol, period, df, meta)

        return df, meta

    def _resolve_symbol(self, symbol: str) -> str:
        info = get_stock_info(symbol)
        if info:
            return info.get("provider_symbol") or symbol
        return symbol.upper().strip()

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard OHLCV column names."""
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj close": "Adj Close",
            "adj_close": "Adj Close",
        }
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        lower_map = {c.lower(): c for c in df.columns}
        for low, std in rename_map.items():
            if low in lower_map and lower_map[low] != std:
                df = df.rename(columns={lower_map[low]: std})
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Downloaded data missing columns: {missing}")
        # Keep only needed columns plus index
        keep = [c for c in required + ["Adj Close"] if c in df.columns]
        return df[keep]
