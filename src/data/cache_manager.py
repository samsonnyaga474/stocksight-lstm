"""File-based cache for downloaded OHLCV data.

Prefers Parquet (via pyarrow). Falls back to CSV if Parquet is unavailable
or write/read fails. Cache failures never discard valid in-memory data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.settings import get_settings
from src.utils.helpers import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def _is_usable_ohlcv(df: Optional[pd.DataFrame]) -> bool:
    """Return True only if df looks like a non-empty OHLCV table."""
    if df is None:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty or len(df) == 0:
        return False
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        return False
    return True


class CacheManager:
    """Manage parquet/csv caches of historical data."""

    def __init__(self, cache_dir: Optional[Path] = None):
        settings = get_settings()
        self.cache_dir = cache_dir or settings.path("data_raw")
        ensure_dir(self.cache_dir)
        self._use_parquet = _parquet_available()
        if not self._use_parquet:
            logger.warning(
                "pyarrow not available — cache will use CSV fallback. "
                "Install pyarrow for better performance."
            )

    def _safe_key(self, symbol: str) -> str:
        return symbol.replace("/", "_").replace("\\", "_").replace(".", "_")

    def _key_path(self, symbol: str, period: str, ext: str = "parquet") -> Path:
        safe = self._safe_key(symbol)
        return self.cache_dir / f"{safe}_{period}.{ext}"

    def _meta_path(self, symbol: str, period: str) -> Path:
        safe = self._safe_key(symbol)
        return self.cache_dir / f"{safe}_{period}_meta.json"

    def get(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Load cached DataFrame. Tries parquet then CSV.

        Returns None when:
        - no cache file exists
        - read fails (corrupt / unreadable)
        - loaded frame is empty
        - required OHLCV columns are missing
        """
        # Prefer parquet
        parquet_path = self._key_path(symbol, period, "parquet")
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                if _is_usable_ohlcv(df):
                    logger.info(
                        "Cache hit (parquet) for %s (%s): %d rows",
                        symbol, period, len(df),
                    )
                    return df
                logger.warning(
                    "Parquet cache for %s (%s) is empty or missing OHLCV columns — ignoring",
                    symbol, period,
                )
            except Exception as e:
                logger.warning("Failed to read parquet cache %s: %s", parquet_path, e)

        # Fallback CSV
        csv_path = self._key_path(symbol, period, "csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                if _is_usable_ohlcv(df):
                    logger.info(
                        "Cache hit (csv) for %s (%s): %d rows",
                        symbol, period, len(df),
                    )
                    return df
                logger.warning(
                    "CSV cache for %s (%s) is empty or missing OHLCV columns — ignoring",
                    symbol, period,
                )
            except Exception as e:
                logger.warning("Failed to read csv cache %s: %s", csv_path, e)

        return None

    def put(
        self,
        symbol: str,
        period: str,
        df: pd.DataFrame,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Save DataFrame to cache. Never raises for cache failures —
        returns the path on success or None if caching failed.
        The caller's in-memory DataFrame is never modified or discarded.
        """
        ensure_dir(self.cache_dir)
        saved_path: Optional[Path] = None

        if self._use_parquet:
            path = self._key_path(symbol, period, "parquet")
            try:
                df.to_parquet(path, index=True)
                saved_path = path
                logger.info("Cached (parquet) %s (%s) -> %s", symbol, period, path)
            except Exception as e:
                logger.warning(
                    "Parquet cache write failed for %s: %s — trying CSV", symbol, e
                )
                self._use_parquet = False  # prefer CSV for remainder of session

        if saved_path is None:
            path = self._key_path(symbol, period, "csv")
            try:
                df.to_csv(path, index=True)
                saved_path = path
                logger.info("Cached (csv) %s (%s) -> %s", symbol, period, path)
            except Exception as e:
                logger.warning(
                    "Cache write failed for %s (%s): %s. Data remains in memory only.",
                    symbol, period, e,
                )
                return None

        # Remove a stale cache in the other format so old data can never be
        # served if the fresh file later becomes unreadable.
        if saved_path is not None:
            other_ext = "csv" if saved_path.suffix == ".parquet" else "parquet"
            stale = self._key_path(symbol, period, other_ext)
            if stale.exists():
                try:
                    stale.unlink()
                except OSError as e:
                    logger.warning("Could not remove stale cache file %s: %s", stale, e)

        # Metadata (best-effort)
        try:
            meta = meta or {}
            meta.update({
                "symbol": symbol,
                "period": period,
                "rows": len(df),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "columns": list(df.columns),
                "format": saved_path.suffix.lstrip("."),
            })
            with open(self._meta_path(symbol, period), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning("Could not write cache metadata: %s", e)

        return saved_path

    def get_meta(self, symbol: str, period: str) -> Optional[Dict[str, Any]]:
        path = self._meta_path(symbol, period)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cache meta %s: %s", path, e)
            return None

    def clear(self, symbol: Optional[str] = None) -> None:
        if symbol:
            safe = self._safe_key(symbol)
            for p in self.cache_dir.glob(f"{safe}*"):
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("Could not delete cache file %s: %s", p, e)
        else:
            for pattern in ("*.parquet", "*.csv", "*_meta.json"):
                for p in self.cache_dir.glob(pattern):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning("Could not delete cache file %s: %s", p, e)
