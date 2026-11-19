"""Persistent store for user-imported (CSV) datasets.

Precedence when loading data for a symbol:
1. Imported CSV registered for that symbol (if present and valid)
2. Yahoo Finance download / cache

Imported data is stored under data/external/imports/ and indexed
in a small JSON registry so it survives Streamlit reruns and restarts.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config.settings import PROJECT_ROOT, get_settings
from src.utils.helpers import ensure_dir, utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMPORTS_DIR = PROJECT_ROOT / "data" / "external" / "imports"
REGISTRY_PATH = IMPORTS_DIR / "registry.json"


def _sanitize_symbol(symbol: str) -> str:
    """Safe filesystem key for a ticker."""
    s = symbol.strip().upper()
    s = re.sub(r"[^A-Z0-9._-]", "_", s)
    return s[:64] or "UNKNOWN"


class DatasetStore:
    """Register and retrieve imported OHLCV datasets."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or IMPORTS_DIR
        ensure_dir(self.base_dir)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        reg_path = self.base_dir / "registry.json"
        if not reg_path.exists():
            return {}
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load import registry: %s", e)
            return {}

    def _save_registry(self) -> None:
        # Always write next to the instance base_dir so tests can isolate storage
        ensure_dir(self.base_dir)
        reg_path = self.base_dir / "registry.json"
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def has_import(self, symbol: str) -> bool:
        key = _sanitize_symbol(symbol)
        entry = self._registry.get(key)
        if not entry:
            return False
        path = Path(entry.get("path", ""))
        return path.exists()

    def get_import(self, symbol: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Return (df, meta) for an imported dataset, or None."""
        key = _sanitize_symbol(symbol)
        entry = self._registry.get(key)
        if not entry:
            return None
        path = Path(entry["path"])
        if not path.exists():
            logger.warning("Registered import missing on disk: %s", path)
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.sort_index()
            required = {"Open", "High", "Low", "Close", "Volume"}
            if df.empty or not required.issubset(df.columns):
                logger.warning("Imported dataset %s is empty or missing OHLCV columns", path)
                return None
            meta = dict(entry)
            meta["source"] = "csv_import"
            meta["provider"] = "csv"
            meta["from_import"] = True
            return df, meta
        except Exception as e:
            logger.error("Failed to load imported dataset %s: %s", path, e)
            return None

    def register_import(
        self,
        symbol: str,
        df: pd.DataFrame,
        original_filename: str = "",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Persist a cleaned DataFrame as the imported dataset for `symbol`.
        Overwrites any previous import for the same symbol.
        """
        key = _sanitize_symbol(symbol)
        ensure_dir(self.base_dir)
        path = self.base_dir / f"{key}_ohlcv.csv"
        df.to_csv(path, index=True)

        meta = {
            "symbol": key,
            "requested_symbol": symbol,
            "path": str(path),
            "rows": len(df),
            "start_date": str(df.index.min().date()) if len(df) else None,
            "end_date": str(df.index.max().date()) if len(df) else None,
            "original_filename": original_filename,
            "registered_at": utc_now_iso(),
            "source": "csv_import",
            "provider": "csv",
        }
        if extra_meta:
            meta.update(extra_meta)

        self._registry[key] = meta
        self._save_registry()
        logger.info("Registered CSV import for %s (%d rows) -> %s", key, len(df), path)
        return meta

    def list_imports(self) -> List[Dict[str, Any]]:
        return list(self._registry.values())

    def remove_import(self, symbol: str) -> bool:
        key = _sanitize_symbol(symbol)
        entry = self._registry.pop(key, None)
        if not entry:
            return False
        path = Path(entry.get("path", ""))
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning("Could not delete import file %s: %s", path, e)
        self._save_registry()
        return True
