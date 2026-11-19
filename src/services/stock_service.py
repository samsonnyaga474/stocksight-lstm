"""Service for loading and inspecting stock data.

Data-source precedence for a symbol:
1. User-imported CSV registered in DatasetStore (if present)
2. Yahoo Finance download / local cache

Canonical symbol identity:
- Application identity = the symbol the user selected / requested
  (registry key or short ticker), stored consistently in the DB.
- provider_symbol = the actual Yahoo Finance ticker that returned data
  (may differ when alternatives were tried).
- Both are recorded; training runs and predictions always use the
  canonical application symbol so foreign keys stay consistent.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import pandas as pd

from config.stock_registry import STOCK_REGISTRY, get_stock_info, list_stocks
from src.data.csv_loader import CSVLoader
from src.data.data_fetcher import DataFetcher
from src.data.data_validator import DataValidator
from src.data.dataset_store import DatasetStore, _sanitize_symbol
from src.database.repositories import StockRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


def canonical_symbol(symbol: str) -> str:
    """Normalise a user-facing symbol to a stable application identity."""
    info = get_stock_info(symbol)
    if info:
        # Prefer the registry key / provider_symbol used as primary identity
        return info.get("provider_symbol") or info.get("ticker") or symbol.upper().strip()
    return symbol.upper().strip()


class StockService:
    """Facade for data retrieval, validation and registry operations."""

    def __init__(self):
        self.fetcher = DataFetcher(use_cache=True)
        self.csv_loader = CSVLoader()
        self.validator = DataValidator()
        self.repo = StockRepository()
        self.dataset_store = DatasetStore()

    def get_registry(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        return list_stocks(market=market)

    def get_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        return get_stock_info(symbol)

    def has_imported_data(self, symbol: str) -> bool:
        return self.dataset_store.has_import(canonical_symbol(symbol))

    def load_data(
        self,
        symbol: str,
        period: str = "5y",
        force_refresh: bool = False,
        prefer_import: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load historical OHLCV for a symbol.

        If prefer_import is True and a CSV has been registered for this
        symbol, that dataset is returned without calling Yahoo Finance.
        """
        canon = canonical_symbol(symbol)

        # 1. Prefer registered CSV import
        if prefer_import and self.dataset_store.has_import(canon):
            result = self.dataset_store.get_import(canon)
            if result is not None:
                df, meta = result
                df = self.validator.clean(df)
                meta["canonical_symbol"] = canon
                meta["data_source"] = "csv_import"
                self._ensure_stock_row(canon, meta, source_note="Imported from CSV")
                logger.info("Using imported CSV data for %s (%d rows)", canon, len(df))
                return df, meta

        # 2. Yahoo Finance
        df, meta = self.fetcher.fetch(symbol, period=period, force_refresh=force_refresh)
        df = self.validator.clean(df)

        # Always register under the *canonical* application symbol, not an
        # alternative that yfinance happened to resolve to.
        provider = meta.get("provider_symbol", symbol)
        meta["canonical_symbol"] = canon
        meta["data_source"] = "yfinance"
        meta["provider_symbol_used"] = provider

        info = get_stock_info(symbol) or get_stock_info(canon) or {}
        self.repo.upsert(
            symbol=canon,
            name=info.get("name", canon),
            market=info.get("market", ""),
            currency=info.get("currency", "USD"),
            country=info.get("country", ""),
            provider_symbol=provider,
            verified=bool(info.get("verified", False)),
            notes=info.get("notes", "") or f"Provider symbol used: {provider}",
        )
        return df, meta

    def load_csv(
        self,
        path: Union[str, Path],
        symbol: str,
        original_filename: str = "",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load, validate, clean and *register* a CSV so other pages can use it."""
        canon = canonical_symbol(symbol)
        df, meta = self.csv_loader.load(path, symbol=canon)
        df = self.validator.clean(df)

        validation = self.validator.validate(df)
        if not validation["is_valid"]:
            raise ValueError(
                "Imported CSV failed validation: "
                + "; ".join(validation["critical_errors"])
            )

        reg_meta = self.dataset_store.register_import(
            symbol=canon,
            df=df,
            original_filename=original_filename or str(path),
            extra_meta={"validation": validation},
        )
        meta.update(reg_meta)
        meta["canonical_symbol"] = canon
        meta["data_source"] = "csv_import"

        self._ensure_stock_row(canon, meta, source_note="Imported from CSV")
        return df, meta

    def _ensure_stock_row(self, canon: str, meta: Dict[str, Any], source_note: str = "") -> None:
        info = get_stock_info(canon) or {}
        self.repo.upsert(
            symbol=canon,
            name=info.get("name", canon),
            market=info.get("market", meta.get("market", "CSV")),
            currency=info.get("currency", "USD"),
            country=info.get("country", ""),
            provider_symbol=meta.get("provider_symbol", canon),
            verified=bool(info.get("verified", False)),
            notes=source_note or info.get("notes", ""),
        )

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.validator.validate(df)

    def list_imports(self) -> List[Dict[str, Any]]:
        return self.dataset_store.list_imports()
