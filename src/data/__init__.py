"""Data acquisition and preprocessing package."""

# Lazy imports to avoid requiring optional network deps at package import time.

def __getattr__(name):
    if name == "DataFetcher":
        from .data_fetcher import DataFetcher
        return DataFetcher
    if name == "CSVLoader":
        from .csv_loader import CSVLoader
        return CSVLoader
    if name == "DataValidator":
        from .data_validator import DataValidator
        return DataValidator
    if name == "compute_min_rows":
        from .data_validator import compute_min_rows
        return compute_min_rows
    if name == "Preprocessor":
        from .preprocessing import Preprocessor
        return Preprocessor
    if name == "CacheManager":
        from .cache_manager import CacheManager
        return CacheManager
    if name == "DatasetStore":
        from .dataset_store import DatasetStore
        return DatasetStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DataFetcher",
    "CSVLoader",
    "DataValidator",
    "compute_min_rows",
    "Preprocessor",
    "CacheManager",
    "DatasetStore",
]
