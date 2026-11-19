"""High-level application services.

Services are exposed lazily so that importing one does not pull in the
training stack (PyTorch) or the data provider (yfinance).
"""

_EXPORTS = {
    "StockService": ("src.services.stock_service", "StockService"),
    "TrainingService": ("src.services.training_service", "TrainingService"),
    "PredictionService": ("src.services.prediction_service", "PredictionService"),
    "HistoryService": ("src.services.history_service", "HistoryService"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        module_name, attr = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
