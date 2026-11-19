"""LSTM model, training and prediction package.

Submodules import heavy dependencies (PyTorch); exposing them lazily keeps
dashboard pages that never touch a model fast to import.
"""

_EXPORTS = {
    "LSTMPredictor": ("src.models.lstm_model", "LSTMPredictor"),
    "StockSequenceDataset": ("src.models.dataset", "StockSequenceDataset"),
    "Trainer": ("src.models.trainer", "Trainer"),
    "Predictor": ("src.models.predictor", "Predictor"),
    "ModelManager": ("src.models.model_manager", "ModelManager"),
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
