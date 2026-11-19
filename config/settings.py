"""Application settings loaded from config.yaml."""

from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional
import yaml


# Project root is one level above the config package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """Central configuration object."""

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        self._config_path = config_path
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reload(self) -> None:
        self._data = self._load()

    @property
    def app_name(self) -> str:
        return self._data.get("app", {}).get("name", "StockSight")

    @property
    def app_version(self) -> str:
        return self._data.get("app", {}).get("version", "1.0.0")

    @property
    def app_description(self) -> str:
        return self._data.get("app", {}).get("description", "")

    @property
    def disclaimer(self) -> str:
        return self._data.get("app", {}).get("disclaimer", "")

    @property
    def default_period(self) -> str:
        return self._data.get("data", {}).get("default_period", "5y")

    @property
    def sequence_length(self) -> int:
        return int(self._data.get("data", {}).get("sequence_length", 60))

    @property
    def train_ratio(self) -> float:
        return float(self._data.get("data", {}).get("train_ratio", 0.70))

    @property
    def val_ratio(self) -> float:
        return float(self._data.get("data", {}).get("val_ratio", 0.15))

    @property
    def test_ratio(self) -> float:
        return float(self._data.get("data", {}).get("test_ratio", 0.15))

    @property
    def cache_enabled(self) -> bool:
        return bool(self._data.get("data", {}).get("cache_enabled", True))

    @property
    def features(self) -> List[str]:
        return list(self._data.get("data", {}).get("features", ["Open", "High", "Low", "Close", "Volume"]))

    @property
    def input_size(self) -> int:
        return int(self._data.get("model", {}).get("input_size", 5))

    @property
    def hidden_size_1(self) -> int:
        return int(self._data.get("model", {}).get("hidden_size_1", 64))

    @property
    def hidden_size_2(self) -> int:
        return int(self._data.get("model", {}).get("hidden_size_2", 32))

    @property
    def dropout(self) -> float:
        return float(self._data.get("model", {}).get("dropout", 0.2))

    @property
    def learning_rate(self) -> float:
        return float(self._data.get("model", {}).get("learning_rate", 0.001))

    @property
    def batch_size(self) -> int:
        return int(self._data.get("model", {}).get("batch_size", 32))

    @property
    def max_epochs(self) -> int:
        return int(self._data.get("model", {}).get("max_epochs", 100))

    @property
    def early_stopping_patience(self) -> int:
        return int(self._data.get("model", {}).get("early_stopping_patience", 10))

    @property
    def random_seed(self) -> int:
        return int(self._data.get("model", {}).get("random_seed", 42))

    def path(self, key: str) -> Path:
        """Return an absolute path for a configured relative path."""
        paths = self._data.get("paths", {})
        rel = paths.get(key)
        if rel is None:
            raise KeyError(f"Path key not found in config: {key}")
        return PROJECT_ROOT / rel

    @property
    def database_path(self) -> Path:
        return self.path("database")

    @property
    def logs_dir(self) -> Path:
        return self.path("logs")

    @property
    def checkpoints_dir(self) -> Path:
        return self.path("checkpoints_dir")

    @property
    def scalers_dir(self) -> Path:
        return self.path("scalers_dir")

    @property
    def metadata_dir(self) -> Path:
        return self.path("metadata_dir")


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
