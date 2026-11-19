"""Application logging setup."""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings, PROJECT_ROOT


_LOGGER_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    """Configure root logging once."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    settings = get_settings()
    log_level = level or settings._data.get("logging", {}).get("level", "INFO")
    log_format = settings._data.get("logging", {}).get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logs_dir = settings.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "stocksight.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _LOGGER_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Ensures setup has run."""
    setup_logging()
    return logging.getLogger(name)
