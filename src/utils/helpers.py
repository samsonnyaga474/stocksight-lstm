"""General helper functions."""

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np

from config.settings import PROJECT_ROOT


def project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return utc_now().isoformat()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it does not exist and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
