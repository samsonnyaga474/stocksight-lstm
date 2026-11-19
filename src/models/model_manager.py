"""Manage model checkpoints, scalers and metadata on disk."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import get_settings
from src.utils.helpers import ensure_dir, utc_now, utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """File-system helpers for trained models."""

    def __init__(self):
        settings = get_settings()
        self.checkpoints_dir = ensure_dir(settings.checkpoints_dir)
        self.scalers_dir = ensure_dir(settings.scalers_dir)
        self.metadata_dir = ensure_dir(settings.metadata_dir)

    def make_version(self, symbol: str) -> str:
        """Generate a unique model version string.

        Microseconds are included so that two training runs for the same
        symbol cannot overwrite each other's artifacts.
        """
        safe = symbol.replace(".", "_").replace("/", "_")
        ts = utc_now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{safe}_{ts}_{uuid.uuid4().hex[:6]}"

    def checkpoint_path(self, version: str) -> Path:
        return self.checkpoints_dir / f"{version}.pt"

    def scaler_path(self, version: str) -> Path:
        return self.scalers_dir / f"{version}_scaler.joblib"

    def metadata_path(self, version: str) -> Path:
        return self.metadata_dir / f"{version}_meta.json"

    def save_metadata(self, version: str, meta: Dict[str, Any]) -> Path:
        path = self.metadata_path(version)
        meta = dict(meta)
        meta["model_version"] = version
        meta["saved_at"] = utc_now_iso()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        return path

    def load_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        path = self.metadata_path(version)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_versions(self, symbol: Optional[str] = None) -> List[str]:
        versions = []
        for p in sorted(self.checkpoints_dir.glob("*.pt"), reverse=True):
            ver = p.stem
            if symbol is None or ver.startswith(symbol.replace(".", "_")):
                versions.append(ver)
        return versions

    def latest_version(self, symbol: str) -> Optional[str]:
        versions = self.list_versions(symbol)
        return versions[0] if versions else None

    def exists(self, version: str) -> bool:
        return (
            self.checkpoint_path(version).exists()
            and self.scaler_path(version).exists()
        )
