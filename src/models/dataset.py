"""PyTorch Dataset for sequence windows."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class StockSequenceDataset(Dataset):
    """Wraps pre-built numpy sequences into a torch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "X and y must have the same number of samples"
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
