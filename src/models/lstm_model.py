"""PyTorch LSTM model for next-day close price prediction."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LSTMPredictor(nn.Module):
    """
    Two-layer LSTM followed by a linear head.

    Input shape:  (batch, sequence_length, input_size)
    Output shape: (batch, 1)  — scaled next-day close
    """

    def __init__(
        self,
        input_size: Optional[int] = None,
        hidden_size_1: Optional[int] = None,
        hidden_size_2: Optional[int] = None,
        dropout: Optional[float] = None,
        output_size: int = 1,
    ):
        super().__init__()
        settings = get_settings()
        self.input_size = input_size or settings.input_size
        self.hidden_size_1 = hidden_size_1 or settings.hidden_size_1
        self.hidden_size_2 = hidden_size_2 or settings.hidden_size_2
        self.dropout_rate = dropout if dropout is not None else settings.dropout

        self.lstm1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size_1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.lstm2 = nn.LSTM(
            input_size=self.hidden_size_1,
            hidden_size=self.hidden_size_2,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.fc = nn.Linear(self.hidden_size_2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)
        out2, _ = self.lstm2(out1)
        # Take the last time step
        last = out2[:, -1, :]
        last = self.dropout2(last)
        out = self.fc(last)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
