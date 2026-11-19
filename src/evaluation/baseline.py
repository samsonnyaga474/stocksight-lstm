"""Naive baseline: tomorrow's close = today's close."""

from typing import Dict

import numpy as np

from src.evaluation.metrics import compute_metrics


class NaiveBaseline:
    """
    Predicts the next close as equal to the last observed close.
    Used as a sanity-check benchmark for the LSTM.
    """

    def predict(self, last_closes: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        last_closes : array of shape (n,)
            The close price observed at the end of each input sequence
            (i.e. the value that the naive model simply repeats).
        """
        return np.asarray(last_closes, dtype=float).copy()

    def evaluate(
        self,
        y_true: np.ndarray,
        last_closes: np.ndarray,
    ) -> Dict[str, float]:
        y_pred = self.predict(last_closes)
        metrics = compute_metrics(y_true, y_pred, y_prev=last_closes)
        metrics["directional_accuracy"] = self._persistence_accuracy(
            y_true, last_closes
        )
        return metrics

    @staticmethod
    def _persistence_accuracy(
        y_true: np.ndarray, last_closes: np.ndarray
    ) -> float:
        """Directional accuracy for a flat forecast, using persistence.

        A naive forecast of "tomorrow = today" makes no directional call, so
        scoring it against ``sign(y_true - last_close)`` would trivially yield
        0%.  Instead, the benchmark direction is the most recent observed
        change (momentum persistence), which is a fair directional baseline.
        """
        y_true = np.asarray(y_true, dtype=float)
        last = np.asarray(last_closes, dtype=float)
        if len(last) < 2 or len(y_true) != len(last):
            return float("nan")
        previous_changes = np.diff(last)
        actual_changes = y_true[1:] - last[1:]
        return float(
            np.mean(np.sign(actual_changes) == np.sign(previous_changes)) * 100.0
        )
