"""Metric display helpers."""

import streamlit as st
from typing import Dict, Optional


def show_metric_row(metrics: Dict[str, float], prefix: str = "") -> None:
    cols = st.columns(4)
    labels = [
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("MAPE (%)", "mape"),
        ("Dir. Acc. (%)", "directional_accuracy"),
    ]
    for col, (label, key) in zip(cols, labels):
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and (val != val)):  # NaN
            display = "—"
        else:
            display = f"{val:.4f}" if key != "directional_accuracy" and key != "mape" else f"{val:.2f}"
        col.metric(f"{prefix}{label}", display)


def explain_metrics() -> None:
    with st.expander("What do these metrics mean?"):
        st.markdown(
            """
            - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual price. Lower is better.
            - **RMSE (Root Mean Squared Error)**: Similar to MAE but penalises large errors more. Lower is better.
            - **MAPE (Mean Absolute Percentage Error)**: Average percentage error. Easier to interpret across different price levels.
            - **Directional Accuracy**: Percentage of times the model correctly predicted whether the price would go up or down.
            
            The **naive baseline** simply predicts that tomorrow's close equals today's close.
            A useful model should beat this baseline on the test period. Because a flat
            forecast has no inherent direction, its directional accuracy is measured with a
            persistence rule (assume the most recent observed move continues).
            """
        )
