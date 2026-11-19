"""Plotly chart helpers."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def price_chart(
    df: pd.DataFrame,
    title: str = "Price History",
    predicted_point: Optional[dict] = None,
) -> go.Figure:
    """Candlestick + volume chart. Optionally mark a prediction."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#4a5568"),
        row=2, col=1,
    )

    if predicted_point:
        fig.add_trace(
            go.Scatter(
                x=[predicted_point.get("date")],
                y=[predicted_point.get("price")],
                mode="markers+text",
                marker=dict(size=12, color="#f6ad55", symbol="star"),
                text=["Prediction"],
                textposition="top center",
                name="Predicted Close",
            ),
            row=1, col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=20, t=60, b=30),
        legend=dict(
            orientation="h",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(11,18,32,0.65)",
        ),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def loss_chart(train_loss: list, val_loss: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_loss, mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(y=val_loss, mode="lines", name="Val Loss"))
    fig.update_layout(
        title="Training & Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="MSE Loss",
        template="plotly_dark",
        height=360,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig


def actual_vs_predicted_chart(
    y_true: list, y_pred: list, title: str = "Actual vs Predicted (Test)"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted"))
    fig.update_layout(
        title=title,
        xaxis_title="Sample",
        yaxis_title="Price",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=50, b=30),
    )
    return fig
