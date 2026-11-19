"""Overview dashboard page."""

import streamlit as st

from config.settings import get_settings
from src.services.history_service import HistoryService


def render():
    settings = get_settings()
    st.title(settings.app_name)
    st.markdown(settings.app_description)

    st.info(
        "This system uses a PyTorch LSTM neural network to forecast the next trading day's "
        "closing price from historical OHLCV data. All predictions are for educational and "
        "research purposes only and are **not** investment advice."
    )

    history = HistoryService()
    stats = history.overview()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered Stocks", stats["n_stocks"])
    col2.metric("Trained Models", stats["n_trained_models"])
    col3.metric("Stored Predictions", stats["n_predictions"])
    col4.metric("Markets", "US + Kenya")

    st.markdown("---")
    st.subheader("Recent Predictions")
    preds = stats.get("recent_predictions") or []
    if preds:
        for p in preds[:8]:
            change = p.get("percent_change") or 0
            sign = "+" if change >= 0 else ""
            st.write(
                f"**{p.get('symbol')}** — "
                f"Last: {p.get('last_close'):.2f} → "
                f"Predicted: {p.get('predicted_close'):.2f} "
                f"({sign}{change:.2f}%)  "
                f"| {p.get('prediction_timestamp', '')[:19]}"
            )
    else:
        st.write("No predictions yet. Train a model and generate a forecast from the Predictions page.")

    st.markdown("---")
    st.subheader("Tracked Stocks")
    stocks = stats.get("stocks") or []
    if stocks:
        for s in stocks:
            st.write(f"• **{s.get('name', s.get('symbol'))}** ({s.get('symbol')}) — {s.get('market')} / {s.get('currency')}")
    else:
        st.write("No stocks registered yet. Load data from the Stock Explorer or Data Management page.")
