"""Data management and registry inspection."""

import streamlit as st
import pandas as pd

from config.stock_registry import list_stocks
from src.services.stock_service import StockService
from src.models.model_manager import ModelManager


def render():
    st.title("Data Management")
    st.markdown("Inspect the stock registry, refresh data, and view model artifacts.")

    service = StockService()
    manager = ModelManager()

    st.subheader("Stock Registry")
    stocks = list_stocks()
    if stocks:
        df = pd.DataFrame(stocks)
        display_cols = [c for c in ["name", "ticker", "market", "currency", "provider_symbol", "country", "verified", "notes"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.write("Registry is empty.")

    st.markdown("---")
    st.subheader("Imported CSV Datasets")
    imports = service.list_imports()
    if imports:
        imported_df = pd.DataFrame(imports)
        cols = [c for c in ["symbol", "rows", "start_date", "end_date", "original_filename", "registered_at"] if c in imported_df.columns]
        st.dataframe(imported_df[cols], use_container_width=True)
        st.caption("Imported datasets take precedence over Yahoo Finance for training and prediction.")
    else:
        st.write("No CSV datasets imported yet. Use the Stock Explorer page to import one.")

    st.markdown("---")
    st.subheader("Refresh / Download Data")
    options = {f"{s['name']} ({s['provider_symbol']})": s["provider_symbol"] for s in stocks}
    if options:
        label = st.selectbox("Symbol", list(options.keys()), key="dm_symbol")
        symbol = options[label]
        period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="dm_period")
        if st.button("Download & Cache"):
            with st.spinner("Downloading..."):
                try:
                    # Explicit refresh: bypass both the cache and any CSV import.
                    df, meta = service.load_data(
                        symbol,
                        period=period,
                        force_refresh=True,
                        prefer_import=False,
                    )
                    st.success(f"Downloaded {len(df)} rows. Source: {meta.get('provider')}")
                    st.json({k: v for k, v in meta.items() if k != "validation"})
                except Exception as e:
                    st.error(str(e))
                    st.info(
                        "Kenyan NSE tickers are not always available on Yahoo Finance. "
                        "Use CSV import on the Stock Explorer page with a file that has "
                        "Date, Open, High, Low, Close, Volume columns."
                    )

    st.markdown("---")
    st.subheader("Trained Model Versions")
    versions = manager.list_versions()
    if versions:
        for v in versions[:15]:
            meta = manager.load_metadata(v)
            extra = ""
            if meta:
                extra = f" | symbol={meta.get('symbol')} | epochs={meta.get('epochs_trained')}"
            st.write(f"• `{v}`{extra}")
    else:
        st.write("No model checkpoints found yet.")

    st.markdown("---")
    st.subheader("CSV Import Notes")
    st.markdown(
        """
        For Kenyan stocks (or any ticker missing from Yahoo Finance):

        1. Obtain historical daily OHLCV data from a reliable source (e.g. NSE data vendors, your broker, or public archives).
        2. Save as CSV with columns: **Date, Open, High, Low, Close, Volume**.
        3. Use the **Stock Explorer** page → "Or import CSV".
        4. Do not invent or fabricate prices.

        The application will never silently replace a Kenyan ticker with an international one.
        """
    )
