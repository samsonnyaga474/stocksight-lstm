"""Stock Explorer page — load and inspect historical data."""

import re
from pathlib import Path

import streamlit as st
import pandas as pd

from config.settings import PROJECT_ROOT
from config.stock_registry import list_stocks
from dashboard.components.charts import price_chart
from src.services.stock_service import StockService, canonical_symbol
from src.utils.helpers import ensure_dir


def _safe_upload_name(name: str) -> str:
    """Sanitize uploaded filename: strip path components, allow only safe chars."""
    base = Path(name).name
    base = re.sub(r"[^\w.\-]", "_", base)
    if not base.lower().endswith(".csv"):
        base = base + ".csv"
    return base[:120] or "upload.csv"


def render():
    st.title("Stock Explorer")
    st.markdown(
        "Download historical OHLCV data, import a CSV, inspect quality, "
        "and view interactive charts."
    )

    service = StockService()
    registry = list_stocks()

    markets = sorted({s.get("market", "Other") for s in registry})
    market = st.selectbox("Market", ["All"] + markets)
    filtered = registry if market == "All" else [s for s in registry if s.get("market") == market]

    options = {f"{s['name']} ({s['provider_symbol']})": s["provider_symbol"] for s in filtered}
    if not options:
        st.warning("No stocks in registry for this filter.")
        return

    label = st.selectbox("Stock", list(options.keys()))
    symbol = options[label]
    canon = canonical_symbol(symbol)

    if service.has_imported_data(canon):
        st.info(
            f"**Data source preference:** A CSV import is registered for `{canon}`. "
            "Training and Predictions will use this imported data unless you force a download."
        )

    period = st.selectbox("History period (Yahoo Finance)", ["1y", "2y", "5y", "max"], index=2)
    force = st.checkbox("Force refresh from Yahoo Finance (ignore cache & import)", value=False)

    col_a, col_b = st.columns(2)
    with col_a:
        load_btn = st.button("Download / Load Data", type="primary")
    with col_b:
        uploaded = st.file_uploader(
            "Or import CSV (Date, Open, High, Low, Close, Volume)", type=["csv"]
        )

    df = None
    meta = None

    if load_btn:
        with st.spinner(f"Loading {symbol}..."):
            try:
                df, meta = service.load_data(
                    symbol, period=period, force_refresh=force, prefer_import=not force
                )
                st.session_state["explorer_df"] = df
                st.session_state["explorer_meta"] = meta
                st.session_state["explorer_symbol"] = canon
                src = meta.get("data_source") or meta.get("provider", "unknown")
                st.success(f"Loaded {len(df)} rows for `{canon}` (source: **{src}**)")
            except Exception as e:
                st.error(f"Could not load data: {e}")
                st.info(
                    "If this is a Kenyan stock, Yahoo Finance may not have the data. "
                    "Use the CSV import option with a file containing Date, Open, High, Low, Close, Volume."
                )

    if uploaded is not None:
        try:
            upload_dir = ensure_dir(PROJECT_ROOT / "data" / "external" / "uploads")
            safe_name = _safe_upload_name(uploaded.name)
            safe_canon = re.sub(r"[^A-Z0-9._-]", "_", canon.upper())
            dest = upload_dir / f"{safe_canon}_{safe_name}"
            raw = uploaded.getbuffer()
            if len(raw) > 10 * 1024 * 1024:
                st.error("File too large (max 10 MB).")
            else:
                with open(dest, "wb") as f:
                    f.write(raw)
                df, meta = service.load_csv(dest, symbol=canon, original_filename=uploaded.name)
                st.session_state["explorer_df"] = df
                st.session_state["explorer_meta"] = meta
                st.session_state["explorer_symbol"] = canon
                try:
                    dest.unlink(missing_ok=True)
                except OSError:
                    pass
                st.success(
                    f"Imported and registered {len(df)} rows for `{canon}`. "
                    "This dataset will be used by Training and Predictions."
                )
        except Exception as e:
            st.error(f"CSV import failed: {e}")

    if df is None and "explorer_df" in st.session_state:
        stored_symbol = st.session_state.get("explorer_symbol")
        if stored_symbol == canon:
            df = st.session_state["explorer_df"]
            meta = st.session_state.get("explorer_meta")
        else:
            st.info(
                f"Showing no data for `{canon}`. Previously loaded data belongs to "
                f"`{stored_symbol}`; click **Download / Load Data** to load this stock."
            )

    if df is not None and not df.empty:
        st.subheader(f"Data — {canon}")
        if meta:
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**Rows:** {meta.get('rows', len(df))}")
            c2.write(f"**Start:** {meta.get('start_date', '')}")
            c3.write(f"**End:** {meta.get('end_date', '')}")
            src = meta.get("data_source") or meta.get("provider", "yfinance")
            c4.write(f"**Source:** {src}")
            if meta.get("provider_symbol_used") and meta.get("provider_symbol_used") != canon:
                st.caption(f"Provider ticker actually used: `{meta['provider_symbol_used']}`")

        validation = service.validate(df)
        if validation["is_valid"]:
            st.success("Data validation passed.")
        else:
            st.error("Critical validation issues: " + "; ".join(validation["critical_errors"]))
        if validation.get("warnings"):
            st.warning("Warnings: " + "; ".join(validation["warnings"]))

        st.plotly_chart(price_chart(df, title=f"{canon} Price & Volume"), use_container_width=True)

        st.subheader("OHLCV Table (latest 30 rows)")
        st.dataframe(
            df.tail(30).style.format("{:.2f}", subset=["Open", "High", "Low", "Close"]),
            use_container_width=True,
        )
    else:
        st.info("Select a stock and click **Download / Load Data**, or import a CSV file.")
        st.caption(
            "Note: `data/samples/sample_ohlcv.csv` is **synthetic demonstration data** only. "
            "Do not treat it as real market history."
        )
