"""Next-day prediction page."""

import streamlit as st

from config.stock_registry import list_stocks
from dashboard.components.charts import price_chart
from src.services.prediction_service import PredictionService
from src.services.stock_service import StockService, canonical_symbol
from src.services.history_service import HistoryService


def render():
    st.title("Predictions")
    st.markdown(
        "Generate a **next trading day** closing-price forecast using a previously trained LSTM model. "
        "If a CSV was imported for this symbol, that dataset is used. "
        "Forecasts are educational only and are **not** guaranteed prices."
    )

    service = StockService()
    pred_service = PredictionService()
    history = HistoryService()

    registry = list_stocks()
    options = {f"{s['name']} ({s['provider_symbol']})": s["provider_symbol"] for s in registry}
    if not options:
        st.warning("No stocks registered.")
        return

    label = st.selectbox("Select stock", list(options.keys()), key="pred_symbol")
    symbol = options[label]
    canon = canonical_symbol(symbol)

    if service.has_imported_data(canon):
        st.info(f"CSV import available for `{canon}` — prediction will use the imported dataset.")

    period = st.selectbox("Data period (if downloading)", ["1y", "2y", "5y"], index=2, key="pred_period")

    if st.button("Generate Prediction", type="primary"):
        with st.spinner("Loading data and running model..."):
            try:
                df, meta = service.load_data(symbol, period=period, prefer_import=True)
                src = meta.get("data_source") or meta.get("provider", "unknown")
                result = pred_service.predict(df, canon)
                result["data_source"] = src
                st.session_state["last_prediction"] = result
                st.session_state["last_pred_df"] = df
                st.session_state["last_pred_symbol"] = canon
                st.success(f"Prediction generated (data source: **{src}**) and saved.")
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Go to **Model Training**, train a model for this stock, then return here.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    result = st.session_state.get("last_prediction")
    df = st.session_state.get("last_pred_df")
    if result and st.session_state.get("last_pred_symbol") != canon:
        result = None
        df = None
        st.caption(
            f"A stored forecast exists for another symbol. Select that symbol to view it, "
            "or generate a new forecast for this one."
        )

    if result:
        st.subheader(f"Forecast — {result.get('symbol')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"{result['last_close']:.2f} {result.get('currency', '')}")
        c2.metric(
            "Predicted Close",
            f"{result['predicted_close']:.2f}",
            delta=f"{result['percent_change']:+.2f}%",
        )
        c3.metric("Abs. Change", f"{result['absolute_change']:+.4f}")
        c4.metric("Target Date", result.get("target_date", "—"))

        st.caption(
            f"Last trading date: {result.get('last_trading_date')} | "
            f"Model: {result.get('model_version')} | "
            f"Source: {result.get('data_source', '—')} | "
            f"Generated: {result.get('prediction_timestamp', '')[:19]} UTC"
        )
        st.warning(result.get("notes", "Educational use only."))

        if df is not None:
            pred_point = {
                "date": result.get("target_date"),
                "price": result["predicted_close"],
            }
            st.plotly_chart(
                price_chart(
                    df.tail(120),
                    title=f"{canon} — Recent History + Prediction",
                    predicted_point=pred_point,
                ),
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Prediction History")
    hist = history.prediction_history(canon, limit=20)
    if hist:
        for h in hist:
            st.write(
                f"{h.get('prediction_timestamp', '')[:19]} | "
                f"Last {h.get('last_close'):.2f} → Pred {h.get('predicted_close'):.2f} "
                f"({h.get('percent_change', 0):+.2f}%) | model {h.get('model_version')}"
            )
    else:
        st.write("No stored predictions for this symbol yet.")
