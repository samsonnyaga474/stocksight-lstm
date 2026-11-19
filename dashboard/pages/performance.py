"""Model performance and evaluation page."""

import streamlit as st

from config.stock_registry import list_stocks
from dashboard.components.charts import actual_vs_predicted_chart, loss_chart
from dashboard.components.metrics import show_metric_row, explain_metrics
from src.services.history_service import HistoryService
from src.services.stock_service import canonical_symbol
from src.models.model_manager import ModelManager


def render():
    st.title("Model Performance")
    st.markdown("Compare the LSTM against the naive baseline on the held-out test period.")

    history = HistoryService()
    manager = ModelManager()

    registry = list_stocks()
    options = {f"{s['name']} ({s['provider_symbol']})": s["provider_symbol"] for s in registry}
    if not options:
        st.warning("No stocks registered.")
        return

    label = st.selectbox("Select stock", list(options.keys()), key="perf_symbol")
    symbol = canonical_symbol(options[label])

    run = history.latest_training(symbol)
    evaluation = history.latest_evaluation(symbol)

    if not run:
        st.info("No completed training runs for this stock. Train a model first.")
        return

    st.subheader("Latest Training Run")
    version = run.get("model_version") or "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Version", version[-15:])
    c2.metric("Epochs Trained", run.get("epochs_trained", "—"))
    c3.metric("Best Epoch", run.get("best_epoch", "—"))
    c4.metric("Val Loss", f"{run.get('val_loss', 0):.6f}" if run.get("val_loss") else "—")
    st.caption(f"Full model version: `{version}`")

    if evaluation:
        st.subheader("LSTM Metrics (Test Set)")
        lstm = {
            "mae": evaluation.get("mae"),
            "rmse": evaluation.get("rmse"),
            "mape": evaluation.get("mape"),
            "directional_accuracy": evaluation.get("directional_accuracy"),
        }
        show_metric_row(lstm, prefix="LSTM ")

        st.subheader("Naive Baseline Metrics")
        baseline = {
            "mae": evaluation.get("baseline_mae"),
            "rmse": evaluation.get("baseline_rmse"),
            "mape": evaluation.get("baseline_mape"),
            "directional_accuracy": evaluation.get("baseline_directional_accuracy"),
        }
        show_metric_row(baseline, prefix="Baseline ")

        # Simple comparison
        if lstm.get("mae") is not None and baseline.get("mae") is not None:
            if lstm["mae"] < baseline["mae"]:
                st.success("LSTM achieved lower MAE than the naive baseline on the test period.")
            else:
                st.warning("LSTM did not beat the naive baseline on MAE for this run.")

        explain_metrics()
    else:
        st.info("No evaluation results stored for this training run.")

    # Try to load metadata for loss curves / predictions if available
    meta = manager.load_metadata(run["model_version"])
    if meta and meta.get("test_metrics"):
        tm = meta["test_metrics"]
        if tm.get("y_true") and tm.get("y_pred"):
            st.subheader("Actual vs Predicted (Test)")
            st.plotly_chart(
                actual_vs_predicted_chart(tm["y_true"], tm["y_pred"]),
                use_container_width=True,
            )
