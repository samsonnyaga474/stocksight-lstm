"""Manual model training page."""

import streamlit as st

from config.stock_registry import list_stocks
from config.settings import get_settings
from dashboard.components.charts import loss_chart
from src.services.stock_service import StockService, canonical_symbol
from src.services.training_service import TrainingService


def render():
    st.title("Model Training")
    st.markdown(
        "Train a PyTorch LSTM on historical data for the selected stock. "
        "Training is **manual** — it only starts when you click the button. "
        "If a CSV has been imported for this symbol, that dataset is used automatically."
    )

    settings = get_settings()
    stock_service = StockService()
    train_service = TrainingService()

    registry = list_stocks()
    options = {f"{s['name']} ({s['provider_symbol']})": s["provider_symbol"] for s in registry}
    if not options:
        st.warning("No stocks registered.")
        return

    label = st.selectbox("Select stock to train", list(options.keys()), key="train_symbol")
    symbol = options[label]
    canon = canonical_symbol(symbol)

    if stock_service.has_imported_data(canon):
        st.info(f"CSV import available for `{canon}` — training will use the imported dataset.")
    else:
        st.caption("No CSV import found. Training will download data via Yahoo Finance.")

    period = st.selectbox("Historical period (if downloading)", ["2y", "5y", "max"], index=1, key="train_period")

    with st.expander("Hyperparameters (optional)", expanded=False):
        seq_len = st.number_input("Sequence length", min_value=20, max_value=120, value=settings.sequence_length)
        h1 = st.number_input("Hidden size layer 1", min_value=16, max_value=256, value=settings.hidden_size_1)
        h2 = st.number_input("Hidden size layer 2", min_value=8, max_value=128, value=settings.hidden_size_2)
        dropout = st.slider("Dropout", 0.0, 0.5, float(settings.dropout), 0.05)
        lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=float(settings.learning_rate), format="%.5f")
        batch = st.number_input("Batch size", min_value=8, max_value=128, value=settings.batch_size)
        epochs = st.number_input("Max epochs", min_value=5, max_value=300, value=settings.max_epochs)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.button("Train Model", type="primary"):
        if st.session_state.get("training_active"):
            st.warning("Training already in progress.")
            return

        progress_bar = progress_placeholder.progress(0)
        status_placeholder.info("Loading data...")
        st.session_state["training_active"] = True

        try:
            # prefer_import=True so registered CSV is used when present
            df, meta = stock_service.load_data(symbol, period=period, prefer_import=True)
            src = meta.get("data_source") or meta.get("provider", "unknown")
            status_placeholder.info(f"Loaded {len(df)} rows (source: {src}). Validating...")

            from src.data.data_validator import DataValidator

            validation = DataValidator(sequence_length=int(seq_len)).validate(df)
            if not validation["is_valid"]:
                st.error("Data validation failed: " + "; ".join(validation["critical_errors"]))
                return

            status_placeholder.info(f"Starting training on CPU/GPU for `{canon}`...")

            history_train = []
            history_val = []

            def on_progress(info):
                epoch = info["epoch"]
                max_e = epochs
                progress_bar.progress(min(epoch / max_e, 1.0))
                status_placeholder.info(
                    f"Epoch {epoch}/{max_e} | "
                    f"train_loss={info['train_loss']:.6f} | "
                    f"val_loss={info['val_loss']:.6f} | "
                    f"best={info['best_epoch']} | "
                    f"elapsed={info['elapsed']:.0f}s"
                )
                history_train.append(info["train_loss"])
                history_val.append(info["val_loss"])
                if len(history_train) > 1:
                    chart_placeholder.plotly_chart(
                        loss_chart(history_train, history_val),
                        use_container_width=True,
                    )

            result = train_service.train(
                df=df,
                symbol=canon,
                sequence_length=int(seq_len),
                hidden_size_1=int(h1),
                hidden_size_2=int(h2),
                dropout=float(dropout),
                learning_rate=float(lr),
                batch_size=int(batch),
                max_epochs=int(epochs),
                progress_callback=on_progress,
            )

            progress_bar.progress(1.0)

            if result.get("success"):
                status_placeholder.success(
                    f"Training completed. Model version: `{result['model_version']}` "
                    f"(data source: {src})"
                )
                st.balloons()
                eval_ = result.get("evaluation", {})
                if eval_.get("lstm"):
                    st.write("**Test metrics (LSTM):**", eval_["lstm"])
                    st.write("**Test metrics (Baseline):**", eval_.get("baseline", {}))
            else:
                status_placeholder.error(f"Training failed: {result.get('error')}")

        except Exception as e:
            status_placeholder.error(f"Error: {e}")
            st.exception(e)
        finally:
            st.session_state["training_active"] = False
