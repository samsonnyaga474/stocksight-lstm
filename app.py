"""
StockSight - Main Streamlit entry point.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from config.settings import get_settings
from src.database.db import init_database
from src.utils.logger import setup_logging
from dashboard.components.sidebar import render_sidebar

# Page config must be first Streamlit command
st.set_page_config(
    page_title="StockSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()
init_database()

settings = get_settings()

# Custom CSS for a clean financial look
st.markdown(
    """
    <style>
    h1, h2, h3 { color: #e6edf7; }
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, #111a2b 0%, #0e1626 100%);
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid #1f2b40;
    }
    [data-testid="stMetricLabel"] p { color: #93a4bd; }
    .disclaimer { color: #8b9bb4; font-size: 0.85rem; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
page = render_sidebar()

# Route to pages
if page == "Overview":
    from dashboard.pages.overview import render
    render()
elif page == "Stock Explorer":
    from dashboard.pages.stock_explorer import render
    render()
elif page == "Predictions":
    from dashboard.pages.predictions import render
    render()
elif page == "Model Performance":
    from dashboard.pages.performance import render
    render()
elif page == "Model Training":
    from dashboard.pages.training import render
    render()
elif page == "Data Management":
    from dashboard.pages.data_management import render
    render()

# Footer disclaimer
st.markdown("---")
st.markdown(
    f'<p class="disclaimer">{settings.disclaimer}</p>',
    unsafe_allow_html=True,
)
