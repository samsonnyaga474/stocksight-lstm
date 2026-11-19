"""Sidebar navigation and global controls."""

import streamlit as st

from config.settings import get_settings


def render_sidebar() -> str:
    settings = get_settings()
    st.sidebar.title(settings.app_name)
    st.sidebar.caption(f"v{settings.app_version}")
    st.sidebar.markdown("---")

    pages = [
        "Overview",
        "Stock Explorer",
        "Predictions",
        "Model Performance",
        "Model Training",
        "Data Management",
    ]
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Markets**  \n"
        "• US (NASDAQ)  \n"
        "• Kenya (NSE)"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Educational & research use only.  \n"
        "Not investment advice."
    )
    return page
