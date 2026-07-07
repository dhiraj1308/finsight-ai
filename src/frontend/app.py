"""Entry point for the FinSight AI Streamlit frontend."""
from __future__ import annotations

import streamlit as st

from frontend.pages import (
    analytics,
    anomalies,
    chat,
    dashboard,
    forecast,
    settings,
    transactions,
    upload,
)
from frontend.services.api import APIClient

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide",
)

_PAGES = {
    "Dashboard": dashboard,
    "Upload": upload,
    "Transactions": transactions,
    "Analytics": analytics,
    "Forecast": forecast,
    "Anomalies": anomalies,
    "Chat": chat,
    "Settings": settings,
}

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()

client: APIClient = st.session_state.api_client

with st.sidebar:
    st.title("FinSight AI")
    selection = st.radio("Navigation", list(_PAGES.keys()), label_visibility="collapsed")

_PAGES[selection].render(client)
