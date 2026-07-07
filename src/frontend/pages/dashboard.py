"""Dashboard page — overview of financial health at a glance."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient

# Columns shown in the recent-transactions table
_TXN_COLUMNS = ["date", "merchant", "amount", "category", "is_anomaly"]


def _check_health(client: APIClient) -> None:
    """Display a backend connectivity badge."""
    try:
        client.health_check()
        st.success("🟢 Backend Online")
    except RuntimeError:
        st.error("🔴 Backend Offline")


def _kpi_cards(transactions: list[dict[str, Any]], anomalies: list[dict[str, Any]]) -> None:
    """Render three KPI metric cards in equal-width columns."""
    total_txns = len(transactions)
    total_anomalies = len(anomalies)
    unique_categories = len({t["category"] for t in transactions}) if transactions else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total_txns)
    col2.metric("Total Anomalies", total_anomalies)
    col3.metric("Categories", unique_categories)


def _recent_transactions(transactions: list[dict[str, Any]]) -> None:
    """Show the 10 most recent transactions in a dataframe."""
    st.subheader("Recent Transactions")

    if not transactions:
        st.info("No transactions found. Upload a statement to get started.")
        return

    df = pd.DataFrame(transactions)[_TXN_COLUMNS].head(10)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df.rename(
        columns={
            "date": "Date",
            "merchant": "Merchant",
            "amount": "Amount (£)",
            "category": "Category",
            "is_anomaly": "Anomaly",
        },
        inplace=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def _quick_actions() -> None:
    """Render a Quick Actions button row."""
    st.subheader("Quick Actions")
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])
    col1.button("📤 Upload Statement", use_container_width=True)
    col2.button("📈 Forecast", use_container_width=True)
    col3.button("🤖 AI Chat", use_container_width=True)


def render(client: APIClient) -> None:
    """Render the FinSight AI dashboard page."""
    st.title("💰 FinSight AI Dashboard")
    _check_health(client)
    st.divider()

    try:
        transactions = client.get_transactions()
    except RuntimeError as exc:
        st.error(f"Could not load transactions: {exc}")
        transactions = []

    try:
        anomalies = client.get_anomalies()
    except RuntimeError as exc:
        st.error(f"Could not load anomalies: {exc}")
        anomalies = []

    _kpi_cards(transactions, anomalies)
    st.divider()
    _recent_transactions(transactions)
    st.divider()
    _quick_actions()
