"""Analytics page — visual insights into spending behaviour."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import page_header

_TOP_MERCHANTS_N = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_transactions(client: APIClient) -> list[dict[str, Any]] | None:
    """Fetch all transactions from the backend.

    Returns the list on success, or ``None`` on error (error rendered inline).
    """
    try:
        return client.get_transactions()
    except RuntimeError as exc:
        st.error(f"Could not load transactions: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return None


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _to_dataframe(transactions: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert the raw API payload to a clean, typed :class:`pd.DataFrame`."""
    df = pd.DataFrame(transactions)

    if df.empty:
        return pd.DataFrame(columns=["date", "merchant", "category", "amount"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day"] = df["date"].dt.date

    return df


def _apply_filters(
    df: pd.DataFrame,
    category: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Apply category and date-range filters to *df*."""
    if category != "All":
        df = df[df["category"] == category]
    df = df[
        (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    ]
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar_controls(df: pd.DataFrame) -> tuple[str, date, date]:
    """Render filter controls in the sidebar.

    Returns
    -------
    tuple of (category, start_date, end_date)
    """
    with st.sidebar:
        st.header("🔍 Filters")

        categories = sorted(df["category"].dropna().unique().tolist())
        category = st.selectbox(
            "Category",
            options=["All"] + categories,
        )

        st.divider()

        if not df.empty and df["date"].notna().any():
            min_date: date = df["date"].dt.date.min()
            max_date: date = df["date"].dt.date.max()
        else:
            min_date = date.today() - timedelta(days=90)
            max_date = date.today()

        start_date = st.date_input(
            "From", value=min_date, min_value=min_date, max_value=max_date
        )
        end_date = st.date_input(
            "To", value=max_date, min_value=min_date, max_value=max_date
        )

    return category, start_date, end_date


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _kpi_cards(df: pd.DataFrame) -> None:
    """Render four KPI metric cards for the filtered dataset.

    All stored transaction amounts are positive spending/debit values —
    the ingestion pipeline maps credit/deposit columns to ignored fields
    and the PDF parser extracts only debit amounts.  Income tracking is
    not part of the current data model, so metrics are derived solely
    from spending amounts.
    """
    total_spending = df["amount"].sum() if not df.empty else 0.0
    total_txns = len(df)
    avg_txn = df["amount"].mean() if total_txns else 0.0
    largest_txn = df["amount"].max() if total_txns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spending", f"₹{total_spending:,.2f}")
    col2.metric("Transaction Count", f"{total_txns:,}")
    col3.metric("Avg Transaction", f"₹{avg_txn:,.2f}")
    col4.metric("Largest Transaction", f"₹{largest_txn:,.2f}")


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _chart_spending_by_category(df: pd.DataFrame) -> None:
    """Bar chart — total spending grouped by category."""
    st.subheader("Spending by Category")

    cat_df = (
        df[df["amount"] > 0]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )

    if cat_df.empty:
        st.info("No spending data available for the selected filters.")
        return

    cat_df = cat_df.rename(columns={"category": "Category", "amount": "Total Spent (₹)"})
    st.bar_chart(cat_df.set_index("Category")["Total Spent (₹)"], use_container_width=True)


def _chart_spending_distribution(df: pd.DataFrame) -> None:
    """Pie chart — category spending share."""
    st.subheader("Spending Distribution")

    cat_df = (
        df[df["amount"] > 0]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )

    if cat_df.empty:
        st.info("No spending data available for the selected filters.")
        return

    # Streamlit natively supports pie via plotly or altair; use a clean
    # dataframe representation with percentage annotation instead so we
    # have zero extra dependencies beyond what the project already uses.
    total = cat_df["amount"].sum()
    cat_df["Share (%)"] = (cat_df["amount"] / total * 100).round(1)
    cat_df = cat_df.rename(columns={"category": "Category", "amount": "Amount (₹)"})

    st.dataframe(
        cat_df[["Category", "Amount (₹)", "Share (%)"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Share (%)": st.column_config.ProgressColumn(
                "Share (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )


def _chart_monthly_trend(df: pd.DataFrame) -> None:
    """Line chart — total spending aggregated by month."""
    st.subheader("Monthly Spending Trend")

    monthly = (
        df[df["amount"] > 0]
        .groupby("month", as_index=False)["amount"]
        .sum()
        .sort_values("month")
    )

    if monthly.empty:
        st.info("Not enough data to show a monthly trend.")
        return

    monthly = monthly.rename(columns={"month": "Month", "amount": "Total Spent (₹)"})
    st.line_chart(monthly.set_index("Month")["Total Spent (₹)"], use_container_width=True)


def _chart_top_merchants(df: pd.DataFrame) -> None:
    """Horizontal bar — top N merchants by total spending."""
    st.subheader(f"Top {_TOP_MERCHANTS_N} Merchants")

    merch_df = (
        df[df["amount"] > 0]
        .groupby("merchant", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .head(_TOP_MERCHANTS_N)
    )

    if merch_df.empty:
        st.info("No merchant data available for the selected filters.")
        return

    merch_df = merch_df.rename(
        columns={"merchant": "Merchant", "amount": "Total Spent (₹)"}
    )
    st.bar_chart(
        merch_df.set_index("Merchant")["Total Spent (₹)"],
        use_container_width=True,
        horizontal=True,
    )


def _chart_daily_spending(df: pd.DataFrame) -> None:
    """Area chart — spending aggregated by day."""
    st.subheader("Daily Spending")

    daily = (
        df[df["amount"] > 0]
        .groupby("day", as_index=False)["amount"]
        .sum()
        .sort_values("day")
    )

    if daily.empty:
        st.info("No daily spending data available for the selected filters.")
        return

    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.rename(columns={"day": "Date", "amount": "Spent (₹)"})
    st.area_chart(daily.set_index("Date")["Spent (₹)"], use_container_width=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(client: APIClient) -> None:
    """Render the Analytics page."""
    page_header(
        "📊 Analytics",
        subtitle="Visual insights into your spending behaviour.",
    )

    raw = _load_transactions(client)
    if raw is None:
        return

    df = _to_dataframe(raw)

    if df.empty:
        st.info("No transactions found. Upload a statement to get started.")
        return

    # Sidebar filters — bounds derived from the full unfiltered dataset
    category, start_date, end_date = _sidebar_controls(df)
    filtered = _apply_filters(df, category, start_date, end_date)

    # KPI cards
    _kpi_cards(filtered)
    st.divider()

    if filtered.empty:
        st.info(
            "No transactions match the current filters. "
            "Try widening the date range or selecting a different category."
        )
        return

    # Charts — two columns for the first row, full-width for the rest
    col_left, col_right = st.columns(2)
    with col_left:
        _chart_spending_by_category(filtered)
    with col_right:
        _chart_spending_distribution(filtered)

    st.divider()
    _chart_monthly_trend(filtered)

    st.divider()
    col_merchants, col_daily = st.columns(2)
    with col_merchants:
        _chart_top_merchants(filtered)
    with col_daily:
        _chart_daily_spending(filtered)
