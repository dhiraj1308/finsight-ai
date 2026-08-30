"""Transactions page — browse, search, filter, sort and export transactions."""
from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import page_header

# Canonical column order for display and export
_DISPLAY_COLUMNS = ["Date", "Merchant", "Category", "Amount"]

# Map canonical display names back to API field names
_COL_API: dict[str, str] = {
    "Date": "date",
    "Merchant": "merchant",
    "Category": "category",
    "Amount": "amount",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_transactions(client: APIClient) -> list[dict[str, Any]] | None:
    """Fetch all transactions from the backend.

    Returns the list on success, or ``None`` when a :class:`RuntimeError`
    is raised (the error is rendered inline before returning).
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
        return pd.DataFrame(columns=list(_COL_API.values()))

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    return df


def _apply_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter rows whose merchant or category contains *query* (case-insensitive)."""
    if not query.strip():
        return df
    q = query.strip().lower()
    mask = (
        df["merchant"].str.lower().str.contains(q, na=False)
        | df["category"].str.lower().str.contains(q, na=False)
    )
    return df[mask]


def _apply_filters(
    df: pd.DataFrame,
    category: str,
    start: date,
    end: date,
    min_amt: float,
    max_amt: float,
) -> pd.DataFrame:
    """Apply category, date-range and amount filters to *df*."""
    if category != "All":
        df = df[df["category"] == category]

    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[(df["amount"] >= min_amt) & (df["amount"] <= max_amt)]
    return df


def _apply_sort(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool,
) -> pd.DataFrame:
    """Sort *df* by the chosen column."""
    api_col = _COL_API[sort_col]
    return df.sort_values(api_col, ascending=ascending)


def _rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with human-readable column names."""
    return df.rename(
        columns={v: k for k, v in _COL_API.items()},
        errors="ignore",
    )[_DISPLAY_COLUMNS]


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _sidebar_controls(
    df: pd.DataFrame,
) -> tuple[str, str, date, date, float, float, str, bool]:
    """Render all search / filter / sort controls in the sidebar.

    Returns
    -------
    tuple of (search_query, category, start_date, end_date,
              min_amount, max_amount, sort_column, sort_ascending)
    """
    with st.sidebar:
        st.header("🔍 Search & Filters")

        search_query = st.text_input(
            "Search merchant or category",
            placeholder="e.g. Swiggy, Food",
        )

        st.divider()

        # Category filter
        categories = sorted(df["category"].dropna().unique().tolist())
        category = st.selectbox(
            "Category",
            options=["All"] + categories,
        )

        # Date range filter
        if not df.empty and df["date"].notna().any():
            min_date: date = df["date"].min()
            max_date: date = df["date"].max()
        else:
            min_date = date.today() - timedelta(days=90)
            max_date = date.today()

        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

        # Amount range filter
        if not df.empty:
            global_min = float(df["amount"].min())
            global_max = float(df["amount"].max())
        else:
            global_min, global_max = 0.0, 100_000.0

        min_amount = st.number_input(
            "Min amount (₹)",
            min_value=0.0,
            max_value=global_max,
            value=global_min,
            step=100.0,
            format="%.2f",
        )
        max_amount = st.number_input(
            "Max amount (₹)",
            min_value=0.0,
            max_value=global_max,
            value=global_max,
            step=100.0,
            format="%.2f",
        )

        st.divider()
        st.header("↕️ Sort")

        sort_col = st.selectbox("Sort by", options=_DISPLAY_COLUMNS, index=0)
        sort_ascending = st.radio(
            "Order",
            options=["Descending", "Ascending"],
            index=0,
        ) == "Ascending"

    return (
        search_query,
        category,
        start_date,
        end_date,
        min_amount,
        max_amount,
        sort_col,
        sort_ascending,
    )


def _statistics(df: pd.DataFrame) -> None:
    """Render KPI metric cards for the current (filtered) dataset."""
    total_txns = len(df)
    total_spending = df["amount"].sum()
    avg_txn = df["amount"].mean() if total_txns else 0.0
    highest_txn = df["amount"].max() if total_txns else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Total Spending", f"₹{total_spending:,.2f}")
    col3.metric("Avg Transaction", f"₹{avg_txn:,.2f}")
    col4.metric("Highest Transaction", f"₹{highest_txn:,.2f}")


def _data_table(display_df: pd.DataFrame) -> None:
    """Render the transactions dataframe with full-width styling."""
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "Amount": st.column_config.NumberColumn(
                "Amount (₹)",
                format="₹%.2f",
            ),
        },
    )


def _export_button(display_df: pd.DataFrame) -> None:
    """Render a Download CSV button for the currently filtered dataset."""
    buffer = io.StringIO()
    display_df.to_csv(buffer, index=False)
    st.download_button(
        label="⬇️ Download CSV",
        data=buffer.getvalue(),
        file_name="finsight_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(client: APIClient) -> None:
    """Render the Transactions page."""
    page_header(
        "💳 Transactions",
        subtitle="Browse, search and filter all imported transactions.",
    )

    raw = _load_transactions(client)
    if raw is None:
        return

    df = _to_dataframe(raw)

    if df.empty:
        st.info("No transactions found. Upload a statement to get started.")
        return

    # Sidebar controls — rendered against the full unfiltered dataframe so
    # range bounds reflect the entire dataset, not just the filtered view.
    (
        search_query,
        category,
        start_date,
        end_date,
        min_amount,
        max_amount,
        sort_col,
        sort_ascending,
    ) = _sidebar_controls(df)

    # Apply search → filters → sort in sequence
    filtered = _apply_search(df, search_query)
    filtered = _apply_filters(
        filtered,
        category,
        start_date,
        end_date,
        min_amount,
        max_amount,
    )
    filtered = _apply_sort(filtered, sort_col, sort_ascending)

    # Statistics for the filtered view
    _statistics(filtered)
    st.divider()

    if filtered.empty:
        st.info("No transactions match the current filters. Try adjusting the search or filter criteria.")
        return

    # Table + export
    display_df = _rename_for_display(filtered)
    _data_table(display_df)
    st.caption(f"Showing {len(display_df):,} of {len(df):,} transactions")
    _export_button(display_df)
