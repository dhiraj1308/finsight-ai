"""Dashboard page — overview of financial health at a glance."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import navigate_to, page_header

# Ordered columns for the recent-transactions table
TRANSACTION_COLUMNS = ["date", "merchant", "category", "amount"]

# Days in the next-month forecast (30 is the default across the app)
_FORECAST_DAYS = 30

# ---------------------------------------------------------------------------
# Income classification
# ---------------------------------------------------------------------------
# The domain model has no transaction_type field.  Income is identified
# heuristically by matching the merchant name against these keywords
# (case-insensitive).  All other transactions are treated as spending.
_INCOME_KEYWORDS: frozenset[str] = frozenset({
    "salary", "credit", "income", "refund", "cashback",
})


def _is_income(transaction: dict[str, Any]) -> bool:
    """Return True if the transaction looks like income/credit rather than spending."""
    merchant = str(transaction.get("merchant", "")).lower()
    return any(kw in merchant for kw in _INCOME_KEYWORDS)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def _check_health(client: APIClient) -> None:
    """Display a backend connectivity badge."""
    try:
        client.health_check()
        st.success("🟢 Backend Online")
    except RuntimeError:
        st.error("🔴 Backend Offline")


# ---------------------------------------------------------------------------
# Financial Summary KPIs
# ---------------------------------------------------------------------------

def _kpi_cards(
    transactions: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    """Render five KPI metric cards: Income, Spending, Net Cash Flow,
    Transaction Count, and Anomaly Count.

    Income transactions are identified by _is_income(); all others are
    treated as spending.  The domain model carries no transaction_type
    field, so merchant-name heuristics are used.
    """
    total_txns = len(transactions)
    total_anomalies = len(anomalies)

    total_income = sum(
        float(t.get("amount", 0) or 0)
        for t in transactions
        if _is_income(t)
    )
    total_spending = sum(
        float(t.get("amount", 0) or 0)
        for t in transactions
        if not _is_income(t)
    )
    net_cash_flow = total_income - total_spending

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Income", f"₹{total_income:,.2f}")
    col2.metric("Total Spending", f"₹{total_spending:,.2f}")
    col3.metric(
        "Net Cash Flow",
        f"₹{net_cash_flow:,.2f}",
        delta=f"₹{net_cash_flow:,.2f}",
        delta_color="normal" if net_cash_flow >= 0 else "inverse",
    )
    col4.metric("Total Transactions", total_txns)
    col5.metric("Total Anomalies", total_anomalies)


# ---------------------------------------------------------------------------
# Spending Behavior
# ---------------------------------------------------------------------------

def _spending_category_df(transactions: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a category-level spending DataFrame, excluding income rows.

    Returns a DataFrame with columns ['Category', 'Total Spent (₹)'] sorted
    descending by spend, or an empty DataFrame if there are no expense rows.
    """
    expense_rows = [t for t in transactions if not _is_income(t)]
    if not expense_rows:
        return pd.DataFrame(columns=["Category", "Total Spent (₹)"])

    df = pd.DataFrame(expense_rows)
    df["amount"] = pd.to_numeric(df.get("amount", 0), errors="coerce").fillna(0.0)
    df["category"] = df.get("category", "").astype(str).str.strip()

    cat_df = (
        df[df["amount"] > 0]
        .groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"category": "Category", "amount": "Total Spent (₹)"})
    )
    return cat_df


def _spending_behavior(transactions: list[dict[str, Any]]) -> None:
    """Render the Spending Behavior section.

    Shows a bar chart of expense spending by category (income excluded),
    plus a compact table with percentage share.  Reuses the same aggregation
    approach as analytics._chart_spending_by_category().
    """
    st.subheader("📊 Spending Behavior")

    cat_df = _spending_category_df(transactions)

    if cat_df.empty:
        st.info(
            "No spending data yet. Upload a bank statement to see your "
            "spending breakdown by category."
        )
        return

    # Bar chart — mirrors analytics._chart_spending_by_category()
    st.bar_chart(
        cat_df.set_index("Category")["Total Spent (₹)"],
        use_container_width=True,
    )

    # Compact share table alongside chart
    total = cat_df["Total Spent (₹)"].sum()
    cat_df = cat_df.copy()
    cat_df["Share (%)"] = (cat_df["Total Spent (₹)"] / total * 100).round(1)

    st.dataframe(
        cat_df[["Category", "Total Spent (₹)", "Share (%)"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Total Spent (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Share (%)": st.column_config.ProgressColumn(
                "Share (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Next-Month Spending Forecast
# ---------------------------------------------------------------------------

def _top_expense_category(transactions: list[dict[str, Any]]) -> str | None:
    """Return the highest-spending expense category, or None if unavailable."""
    cat_df = _spending_category_df(transactions)
    if cat_df.empty:
        return None
    return str(cat_df.iloc[0]["Category"])


def _next_month_forecast(
    client: APIClient,
) -> None:
    """Render the Next-Month Spending Forecast section.

    Calls /forecast/aggregate to get a 30-day total-expense forecast that
    excludes income transactions.  Uses the existing ForecastDTO response
    schema and the same EWMA + linear trend algorithm as the Forecast page.

    Shows KPI cards (predicted total, avg daily, horizon) plus a line chart
    with confidence bounds.  If there is insufficient history the backend
    returns a 422 which is shown as a user-friendly message.
    """
    st.subheader("📈 Next-Month Spending Forecast")

    with st.spinner(f"Generating {_FORECAST_DAYS}-day total expense forecast…"):
        try:
            forecast = client.get_forecast_aggregate(_FORECAST_DAYS)
        except RuntimeError as exc:
            err = str(exc)
            if "422" in err:
                st.info(
                    "Not enough historical data to forecast yet. "
                    "At least 14 days of expense transactions are needed. "
                    "Upload more statements to enable forecasting."
                )
            else:
                st.error(f"Could not load forecast: {err}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load forecast: {exc}")
            return

    points = forecast.get("points", [])
    if not points:
        st.info("No forecast points returned. Try uploading more historical data.")
        return

    df = pd.DataFrame(points)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)

    predicted_total = df["yhat"].sum()
    avg_daily = df["yhat"].mean() if len(df) else 0.0
    horizon = forecast.get("horizon_days", _FORECAST_DAYS)

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Total Spend", f"₹{predicted_total:,.2f}")
    k2.metric("Avg Daily Spend", f"₹{avg_daily:,.2f}")
    k3.metric("Forecast Period", f"{horizon} days")

    chart_df = (
        df[["date", "yhat", "yhat_lower", "yhat_upper"]]
        .set_index("date")
        .rename(
            columns={
                "yhat": "Forecast",
                "yhat_lower": "Lower bound",
                "yhat_upper": "Upper bound",
            }
        )
    )
    st.line_chart(chart_df, use_container_width=True)
    st.caption(
        f"Total expense spending forecast for the next {horizon} days. "
        "Shaded bounds show the 95% confidence interval."
    )


# ---------------------------------------------------------------------------
# Recent Transactions
# ---------------------------------------------------------------------------

def _recent_transactions(transactions: list[dict[str, Any]]) -> None:
    """Show the 10 most recent transactions in a dataframe."""
    st.subheader("Recent Transactions")

    if not transactions:
        st.info("No transactions found. Upload a statement to get started.")
        return

    df = pd.DataFrame(transactions)
    available_cols = [c for c in TRANSACTION_COLUMNS if c in df.columns]
    df = df.sort_values("date", ascending=False)[available_cols].head(10)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df.rename(
        columns={
            "date": "Date",
            "merchant": "Merchant",
            "category": "Category",
            "amount": "Amount",
        },
        inplace=True,
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Quick Actions
# ---------------------------------------------------------------------------

def _quick_actions() -> None:
    """Render a Quick Actions button row that navigates to the target page."""
    st.subheader("Quick Actions")
    col1, col2, col3, _ = st.columns([1, 1, 1, 3])

    if col1.button("📤 Upload Statement", use_container_width=True):
        navigate_to("Upload")
    if col2.button("📈 Forecast", use_container_width=True):
        navigate_to("Forecast")
    if col3.button("🤖 AI Chat", use_container_width=True):
        navigate_to("Chat")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render(client: APIClient) -> None:
    """Render the FinSight AI dashboard page."""
    page_header("💰 FinSight AI", subtitle="Personal Finance Intelligence Dashboard")
    _check_health(client)

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

    # ── 1. Financial Summary ──────────────────────────────────────────────
    _kpi_cards(transactions, anomalies)
    st.divider()

    # ── 2. Spending Behavior ──────────────────────────────────────────────
    _spending_behavior(transactions)
    st.divider()

    # ── 3. Next-Month Forecast ────────────────────────────────────────────
    _next_month_forecast(client)
    st.divider()

    # ── 4. Recent Transactions ────────────────────────────────────────────
    _recent_transactions(transactions)
    st.divider()

    # ── 5. Quick Actions ─────────────────────────────────────────────────
    _quick_actions()
