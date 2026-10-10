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
# Financial Insights
# ---------------------------------------------------------------------------

def _compute_insights(
    transactions: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute deterministic financial insights from already-fetched data.

    All calculations use expense-only transactions (income excluded via
    ``_is_income()``).  The anomaly count comes directly from the anomalies
    list, which is already loaded by ``render()``.

    Returns a dict with keys:
        top_category      str | None
        top_category_amt  float
        top_category_pct  float          # % of total expense spending
        top_merchant      str | None
        top_merchant_amt  float
        largest_merchant  str | None
        largest_amount    float
        largest_date      str            # ISO date string, "" if none
        anomaly_count     int
    """
    from collections import defaultdict

    expense = [t for t in transactions if not _is_income(t)]

    # Category totals
    cat_totals: dict[str, float] = defaultdict(float)
    for t in expense:
        amt = float(t.get("amount", 0) or 0)
        if amt > 0:
            cat_totals[str(t.get("category") or "")] += amt

    total_spend = sum(cat_totals.values())

    if cat_totals:
        top_cat = max(cat_totals, key=cat_totals.__getitem__)
        top_cat_amt = cat_totals[top_cat]
        top_cat_pct = top_cat_amt / total_spend * 100 if total_spend else 0.0
    else:
        top_cat = None
        top_cat_amt = 0.0
        top_cat_pct = 0.0

    # Merchant totals
    merch_totals: dict[str, float] = defaultdict(float)
    for t in expense:
        amt = float(t.get("amount", 0) or 0)
        if amt > 0:
            merch_totals[str(t.get("merchant") or "")] += amt

    if merch_totals:
        top_merch = max(merch_totals, key=merch_totals.__getitem__)
        top_merch_amt = merch_totals[top_merch]
    else:
        top_merch = None
        top_merch_amt = 0.0

    # Largest single expense transaction
    positive_expense = [t for t in expense if float(t.get("amount", 0) or 0) > 0]
    if positive_expense:
        largest = max(positive_expense, key=lambda t: float(t.get("amount", 0) or 0))
        largest_merchant = str(largest.get("merchant") or "")
        largest_amount = float(largest.get("amount", 0) or 0)
        largest_date = str(largest.get("date") or "")
    else:
        largest_merchant = None
        largest_amount = 0.0
        largest_date = ""

    return {
        "top_category": top_cat,
        "top_category_amt": top_cat_amt,
        "top_category_pct": top_cat_pct,
        "top_merchant": top_merch,
        "top_merchant_amt": top_merch_amt,
        "largest_merchant": largest_merchant,
        "largest_amount": largest_amount,
        "largest_date": largest_date,
        "anomaly_count": len(anomalies),
    }


def _financial_insights(
    transactions: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> None:
    """Render the Financial Insights section.

    Interprets the existing transaction and anomaly data for the user using
    four deterministic insights: top spending category, top merchant, largest
    single expense, and anomaly count.  No additional API calls are made —
    both lists are already fetched by ``render()``.
    """
    st.subheader("💡 Financial Insights")

    expense = [t for t in transactions if not _is_income(t)]
    if not expense:
        st.info("Upload a bank statement to see financial insights.")
        return

    ins = _compute_insights(transactions, anomalies)

    col1, col2, col3, col4 = st.columns(4)

    # ── 1. Top spending category ──────────────────────────────────────────
    with col1:
        st.markdown("**Highest Spending Category**")
        if ins["top_category"]:
            st.markdown(f"### {ins['top_category']}")
            st.markdown(f"₹{ins['top_category_amt']:,.2f}")
            st.caption(f"{ins['top_category_pct']:.1f}% of total spending")
        else:
            st.caption("No expense data")

    # ── 2. Top merchant ───────────────────────────────────────────────────
    with col2:
        st.markdown("**Top Merchant**")
        if ins["top_merchant"]:
            st.markdown(f"### {ins['top_merchant']}")
            st.markdown(f"₹{ins['top_merchant_amt']:,.2f}")
        else:
            st.caption("No expense data")

    # ── 3. Largest single expense ─────────────────────────────────────────
    with col3:
        st.markdown("**Largest Expense**")
        if ins["largest_merchant"]:
            st.markdown(f"### {ins['largest_merchant']}")
            st.markdown(f"₹{ins['largest_amount']:,.2f}")
            st.caption(ins["largest_date"])
        else:
            st.caption("No expense data")

    # ── 4. Anomalies ──────────────────────────────────────────────────────
    with col4:
        st.markdown("**Anomalies Flagged**")
        count = ins["anomaly_count"]
        st.markdown(f"### {count}")
        if count == 0:
            st.caption("No unusual transactions")
        elif count == 1:
            st.caption("1 transaction flagged")
        else:
            st.caption(f"{count} transactions flagged")


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

    Calls /forecast/aggregate/next-month so the prediction always covers
    the next complete calendar month (e.g. 1–30 Nov 2026) rather than a
    rolling 30-day window from the last transaction.  The backend determines
    the target month from today's date.

    Shows the month name, predicted total, avg daily spend, and a line chart
    with 95% confidence bounds.  If history is insufficient the backend
    returns 422 with a clear message.
    """
    from datetime import date
    from calendar import month_name as _month_name

    # Compute the target month label locally so the spinner text is informative
    # without requiring an extra API call.
    _today = date.today()
    _target_month = _today.month + 1
    _target_year = _today.year + (1 if _target_month > 12 else 0)
    _target_month = 1 if _target_month > 12 else _target_month
    _period_label = f"{_month_name[_target_month]} {_target_year}"

    st.subheader("📈 Next-Month Spending Forecast")

    with st.spinner(f"Generating forecast for {_period_label}…"):
        try:
            forecast = client.get_forecast_aggregate_next_month()
        except RuntimeError as exc:
            err = str(exc)
            if "422" in err:
                if "month" in err.lower():
                    st.info(
                        "📅 **Forecast unavailable** — upload at least 2 months of "
                        "transaction history to generate a spending forecast. "
                        "Spending analysis and financial insights remain available above."
                    )
                else:
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
    horizon = forecast.get("horizon_days", len(df))

    # Build the display period from the actual forecast dates
    first_date = df["date"].iloc[0].date() if not df.empty else None
    last_date = df["date"].iloc[-1].date() if not df.empty else None
    period_str = (
        f"{first_date.strftime('%d %b')} – {last_date.strftime('%d %b %Y')}"
        if first_date and last_date
        else _period_label
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Total Spend", f"₹{predicted_total:,.2f}")
    k2.metric("Avg Daily Spend", f"₹{avg_daily:,.2f}")
    k3.metric("Forecast Period", period_str)

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
        f"Estimated total expense spending for **{_period_label}** "
        f"({horizon} days). "
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

    # ── 3. Financial Insights ─────────────────────────────────────────────
    _financial_insights(transactions, anomalies)
    st.divider()

    # ── 4. Next-Month Forecast ────────────────────────────────────────────
    _next_month_forecast(client)
    st.divider()

    # ── 5. Recent Transactions ────────────────────────────────────────────
    _recent_transactions(transactions)
    st.divider()

    # ── 6. Quick Actions ─────────────────────────────────────────────────
    _quick_actions()
