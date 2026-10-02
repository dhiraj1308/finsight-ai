"""Anomalies page — AI-detected unusual spending patterns."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import page_header

# Anomaly score thresholds for severity bands.
# Scores come from IsolationForest: clip(-decision_function, 0, 1),
# so higher value = more anomalous.
_HIGH_THRESHOLD: float = 0.6
_MEDIUM_THRESHOLD: float = 0.3

_SEVERITY_HIGH = "🔴 High"
_SEVERITY_MEDIUM = "🟡 Medium"
_SEVERITY_LOW = "🟢 Low"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_anomalies(client: APIClient) -> list[dict[str, Any]] | None:
    """Fetch anomalous transactions from the backend.

    Returns the list on success, or ``None`` on error (error rendered inline
    before returning).
    """
    try:
        return client.get_anomalies()
    except RuntimeError as exc:
        st.error(f"Could not load anomalies: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error loading anomalies: {type(exc).__name__}")
        return None


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _to_dataframe(anomalies: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert the raw API payload to a clean, typed :class:`pd.DataFrame`."""
    df = pd.DataFrame(anomalies)

    if df.empty:
        return pd.DataFrame(
            columns=["date", "merchant", "category", "amount", "anomaly_score",
                     "needs_review", "severity"]
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["anomaly_score"] = (
        pd.to_numeric(df.get("anomaly_score", 0.0), errors="coerce").fillna(0.0)
    )
    df["merchant"] = df["merchant"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["needs_review"] = df.get("needs_review", False).astype(bool)

    # Derive severity from anomaly_score
    df["severity"] = df["anomaly_score"].apply(_score_to_severity)

    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def _score_to_severity(score: float) -> str:
    """Map a normalised anomaly score to a human-readable severity label."""
    if score >= _HIGH_THRESHOLD:
        return _SEVERITY_HIGH
    if score >= _MEDIUM_THRESHOLD:
        return _SEVERITY_MEDIUM
    return _SEVERITY_LOW


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def _sidebar_controls(df: pd.DataFrame) -> tuple[str, str]:
    """Render filter controls in the sidebar.

    Returns
    -------
    tuple of (severity_filter, category_filter)
    """
    with st.sidebar:
        st.header("🔍 Filters")

        severity_options = [
            "All",
            _SEVERITY_HIGH,
            _SEVERITY_MEDIUM,
            _SEVERITY_LOW,
        ]
        severity_filter = st.selectbox("Severity", options=severity_options)

        st.divider()

        categories = sorted(df["category"].dropna().unique().tolist())
        category_filter = st.selectbox(
            "Category",
            options=["All"] + categories,
        )

    return severity_filter, category_filter


def _apply_filters(
    df: pd.DataFrame,
    severity: str,
    category: str,
) -> pd.DataFrame:
    """Apply severity and category filters to *df*."""
    if severity != "All":
        df = df[df["severity"] == severity]
    if category != "All":
        df = df[df["category"] == category]
    return df


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _kpi_cards(df: pd.DataFrame, total_anomalies: int) -> None:
    """Render KPI summary cards derived from the backend response fields.

    Severity bands are computed from ``anomaly_score`` — the only numeric
    field returned by the backend that quantifies how anomalous each
    transaction is.
    """
    high = int((df["anomaly_score"] >= _HIGH_THRESHOLD).sum())
    medium = int(
        ((df["anomaly_score"] >= _MEDIUM_THRESHOLD) & (df["anomaly_score"] < _HIGH_THRESHOLD)).sum()
    )
    low = int((df["anomaly_score"] < _MEDIUM_THRESHOLD).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Anomalies", f"{total_anomalies:,}")
    col2.metric("🔴 High Severity", f"{high:,}")
    col3.metric("🟡 Medium Severity", f"{medium:,}")
    col4.metric("🟢 Low Severity", f"{low:,}")


# ---------------------------------------------------------------------------
# Anomaly table
# ---------------------------------------------------------------------------


def _anomaly_table(df: pd.DataFrame) -> None:
    """Render the anomalies dataframe with severity-aware column config."""
    st.subheader("Anomalous Transactions")

    display = df[
        ["date", "merchant", "category", "amount", "anomaly_score", "severity", "needs_review"]
    ].rename(
        columns={
            "date": "Date",
            "merchant": "Merchant",
            "category": "Category",
            "amount": "Amount (₹)",
            "anomaly_score": "Anomaly Score",
            "severity": "Severity",
            "needs_review": "Needs Review",
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "Amount (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            "Anomaly Score": st.column_config.ProgressColumn(
                "Anomaly Score",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "Needs Review": st.column_config.CheckboxColumn("Needs Review"),
        },
    )


# ---------------------------------------------------------------------------
# Score distribution chart
# ---------------------------------------------------------------------------


def _score_distribution_chart(df: pd.DataFrame) -> None:
    """Bar chart — number of anomalies per severity band."""
    st.subheader("Anomaly Severity Distribution")

    counts = (
        df["severity"]
        .value_counts()
        .reindex([_SEVERITY_HIGH, _SEVERITY_MEDIUM, _SEVERITY_LOW], fill_value=0)
        .reset_index()
    )
    counts.columns = ["Severity", "Count"]

    st.bar_chart(counts.set_index("Severity")["Count"], use_container_width=True)


# ---------------------------------------------------------------------------
# Category breakdown chart
# ---------------------------------------------------------------------------


def _category_chart(df: pd.DataFrame) -> None:
    """Horizontal bar — anomaly count per spending category."""
    st.subheader("Anomalies by Category")

    cat_counts = (
        df.groupby("category", as_index=False)
        .size()
        .rename(columns={"category": "Category", "size": "Anomalies"})
        .sort_values("Anomalies", ascending=False)
    )

    if cat_counts.empty:
        st.info("No category data available.")
        return

    st.bar_chart(
        cat_counts.set_index("Category")["Anomalies"],
        use_container_width=True,
        horizontal=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(client: APIClient) -> None:
    """Render the Anomalies page."""
    page_header(
        "🚨 Anomalies",
        subtitle="AI-detected unusual spending patterns.",
    )

    raw = _load_anomalies(client)
    if raw is None:
        return

    if not raw:
        st.info(
            "No anomalies detected. This could mean:\n"
            "- Not enough transactions have been uploaded yet "
            "(at least 10 are required to run the model).\n"
            "- All transactions look normal based on your spending history."
        )
        return

    df = _to_dataframe(raw)
    total_anomalies = len(df)

    # Sidebar filters
    severity_filter, category_filter = _sidebar_controls(df)
    filtered = _apply_filters(df, severity_filter, category_filter)

    # KPI cards always reflect the full (unfiltered) dataset so the totals
    # are stable regardless of what the user has filtered to
    _kpi_cards(df, total_anomalies)
    st.divider()

    if filtered.empty:
        st.info(
            "No anomalies match the current filters. "
            "Try selecting a different severity or category."
        )
        return

    # Main table
    _anomaly_table(filtered)
    st.caption(f"Showing {len(filtered):,} of {total_anomalies:,} anomalies")

    st.divider()

    # Charts side by side
    col_dist, col_cat = st.columns(2)
    with col_dist:
        _score_distribution_chart(filtered)
    with col_cat:
        _category_chart(filtered)
