"""Forecast page — predicted future spending by category."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import page_header

# Horizon options shown to the user: label → days value
_HORIZON_OPTIONS: dict[str, int] = {
    "7 days": 7,
    "30 days": 30,
    "60 days": 60,
    "90 days": 90,
}

_DEFAULT_HORIZON = "30 days"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_categories(client: APIClient) -> list[str] | None:
    """Fetch all transactions and extract unique non-empty categories.

    Returns a sorted list on success, or ``None`` on error (error rendered
    inline before returning).
    """
    try:
        transactions = client.get_transactions()
    except RuntimeError as exc:
        st.error(f"Could not load transactions: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return None

    if not transactions:
        return []

    categories = sorted(
        {
            t.get("category", "").strip()
            for t in transactions
            if t.get("category", "").strip()
        }
    )
    return categories


def _load_forecast(
    client: APIClient, category: str, days: int
) -> dict[str, Any] | None:
    """Fetch the forecast for *category* over *days* from the backend.

    Returns the parsed payload dict on success, or ``None`` on error (error
    rendered inline before returning).
    """
    try:
        return client.get_forecast(category, days)
    except RuntimeError as exc:
        st.error(f"Could not load forecast: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)
        return None


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def _points_to_dataframe(points: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ``ForecastDTO.points`` to a clean :class:`pd.DataFrame`.

    Columns: ``date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    """
    df = pd.DataFrame(points)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Controls (rendered inline, not in sidebar, so category + horizon are
# visually paired at the top of the page — matching the requirement).
# ---------------------------------------------------------------------------


def _top_controls(categories: list[str]) -> tuple[str, int]:
    """Render the category selector and horizon selector below the header.

    Returns
    -------
    tuple of (selected_category, horizon_days)
    """
    col_cat, col_horizon, _ = st.columns([2, 2, 4])

    with col_cat:
        selected_category = st.selectbox(
            "Category",
            options=categories,
            help="Select a spending category to forecast.",
        )

    with col_horizon:
        horizon_label = st.selectbox(
            "Forecast horizon",
            options=list(_HORIZON_OPTIONS.keys()),
            index=list(_HORIZON_OPTIONS.keys()).index(_DEFAULT_HORIZON),
            help="How many days ahead to predict.",
        )

    return selected_category, _HORIZON_OPTIONS[horizon_label]


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _kpi_cards(forecast: dict[str, Any], df: pd.DataFrame) -> None:
    """Render KPI cards from the fields actually present in the response.

    Only values that exist in the backend response are displayed — no
    invented data.
    """
    points = df[df["yhat"] > 0] if not df.empty else df

    predicted_total = points["yhat"].sum() if not points.empty else 0.0
    avg_forecast = points["yhat"].mean() if not points.empty else 0.0
    horizon_days: int = forecast.get("horizon_days", len(df))

    # Trend: compare second-half average to first-half average
    if len(points) >= 4:
        mid = len(points) // 2
        first_half_avg = points["yhat"].iloc[:mid].mean()
        second_half_avg = points["yhat"].iloc[mid:].mean()
        if first_half_avg > 0:
            trend_pct = (second_half_avg - first_half_avg) / first_half_avg * 100
            trend_label = f"{'↑' if trend_pct > 0 else '↓'} {abs(trend_pct):.1f}%"
            trend_delta_color = "inverse" if trend_pct > 0 else "normal"
        else:
            trend_label = "—"
            trend_delta_color = "off"
    else:
        trend_label = "—"
        trend_delta_color = "off"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted Total", f"₹{predicted_total:,.2f}")
    col2.metric("Avg Daily Forecast", f"₹{avg_forecast:,.2f}")
    col3.metric("Forecast Period", f"{horizon_days} days")
    col4.metric(
        "Expected Trend",
        trend_label,
        delta_color=trend_delta_color,
    )


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def _forecast_chart(df: pd.DataFrame, category: str) -> None:
    """Render the forecast chart with predicted values and confidence band.

    Shows:
    - Forecast line (``yhat``)
    - Lower bound (``yhat_lower``)
    - Upper bound (``yhat_upper``)

    All three series are plotted as a line chart so the confidence interval
    is visible alongside the central prediction — no extra dependencies
    beyond pandas and Streamlit.
    """
    st.subheader(f"Spending Forecast — {category}")

    if df.empty:
        st.info("No forecast points returned by the backend.")
        return

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
        "**Forecast** — predicted daily spending. "
        "**Lower / Upper bound** — 80 % confidence interval."
    )


# ---------------------------------------------------------------------------
# Detailed data table
# ---------------------------------------------------------------------------


def _forecast_table(df: pd.DataFrame) -> None:
    """Render a collapsible table of raw forecast values."""
    with st.expander("View raw forecast data"):
        display = df[["date", "yhat", "yhat_lower", "yhat_upper"]].copy()
        display["date"] = display["date"].dt.date
        display = display.rename(
            columns={
                "date": "Date",
                "yhat": "Forecast (₹)",
                "yhat_lower": "Lower (₹)",
                "yhat_upper": "Upper (₹)",
            }
        )
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Forecast (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Lower (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                "Upper (₹)": st.column_config.NumberColumn(format="₹%.2f"),
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(client: APIClient) -> None:
    """Render the Forecast page."""
    page_header(
        "📈 Forecast",
        subtitle="Predicted future spending by category, powered by Prophet.",
    )

    # Load categories from transaction history
    categories = _load_categories(client)
    if categories is None:
        # Error already rendered by _load_categories
        return

    if not categories:
        st.info(
            "No transaction categories found. "
            "Upload a statement first so there is data to forecast."
        )
        return

    # Category + horizon controls sit directly below the header
    selected_category, horizon_days = _top_controls(categories)
    st.divider()

    # Fetch forecast — show a spinner while waiting
    with st.spinner(f"Generating {horizon_days}-day forecast for {selected_category}…"):
        forecast = _load_forecast(client, selected_category, horizon_days)

    if forecast is None:
        # Error already rendered by _load_forecast
        return

    points: list[dict[str, Any]] = forecast.get("points", [])
    if not points:
        st.info(
            f"The backend returned no forecast points for **{selected_category}**. "
            "This category may have too few historical transactions to model."
        )
        return

    df = _points_to_dataframe(points)

    # KPI cards
    _kpi_cards(forecast, df)
    st.divider()

    # Chart + raw data table
    _forecast_chart(df, selected_category)
    _forecast_table(df)
