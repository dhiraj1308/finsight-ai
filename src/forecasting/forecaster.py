from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Union

import numpy as np
import pandas as pd

from ingestion.transaction_store import TransactionStore

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 14
MIN_HISTORY_MONTHS = 2
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 365

# ---------------------------------------------------------------------------
# Calendar-month helper
# ---------------------------------------------------------------------------

def next_calendar_month_range(today: date) -> tuple[date, date]:
    """Return (first_day, last_day) of the calendar month after *today*.

    Handles year boundaries and variable month lengths (28/29/30/31 days).

    Parameters
    ----------
    today:
        Reference date.  Typically ``date.today()`` at call time.

    Returns
    -------
    tuple[date, date]
        First and last day of the next calendar month, inclusive.

    Examples
    --------
    >>> next_calendar_month_range(date(2026, 10, 9))
    (date(2026, 11, 1), date(2026, 11, 30))
    >>> next_calendar_month_range(date(2026, 12, 31))
    (date(2027, 1, 1), date(2027, 1, 31))
    >>> next_calendar_month_range(date(2027, 1, 20))   # Feb non-leap
    (date(2027, 2, 1), date(2027, 2, 28))
    >>> next_calendar_month_range(date(2028, 1, 20))   # Feb leap
    (date(2028, 2, 1), date(2028, 2, 29))
    """
    month = today.month + 1
    year = today.year
    if month > 12:
        month = 1
        year += 1
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])
    return first_day, last_day


# ---------------------------------------------------------------------------
# Income-keyword filter — mirrors frontend.views.dashboard._INCOME_KEYWORDS.
# Transactions whose merchant name contains any of these words (case-insensitive)
# are treated as income/credit and excluded from expense-level forecasting.
# ---------------------------------------------------------------------------
_INCOME_KEYWORDS: frozenset[str] = frozenset({
    "salary", "credit", "income", "refund", "cashback",
})


@dataclass
class ForecastPoint:
    date: date
    yhat: float        # point estimate, floored at 0.0
    yhat_lower: float  # 95% CI lower bound, floored at 0.0
    yhat_upper: float  # 95% CI upper bound


@dataclass
class Forecast:
    category: str
    horizon_days: int
    points: list[ForecastPoint]


class Forecaster:
    """
    Forecasts future spending per category using exponentially weighted
    moving average with linear trend. Pure numpy — no external ML
    dependencies, CPU-only, reliable on all platforms.
    """

    def __init__(self, alpha: float = 0.3):
        # Smoothing factor: higher = more weight on recent observations
        self._alpha = alpha

    def _get_daily_totals(
        self, category: str, store: TransactionStore
    ) -> pd.DataFrame:
        """
        Aggregates transactions for a category into daily totals.
        Returns DataFrame with columns: ds (date), y (total amount).
        """
        transactions = store.query_by_category(category)
        if not transactions:
            return pd.DataFrame(columns=["ds", "y"])

        daily: dict[date, float] = {}
        for txn in transactions:
            daily[txn.date] = daily.get(txn.date, 0.0) + txn.amount

        df = pd.DataFrame(
            [{"ds": d, "y": total} for d, total in daily.items()]
        )
        return df.sort_values("ds").reset_index(drop=True)

    def _ewma_with_trend(
        self, y: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, float]:
        """
        Exponentially weighted moving average with linear trend projection.
        Returns (point_estimates array, residual_std float).
        """
        n = len(y)
        alpha = self._alpha

        # Compute exponentially decaying weights (most recent = highest weight)
        weights = np.array([(1 - alpha) ** i for i in range(n)])
        weights = weights[::-1]
        weights /= weights.sum()
        smoothed_baseline = float(np.dot(weights, y))

        # Estimate trend from second half of history via linear regression
        half = max(n // 2, 2)
        x = np.arange(half, dtype=float)
        slope = float(np.polyfit(x, y[-half:], 1)[0])

        # Project forward: baseline + trend * step
        point_estimates = np.array(
            [max(0.0, smoothed_baseline + slope * (i + 1)) for i in range(horizon)]
        )

        # Residual std from baseline (used for confidence intervals)
        residuals = y - smoothed_baseline
        std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        return point_estimates, std

    def forecast_category(
        self,
        category: str,
        horizon_days: int,
        store: TransactionStore,
    ) -> Forecast:
        """
        Forecasts spending for a single category.

        Raises ValueError for:
        - horizon_days outside [1, 365]
        - category with no transactions
        - category with fewer than 14 distinct calendar days of history
        """
        if not (MIN_HORIZON_DAYS <= horizon_days <= MAX_HORIZON_DAYS):
            raise ValueError(
                f"horizon_days must be between {MIN_HORIZON_DAYS} and "
                f"{MAX_HORIZON_DAYS}, got {horizon_days}."
            )

        df = self._get_daily_totals(category, store)

        if df.empty:
            raise ValueError(
                f"No transactions found for category '{category}'."
            )

        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Category '{category}' has only {len(df)} distinct calendar "
                f"days of history. At least {MIN_HISTORY_DAYS} are required."
            )

        y = df["y"].values.astype(float)
        point_estimates, std = self._ewma_with_trend(y, horizon_days)

        z95 = 1.96  # 95% confidence interval z-score
        margin = z95 * std

        last_date = df["ds"].iloc[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()

        points = []
        for i in range(horizon_days):
            forecast_date = last_date + timedelta(days=i + 1)
            yhat = float(point_estimates[i])
            points.append(
                ForecastPoint(
                    date=forecast_date,
                    yhat=yhat,
                    yhat_lower=max(0.0, yhat - margin),
                    yhat_upper=yhat + margin,
                )
            )

        return Forecast(
            category=category,
            horizon_days=horizon_days,
            points=points,
        )

    def forecast_all(
        self,
        horizon_days: int,
        store: TransactionStore,
    ) -> dict[str, Union[Forecast, str]]:
        """
        Forecasts all categories with at least one transaction.
        Returns dict mapping category -> Forecast on success,
        or category -> error string on insufficient data.
        Never raises.
        """
        all_txns = store.get_all()
        categories = {txn.category for txn in all_txns if txn.category}

        results: dict[str, Union[Forecast, str]] = {}
        for category in categories:
            try:
                results[category] = self.forecast_category(
                    category, horizon_days, store
                )
            except ValueError as e:
                results[category] = str(e)

        return results

    def forecast_aggregate(
        self,
        horizon_days: int,
        store: TransactionStore,
    ) -> Forecast:
        """Forecast total expense spending across all categories.

        Aggregates ALL non-income transactions by calendar date, then runs
        the same EWMA + linear trend algorithm used by ``forecast_category``.
        Income is identified by the ``_INCOME_KEYWORDS`` merchant-name heuristic.

        Returns a :class:`Forecast` with ``category="Total Expenses"``.

        Raises
        ------
        ValueError
            If horizon_days is out of range, no expense transactions exist,
            or fewer than ``MIN_HISTORY_DAYS`` distinct expense days are found.
        """
        if not (MIN_HORIZON_DAYS <= horizon_days <= MAX_HORIZON_DAYS):
            raise ValueError(
                f"horizon_days must be between {MIN_HORIZON_DAYS} and "
                f"{MAX_HORIZON_DAYS}, got {horizon_days}."
            )

        all_txns = store.get_all()
        expense_txns = [
            t for t in all_txns
            if not any(kw in t.merchant.lower() for kw in _INCOME_KEYWORDS)
        ]

        if not expense_txns:
            raise ValueError(
                "No expense transactions found. "
                "Upload a bank statement to enable forecasting."
            )

        # Aggregate by calendar date
        daily: dict[date, float] = {}
        for txn in expense_txns:
            daily[txn.date] = daily.get(txn.date, 0.0) + txn.amount

        df = pd.DataFrame(
            [{"ds": d, "y": total} for d, total in daily.items()]
        ).sort_values("ds").reset_index(drop=True)

        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Only {len(df)} distinct calendar days of expense history found. "
                f"At least {MIN_HISTORY_DAYS} are required."
            )

        distinct_months = len({(d.year, d.month) for d in df["ds"]})
        if distinct_months < MIN_HISTORY_MONTHS:
            raise ValueError(
                f"Expense history spans only {distinct_months} calendar month(s). "
                f"At least {MIN_HISTORY_MONTHS} months are required for forecasting. "
                "Upload statements from at least 2 different months."
            )

        y = df["y"].values.astype(float)
        point_estimates, std = self._ewma_with_trend(y, horizon_days)

        z95 = 1.96
        margin = z95 * std

        last_date = df["ds"].iloc[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()

        points = []
        for i in range(horizon_days):
            forecast_date = last_date + timedelta(days=i + 1)
            yhat = float(point_estimates[i])
            points.append(
                ForecastPoint(
                    date=forecast_date,
                    yhat=yhat,
                    yhat_lower=max(0.0, yhat - margin),
                    yhat_upper=yhat + margin,
                )
            )

        return Forecast(
            category="Total Expenses",
            horizon_days=horizon_days,
            points=points,
        )

    def forecast_aggregate_next_month(
        self,
        store: TransactionStore,
        *,
        today: date | None = None,
    ) -> Forecast:
        """Forecast total expense spending for the next complete calendar month.

        Applies the same eligibility guards as ``forecast_aggregate`` (14
        distinct expense days, 2 distinct calendar months), then generates one
        daily prediction for every day in the next calendar month relative to
        *today* (defaults to ``date.today()``).

        The forecast dates are exactly ``[first_day … last_day]`` of the next
        calendar month, so the returned ``horizon_days`` equals the number of
        days in that month (28, 29, 30, or 31).

        Parameters
        ----------
        store:
            The transaction store to read history from.
        today:
            Override for the reference date.  Defaults to ``date.today()``.
            Exposed for testing.

        Returns
        -------
        Forecast
            ``category="Total Expenses"``, ``horizon_days`` = days in next
            month, one ``ForecastPoint`` per day.

        Raises
        ------
        ValueError
            If eligibility guards fail (no expenses, < 14 distinct expense
            days, or < 2 distinct expense months).
        """
        if today is None:
            today = date.today()

        first_day, last_day = next_calendar_month_range(today)
        horizon_days = (last_day - first_day).days + 1

        all_txns = store.get_all()
        expense_txns = [
            t for t in all_txns
            if not any(kw in t.merchant.lower() for kw in _INCOME_KEYWORDS)
        ]

        if not expense_txns:
            raise ValueError(
                "No expense transactions found. "
                "Upload a bank statement to enable forecasting."
            )

        # Aggregate by calendar date
        daily: dict[date, float] = {}
        for txn in expense_txns:
            daily[txn.date] = daily.get(txn.date, 0.0) + txn.amount

        df = pd.DataFrame(
            [{"ds": d, "y": total} for d, total in daily.items()]
        ).sort_values("ds").reset_index(drop=True)

        if len(df) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Only {len(df)} distinct calendar days of expense history found. "
                f"At least {MIN_HISTORY_DAYS} are required."
            )

        distinct_months = len({(d.year, d.month) for d in df["ds"]})
        if distinct_months < MIN_HISTORY_MONTHS:
            raise ValueError(
                f"Expense history spans only {distinct_months} calendar month(s). "
                f"At least {MIN_HISTORY_MONTHS} months are required for forecasting. "
                "Upload statements from at least 2 different months."
            )

        y = df["y"].values.astype(float)
        point_estimates, std = self._ewma_with_trend(y, horizon_days)

        z95 = 1.96
        margin = z95 * std

        points = []
        for i in range(horizon_days):
            forecast_date = first_day + timedelta(days=i)
            yhat = float(point_estimates[i])
            points.append(
                ForecastPoint(
                    date=forecast_date,
                    yhat=yhat,
                    yhat_lower=max(0.0, yhat - margin),
                    yhat_upper=yhat + margin,
                )
            )

        return Forecast(
            category="Total Expenses",
            horizon_days=horizon_days,
            points=points,
        )
