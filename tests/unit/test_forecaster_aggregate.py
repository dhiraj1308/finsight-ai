"""Unit tests for Forecaster.forecast_aggregate().

Covers:
- income exclusion from aggregation
- daily expense aggregation logic
- insufficient-history guard
- valid output structure (Forecast / ForecastPoint)
- category label = "Total Expenses"
- horizon bounds validation
- no-expense-transactions guard
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from domain import Transaction
from forecasting.forecaster import (
    Forecaster,
    Forecast,
    ForecastPoint,
    MIN_HISTORY_DAYS,
    _INCOME_KEYWORDS,
)
from ingestion.transaction_store import TransactionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_with_transactions(tmp_path, txns: list[Transaction]) -> TransactionStore:
    store = TransactionStore(str(tmp_path / "test.db"))
    store.insert(txns)
    return store


def _expense_txns(n: int = MIN_HISTORY_DAYS + 2, start: date = date(2026, 1, 1)) -> list[Transaction]:
    """Return n expense transactions on distinct consecutive days."""
    from datetime import timedelta
    return [
        Transaction(
            date=start + timedelta(days=i),
            merchant="Swiggy",
            amount=500.0 + i,
            category="Dining",
            source_file="test.csv",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TEST 1 — income exclusion
# ---------------------------------------------------------------------------

class TestIncomeExclusion:

    def test_salary_credit_excluded(self, tmp_path):
        """Salary Credit rows must not be counted in expense aggregation."""
        txns = _expense_txns(MIN_HISTORY_DAYS + 2)
        # Add two income rows
        txns += [
            Transaction(date=date(2026, 2, 1), merchant="Salary Credit",
                        amount=65000.0, category="Other", source_file="test"),
            Transaction(date=date(2026, 2, 2), merchant="Salary Credit",
                        amount=65000.0, category="Other", source_file="test"),
        ]
        store = _store_with_transactions(tmp_path, txns)
        f = Forecaster()
        forecast = f.forecast_aggregate(30, store)

        # Total forecast must be much less than 130000
        total = sum(p.yhat for p in forecast.points)
        assert total < 130_000, (
            f"Income appears to be included: 30-day total={total:.2f}"
        )

    def test_all_income_keywords_excluded(self, tmp_path):
        """All five income keywords must be recognised."""
        income_merchants = [
            "Salary Credit",
            "Bank Credit",
            "Side Income",
            "Amazon Refund",
            "Cashback Reward",
        ]
        # Build a minimal store with expense data passing the MIN_HISTORY_DAYS guard
        txns = _expense_txns(MIN_HISTORY_DAYS + 2)
        for m in income_merchants:
            txns.append(Transaction(
                date=date(2025, 12, 1), merchant=m,
                amount=50000.0, category="Other", source_file="test",
            ))
        store = _store_with_transactions(tmp_path, txns)
        f = Forecaster()
        forecast = f.forecast_aggregate(30, store)

        # Forecast total must only reflect ~500-odd daily amounts, not 50000×5
        total = sum(p.yhat for p in forecast.points)
        assert total < 250_000, (
            f"Income keyword rows may not all be excluded; total={total:.2f}"
        )

    def test_income_keyword_set_matches_frontend(self):
        """Backend income keywords must match the frontend _INCOME_KEYWORDS."""
        frontend_keywords = frozenset({
            "salary", "credit", "income", "refund", "cashback",
        })
        assert _INCOME_KEYWORDS == frontend_keywords, (
            "Backend and frontend income keywords are out of sync.\n"
            f"Backend : {sorted(_INCOME_KEYWORDS)}\n"
            f"Frontend: {sorted(frontend_keywords)}"
        )


# ---------------------------------------------------------------------------
# TEST 2 — daily aggregation
# ---------------------------------------------------------------------------

class TestDailyAggregation:

    def test_multiple_transactions_same_day_aggregated(self, tmp_path):
        """Two expense rows on the same day must be summed into one data point."""
        from datetime import timedelta
        # First build enough days to pass MIN_HISTORY_DAYS
        txns = _expense_txns(MIN_HISTORY_DAYS)
        # Override last two rows to the same date
        shared_date = txns[-1].date
        extra = Transaction(
            date=shared_date, merchant="FreshMart",
            amount=1000.0, category="Groceries", source_file="test",
        )
        txns.append(extra)
        store = _store_with_transactions(tmp_path, txns)

        f = Forecaster()
        # Must not raise — aggregation prevents a duplicate date issue
        forecast = f.forecast_aggregate(7, store)
        assert len(forecast.points) == 7


# ---------------------------------------------------------------------------
# TEST 3 — insufficient history guard
# ---------------------------------------------------------------------------

class TestInsufficientHistory:

    def test_raises_when_below_min_history_days(self, tmp_path):
        """Fewer than MIN_HISTORY_DAYS distinct expense days must raise ValueError."""
        txns = _expense_txns(MIN_HISTORY_DAYS - 1)
        store = _store_with_transactions(tmp_path, txns)

        f = Forecaster()
        with pytest.raises(ValueError) as exc_info:
            f.forecast_aggregate(30, store)
        assert str(MIN_HISTORY_DAYS) in str(exc_info.value)

    def test_raises_when_no_expense_transactions(self, tmp_path):
        """Only income rows → no expense data → ValueError."""
        txns = [
            Transaction(date=date(2026, 9, 1), merchant="Salary Credit",
                        amount=65000.0, category="Other", source_file="test"),
        ]
        store = _store_with_transactions(tmp_path, txns)

        f = Forecaster()
        with pytest.raises(ValueError):
            f.forecast_aggregate(30, store)

    def test_raises_when_store_is_empty(self, tmp_path):
        """Empty store → ValueError."""
        store = TransactionStore(str(tmp_path / "empty.db"))

        f = Forecaster()
        with pytest.raises(ValueError):
            f.forecast_aggregate(30, store)

    def test_raises_on_invalid_horizon_zero(self, tmp_path):
        store = _store_with_transactions(tmp_path, _expense_txns())
        f = Forecaster()
        with pytest.raises(ValueError, match="horizon_days"):
            f.forecast_aggregate(0, store)

    def test_raises_on_invalid_horizon_too_large(self, tmp_path):
        store = _store_with_transactions(tmp_path, _expense_txns())
        f = Forecaster()
        with pytest.raises(ValueError):
            f.forecast_aggregate(366, store)


# ---------------------------------------------------------------------------
# TEST 4 — output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:

    @pytest.fixture()
    def forecast(self, tmp_path):
        store = _store_with_transactions(tmp_path, _expense_txns(MIN_HISTORY_DAYS + 5))
        return Forecaster().forecast_aggregate(30, store)

    def test_returns_forecast_instance(self, forecast):
        assert isinstance(forecast, Forecast)

    def test_category_is_total_expenses(self, forecast):
        assert forecast.category == "Total Expenses"

    def test_horizon_days_matches_request(self, forecast):
        assert forecast.horizon_days == 30

    def test_correct_number_of_points(self, forecast):
        assert len(forecast.points) == 30

    def test_all_points_are_forecast_point(self, forecast):
        for p in forecast.points:
            assert isinstance(p, ForecastPoint)

    def test_all_yhat_non_negative(self, forecast):
        for p in forecast.points:
            assert p.yhat >= 0.0
            assert p.yhat_lower >= 0.0

    def test_confidence_interval_valid(self, forecast):
        for p in forecast.points:
            assert p.yhat_lower <= p.yhat <= p.yhat_upper

    def test_dates_are_consecutive(self, forecast):
        from datetime import timedelta
        for i in range(1, len(forecast.points)):
            expected = forecast.points[i - 1].date + timedelta(days=1)
            assert forecast.points[i].date == expected, (
                f"Gap between point {i-1} and {i}"
            )
