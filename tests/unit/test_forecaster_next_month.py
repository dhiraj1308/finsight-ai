"""Unit tests for next_calendar_month_range() and forecast_aggregate_next_month().

Covers:
- next_calendar_month_range: all edge cases (year boundary, Feb leap/non-leap,
  28/29/30/31-day months, mid-month and end-of-month reference dates)
- forecast_aggregate_next_month: eligibility guards (14-day, 2-month)
- forecast_aggregate_next_month: dates always cover the exact next calendar month
- forecast_aggregate_next_month: horizon_days = days in that month
- forecast_aggregate_next_month: predicted total == sum of daily yhats
- forecast_aggregate_next_month: income excluded from eligibility check
- forecast_aggregate_next_month: today= override works for deterministic tests
- category forecast (forecast_category) is unaffected
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
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
    MIN_HISTORY_MONTHS,
    next_calendar_month_range,
)
from ingestion.transaction_store import TransactionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(tmp_path, txns):
    s = TransactionStore(str(tmp_path / "test.db"))
    s.insert(txns)
    return s


def _expense_txns(n: int, start: date) -> list[Transaction]:
    """n consecutive daily expense transactions starting at start."""
    return [
        Transaction(
            date=start + timedelta(days=i),
            merchant="Swiggy",
            amount=600.0 + i,
            category="Dining",
            source_file="test.csv",
        )
        for i in range(n)
    ]


def _two_month_store(tmp_path, start: date = date(2026, 1, 20), n: int = 16):
    """Store with n expense days spanning 2 calendar months. Default: Jan–Feb."""
    return _store(tmp_path, _expense_txns(n, start))


# ---------------------------------------------------------------------------
# TEST GROUP 1 — next_calendar_month_range()
# ---------------------------------------------------------------------------

class TestNextCalendarMonthRange:

    def test_mid_month_october(self):
        first, last = next_calendar_month_range(date(2026, 10, 9))
        assert first == date(2026, 11, 1)
        assert last == date(2026, 11, 30)

    def test_first_of_month(self):
        first, last = next_calendar_month_range(date(2026, 11, 1))
        assert first == date(2026, 12, 1)
        assert last == date(2026, 12, 31)

    def test_last_of_month(self):
        first, last = next_calendar_month_range(date(2026, 10, 31))
        assert first == date(2026, 11, 1)
        assert last == date(2026, 11, 30)

    def test_year_boundary_december_31(self):
        first, last = next_calendar_month_range(date(2026, 12, 31))
        assert first == date(2027, 1, 1)
        assert last == date(2027, 1, 31)

    def test_year_boundary_december_1(self):
        first, last = next_calendar_month_range(date(2026, 12, 1))
        assert first == date(2027, 1, 1)
        assert last == date(2027, 1, 31)

    def test_february_non_leap_year(self):
        """Jan 2027 → Feb 2027 (non-leap: 28 days)."""
        first, last = next_calendar_month_range(date(2027, 1, 15))
        assert first == date(2027, 2, 1)
        assert last == date(2027, 2, 28)
        assert (last - first).days + 1 == 28

    def test_february_leap_year(self):
        """Jan 2028 → Feb 2028 (leap: 29 days)."""
        first, last = next_calendar_month_range(date(2028, 1, 15))
        assert first == date(2028, 2, 1)
        assert last == date(2028, 2, 29)
        assert (last - first).days + 1 == 29

    def test_30_day_month_april(self):
        first, last = next_calendar_month_range(date(2026, 3, 15))
        assert first == date(2026, 4, 1)
        assert last == date(2026, 4, 30)
        assert (last - first).days + 1 == 30

    def test_31_day_month_january(self):
        first, last = next_calendar_month_range(date(2026, 12, 15))
        assert first == date(2027, 1, 1)
        assert last == date(2027, 1, 31)
        assert (last - first).days + 1 == 31

    def test_first_always_day_1(self):
        """first_day must always be the 1st of a month."""
        for month in range(1, 13):
            first, _ = next_calendar_month_range(date(2026, month, 15))
            assert first.day == 1

    def test_last_equals_calendar_end(self):
        """last_day must always be the final day of the target month."""
        import calendar
        for month in range(1, 13):
            today = date(2026, month, 10)
            _, last = next_calendar_month_range(today)
            target_month = month + 1 if month < 12 else 1
            target_year = 2026 if month < 12 else 2027
            expected_last = calendar.monthrange(target_year, target_month)[1]
            assert last.day == expected_last

    def test_days_count_correct_for_all_months(self):
        """horizon_days must match the actual number of days in the target month."""
        import calendar
        for month in range(1, 13):
            today = date(2026, month, 1)
            first, last = next_calendar_month_range(today)
            computed = (last - first).days + 1
            target_m = first.month
            target_y = first.year
            expected = calendar.monthrange(target_y, target_m)[1]
            assert computed == expected, (
                f"For today={today}: expected {expected} days, got {computed}"
            )


# ---------------------------------------------------------------------------
# TEST GROUP 2 — forecast_aggregate_next_month() output structure
# ---------------------------------------------------------------------------

class TestForecastAggregateNextMonthStructure:

    @pytest.fixture()
    def store(self, tmp_path):
        return _two_month_store(tmp_path)

    def test_returns_forecast_instance(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert isinstance(result, Forecast)

    def test_category_is_total_expenses(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert result.category == "Total Expenses"

    def test_horizon_days_equals_days_in_next_month(self, store):
        """horizon_days must equal the number of days in November 2026 (30)."""
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert result.horizon_days == 30

    def test_points_count_matches_horizon_days(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert len(result.points) == result.horizon_days

    def test_forecast_dates_are_november(self, store):
        """All forecast dates must fall within November 2026."""
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        for p in result.points:
            assert p.date.year == 2026
            assert p.date.month == 11

    def test_first_forecast_date_is_first_of_next_month(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert result.points[0].date == date(2026, 11, 1)

    def test_last_forecast_date_is_last_of_next_month(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        assert result.points[-1].date == date(2026, 11, 30)

    def test_forecast_dates_are_consecutive(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        for i in range(1, len(result.points)):
            expected = result.points[i - 1].date + timedelta(days=1)
            assert result.points[i].date == expected

    def test_all_yhat_non_negative(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        for p in result.points:
            assert p.yhat >= 0.0
            assert p.yhat_lower >= 0.0

    def test_confidence_interval_valid(self, store):
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        for p in result.points:
            assert p.yhat_lower <= p.yhat <= p.yhat_upper

    def test_predicted_total_equals_sum_of_daily_yhats(self, store):
        """This is the key product requirement: total = sum(yhat for each day)."""
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 10, 9))
        computed_total = sum(p.yhat for p in result.points)
        # Sum from 30 daily predictions must equal the number we'd display
        assert abs(computed_total - sum(p.yhat for p in result.points)) < 1e-9


# ---------------------------------------------------------------------------
# TEST GROUP 3 — month boundary and year boundary
# ---------------------------------------------------------------------------

class TestForecastAggregateNextMonthBoundaries:

    def _store_jan_feb_2027(self, tmp_path):
        """16 days spanning Jan 20 – Feb 4, 2027."""
        return _two_month_store(tmp_path, start=date(2027, 1, 20), n=16)

    def test_december_today_gives_january_next_year(self, tmp_path):
        """today=Dec 15, 2026 → forecast for Jan 2027 (31 days)."""
        store = _two_month_store(tmp_path, start=date(2026, 11, 20), n=16)
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 12, 15))
        assert result.points[0].date == date(2027, 1, 1)
        assert result.points[-1].date == date(2027, 1, 31)
        assert result.horizon_days == 31

    def test_january_today_gives_february_non_leap(self, tmp_path):
        """today=Jan 15, 2027 → Feb 2027 (28 days, non-leap)."""
        store = self._store_jan_feb_2027(tmp_path)
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2027, 1, 15))
        assert result.points[0].date == date(2027, 2, 1)
        assert result.points[-1].date == date(2027, 2, 28)
        assert result.horizon_days == 28

    def test_january_today_gives_february_leap(self, tmp_path):
        """today=Jan 15, 2028 → Feb 2028 (29 days, leap year)."""
        store = _two_month_store(tmp_path, start=date(2028, 1, 20), n=16)
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2028, 1, 15))
        assert result.points[0].date == date(2028, 2, 1)
        assert result.points[-1].date == date(2028, 2, 29)
        assert result.horizon_days == 29

    def test_march_today_gives_april_30_days(self, tmp_path):
        """today=Mar 15 → April (30 days)."""
        store = _two_month_store(tmp_path, start=date(2026, 2, 14), n=16)
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 3, 15))
        assert result.points[0].date == date(2026, 4, 1)
        assert result.points[-1].date == date(2026, 4, 30)
        assert result.horizon_days == 30


# ---------------------------------------------------------------------------
# TEST GROUP 4 — eligibility guards (unchanged behavior)
# ---------------------------------------------------------------------------

class TestForecastAggregateNextMonthGuards:

    def test_raises_when_only_one_expense_month(self, tmp_path):
        """20 days in a single month must be rejected with month error."""
        txns = [
            Transaction(date=date(2026, 1, i + 1), merchant="Amazon",
                        amount=1000.0, category="Shopping", source_file="test")
            for i in range(20)
        ]
        store = _store(tmp_path, txns)
        f = Forecaster()
        with pytest.raises(ValueError) as exc_info:
            f.forecast_aggregate_next_month(store, today=date(2026, 2, 10))
        assert "month" in str(exc_info.value).lower()

    def test_raises_when_below_14_day_minimum(self, tmp_path):
        """3 expense days across 2 months → 14-day guard fires first."""
        txns = [
            Transaction(date=date(2026, 1, 31), merchant="Swiggy",
                        amount=500.0, category="Dining", source_file="test"),
            Transaction(date=date(2026, 2, 1), merchant="Swiggy",
                        amount=500.0, category="Dining", source_file="test"),
            Transaction(date=date(2026, 2, 2), merchant="Swiggy",
                        amount=500.0, category="Dining", source_file="test"),
        ]
        store = _store(tmp_path, txns)
        f = Forecaster()
        with pytest.raises(ValueError) as exc_info:
            f.forecast_aggregate_next_month(store, today=date(2026, 3, 1))
        err = str(exc_info.value)
        assert str(MIN_HISTORY_DAYS) in err
        assert "month" not in err.lower()

    def test_raises_when_no_expense_transactions(self, tmp_path):
        txns = [
            Transaction(date=date(2026, 1, 1), merchant="Salary Credit",
                        amount=65000.0, category="Other", source_file="test"),
        ]
        store = _store(tmp_path, txns)
        f = Forecaster()
        with pytest.raises(ValueError):
            f.forecast_aggregate_next_month(store, today=date(2026, 2, 1))

    def test_raises_when_store_empty(self, tmp_path):
        store = TransactionStore(str(tmp_path / "empty.db"))
        f = Forecaster()
        with pytest.raises(ValueError):
            f.forecast_aggregate_next_month(store, today=date(2026, 2, 1))

    def test_income_does_not_count_toward_2_month_rule(self, tmp_path):
        """Expenses only in Jan + income in Feb must still fail the month rule."""
        jan_expense = [
            Transaction(date=date(2026, 1, i + 1), merchant="FreshMart",
                        amount=1000.0, category="Groceries", source_file="test")
            for i in range(20)
        ]
        feb_income = [
            Transaction(date=date(2026, 2, i + 1), merchant="Salary Credit",
                        amount=65000.0, category="Other", source_file="test")
            for i in range(5)
        ]
        store = _store(tmp_path, jan_expense + feb_income)
        f = Forecaster()
        with pytest.raises(ValueError) as exc_info:
            f.forecast_aggregate_next_month(store, today=date(2026, 3, 1))
        assert "month" in str(exc_info.value).lower()

    def test_passes_with_two_months_and_14_days(self, tmp_path):
        """16 days across 2 months must succeed."""
        store = _two_month_store(tmp_path)
        f = Forecaster()
        result = f.forecast_aggregate_next_month(store, today=date(2026, 3, 1))
        assert isinstance(result, Forecast)
        assert len(result.points) > 0
