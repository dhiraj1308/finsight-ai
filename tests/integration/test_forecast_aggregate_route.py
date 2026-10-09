"""Integration tests: GET /forecast/aggregate through the real HTTP API.

Exercises:
  - real TransactionStore (isolated SQLite)
  - real Forecaster.forecast_aggregate()
  - ForecastDTO serialization
  - sufficient-history success path
  - insufficient-history 422 path
  - /forecast/{category} unchanged after route addition
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

from forecasting.forecaster import MIN_HISTORY_DAYS


# ---------------------------------------------------------------------------
# No-op lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Client factory helpers
# ---------------------------------------------------------------------------

def _make_client(store) -> TestClient:
    from api.app import app
    from api.dependencies import AppComponents
    from categorization.categorizer import Categorizer
    from forecasting.forecaster import Forecaster

    components = AppComponents(
        store=store,
        vector_store=_mock_vector_store(),
        categorizer=_untrained_categorizer(),
        anomaly_detector=MagicMock(),
        forecaster=Forecaster(),
    )

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        components.agent = MagicMock()
        app.state.components = components
    finally:
        app.router.lifespan_context = original_lifespan

    return client


def _mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.indexed_ids = frozenset()
    return vs


def _untrained_categorizer():
    from categorization.categorizer import Categorizer
    cat = Categorizer()
    assert not cat._is_trained
    return cat


def _store_with_expenses(tmp_path: Path, n_days: int = MIN_HISTORY_DAYS + 5) -> object:
    """Create a TransactionStore with n_days distinct expense days.

    Default start date (Jan 20) ensures the default 19-day range spans
    Jan 20 – Feb 7, covering 2 calendar months (satisfying the 2-month rule).
    """
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "agg_test.db"))
    for i in range(n_days):
        store.insert([Transaction(
            date=date(2026, 1, 20) + timedelta(days=i),
            merchant="Swiggy",
            amount=500.0 + i,
            category="Dining",
            source_file="test.csv",
        )])
    return store


def _store_with_mixed(tmp_path: Path) -> object:
    """Store with both income and expense rows; only expenses count toward history.

    Expenses span Feb 1 – Feb (MIN_HISTORY_DAYS+3) in year 2026 and
    Jan 20–31 in 2026 — two distinct months — so both guards pass.
    """
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "mixed.db"))
    # Income rows in January (must be excluded from expense month count)
    for i in range(5):
        store.insert([Transaction(
            date=date(2026, 1, i + 1),
            merchant="Salary Credit",
            amount=65000.0,
            category="Other",
            source_file="test.csv",
        )])
    # Expense rows spanning Jan 20 – Feb (MIN_HISTORY_DAYS+3) → 2 months
    for i in range(MIN_HISTORY_DAYS + 2):
        store.insert([Transaction(
            date=date(2026, 1, 20) + timedelta(days=i),
            merchant="FreshMart",
            amount=1200.0 + i,
            category="Groceries",
            source_file="test.csv",
        )])
    return store


def _store_too_few_days(tmp_path: Path) -> object:
    """Store with fewer expense days than MIN_HISTORY_DAYS."""
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "thin.db"))
    for i in range(MIN_HISTORY_DAYS - 2):
        store.insert([Transaction(
            date=date(2026, 3, i + 1),
            merchant="Zomato",
            amount=300.0,
            category="Dining",
            source_file="test.csv",
        )])
    return store


def _store_one_month_only(tmp_path: Path) -> object:
    """Store with 20 expense days all in one calendar month (January 2026).

    Passes the 14-day guard (20 >= 14) but fails the 2-month rule.
    """
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "one_month.db"))
    for i in range(20):
        store.insert([Transaction(
            date=date(2026, 1, i + 1),
            merchant="Swiggy",
            amount=500.0 + i,
            category="Dining",
            source_file="test.csv",
        )])
    return store


def _store_two_months(tmp_path: Path) -> object:
    """Store with 16 expense days spanning January + February 2026.

    Passes both the 14-day guard (16 >= 14) and the 2-month rule (Jan + Feb).
    """
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "two_months.db"))
    for i in range(16):
        store.insert([Transaction(
            date=date(2026, 1, 20) + timedelta(days=i),
            merchant="FreshMart",
            amount=1200.0 + i,
            category="Groceries",
            source_file="test.csv",
        )])
    return store


# ---------------------------------------------------------------------------
# TEST 1 — successful aggregate forecast
# ---------------------------------------------------------------------------

class TestForecastAggregateSuccess:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(_store_with_expenses(tmp_path))

    def test_returns_200(self, client):
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 200, resp.text

    def test_category_is_total_expenses(self, client):
        body = client.get("/forecast/aggregate").json()
        assert body["category"] == "Total Expenses"

    def test_default_horizon_is_30(self, client):
        body = client.get("/forecast/aggregate").json()
        assert body["horizon_days"] == 30

    def test_custom_horizon(self, client):
        body = client.get("/forecast/aggregate", params={"days": 7}).json()
        assert body["horizon_days"] == 7
        assert len(body["points"]) == 7

    def test_points_count_matches_horizon(self, client):
        body = client.get("/forecast/aggregate", params={"days": 14}).json()
        assert len(body["points"]) == 14

    def test_all_points_have_required_fields(self, client):
        body = client.get("/forecast/aggregate").json()
        for p in body["points"]:
            assert "date" in p
            assert "yhat" in p
            assert "yhat_lower" in p
            assert "yhat_upper" in p

    def test_yhat_non_negative(self, client):
        body = client.get("/forecast/aggregate").json()
        for p in body["points"]:
            assert p["yhat"] >= 0.0
            assert p["yhat_lower"] >= 0.0

    def test_confidence_interval_valid(self, client):
        body = client.get("/forecast/aggregate").json()
        for p in body["points"]:
            assert p["yhat_lower"] <= p["yhat"] <= p["yhat_upper"]


# ---------------------------------------------------------------------------
# TEST 2 — income excluded from aggregate
# ---------------------------------------------------------------------------

class TestForecastAggregateIncomeExcluded:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(_store_with_mixed(tmp_path))

    def test_returns_200_with_mixed_data(self, client):
        """Income rows don't block the route — expense days are sufficient."""
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 200, resp.text

    def test_forecast_total_excludes_income_magnitude(self, client):
        """30-day total must be in the expense range, not inflated by salary."""
        body = client.get("/forecast/aggregate").json()
        total = sum(p["yhat"] for p in body["points"])
        # Expenses are ~1200-1216/day; income is 65000 per entry.
        # If income were included, daily totals would be >> 10000.
        assert total < 200_000, (
            f"Forecast total {total:.2f} looks like income was included"
        )


# ---------------------------------------------------------------------------
# TEST 3 — insufficient history returns 422
# ---------------------------------------------------------------------------

class TestForecastAggregateInsufficientHistory:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(_store_too_few_days(tmp_path))

    def test_returns_422_on_insufficient_history(self, client):
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 422, resp.text

    def test_422_detail_mentions_min_history(self, client):
        body = client.get("/forecast/aggregate").json()
        assert "detail" in body
        assert str(MIN_HISTORY_DAYS) in body["detail"]


# ---------------------------------------------------------------------------
# TEST 4 — existing /forecast/{category} unaffected
# ---------------------------------------------------------------------------

class TestCategoryForecastUnchanged:
    """The original per-category route must continue to work after the new route."""

    @pytest.fixture()
    def client(self, tmp_path):
        # Use the seeded synthetic store (same pattern as test_forecast_round_trip)
        from ingestion.transaction_store import TransactionStore
        from ingestion.synthetic_generator import SyntheticGenerator
        store = TransactionStore(str(tmp_path / "cat_test.db"))
        gen = SyntheticGenerator()
        txns = gen.generate(n=500, seed=42)
        for t in txns:
            t.source_file = "synthetic"
        store.insert(txns)
        return _make_client(store)

    def test_category_route_still_returns_200(self, client):
        resp = client.get("/forecast/Groceries", params={"days": 7})
        assert resp.status_code == 200, resp.text

    def test_aggregate_does_not_shadow_category_route(self, client):
        """'aggregate' must not be treated as a category name."""
        # The aggregate route must respond correctly
        resp_agg = client.get("/forecast/aggregate")
        assert resp_agg.status_code == 200
        assert resp_agg.json()["category"] == "Total Expenses"

        # The Groceries category route must still respond correctly
        resp_cat = client.get("/forecast/Groceries")
        assert resp_cat.status_code == 200
        assert resp_cat.json()["category"] == "Groceries"


# ---------------------------------------------------------------------------
# TEST 5 — invalid horizon values
# ---------------------------------------------------------------------------

class TestForecastAggregateHorizonValidation:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(_store_with_expenses(tmp_path))

    def test_horizon_zero_returns_422(self, client):
        resp = client.get("/forecast/aggregate", params={"days": 0})
        assert resp.status_code == 422

    def test_horizon_366_returns_422(self, client):
        resp = client.get("/forecast/aggregate", params={"days": 366})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TEST 6 — 2-month minimum history rule via HTTP
# ---------------------------------------------------------------------------

class TestForecastAggregateTwoMonthRule:
    """Verify the 2-month product rule is enforced at the API level.

    These tests exercise the exact scenario the product requirement describes:
    a user uploads one month of statements and must see a 422, while a user
    with two months gets a 200.
    """

    def test_one_month_14_plus_days_returns_422(self, tmp_path):
        """20 expense days all in January → 14-day guard passes, 2-month guard fails → 422."""
        client = _make_client(_store_one_month_only(tmp_path))
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 422, (
            f"Expected 422 for single-month data, got {resp.status_code}: {resp.text}"
        )

    def test_one_month_422_detail_mentions_month(self, tmp_path):
        """The 422 detail for single-month data must mention 'month'."""
        client = _make_client(_store_one_month_only(tmp_path))
        body = client.get("/forecast/aggregate").json()
        assert "detail" in body
        assert "month" in body["detail"].lower(), (
            f"Expected 'month' in detail, got: {body['detail']!r}"
        )

    def test_two_months_with_enough_days_returns_200(self, tmp_path):
        """16 days spanning Jan + Feb → both guards pass → 200."""
        client = _make_client(_store_two_months(tmp_path))
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 200, (
            f"Expected 200 for 2-month data, got {resp.status_code}: {resp.text}"
        )

    def test_two_months_response_is_valid_forecast(self, tmp_path):
        """The 2-month success response must contain valid ForecastDTO data."""
        client = _make_client(_store_two_months(tmp_path))
        body = client.get("/forecast/aggregate").json()
        assert body["category"] == "Total Expenses"
        assert body["horizon_days"] == 30
        assert len(body["points"]) == 30

    def test_income_in_second_month_does_not_satisfy_rule(self, tmp_path):
        """Salary Credit in Feb must not count as a second expense month."""
        from ingestion.transaction_store import TransactionStore
        from domain import Transaction

        store = TransactionStore(str(tmp_path / "income_feb.db"))
        # 20 Jan expense days
        for i in range(20):
            store.insert([Transaction(
                date=date(2026, 1, i + 1), merchant="Amazon",
                amount=1000.0, category="Shopping", source_file="test.csv",
            )])
        # 5 Feb income rows
        for i in range(5):
            store.insert([Transaction(
                date=date(2026, 2, i + 1), merchant="Salary Credit",
                amount=65000.0, category="Other", source_file="test.csv",
            )])

        client = _make_client(store)
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 422, (
            "Income in Feb must not satisfy the 2-month expense requirement"
        )
        body = resp.json()
        assert "month" in body["detail"].lower()
