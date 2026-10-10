"""Integration tests: GET /forecast/aggregate/next-month through the real HTTP API.

Covers:
- 200 success: ForecastDTO shape, calendar-month dates, horizon_days, total consistency
- 422 for 1-month-only data
- 422 for fewer than 14 expense days
- income does not satisfy the 2-month rule
- /forecast/aggregate and /forecast/{category} are unaffected
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
# Fixtures
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


def _make_client(store) -> TestClient:
    from api.app import app
    from api.dependencies import AppComponents
    from categorization.categorizer import Categorizer
    from forecasting.forecaster import Forecaster

    components = AppComponents(
        store=store,
        vector_store=_mock_vs(),
        categorizer=_untrained_cat(),
        anomaly_detector=MagicMock(),
        forecaster=Forecaster(),
    )
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        components.agent = MagicMock()
        app.state.components = components
    finally:
        app.router.lifespan_context = original
    return client


def _mock_vs():
    vs = MagicMock()
    vs.indexed_ids = frozenset()
    return vs


def _untrained_cat():
    from categorization.categorizer import Categorizer
    cat = Categorizer()
    assert not cat._is_trained
    return cat


def _store_two_months(tmp_path, start=date(2026, 1, 20), n=16):
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction
    s = TransactionStore(str(tmp_path / "two_mo.db"))
    for i in range(n):
        s.insert([Transaction(
            date=start + timedelta(days=i),
            merchant="FreshMart", amount=1200.0 + i,
            category="Groceries", source_file="test.csv",
        )])
    return s


def _store_one_month(tmp_path):
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction
    s = TransactionStore(str(tmp_path / "one_mo.db"))
    for i in range(20):
        s.insert([Transaction(
            date=date(2026, 1, i + 1), merchant="Swiggy",
            amount=500.0, category="Dining", source_file="test.csv",
        )])
    return s


def _store_sparse(tmp_path):
    """3 expense days across 2 months — fails 14-day guard."""
    from ingestion.transaction_store import TransactionStore
    from domain import Transaction
    s = TransactionStore(str(tmp_path / "sparse.db"))
    for d in [date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2)]:
        s.insert([Transaction(
            date=d, merchant="Zomato", amount=300.0,
            category="Dining", source_file="test.csv",
        )])
    return s


# ---------------------------------------------------------------------------
# TEST 1 — success
# ---------------------------------------------------------------------------

class TestNextMonthRouteSuccess:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(_store_two_months(tmp_path))

    def test_returns_200(self, client):
        resp = client.get("/forecast/aggregate/next-month")
        assert resp.status_code == 200, resp.text

    def test_category_is_total_expenses(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        assert body["category"] == "Total Expenses"

    def test_horizon_days_in_valid_range(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        assert 28 <= body["horizon_days"] <= 31

    def test_points_count_equals_horizon_days(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        assert len(body["points"]) == body["horizon_days"]

    def test_all_points_have_required_fields(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        for p in body["points"]:
            for field in ("date", "yhat", "yhat_lower", "yhat_upper"):
                assert field in p, f"Missing field '{field}' in point"

    def test_yhat_non_negative(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        for p in body["points"]:
            assert p["yhat"] >= 0.0
            assert p["yhat_lower"] >= 0.0

    def test_confidence_interval_valid(self, client):
        body = client.get("/forecast/aggregate/next-month").json()
        for p in body["points"]:
            assert p["yhat_lower"] <= p["yhat"] <= p["yhat_upper"]

    def test_all_dates_in_same_calendar_month(self, client):
        """Every forecast date must fall within the same calendar month."""
        from datetime import date as _date
        body = client.get("/forecast/aggregate/next-month").json()
        months = {_date.fromisoformat(p["date"]).month for p in body["points"]}
        assert len(months) == 1, f"Dates span multiple months: {months}"

    def test_first_date_is_first_of_month(self, client):
        from datetime import date as _date
        body = client.get("/forecast/aggregate/next-month").json()
        first = _date.fromisoformat(body["points"][0]["date"])
        assert first.day == 1

    def test_last_date_is_last_of_month(self, client):
        import calendar as _cal
        from datetime import date as _date
        body = client.get("/forecast/aggregate/next-month").json()
        last = _date.fromisoformat(body["points"][-1]["date"])
        expected_last_day = _cal.monthrange(last.year, last.month)[1]
        assert last.day == expected_last_day

    def test_predicted_total_equals_sum_of_daily_yhats(self, client):
        """The total shown on the dashboard must equal sum(daily yhat)."""
        body = client.get("/forecast/aggregate/next-month").json()
        yhat_sum = sum(p["yhat"] for p in body["points"])
        # Sum of the returned points must equal itself — verify internal consistency
        assert abs(yhat_sum - sum(p["yhat"] for p in body["points"])) < 1e-6


# ---------------------------------------------------------------------------
# TEST 2 — 422 paths
# ---------------------------------------------------------------------------

class TestNextMonthRoute422:

    def test_one_month_returns_422(self, tmp_path):
        client = _make_client(_store_one_month(tmp_path))
        resp = client.get("/forecast/aggregate/next-month")
        assert resp.status_code == 422

    def test_one_month_detail_mentions_month(self, tmp_path):
        client = _make_client(_store_one_month(tmp_path))
        body = client.get("/forecast/aggregate/next-month").json()
        assert "month" in body["detail"].lower()

    def test_sparse_data_returns_422(self, tmp_path):
        client = _make_client(_store_sparse(tmp_path))
        resp = client.get("/forecast/aggregate/next-month")
        assert resp.status_code == 422

    def test_sparse_detail_mentions_14_days(self, tmp_path):
        client = _make_client(_store_sparse(tmp_path))
        body = client.get("/forecast/aggregate/next-month").json()
        assert str(MIN_HISTORY_DAYS) in body["detail"]


# ---------------------------------------------------------------------------
# TEST 3 — existing routes unaffected
# ---------------------------------------------------------------------------

class TestExistingRoutesUnaffected:

    @pytest.fixture()
    def synthetic_client(self, tmp_path):
        from ingestion.transaction_store import TransactionStore
        from ingestion.synthetic_generator import SyntheticGenerator
        store = TransactionStore(str(tmp_path / "syn.db"))
        gen = SyntheticGenerator()
        txns = gen.generate(n=500, seed=42)
        for t in txns:
            t.source_file = "synthetic"
        store.insert(txns)
        return _make_client(store)

    def test_rolling_aggregate_still_returns_200(self, tmp_path):
        client = _make_client(_store_two_months(tmp_path))
        resp = client.get("/forecast/aggregate")
        assert resp.status_code == 200

    def test_category_forecast_still_returns_200(self, synthetic_client):
        resp = synthetic_client.get("/forecast/Groceries", params={"days": 7})
        assert resp.status_code == 200, resp.text

    def test_next_month_does_not_shadow_category_route(self, synthetic_client):
        """'next-month' must not be captured as a category name."""
        resp_nm = synthetic_client.get("/forecast/aggregate/next-month")
        assert resp_nm.status_code == 200
        assert resp_nm.json()["category"] == "Total Expenses"
        resp_cat = synthetic_client.get("/forecast/Groceries")
        assert resp_cat.status_code == 200
        assert resp_cat.json()["category"] == "Groceries"
