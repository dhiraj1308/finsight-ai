"""Integration test: forecast round-trip through the real HTTP API.

Exercises:

  real TransactionStore (isolated SQLite)
  → real Forecaster.forecast_category()
  → ForecastDTO + nested ForecastPointDTO
  → FastAPI/Pydantic JSON serialization

  GET /forecast/{category}?days=N
  → HTTP 200 with validated response shape

  Error paths:
  GET /forecast/NonExistentCategory → real ValueError → HTTP 422
  GET /forecast/Groceries?days=0   → FastAPI ge=1 validation → HTTP 422
  GET /forecast/Groceries?days=366 → FastAPI le=365 validation → HTTP 422
  insufficient history             → real ValueError → HTTP 422

Components that are REAL:
  - TransactionStore — SQLite database under pytest's tmp_path (isolated)
  - Forecaster       — EWMA with linear trend (pure numpy, no ML)
  - FastAPI handler  — HTTP param binding, Pydantic serialization

Components that are MOCKED:
  - VectorStore  — prevent SentenceTransformer/torch loading
  - Categorizer  — untrained (_is_trained=False)
  - AnomalyDetector — not involved in forecast requests

Dataset:
  SyntheticGenerator().generate(n=500, seed=42) with default date range
  2024-01-01 → 2024-12-31.  This is the same dataset used by
  tests/unit/test_forecaster.py and is proven to produce well over 14
  distinct calendar days for every category including "Groceries".

  For the insufficient-history test a hand-crafted store is used containing
  exactly 3 transactions on 3 distinct days for a custom category, identical
  to the approach used by test_forecaster.py::test_insufficient_history_raises_value_error.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# No-op lifespan — skips heavy ML/env-var startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Client factory helpers
# ---------------------------------------------------------------------------

def _make_client(store) -> TestClient:
    """Return a TestClient with a real store + real Forecaster, mocked rest."""
    from api.app import app
    from api.dependencies import AppComponents
    from categorization.categorizer import Categorizer
    from forecasting.forecaster import Forecaster

    components = AppComponents(
        store=store,
        vector_store=_mock_vector_store(),
        categorizer=_untrained_categorizer(),
        anomaly_detector=MagicMock(),
        forecaster=Forecaster(),      # REAL forecaster
    )

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        app.state.components = components
        app.state.agent = MagicMock()
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


def _seeded_store(tmp_path: Path):
    """
    Create a real TransactionStore populated with SyntheticGenerator data.

    n=500, seed=42, default date range 2024-01-01 → 2024-12-31.
    This mirrors exactly what test_forecaster.py uses and is proven to
    provide ≥ 14 distinct calendar days per category.
    """
    from ingestion.transaction_store import TransactionStore
    from ingestion.synthetic_generator import SyntheticGenerator

    store = TransactionStore(str(tmp_path / "forecast_test.db"))
    gen = SyntheticGenerator()
    txns = gen.generate(n=500, seed=42)
    for t in txns:
        t.source_file = "synthetic"
    store.insert(txns)
    return store


# ---------------------------------------------------------------------------
# TEST 1 — Successful forecast: correct shape, types, and values
# ---------------------------------------------------------------------------

class TestForecastSuccess:

    @pytest.fixture()
    def client(self, tmp_path):
        store = _seeded_store(tmp_path)
        return _make_client(store)

    def test_successful_forecast_returns_200(self, client):
        """GET /forecast/Groceries?days=7 must return HTTP 200."""
        response = client.get("/forecast/Groceries", params={"days": 7})
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_forecast_category_matches_request(self, client):
        """response['category'] must equal the URL path segment."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        assert body["category"] == "Groceries", (
            f"Expected category='Groceries', got {body['category']!r}"
        )

    def test_forecast_horizon_days_matches_request(self, client):
        """response['horizon_days'] must equal the days query param."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        assert body["horizon_days"] == 7, (
            f"Expected horizon_days=7, got {body['horizon_days']!r}"
        )

    def test_forecast_points_count_matches_horizon(self, client):
        """len(response['points']) must equal horizon_days."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        assert "points" in body, "Response must contain 'points'"
        assert isinstance(body["points"], list)
        assert len(body["points"]) == 7, (
            f"Expected 7 forecast points, got {len(body['points'])}"
        )

    def test_every_point_has_required_fields(self, client):
        """Each ForecastPointDTO must contain date, yhat, yhat_lower, yhat_upper."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        required = {"date", "yhat", "yhat_lower", "yhat_upper"}
        for i, point in enumerate(body["points"]):
            missing = required - point.keys()
            assert not missing, (
                f"Point {i} missing fields: {missing}"
            )

    def test_every_point_date_is_iso_string(self, client):
        """date field in each point must be a valid ISO-format string."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        for i, point in enumerate(body["points"]):
            try:
                date.fromisoformat(point["date"])
            except (ValueError, TypeError) as exc:
                pytest.fail(
                    f"Point {i} date={point['date']!r} is not a valid ISO date: {exc}"
                )

    def test_every_point_numeric_fields_are_float(self, client):
        """yhat, yhat_lower, yhat_upper must be numeric."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        for i, point in enumerate(body["points"]):
            for field in ("yhat", "yhat_lower", "yhat_upper"):
                assert isinstance(point[field], (int, float)), (
                    f"Point {i} {field}={point[field]!r} is not numeric"
                )

    def test_every_point_yhat_is_non_negative(self, client):
        """yhat and yhat_lower must be >= 0 (floored in Forecaster)."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        for i, point in enumerate(body["points"]):
            assert point["yhat"] >= 0.0, (
                f"Point {i} yhat={point['yhat']} is negative"
            )
            assert point["yhat_lower"] >= 0.0, (
                f"Point {i} yhat_lower={point['yhat_lower']} is negative"
            )

    def test_every_point_confidence_interval_is_valid(self, client):
        """yhat_lower <= yhat <= yhat_upper for every point."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        for i, point in enumerate(body["points"]):
            assert point["yhat_lower"] <= point["yhat"], (
                f"Point {i}: yhat_lower={point['yhat_lower']} > yhat={point['yhat']}"
            )
            assert point["yhat"] <= point["yhat_upper"], (
                f"Point {i}: yhat={point['yhat']} > yhat_upper={point['yhat_upper']}"
            )

    def test_forecast_dates_are_consecutive_days(self, client):
        """Forecast dates must be consecutive one-day increments."""
        body = client.get("/forecast/Groceries", params={"days": 7}).json()
        dates = [date.fromisoformat(p["date"]) for p in body["points"]]
        for i in range(1, len(dates)):
            expected = dates[i - 1] + timedelta(days=1)
            assert dates[i] == expected, (
                f"Gap at index {i}: {dates[i - 1]} → {dates[i]} "
                f"(expected {expected})"
            )


# ---------------------------------------------------------------------------
# TEST 2 — Nonexistent category → real ValueError → HTTP 422
# ---------------------------------------------------------------------------

class TestForecastNonExistentCategory:

    @pytest.fixture()
    def client(self, tmp_path):
        store = _seeded_store(tmp_path)
        return _make_client(store)

    def test_nonexistent_category_returns_422(self, client):
        """GET /forecast/NonExistentCategory must return HTTP 422."""
        response = client.get("/forecast/NonExistentCategory", params={"days": 30})
        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )

    def test_nonexistent_category_detail_contains_category_name(self, client):
        """The 422 detail must mention the unknown category name."""
        body = client.get("/forecast/NonExistentCategory", params={"days": 30}).json()
        assert "detail" in body, "422 response must contain 'detail'"
        assert "NonExistentCategory" in body["detail"], (
            f"Expected 'NonExistentCategory' in detail, got: {body['detail']!r}"
        )


# ---------------------------------------------------------------------------
# TEST 3 — Invalid horizon parameter → FastAPI validation → HTTP 422
# ---------------------------------------------------------------------------

class TestForecastInvalidHorizon:

    @pytest.fixture()
    def client(self, tmp_path):
        store = _seeded_store(tmp_path)
        return _make_client(store)

    def test_horizon_zero_returns_422(self, client):
        """days=0 is below the ge=1 constraint → FastAPI must return 422."""
        response = client.get("/forecast/Groceries", params={"days": 0})
        assert response.status_code == 422, (
            f"Expected 422 for days=0, got {response.status_code}"
        )

    def test_horizon_366_returns_422(self, client):
        """days=366 exceeds the le=365 constraint → FastAPI must return 422."""
        response = client.get("/forecast/Groceries", params={"days": 366})
        assert response.status_code == 422, (
            f"Expected 422 for days=366, got {response.status_code}"
        )


# ---------------------------------------------------------------------------
# TEST 4 — Insufficient history → real ValueError → HTTP 422
# ---------------------------------------------------------------------------

class TestForecastInsufficientHistory:
    """
    Store has only 3 distinct calendar days for 'ThinHistory'.
    Forecaster requires MIN_HISTORY_DAYS=14 → raises ValueError.
    Handler must convert that to HTTP 422.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        from ingestion.transaction_store import TransactionStore
        from domain import Transaction

        store = TransactionStore(str(tmp_path / "thin.db"))
        # Insert exactly 3 transactions on 3 distinct days — far below MIN_HISTORY_DAYS=14
        for i in range(3):
            store.insert([Transaction(
                date=date(2024, 1, i + 1),
                merchant="Test Merchant",
                amount=50.0 + i,
                category="ThinHistory",
                source_file="test",
            )])
        return _make_client(store)

    def test_insufficient_history_returns_422(self, client):
        """GET /forecast/ThinHistory with < 14 days of data must return HTTP 422."""
        response = client.get("/forecast/ThinHistory", params={"days": 30})
        assert response.status_code == 422, (
            f"Expected 422 for insufficient history, "
            f"got {response.status_code}: {response.text}"
        )

    def test_insufficient_history_detail_mentions_minimum(self, client):
        """The 422 detail must indicate how many days are required."""
        from forecasting.forecaster import MIN_HISTORY_DAYS
        body = client.get("/forecast/ThinHistory", params={"days": 30}).json()
        assert "detail" in body
        assert str(MIN_HISTORY_DAYS) in body["detail"], (
            f"Expected MIN_HISTORY_DAYS ({MIN_HISTORY_DAYS}) in detail, "
            f"got: {body['detail']!r}"
        )
