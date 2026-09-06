"""Integration tests: GET /transactions query parameters through the real HTTP API.

Exercises the three filtering branches in the /transactions handler:

  1. start_date + end_date  → store.query_by_date_range()
  2. category               → store.query_by_category()
  3. neither / one-only     → store.get_all()

Also covers error paths:
  - invalid ISO date string  → date.fromisoformat() raises ValueError → HTTP 422
  - start_date > end_date    → TransactionStore raises ValueError    → HTTP 422

Components that are REAL:
  - TransactionStore — SQLite database under pytest's tmp_path (isolated)
  - FastAPI handler  — HTTP query-param binding and branching logic
  - date.fromisoformat() — called inside the handler body

Components that are MOCKED:
  - VectorStore     — no embedding model needed
  - Categorizer     — untrained
  - AnomalyDetector — not involved in GET /transactions
  - Forecaster      — not involved in GET /transactions

Dataset — 5 deterministic transactions inserted as real Transaction objects:

  T1  2024-01-10  Store A  100.0  Groceries
  T2  2024-02-15  Store B  200.0  Dining
  T3  2024-03-20  Store C  300.0  Groceries
  T4  2024-04-25  Store D  400.0  Transport
  T5  2024-05-30  Store E  500.0  Dining
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import date
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
# Deterministic test dataset
# ---------------------------------------------------------------------------
_ROWS = [
    # (date_str,    merchant,  amount,  category)
    ("2024-01-10", "Store A", 100.0, "Groceries"),
    ("2024-02-15", "Store B", 200.0, "Dining"),
    ("2024-03-20", "Store C", 300.0, "Groceries"),
    ("2024-04-25", "Store D", 400.0, "Transport"),
    ("2024-05-30", "Store E", 500.0, "Dining"),
]
_N = len(_ROWS)  # 5


# ---------------------------------------------------------------------------
# No-op lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path) -> TestClient:
    """Return a TestClient backed by a real isolated store, mocked ML."""
    from api.app import app
    from api.dependencies import AppComponents
    from ingestion.transaction_store import TransactionStore
    from categorization.categorizer import Categorizer
    from domain import Transaction

    store = TransactionStore(str(tmp_path / "txn_query_test.db"))
    for date_str, merchant, amount, category in _ROWS:
        store.insert([Transaction(
            date=date.fromisoformat(date_str),
            merchant=merchant,
            amount=amount,
            category=category,
            source_file="test.csv",
        )])

    mock_vs = MagicMock()
    mock_vs.indexed_ids = frozenset()

    cat = Categorizer()
    assert not cat._is_trained

    components = AppComponents(
        store=store,
        vector_store=mock_vs,
        categorizer=cat,
        anomaly_detector=MagicMock(),
        forecaster=MagicMock(),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTransactionsQueryParams:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    # ------------------------------------------------------------------
    # Test 1 — date-range filter
    # ------------------------------------------------------------------

    def test_date_range_returns_matching_transactions(self, client):
        """
        start_date=2024-02-01 & end_date=2024-04-30 must return T2, T3, T4.
        T1 (Jan 10) is before start; T5 (May 30) is after end.
        """
        response = client.get(
            "/transactions",
            params={"start_date": "2024-02-01", "end_date": "2024-04-30"},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        txns = response.json()
        assert len(txns) == 3, f"Expected 3, got {len(txns)}"

        merchants = {t["merchant"] for t in txns}
        assert merchants == {"Store B", "Store C", "Store D"}
        assert "Store A" not in merchants, "Store A (Jan 10) must not appear"
        assert "Store E" not in merchants, "Store E (May 30) must not appear"

    def test_date_range_results_ordered_descending(self, client):
        """TransactionStore returns ORDER BY date DESC — newest first."""
        txns = client.get(
            "/transactions",
            params={"start_date": "2024-02-01", "end_date": "2024-04-30"},
        ).json()
        assert len(txns) == 3
        dates = [t["date"] for t in txns]
        assert dates == sorted(dates, reverse=True), (
            f"Expected descending dates, got: {dates}"
        )

    # ------------------------------------------------------------------
    # Test 2 — boundary inclusivity
    # ------------------------------------------------------------------

    def test_date_range_includes_boundary_dates(self, client):
        """
        start_date == end_date == 2024-01-10 (exact boundary).
        The SQL uses >= and <= so the boundary date must be returned.
        """
        response = client.get(
            "/transactions",
            params={"start_date": "2024-01-10", "end_date": "2024-01-10"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == 1, f"Expected 1 (boundary inclusive), got {len(txns)}"
        assert txns[0]["merchant"] == "Store A"
        assert txns[0]["date"] == "2024-01-10"

    # ------------------------------------------------------------------
    # Test 3 — start_date only (no end_date) → get_all()
    # ------------------------------------------------------------------

    def test_start_date_only_returns_all_transactions(self, client):
        """
        GET /transactions?start_date=2024-03-01 (end_date absent).

        DOCUMENTS CURRENT HANDLER BEHAVIOR:
        The handler condition `if start_date and end_date:` is False when
        end_date is missing.  The category branch is also False.  The request
        falls through to store.get_all() and returns all 5 transactions —
        the start_date value is completely ignored.

        This test documents the behavior; the implementation is not changed.
        """
        response = client.get(
            "/transactions",
            params={"start_date": "2024-03-01"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == _N, (
            f"Expected all {_N} transactions when only start_date supplied "
            f"(handler falls through to get_all()), got {len(txns)}"
        )

    # ------------------------------------------------------------------
    # Test 4 — end_date only (no start_date) → get_all()
    # ------------------------------------------------------------------

    def test_end_date_only_returns_all_transactions(self, client):
        """
        GET /transactions?end_date=2024-03-01 (start_date absent).

        DOCUMENTS CURRENT HANDLER BEHAVIOR:
        Same fallthrough as the start_date-only case — all 5 transactions
        are returned via get_all(), regardless of the end_date value.

        This test documents the behavior; the implementation is not changed.
        """
        response = client.get(
            "/transactions",
            params={"end_date": "2024-03-01"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == _N, (
            f"Expected all {_N} transactions when only end_date supplied "
            f"(handler falls through to get_all()), got {len(txns)}"
        )

    # ------------------------------------------------------------------
    # Test 5 — category filter
    # ------------------------------------------------------------------

    def test_category_filter_returns_matching_transactions(self, client):
        """GET /transactions?category=Groceries must return T1 and T3."""
        response = client.get(
            "/transactions",
            params={"category": "Groceries"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == 2, f"Expected 2 Groceries, got {len(txns)}"
        assert all(t["category"] == "Groceries" for t in txns)
        assert {t["merchant"] for t in txns} == {"Store A", "Store C"}

    # ------------------------------------------------------------------
    # Test 6 — category filter is case-insensitive
    # ------------------------------------------------------------------

    def test_category_filter_case_insensitive_lowercase(self, client):
        """category=groceries (lowercase) must return the same 2 Groceries rows."""
        response = client.get(
            "/transactions",
            params={"category": "groceries"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == 2, f"Expected 2 for lowercase 'groceries', got {len(txns)}"
        assert {t["merchant"] for t in txns} == {"Store A", "Store C"}

    def test_category_filter_case_insensitive_uppercase(self, client):
        """category=GROCERIES (uppercase) must return the same 2 Groceries rows."""
        response = client.get(
            "/transactions",
            params={"category": "GROCERIES"},
        )

        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == 2, f"Expected 2 for uppercase 'GROCERIES', got {len(txns)}"
        assert {t["merchant"] for t in txns} == {"Store A", "Store C"}

    # ------------------------------------------------------------------
    # Test 7 — category with no matches
    # ------------------------------------------------------------------

    def test_category_filter_no_matches_returns_empty_list(self, client):
        """
        GET /transactions?category=NonExistentCategory must return HTTP 200
        with an empty list — not 404 or 422.
        """
        response = client.get(
            "/transactions",
            params={"category": "NonExistentCategory"},
        )

        assert response.status_code == 200
        assert response.json() == [], (
            f"Expected [] for unknown category, got {response.json()!r}"
        )

    # ------------------------------------------------------------------
    # Test 8 — invalid ISO date string → HTTP 422
    # ------------------------------------------------------------------

    def test_invalid_start_date_returns_422(self, client):
        """
        GET /transactions?start_date=not-a-date&end_date=2024-03-01

        date.fromisoformat("not-a-date") raises ValueError inside the handler.
        The handler's except ValueError block converts this to HTTP 422.
        This is handler-specific code — not reachable via store unit tests.
        """
        response = client.get(
            "/transactions",
            params={"start_date": "not-a-date", "end_date": "2024-03-01"},
        )

        assert response.status_code == 422, (
            f"Expected 422 for invalid date, got {response.status_code}"
        )
        body = response.json()
        assert "detail" in body, "422 response must include 'detail'"

    # ------------------------------------------------------------------
    # Test 9 — start_date > end_date → HTTP 422
    # ------------------------------------------------------------------

    def test_inverted_date_range_returns_422(self, client):
        """
        GET /transactions?start_date=2024-12-31&end_date=2024-01-01

        TransactionStore.query_by_date_range raises ValueError when
        start_date > end_date.  The handler's except ValueError block
        converts this to HTTP 422.
        """
        response = client.get(
            "/transactions",
            params={"start_date": "2024-12-31", "end_date": "2024-01-01"},
        )

        assert response.status_code == 422, (
            f"Expected 422 for inverted range, got {response.status_code}"
        )
        body = response.json()
        assert "detail" in body, "422 response must include 'detail'"
        # The ValueError message from TransactionStore includes the date values
        assert "2024-12-31" in body["detail"] or "start_date" in body["detail"], (
            f"Expected detail to mention the inverted start date, got: {body['detail']!r}"
        )
