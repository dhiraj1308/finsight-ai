"""Integration test: CSV ingest round-trip through the real HTTP API.

This test exercises the full ingest pipeline with REAL components:

  POST /ingest  → real CSVParser  → real TransactionStore (isolated SQLite)
                                                ↓
  GET  /transactions  ← real TransactionStore ←┘

It also verifies real duplicate detection by uploading the same file twice.

Components that are REAL (not mocked):
  - CSVParser        — parses the uploaded bytes on disk, no ML dependencies
  - TransactionStore — SQLite database under pytest's tmp_path (fully isolated)
  - FastAPI handler  — HTTP request/response, Pydantic serialization

Components that are MOCKED (to avoid heavy ML/model loading):
  - VectorStore      — embedding is not the subject of this test; mock returns
                       an empty indexed_ids frozenset and silently accepts
                       index() calls so the handler never tries to load
                       sentence-transformers/torch
  - Categorizer      — _is_trained=False; predict_batch() is never called,
                       no joblib model needed
  - AnomalyDetector  — fit_and_score() is never called because the CSV has
                       fewer than 10 transactions (MIN_TRANSACTIONS = 10)
  - Forecaster       — not involved in ingestion
"""
from __future__ import annotations

import csv
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable regardless of how pytest is invoked
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test data: 4 deterministic transactions
# ---------------------------------------------------------------------------
_UPLOADED_FILENAME = "bank_march_2024.csv"

_ROWS = [
    ("2024-03-01", "Whole Foods",        "87.43",  "Groceries"),
    ("2024-03-05", "Netflix",            "15.99",  "Entertainment"),
    ("2024-03-10", "Shell Gas Station",  "62.00",  "Transport"),
    ("2024-03-15", "City Clinic",        "120.50", "Healthcare"),
]

N = len(_ROWS)   # 4 — safely below MIN_TRANSACTIONS=10 so anomaly detection is skipped


def _make_csv_bytes() -> bytes:
    """Return valid UTF-8 CSV bytes for the 4 test transactions."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "merchant", "amount", "category"])
    for row in _ROWS:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# No-op lifespan: skips the real startup that loads heavy ML models
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Build a TestClient backed by real SQLite + real CSVParser, mocked ML/vector
# ---------------------------------------------------------------------------

def _make_client_with_real_store(tmp_path: Path) -> TestClient:
    """
    Construct a TestClient whose AppComponents contains:
      - real TransactionStore at tmp_path/test.db
      - mock VectorStore (indexed_ids=frozenset(), index() is a no-op)
      - untrained Categorizer (_is_trained=False)
      - mock AnomalyDetector (fit_and_score() never called here)
      - mock Forecaster
    """
    from api.app import app
    from api.dependencies import AppComponents
    from ingestion.transaction_store import TransactionStore
    from categorization.categorizer import Categorizer

    # Real isolated SQLite database
    real_store = TransactionStore(str(tmp_path / "test.db"))

    # Mock VectorStore: return an empty frozenset so the incremental-index
    # logic thinks nothing is indexed yet, and accept index() silently.
    mock_vector_store = MagicMock()
    mock_vector_store.indexed_ids = frozenset()

    # Untrained categorizer — predict_batch() is never invoked
    untrained_categorizer = Categorizer()
    assert not untrained_categorizer._is_trained  # sanity check

    # Mock anomaly detector and forecaster — not exercised for < 10 transactions
    mock_anomaly_detector = MagicMock()
    mock_forecaster = MagicMock()

    components = AppComponents(
        store=real_store,
        vector_store=mock_vector_store,
        categorizer=untrained_categorizer,
        anomaly_detector=mock_anomaly_detector,
        forecaster=mock_forecaster,
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
# The integration test
# ---------------------------------------------------------------------------

class TestCsvIngestRoundTrip:
    """Full round-trip: ingest → query → duplicate detection."""

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client_with_real_store(tmp_path)

    # ------------------------------------------------------------------
    # 1. POST /ingest — first upload
    # ------------------------------------------------------------------

    def test_first_ingest_returns_correct_counts(self, client):
        """POST /ingest with a valid 4-row CSV must return ingested=4, skipped=0."""
        response = client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()

        assert body["ingested"] == N, (
            f"Expected ingested={N}, got {body['ingested']}"
        )
        assert body["skipped"] == 0, (
            f"Expected skipped=0, got {body['skipped']}"
        )
        assert body["warnings"] == [], (
            f"Expected no warnings, got {body['warnings']}"
        )
        # anomaly detection skipped (< 10 rows) → None
        assert body["anomalies_detected"] is None
        # categorizer not trained → None
        assert body["needs_review_count"] is None

    # ------------------------------------------------------------------
    # 2. GET /transactions — verify stored records
    # ------------------------------------------------------------------

    def test_get_transactions_returns_all_ingested_rows(self, client):
        """After ingest, GET /transactions must return exactly the 4 stored rows."""
        # Ingest first
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        response = client.get("/transactions")

        assert response.status_code == 200, (
            f"Expected 200 from GET /transactions, got {response.status_code}"
        )
        txns = response.json()
        assert isinstance(txns, list)
        assert len(txns) == N, (
            f"Expected {N} transactions, got {len(txns)}"
        )

    def test_get_transactions_merchants_match(self, client):
        """Merchant names returned by GET /transactions must match the uploaded CSV."""
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()
        returned_merchants = {t["merchant"] for t in txns}
        expected_merchants = {row[1] for row in _ROWS}

        assert returned_merchants == expected_merchants, (
            f"Merchant mismatch.\n  Expected: {expected_merchants}\n  Got: {returned_merchants}"
        )

    def test_get_transactions_amounts_match(self, client):
        """Amounts returned by GET /transactions must match the uploaded CSV."""
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()
        returned_amounts = {round(t["amount"], 2) for t in txns}
        expected_amounts = {round(float(row[2]), 2) for row in _ROWS}

        assert returned_amounts == expected_amounts, (
            f"Amount mismatch.\n  Expected: {expected_amounts}\n  Got: {returned_amounts}"
        )

    def test_get_transactions_dates_match(self, client):
        """Dates returned by GET /transactions must match the uploaded CSV."""
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()
        returned_dates = {t["date"] for t in txns}
        expected_dates = {row[0] for row in _ROWS}

        assert returned_dates == expected_dates, (
            f"Date mismatch.\n  Expected: {expected_dates}\n  Got: {returned_dates}"
        )

    def test_get_transactions_source_file_is_original_filename(self, client):
        """
        source_file in every returned transaction must be the original
        uploaded filename, not the safe temp basename.
        """
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()

        for txn in txns:
            assert "source_file" in txn, "source_file field missing from TransactionDTO"
            assert txn["source_file"] == _UPLOADED_FILENAME, (
                f"Expected source_file={_UPLOADED_FILENAME!r}, "
                f"got {txn['source_file']!r}"
            )

    def test_get_transactions_category_field_is_present(self, client):
        """
        Every transaction must have a category field.
        When the categorizer is not trained, the CSV-supplied category is stored
        (or empty string if the CSV had no category column).
        """
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()

        for txn in txns:
            assert "category" in txn, "category field missing from TransactionDTO"
            assert isinstance(txn["category"], str), (
                f"category must be a string, got {type(txn['category'])}"
            )

    def test_get_transactions_schema_fields_complete(self, client):
        """
        Each transaction in the GET /transactions response must contain all
        expected TransactionDTO fields.
        """
        client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        txns = client.get("/transactions").json()
        required_fields = {
            "id", "date", "merchant", "amount", "category",
            "is_anomaly", "anomaly_score", "needs_review", "source_file",
        }

        for txn in txns:
            missing = required_fields - txn.keys()
            assert not missing, (
                f"Transaction response is missing fields: {missing}\n  Got: {list(txn.keys())}"
            )

    # ------------------------------------------------------------------
    # 3. Second POST /ingest — duplicate detection via real SQLite
    # ------------------------------------------------------------------

    def test_duplicate_ingest_returns_zero_inserted(self, client):
        """
        Uploading the same CSV twice must result in ingested=0, skipped=N
        on the second upload — real SQLite unique-index enforcement.
        """
        csv_bytes = _make_csv_bytes()

        # First upload
        first = client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, csv_bytes, "text/csv")},
        )
        assert first.status_code == 200
        assert first.json()["ingested"] == N

        # Second upload — exact same content
        second = client.post(
            "/ingest",
            files={"file": (_UPLOADED_FILENAME, csv_bytes, "text/csv")},
        )

        assert second.status_code == 200, (
            f"Expected 200 on duplicate upload, got {second.status_code}"
        )
        body = second.json()
        assert body["ingested"] == 0, (
            f"Expected ingested=0 on duplicate, got {body['ingested']}"
        )
        assert body["skipped"] == N, (
            f"Expected skipped={N} on duplicate, got {body['skipped']}"
        )

    def test_duplicate_ingest_does_not_create_extra_transactions(self, client):
        """
        After two uploads of the same CSV, GET /transactions must still
        return exactly N transactions — no duplicates in the database.
        """
        csv_bytes = _make_csv_bytes()
        client.post("/ingest", files={"file": (_UPLOADED_FILENAME, csv_bytes, "text/csv")})
        client.post("/ingest", files={"file": (_UPLOADED_FILENAME, csv_bytes, "text/csv")})

        txns = client.get("/transactions").json()

        assert len(txns) == N, (
            f"Expected {N} transactions after duplicate upload, got {len(txns)} — "
            "duplicate transactions must not be stored"
        )
