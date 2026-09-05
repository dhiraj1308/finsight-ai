"""Integration test: anomaly detection round-trip through the real HTTP API.

Exercises the full anomaly workflow with REAL components:

  POST /ingest  → real CSVParser  → real TransactionStore (isolated SQLite)
                → real AnomalyDetector.fit_and_score() (IsolationForest)
                → SQLite anomaly flags/scores written back to rows
                                             ↓
  GET  /anomalies  ← real TransactionStore filtered by is_anomaly=True
                   ← _txn_to_dto() Pydantic serialization

Also exercises cross-endpoint consistency:
  GET /transactions vs GET /anomalies → anomaly IDs must be a subset of all IDs.

Components that are REAL (not mocked):
  - CSVParser        — parses the uploaded bytes from disk
  - TransactionStore — SQLite database under pytest's tmp_path (isolated)
  - AnomalyDetector  — real IsolationForest (contamination=0.05, random_state=42)
  - FastAPI handler  — HTTP request/response, Pydantic serialization

Components that are MOCKED:
  - VectorStore  — no SentenceTransformer/torch loading
  - Categorizer  — untrained (_is_trained=False), so predict_batch() is never called
  - Forecaster   — not involved in ingestion

Dataset design (deterministic, verified by direct IsolationForest probe):
  20 transactions — 9 normal Groceries (~45-55), 9 normal Dining (~25-33),
  1 moderate outlier Groceries (500.0), 1 extreme outlier Dining (2500.0).

  IsolationForest(contamination=0.05, random_state=42) on this dataset:
    - Flags exactly 1 anomaly: index 19, amount=2500.0, category="Dining"
    - Anomaly score ≈ 0.060 (> 0.0, within [0.0, 1.0])
    - floor(0.05 * 20) = 1  →  deterministic, verified against actual sklearn output

  MIN_TRANSACTIONS = 10, so the 20-row CSV guarantees fit_and_score() is called.
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
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Deterministic test dataset — 20 transactions, exactly 1 anomaly
#
# Verified empirically: IsolationForest(contamination=0.05, random_state=42)
# flags exactly row index 19 (amount=2500.0, Dining) as the single anomaly.
# The expected_anomaly_count is therefore exactly 1.
# ---------------------------------------------------------------------------
_FILENAME = "march_2024_statement.csv"
_EXPECTED_ANOMALY_COUNT = 1  # floor(0.05 * 20) = 1, verified by direct probe

_ROWS: list[tuple[str, str, str, str]] = [
    # 9 normal Groceries (~45–55)
    ("2024-03-01", "Whole Foods",       "45.00", "Groceries"),
    ("2024-03-02", "Trader Joe's",      "52.00", "Groceries"),
    ("2024-03-03", "Safeway",           "48.50", "Groceries"),
    ("2024-03-04", "Kroger",            "55.00", "Groceries"),
    ("2024-03-05", "Local Mart",        "47.00", "Groceries"),
    ("2024-03-06", "Aldi",              "51.50", "Groceries"),
    ("2024-03-07", "Publix",            "49.00", "Groceries"),
    ("2024-03-08", "Whole Foods",       "53.50", "Groceries"),
    ("2024-03-09", "Safeway",           "46.50", "Groceries"),
    # 9 normal Dining (~25–33)
    ("2024-03-10", "Chipotle",          "25.00", "Dining"),
    ("2024-03-11", "Olive Garden",      "30.00", "Dining"),
    ("2024-03-12", "Local Diner",       "28.50", "Dining"),
    ("2024-03-13", "Sushi House",       "32.00", "Dining"),
    ("2024-03-14", "Pizza Place",       "27.00", "Dining"),
    ("2024-03-15", "McDonald's",        "29.50", "Dining"),
    ("2024-03-16", "Starbucks",         "31.00", "Dining"),
    ("2024-03-17", "Subway",            "26.50", "Dining"),
    ("2024-03-18", "Chipotle",          "33.00", "Dining"),
    # 1 moderate outlier — amount far above normal but NOT flagged at 5% threshold
    ("2024-03-19", "Whole Foods Bulk",  "500.00", "Groceries"),
    # 1 extreme outlier — flagged as THE anomaly
    ("2024-03-20", "Luxury Restaurant", "2500.00", "Dining"),
]

assert len(_ROWS) == 20, "Dataset must have exactly 20 rows"

N = len(_ROWS)


def _make_csv_bytes() -> bytes:
    """Return valid UTF-8 CSV bytes for the 20-row test dataset."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "merchant", "amount", "category"])
    for row in _ROWS:
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# No-op lifespan — skips heavy ML startup (SentenceTransformer, env vars)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Build TestClient with real store + real anomaly detector, mocked ML
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path) -> TestClient:
    """
    AppComponents:
      - real TransactionStore (tmp_path / "test.db")
      - real AnomalyDetector  (contamination=0.05, random_state=42)
      - mock VectorStore      (no embedding model loaded)
      - untrained Categorizer (_is_trained=False)
      - mock Forecaster
    """
    from api.app import app
    from api.dependencies import AppComponents
    from ingestion.transaction_store import TransactionStore
    from anomaly.anomaly_detector import AnomalyDetector
    from categorization.categorizer import Categorizer

    real_store = TransactionStore(str(tmp_path / "test.db"))
    real_detector = AnomalyDetector()   # contamination=0.05, random_state=42

    mock_vector_store = MagicMock()
    mock_vector_store.indexed_ids = frozenset()  # nothing indexed yet

    untrained_cat = Categorizer()
    assert not untrained_cat._is_trained  # sanity: no model loaded

    components = AppComponents(
        store=real_store,
        vector_store=mock_vector_store,
        categorizer=untrained_cat,
        anomaly_detector=real_detector,
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

class TestAnomalyDetectionRoundTrip:
    """Anomaly detection: ingest → fit_and_score → GET /anomalies."""

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    # ------------------------------------------------------------------
    # TEST 1 — empty database returns empty anomaly list
    # ------------------------------------------------------------------

    def test_get_anomalies_empty_database_returns_empty_list(self, client):
        """GET /anomalies with no data must return HTTP 200 and an empty list."""
        response = client.get("/anomalies")

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body == [], (
            f"Expected [] on empty database, got: {body}"
        )

    # ------------------------------------------------------------------
    # TEST 2 — full round trip: ingest → real fit_and_score → GET /anomalies
    # ------------------------------------------------------------------

    def test_ingest_triggers_real_anomaly_detection(self, client):
        """
        POST /ingest with 20 rows must actually invoke fit_and_score()
        (N >= MIN_TRANSACTIONS=10) and set anomalies_detected in the response.
        """
        response = client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        assert response.status_code == 200, (
            f"Expected 200 from POST /ingest, got {response.status_code}: {response.text}"
        )
        body = response.json()

        assert body["ingested"] == N, (
            f"Expected ingested={N}, got {body['ingested']}"
        )
        assert body["skipped"] == 0
        assert body["warnings"] == []

        # fit_and_score() was called (N=20 >= MIN_TRANSACTIONS=10) →
        # anomalies_detected must be an integer, not None
        assert body["anomalies_detected"] is not None, (
            "anomalies_detected must not be None — fit_and_score() should have run"
        )
        assert isinstance(body["anomalies_detected"], int), (
            f"anomalies_detected must be int, got {type(body['anomalies_detected'])}"
        )

    def test_ingest_anomalies_detected_count_matches_expected(self, client):
        """
        The ingest response anomalies_detected must equal the deterministically
        expected count for this dataset (1 anomaly confirmed by direct probe).
        """
        body = client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        ).json()

        assert body["anomalies_detected"] == _EXPECTED_ANOMALY_COUNT, (
            f"Expected anomalies_detected={_EXPECTED_ANOMALY_COUNT} "
            f"(IsolationForest contamination=0.05 on 20 rows), "
            f"got {body['anomalies_detected']}"
        )

    def test_get_anomalies_returns_correct_count(self, client):
        """
        After ingest, GET /anomalies must return exactly anomalies_detected records.
        """
        ingest_body = client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        ).json()
        expected_count = ingest_body["anomalies_detected"]

        response = client.get("/anomalies")

        assert response.status_code == 200
        anomalies = response.json()
        assert isinstance(anomalies, list)
        assert len(anomalies) == expected_count, (
            f"GET /anomalies returned {len(anomalies)} records "
            f"but ingest said anomalies_detected={expected_count}"
        )

    def test_every_anomaly_has_is_anomaly_true(self, client):
        """Every transaction returned by GET /anomalies must have is_anomaly=True."""
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        anomalies = client.get("/anomalies").json()

        assert len(anomalies) > 0, "Expected at least one anomaly for this dataset"
        for txn in anomalies:
            assert txn["is_anomaly"] is True, (
                f"Transaction {txn.get('id')} has is_anomaly={txn['is_anomaly']!r}, "
                "expected True"
            )

    def test_every_anomaly_has_valid_score(self, client):
        """Every anomaly must have a non-None score in [0.0, 1.0]."""
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        anomalies = client.get("/anomalies").json()

        for txn in anomalies:
            score = txn["anomaly_score"]
            assert score is not None, (
                f"Transaction {txn.get('id')}: anomaly_score must not be None"
            )
            assert isinstance(score, float), (
                f"Transaction {txn.get('id')}: anomaly_score must be float, "
                f"got {type(score)}"
            )
            assert 0.0 <= score <= 1.0, (
                f"Transaction {txn.get('id')}: anomaly_score {score} not in [0.0, 1.0]"
            )
            # Confirmed by probe: the one anomaly (2500.0 Dining) has score ≈ 0.060
            assert score > 0.0, (
                f"Transaction {txn.get('id')}: anomaly_score should be > 0 for a "
                f"genuine anomaly, got {score}"
            )

    def test_anomalies_sorted_by_score_descending(self, client):
        """GET /anomalies must return records sorted by anomaly_score descending."""
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        anomalies = client.get("/anomalies").json()

        scores = [t["anomaly_score"] for t in anomalies]
        assert scores == sorted(scores, reverse=True), (
            f"Anomalies not sorted by score descending: {scores}"
        )

    def test_anomaly_schema_fields_complete(self, client):
        """Every anomaly transaction must contain all required TransactionDTO fields."""
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        anomalies = client.get("/anomalies").json()

        required_fields = {
            "id", "date", "merchant", "amount", "category",
            "is_anomaly", "anomaly_score", "needs_review", "source_file",
        }
        for txn in anomalies:
            missing = required_fields - txn.keys()
            assert not missing, (
                f"Anomaly transaction missing fields: {missing}"
            )

    def test_anomaly_merchant_is_extreme_outlier(self, client):
        """
        The deterministic dataset has exactly one anomaly: 'Luxury Restaurant'
        (amount=2500.0, the extreme outlier).  Verify the flagged transaction
        is the expected one.
        """
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        anomalies = client.get("/anomalies").json()

        assert len(anomalies) == _EXPECTED_ANOMALY_COUNT
        flagged_merchants = [t["merchant"] for t in anomalies]
        assert "Luxury Restaurant" in flagged_merchants, (
            f"Expected 'Luxury Restaurant' (2500.0) to be flagged as anomaly. "
            f"Got: {flagged_merchants}"
        )
        flagged_amounts = [t["amount"] for t in anomalies]
        assert 2500.0 in flagged_amounts, (
            f"Expected amount 2500.0 in anomaly list, got: {flagged_amounts}"
        )

    # ------------------------------------------------------------------
    # TEST 3 — cross-endpoint consistency: anomaly IDs ⊂ all transaction IDs
    # ------------------------------------------------------------------

    def test_anomaly_ids_are_subset_of_all_transaction_ids(self, client):
        """
        Every ID returned by GET /anomalies must also appear in GET /transactions.
        Anomalies must be a strict subset (not all transactions are anomalies).
        """
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        all_txns = client.get("/transactions").json()
        anomalies = client.get("/anomalies").json()

        all_ids = {t["id"] for t in all_txns}
        anomaly_ids = {t["id"] for t in anomalies}

        assert anomaly_ids, "Expected at least one anomaly ID"
        assert anomaly_ids.issubset(all_ids), (
            f"Anomaly IDs {anomaly_ids} are not a subset of all transaction IDs {all_ids}"
        )
        assert len(anomaly_ids) < len(all_ids), (
            f"Expected anomalies ({len(anomaly_ids)}) to be fewer than all "
            f"transactions ({len(all_ids)}) — not all rows should be flagged"
        )

    def test_non_anomaly_transactions_have_is_anomaly_false(self, client):
        """
        Transactions NOT returned by GET /anomalies must have is_anomaly=False
        in GET /transactions.
        """
        client.post(
            "/ingest",
            files={"file": (_FILENAME, _make_csv_bytes(), "text/csv")},
        )

        all_txns = client.get("/transactions").json()
        anomalies = client.get("/anomalies").json()

        anomaly_ids = {t["id"] for t in anomalies}
        non_anomaly_txns = [t for t in all_txns if t["id"] not in anomaly_ids]

        assert len(non_anomaly_txns) == N - _EXPECTED_ANOMALY_COUNT, (
            f"Expected {N - _EXPECTED_ANOMALY_COUNT} non-anomaly transactions, "
            f"got {len(non_anomaly_txns)}"
        )
        for txn in non_anomaly_txns:
            assert txn["is_anomaly"] is False, (
                f"Transaction {txn['id']} ({txn['merchant']}) is not in the anomaly "
                f"list but has is_anomaly=True"
            )
