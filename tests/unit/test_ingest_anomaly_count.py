"""Tests: anomalies_detected field in /ingest API response.

Verifies that the ingest endpoint correctly propagates the integer count
returned by AnomalyDetector.fit_and_score() into the IngestResponse, and
that it is None when fewer than 10 transactions are present (anomaly
detection is skipped) or when the detector raises.

Uses FastAPI's TestClient with the lifespan patched to a no-op and
mock components injected into app.state before each request.
"""
from __future__ import annotations

import csv
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_bytes(n_rows: int) -> bytes:
    """Generate a minimal valid CSV with n_rows transaction rows."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "merchant", "amount", "category"])
    for i in range(n_rows):
        writer.writerow(
            [f"2024-01-{(i % 28) + 1:02d}", f"Store {i}", f"{10 + i}.00", "Groceries"]
        )
    return buf.getvalue().encode("utf-8")


def _make_components(n_stored: int, anomaly_count: int):
    """Build mock AppComponents whose store returns n_stored transactions."""
    from domain import Transaction
    from datetime import date

    stored = [
        Transaction(
            id=i + 1,
            date=date(2024, 1, (i % 28) + 1),
            merchant=f"Store {i}",
            amount=float(10 + i),
            category="Groceries",
            source_file="test.csv",
        )
        for i in range(n_stored)
    ]

    store = MagicMock()
    store.insert.return_value = (n_stored, 0)
    store.get_all.return_value = stored

    vector_store = MagicMock()
    vector_store.indexed_ids = frozenset()

    categorizer = MagicMock()
    categorizer._is_trained = False          # skip categorization in handler

    anomaly_detector = MagicMock()
    anomaly_detector.fit_and_score.return_value = anomaly_count

    forecaster = MagicMock()

    components = MagicMock()
    components.store = store
    components.vector_store = vector_store
    components.categorizer = categorizer
    components.anomaly_detector = anomaly_detector
    components.forecaster = forecaster

    return components


@asynccontextmanager
async def _noop_lifespan(app):
    """No-op lifespan: skip the real startup (avoids loading sentence-transformers)."""
    yield


def _make_client(components) -> TestClient:
    """Return a TestClient with the no-op lifespan and mock components injected."""
    from api.app import app

    # Swap the lifespan before TestClient is constructed so the heavy startup
    # (SentenceTransformer, settings validation) is never triggered.
    original_router = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        # Set app.state so _get_components() returns our mocks
        app.state.components = components
        app.state.agent = MagicMock()
        return client
    finally:
        # Restore so other tests / runs are not affected
        app.router.lifespan_context = original_router


# ---------------------------------------------------------------------------
# TEST 1 — anomalies_detected is an integer when >= 10 transactions exist
# ---------------------------------------------------------------------------

def test_ingest_returns_anomalies_detected_when_enough_transactions():
    """
    >= 10 transactions → fit_and_score() runs → its return value appears
    as anomalies_detected in the JSON response body.
    """
    components = _make_components(n_stored=20, anomaly_count=3)
    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(20), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "anomalies_detected" in body, (
        f"Expected 'anomalies_detected' key in response, got: {list(body.keys())}"
    )
    assert body["anomalies_detected"] == 3, (
        f"Expected anomalies_detected=3, got {body['anomalies_detected']!r}"
    )
    assert isinstance(body["anomalies_detected"], int)
    # Core counts unchanged
    assert body["ingested"] == 20
    assert body["skipped"] == 0


# ---------------------------------------------------------------------------
# TEST 2 — anomalies_detected is None when fewer than 10 transactions
# ---------------------------------------------------------------------------

def test_ingest_anomalies_detected_is_none_when_below_threshold():
    """
    < 10 transactions → anomaly detection is skipped →
    anomalies_detected must be null in the JSON response.
    """
    components = _make_components(n_stored=5, anomaly_count=0)
    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(5), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "anomalies_detected" in body
    assert body["anomalies_detected"] is None, (
        f"Expected null for small dataset, got {body['anomalies_detected']!r}"
    )
    # fit_and_score must not have been called at all
    components.anomaly_detector.fit_and_score.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 3 — anomalies_detected is None when the detector raises
# ---------------------------------------------------------------------------

def test_ingest_anomalies_detected_is_none_when_detector_raises():
    """
    fit_and_score() raises → existing warning is logged → anomalies_detected
    is None and the overall 200 response is preserved.
    """
    components = _make_components(n_stored=15, anomaly_count=0)
    components.anomaly_detector.fit_and_score.side_effect = RuntimeError("detector broken")

    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(15), "text/csv")},
    )

    assert response.status_code == 200, (
        "Ingest must still return 200 even when anomaly detection raises"
    )
    body = response.json()
    assert body["anomalies_detected"] is None
    assert body["ingested"] == 15   # core ingestion unaffected


# ===========================================================================
# needs_review_count tests
# ===========================================================================

def _make_components_with_categorizer(
    n_stored: int,
    anomaly_count: int,
    is_trained: bool,
    needs_review_flags: list[bool],
):
    """Build mock AppComponents where the categorizer is configurable.

    The categorizer mock's predict_batch() mutates each transaction in-place
    to set .needs_review according to needs_review_flags (cycled if shorter
    than n_stored), mirroring the real Categorizer.predict_batch() behaviour.
    """
    from domain import Transaction
    from datetime import date

    stored = [
        Transaction(
            id=i + 1,
            date=date(2024, 1, (i % 28) + 1),
            merchant=f"Store {i}",
            amount=float(10 + i),
            category="Groceries",
            source_file="test.csv",
        )
        for i in range(n_stored)
    ]

    store = MagicMock()
    store.insert.return_value = (n_stored, 0)
    store.get_all.return_value = stored

    vector_store = MagicMock()
    vector_store.indexed_ids = frozenset()

    categorizer = MagicMock()
    categorizer._is_trained = is_trained

    if is_trained:
        def _fake_predict_batch(txns):
            for idx, txn in enumerate(txns):
                txn.needs_review = needs_review_flags[idx % len(needs_review_flags)]
                txn.category = "Other" if txn.needs_review else "Groceries"
            return txns
        categorizer.predict_batch.side_effect = _fake_predict_batch

    anomaly_detector = MagicMock()
    anomaly_detector.fit_and_score.return_value = anomaly_count

    forecaster = MagicMock()

    components = MagicMock()
    components.store = store
    components.vector_store = vector_store
    components.categorizer = categorizer
    components.anomaly_detector = anomaly_detector
    components.forecaster = forecaster
    return components


# ---------------------------------------------------------------------------
# TEST A — categorizer trained, some transactions need review
# ---------------------------------------------------------------------------

def test_ingest_needs_review_count_when_some_need_review():
    """
    Categorizer trained + 2 of 5 transactions flagged with needs_review=True
    → response contains needs_review_count = 2.
    """
    # Flags: True, False, True, False, False → 2 needing review
    flags = [True, False, True, False, False]
    components = _make_components_with_categorizer(
        n_stored=5,
        anomaly_count=0,
        is_trained=True,
        needs_review_flags=flags,
    )
    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(5), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "needs_review_count" in body, (
        f"Expected 'needs_review_count' in response, got: {list(body.keys())}"
    )
    assert body["needs_review_count"] == 2, (
        f"Expected needs_review_count=2, got {body['needs_review_count']!r}"
    )
    assert isinstance(body["needs_review_count"], int)


# ---------------------------------------------------------------------------
# TEST B — categorizer trained, no transactions need review
# ---------------------------------------------------------------------------

def test_ingest_needs_review_count_zero_when_all_high_confidence():
    """
    Categorizer trained + all transactions have needs_review=False
    → response contains needs_review_count = 0.
    """
    flags = [False, False, False]
    components = _make_components_with_categorizer(
        n_stored=3,
        anomaly_count=0,
        is_trained=True,
        needs_review_flags=flags,
    )
    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(3), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["needs_review_count"] == 0, (
        f"Expected needs_review_count=0, got {body['needs_review_count']!r}"
    )


# ---------------------------------------------------------------------------
# TEST C — categorizer not trained
# ---------------------------------------------------------------------------

def test_ingest_needs_review_count_is_none_when_categorizer_not_trained():
    """
    Categorizer not trained (_is_trained=False) → predict_batch() is not
    called → needs_review_count must be null in the JSON response.
    """
    components = _make_components_with_categorizer(
        n_stored=5,
        anomaly_count=0,
        is_trained=False,
        needs_review_flags=[],
    )
    client = _make_client(components)

    response = client.post(
        "/ingest",
        files={"file": ("statement.csv", _csv_bytes(5), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "needs_review_count" in body
    assert body["needs_review_count"] is None, (
        f"Expected null when categorizer not trained, "
        f"got {body['needs_review_count']!r}"
    )
    components.categorizer.predict_batch.assert_not_called()
