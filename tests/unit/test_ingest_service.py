"""Unit tests for IngestService.

Tests call IngestService.process() directly — no HTTP, no TestClient.
All components (store, vector_store, categorizer, anomaly_detector) are
either real (TransactionStore backed by tmp_path SQLite) or MagicMock,
depending on what each test is verifying.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.models import IngestResponse
from api.services.ingest_service import IngestService, _ANOMALY_MIN_TRANSACTIONS
from domain import Transaction
from ingestion.transaction_store import TransactionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transaction(
    merchant: str = "Whole Foods",
    amount: float = 50.0,
    category: str = "",
    txn_date: date = date(2024, 3, 15),
) -> Transaction:
    return Transaction(date=txn_date, merchant=merchant, amount=amount, category=category)


def _make_store(tmp_path: Path) -> TransactionStore:
    return TransactionStore(str(tmp_path / "test.db"))


def _mock_vector_store(already_indexed: frozenset[int] = frozenset()) -> MagicMock:
    vs = MagicMock()
    vs.indexed_ids = already_indexed
    return vs


def _untrained_categorizer() -> MagicMock:
    cat = MagicMock()
    cat._is_trained = False
    return cat


def _trained_categorizer(needs_review_flags: list[bool]) -> MagicMock:
    """Return a mock categorizer that sets needs_review per the given flags list."""
    cat = MagicMock()
    cat._is_trained = True

    def _fake_predict_batch(txns):
        for i, txn in enumerate(txns):
            flag = needs_review_flags[i % len(needs_review_flags)]
            txn.needs_review = flag
            txn.category = "Other" if flag else "Groceries"
        return txns

    cat.predict_batch.side_effect = _fake_predict_batch
    return cat


# ---------------------------------------------------------------------------
# TEST 1 — successful processing returns correct IngestResponse
# ---------------------------------------------------------------------------

def test_process_returns_ingest_response(tmp_path):
    """process() must return an IngestResponse instance."""
    store = _make_store(tmp_path)
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), MagicMock())

    result = service.process(
        [_make_transaction()], "statement.csv", []
    )

    assert isinstance(result, IngestResponse)
    assert result.ingested == 1
    assert result.skipped == 0
    assert result.warnings == []


# ---------------------------------------------------------------------------
# TEST 2 — source_file is assigned from the filename argument
# ---------------------------------------------------------------------------

def test_process_assigns_source_file(tmp_path):
    """Every transaction must have source_file set to the provided filename."""
    store = _make_store(tmp_path)
    txn = _make_transaction()
    assert txn.source_file == ""  # default before processing

    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), MagicMock())
    service.process([txn], "my_statement.pdf", [])

    stored = store.get_all()
    assert len(stored) == 1
    assert stored[0].source_file == "my_statement.pdf"


# ---------------------------------------------------------------------------
# TEST 3 — categorization runs when trained
# ---------------------------------------------------------------------------

def test_process_categorizes_when_trained(tmp_path):
    """When categorizer._is_trained is True, predict_batch() must be called."""
    store = _make_store(tmp_path)
    cat = _trained_categorizer([False])  # all high confidence
    service = IngestService(store, _mock_vector_store(), cat, MagicMock())

    result = service.process(
        [_make_transaction(), _make_transaction(merchant="Netflix", amount=15.99)],
        "s.csv",
        [],
    )

    cat.predict_batch.assert_called_once()
    assert result.needs_review_count == 0


# ---------------------------------------------------------------------------
# TEST 4 — needs_review_count reflects low-confidence categorizations
# ---------------------------------------------------------------------------

def test_process_needs_review_count(tmp_path):
    """needs_review_count must equal the number of transactions flagged as needing review."""
    store = _make_store(tmp_path)
    # 2 transactions; first needs review, second does not
    cat = _trained_categorizer([True, False])
    service = IngestService(store, _mock_vector_store(), cat, MagicMock())

    result = service.process(
        [_make_transaction("A"), _make_transaction("B")], "s.csv", []
    )

    assert result.needs_review_count == 1


# ---------------------------------------------------------------------------
# TEST 5 — needs_review_count is None when categorizer is untrained
# ---------------------------------------------------------------------------

def test_process_needs_review_count_none_when_untrained(tmp_path):
    """When the categorizer is not trained, needs_review_count must be None."""
    store = _make_store(tmp_path)
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), MagicMock())

    result = service.process([_make_transaction()], "s.csv", [])

    assert result.needs_review_count is None


# ---------------------------------------------------------------------------
# TEST 6 — database insertion counts are correct
# ---------------------------------------------------------------------------

def test_process_database_insertion_counts(tmp_path):
    """ingested and skipped counts in IngestResponse must reflect store.insert()."""
    store = _make_store(tmp_path)
    txn = _make_transaction()
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), MagicMock())

    # First insert — should succeed
    r1 = service.process([txn], "s.csv", [])
    assert r1.ingested == 1
    assert r1.skipped == 0

    # Second insert of same data — should be skipped (duplicate)
    r2 = service.process([txn], "s.csv", [])
    assert r2.ingested == 0
    assert r2.skipped == 1


# ---------------------------------------------------------------------------
# TEST 7 — vector indexing called only for newly inserted transactions
# ---------------------------------------------------------------------------

def test_process_vector_indexes_only_new_transactions(tmp_path):
    """index() must be called only for IDs not already in vector_store.indexed_ids."""
    store = _make_store(tmp_path)
    txn = _make_transaction()
    store.insert([txn])
    stored_id = store.get_all()[0].id

    # Pre-populate indexed_ids so the already-stored transaction is skipped
    vs = _mock_vector_store(already_indexed=frozenset({stored_id}))
    service = IngestService(store, vs, _untrained_categorizer(), MagicMock())

    # Insert a second transaction — only this one should be indexed
    txn2 = _make_transaction(merchant="Netflix", txn_date=date(2024, 3, 16))
    service.process([txn2], "s.csv", [])

    # index() called once, for the new transaction only
    assert vs.index.call_count == 1
    indexed_txn = vs.index.call_args[0][0]
    assert indexed_txn.merchant == "Netflix"


# ---------------------------------------------------------------------------
# TEST 8 — anomaly detection runs when threshold is met
# ---------------------------------------------------------------------------

def test_process_anomaly_detection_runs_above_threshold(tmp_path):
    """fit_and_score() must be called when total transaction count >= threshold."""
    store = _make_store(tmp_path)

    # Pre-populate the store so get_all() returns >= threshold rows
    for i in range(_ANOMALY_MIN_TRANSACTIONS):
        store.insert([_make_transaction(
            merchant=f"Merchant {i}",
            txn_date=date(2024, 1, i + 1),
        )])

    mock_detector = MagicMock()
    mock_detector.fit_and_score.return_value = 2
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), mock_detector)

    result = service.process([], "s.csv", [])

    mock_detector.fit_and_score.assert_called_once_with(store)
    assert result.anomalies_detected == 2


# ---------------------------------------------------------------------------
# TEST 9 — anomaly detection skipped below threshold
# ---------------------------------------------------------------------------

def test_process_anomaly_detection_skipped_below_threshold(tmp_path):
    """fit_and_score() must NOT be called when total row count < threshold."""
    store = _make_store(tmp_path)

    # Insert fewer than the minimum required transactions
    for i in range(_ANOMALY_MIN_TRANSACTIONS - 1):
        store.insert([_make_transaction(
            merchant=f"Merchant {i}",
            txn_date=date(2024, 1, i + 1),
        )])

    mock_detector = MagicMock()
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), mock_detector)

    result = service.process([], "s.csv", [])

    mock_detector.fit_and_score.assert_not_called()
    assert result.anomalies_detected is None


# ---------------------------------------------------------------------------
# TEST 10 — vector indexing failure becomes a warning, not an error
# ---------------------------------------------------------------------------

def test_process_vector_indexing_failure_does_not_abort(tmp_path):
    """A vector_store.index() exception must be caught and logged; process() must succeed."""
    store = _make_store(tmp_path)
    vs = _mock_vector_store()
    vs.index.side_effect = RuntimeError("embedding model unavailable")
    service = IngestService(store, vs, _untrained_categorizer(), MagicMock())

    # Must not raise
    result = service.process([_make_transaction()], "s.csv", [])

    assert isinstance(result, IngestResponse)
    assert result.ingested == 1  # transaction was inserted despite index failure


# ---------------------------------------------------------------------------
# TEST 11 — anomaly detection failure becomes a warning, anomalies_detected=None
# ---------------------------------------------------------------------------

def test_process_anomaly_detection_failure_does_not_abort(tmp_path):
    """An anomaly_detector.fit_and_score() exception must not abort ingestion."""
    store = _make_store(tmp_path)

    for i in range(_ANOMALY_MIN_TRANSACTIONS):
        store.insert([_make_transaction(
            merchant=f"Merchant {i}",
            txn_date=date(2024, 1, i + 1),
        )])

    mock_detector = MagicMock()
    mock_detector.fit_and_score.side_effect = RuntimeError("sklearn error")
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), mock_detector)

    result = service.process([], "s.csv", [])

    assert isinstance(result, IngestResponse)
    assert result.anomalies_detected is None  # failure → None


# ---------------------------------------------------------------------------
# TEST 12 — parse warnings are capped at 10 and passed through
# ---------------------------------------------------------------------------

def test_process_warnings_capped_at_ten(tmp_path):
    """process() must cap parse warnings at 10 in the IngestResponse."""
    store = _make_store(tmp_path)
    service = IngestService(store, _mock_vector_store(), _untrained_categorizer(), MagicMock())

    many_warnings = [f"Row {i}: bad value" for i in range(20)]
    result = service.process([], "s.csv", many_warnings)

    assert len(result.warnings) == 10
    assert result.warnings == many_warnings[:10]
