"""Unit tests for VectorStore.

SentenceTransformer / torch are mocked out entirely so these tests can run
in the same process as sklearn-based tests.  On Windows, sklearn's BLAS/LAPACK
DLLs mutate the DLL loader state in a way that prevents torch's c10.dll from
initialising afterwards (WinError 1114).  Mocking avoids that problem while
still exercising every line of VectorStore logic.
"""
import shutil
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api.vector_store import VectorStore
from domain import Transaction

pytestmark = pytest.mark.vector

# ---------------------------------------------------------------------------
# Shared mock: SentenceTransformer.encode returns a deterministic unit vector
# so similarity calculations inside VectorStore still work correctly.
# ---------------------------------------------------------------------------

def _fake_encode(texts, **kwargs):
    """Return one 8-d unit vector per input text (deterministic, non-zero)."""
    vecs = []
    for i, t in enumerate(texts):
        v = np.zeros(8, dtype=np.float32)
        v[i % 8] = 1.0
        vecs.append(v)
    return np.array(vecs, dtype=np.float32)


@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Patch SentenceTransformer for every test in this module."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = _fake_encode
    with patch(
        "api.vector_store.VectorStore._get_model",
        return_value=mock_model,
    ):
        yield mock_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path) -> VectorStore:
    persist_dir = str(tmp_path / "vector_store")
    return VectorStore(persist_dir=persist_dir)


def _make_transaction(
    txn_id: int,
    merchant: str,
    category: str = "Groceries",
    amount: float = 50.0,
) -> Transaction:
    return Transaction(
        id=txn_id,
        date=date(2024, 3, 15),
        merchant=merchant,
        amount=amount,
        category=category,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_count_is_zero(tmp_path):
    vs = _make_store(tmp_path)
    assert vs.count == 0


def test_index_increases_count(tmp_path):
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.index(_make_transaction(2, "Netflix", "Entertainment"))
    assert vs.count == 2


def test_embedding_text_format_is_canonical(tmp_path):
    """Verify the canonical text format: merchant category amount date."""
    vs = _make_store(tmp_path)
    txn = _make_transaction(1, "Whole Foods", "Groceries", 87.43)
    text = vs._transaction_text(txn)
    assert "Whole Foods" in text
    assert "Groceries" in text
    assert "87.43" in text
    assert "2024-03-15" in text


def test_search_returns_at_most_k_results(tmp_path):
    vs = _make_store(tmp_path)
    for i in range(5):
        vs.index(_make_transaction(i + 1, f"Store {i}", "Shopping"))
    results = vs.search("shopping", k=3)
    assert len(results) <= 3


def test_search_returns_all_when_fewer_than_k(tmp_path):
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.index(_make_transaction(2, "Netflix", "Entertainment"))
    results = vs.search("anything", k=10)
    assert len(results) == 2


def test_search_returns_empty_when_store_is_empty(tmp_path):
    vs = _make_store(tmp_path)
    results = vs.search("food", k=5)
    assert results == []


def test_upsert_does_not_increase_count(tmp_path):
    vs = _make_store(tmp_path)
    txn = _make_transaction(1, "Whole Foods", "Groceries")
    vs.index(txn)
    assert vs.count == 1
    # Index same ID again — should update, not append
    updated = _make_transaction(1, "Whole Foods Market", "Groceries")
    vs.index(updated)
    assert vs.count == 1


def test_delete_decreases_count(tmp_path):
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.index(_make_transaction(2, "Netflix", "Entertainment"))
    vs.delete(1)
    assert vs.count == 1


def test_delete_nonexistent_id_is_noop(tmp_path):
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.delete(999)  # Should not raise
    assert vs.count == 1


def test_deleted_transaction_not_in_search_results(tmp_path):
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.index(_make_transaction(2, "Netflix", "Entertainment"))
    vs.delete(2)
    results = vs.search("entertainment streaming", k=5)
    ids = [r.id for r in results]
    assert 2 not in ids


# ---------------------------------------------------------------------------
# Incremental-indexing tests
# Verifies that VectorStore.index() skips re-embedding unchanged transactions,
# re-embeds when content changes, and never creates duplicates.
# ---------------------------------------------------------------------------

def test_skip_if_unchanged_does_not_call_encode(tmp_path, mock_sentence_transformer):
    """
    Indexing the same transaction twice (identical text) must NOT call encode
    on the second call.
    """
    vs = _make_store(tmp_path)
    txn = _make_transaction(1, "Whole Foods", "Groceries", 50.0)

    vs.index(txn)
    first_call_count = mock_sentence_transformer.encode.call_count

    # Index the exact same transaction again — text is unchanged
    vs.index(txn)
    second_call_count = mock_sentence_transformer.encode.call_count

    assert second_call_count == first_call_count, (
        f"encode() called {second_call_count - first_call_count} extra time(s) "
        "on a transaction with no text change — expected 0"
    )


def test_skip_if_unchanged_does_not_increase_count(tmp_path):
    """Re-indexing an unchanged transaction must not increase the store count."""
    vs = _make_store(tmp_path)
    txn = _make_transaction(1, "Whole Foods", "Groceries", 50.0)
    vs.index(txn)
    vs.index(txn)
    assert vs.count == 1


def test_upsert_on_changed_text_calls_encode_again(tmp_path, mock_sentence_transformer):
    """
    When the text representation changes (e.g. category updated), encode()
    must be called again to regenerate the embedding.
    """
    vs = _make_store(tmp_path)
    txn_original = _make_transaction(1, "Whole Foods", "Groceries", 50.0)
    vs.index(txn_original)
    before = mock_sentence_transformer.encode.call_count

    # Same ID, different category → text changes
    txn_updated = _make_transaction(1, "Whole Foods", "Shopping", 50.0)
    vs.index(txn_updated)
    after = mock_sentence_transformer.encode.call_count

    assert after > before, (
        "encode() should be called again when the transaction text changes"
    )
    # Count must still be 1 — it is an upsert, not a new entry
    assert vs.count == 1


def test_upsert_on_changed_text_updates_metadata(tmp_path):
    """After a category change, the stored metadata must reflect the new category."""
    vs = _make_store(tmp_path)
    vs.index(_make_transaction(1, "Whole Foods", "Groceries", 50.0))
    vs.index(_make_transaction(1, "Whole Foods", "Shopping", 50.0))

    stored_category = vs._metadata[0]["category"]
    assert stored_category == "Shopping", (
        f"Expected category 'Shopping' after upsert, got '{stored_category}'"
    )


def test_incremental_indexing_adds_only_new_transactions(tmp_path, mock_sentence_transformer):
    """
    Initial batch: transactions 1-5.
    Second batch: transactions 6-8.
    encode() must be called exactly 5 times for the first batch and 3 for
    the second — existing entries must not be re-embedded.
    """
    vs = _make_store(tmp_path)

    first_batch = [_make_transaction(i, f"Store {i}") for i in range(1, 6)]
    for txn in first_batch:
        vs.index(txn)

    calls_after_first = mock_sentence_transformer.encode.call_count
    assert calls_after_first == 5, (
        f"Expected 5 encode() calls for the first batch, got {calls_after_first}"
    )

    second_batch = [_make_transaction(i, f"Store {i}") for i in range(6, 9)]
    for txn in second_batch:
        vs.index(txn)

    calls_after_second = mock_sentence_transformer.encode.call_count
    assert calls_after_second == 8, (
        f"Expected 8 total encode() calls after second batch, "
        f"got {calls_after_second}"
    )
    assert vs.count == 8


def test_repeated_ingestion_of_same_batch_does_not_duplicate(tmp_path):
    """
    Ingesting the same five transactions twice must result in exactly 5 entries,
    not 10.
    """
    vs = _make_store(tmp_path)
    batch = [_make_transaction(i, f"Merchant {i}") for i in range(1, 6)]

    for txn in batch:
        vs.index(txn)
    for txn in batch:
        vs.index(txn)  # identical — all should be skipped

    assert vs.count == 5, (
        f"Expected 5 entries after duplicate ingestion, got {vs.count}"
    )


def test_repeated_ingestion_does_not_call_encode_extra(tmp_path, mock_sentence_transformer):
    """
    Calling index() on the same unchanged transactions a second time must not
    trigger any additional encode() calls.
    """
    vs = _make_store(tmp_path)
    batch = [_make_transaction(i, f"Merchant {i}") for i in range(1, 4)]

    for txn in batch:
        vs.index(txn)
    baseline = mock_sentence_transformer.encode.call_count  # should be 3

    for txn in batch:
        vs.index(txn)
    assert mock_sentence_transformer.encode.call_count == baseline, (
        "encode() must not be called again for unchanged already-indexed transactions"
    )


def test_incremental_index_search_still_finds_all_transactions(tmp_path):
    """
    After incremental indexing across two separate batches, semantic search
    must return all indexed transactions when k equals the total count.

    The mock encode assigns a unique unit vector to each text, so every
    indexed transaction has a non-zero similarity to any query and must appear
    in a k=4 search over a 4-entry store.
    """
    vs = _make_store(tmp_path)

    # First batch
    vs.index(_make_transaction(1, "Whole Foods", "Groceries"))
    vs.index(_make_transaction(2, "Netflix", "Entertainment"))
    # Second batch — new transactions only
    vs.index(_make_transaction(3, "Uber", "Transport"))
    vs.index(_make_transaction(4, "Starbucks", "Dining"))

    assert vs.count == 4

    # With k equal to the store size, every entry must be returned.
    results = vs.search("spending query", k=4)
    assert len(results) == 4, (
        f"Expected all 4 indexed transactions to be returned by search, "
        f"got {len(results)}"
    )

    returned_ids = {r.id for r in results}
    expected_ids = {1, 2, 3, 4}
    assert returned_ids == expected_ids, (
        f"Expected IDs {expected_ids} in search results, got {returned_ids}. "
        "Transactions from both batches must remain searchable after "
        "incremental indexing."
    )
