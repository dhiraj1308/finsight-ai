import shutil
from datetime import date
from pathlib import Path

import pytest

from api.vector_store import VectorStore
from domain import Transaction

pytestmark = pytest.mark.vector


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
    # Index same ID again
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
