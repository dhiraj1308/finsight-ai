from datetime import date

import pytest

from domain import Transaction
from ingestion.transaction_store import TransactionStore


def _make_store(tmp_path) -> TransactionStore:
    db_path = tmp_path / "test.db"
    return TransactionStore(str(db_path))


def _sample_transaction(
    txn_date=date(2024, 3, 15),
    merchant="Whole Foods",
    amount=87.43,
    category="Groceries",
    source_file="test.csv",
) -> Transaction:
    return Transaction(
        date=txn_date,
        merchant=merchant,
        amount=amount,
        category=category,
        source_file=source_file,
    )


def test_insert_assigns_unique_ids(tmp_path):
    store = _make_store(tmp_path)
    txns = [
        _sample_transaction(merchant="Store A"),
        _sample_transaction(merchant="Store B"),
        _sample_transaction(merchant="Store C"),
    ]

    inserted, skipped = store.insert(txns)

    assert inserted == 3
    assert skipped == 0

    all_txns = store.get_all()
    ids = [t.id for t in all_txns]
    assert len(set(ids)) == 3  # all unique
    assert all(i is not None for i in ids)


def test_insert_deduplicates_identical_transactions(tmp_path):
    store = _make_store(tmp_path)
    txn = _sample_transaction()

    inserted1, skipped1 = store.insert([txn])
    inserted2, skipped2 = store.insert([txn])

    assert inserted1 == 1
    assert skipped1 == 0
    assert inserted2 == 0
    assert skipped2 == 1
    assert len(store.get_all()) == 1


def test_query_by_date_range_returns_only_in_range_records(tmp_path):
    store = _make_store(tmp_path)
    store.insert(
        [
            _sample_transaction(txn_date=date(2024, 1, 15), merchant="January"),
            _sample_transaction(txn_date=date(2024, 3, 15), merchant="March"),
            _sample_transaction(txn_date=date(2024, 6, 15), merchant="June"),
        ]
    )

    results = store.query_by_date_range(date(2024, 2, 1), date(2024, 4, 30))

    assert len(results) == 1
    assert results[0].merchant == "March"


def test_query_by_date_range_includes_boundary_dates(tmp_path):
    store = _make_store(tmp_path)
    store.insert(
        [
            _sample_transaction(txn_date=date(2024, 1, 1), merchant="StartBoundary"),
            _sample_transaction(txn_date=date(2024, 1, 31), merchant="EndBoundary"),
        ]
    )

    results = store.query_by_date_range(date(2024, 1, 1), date(2024, 1, 31))

    assert len(results) == 2


def test_query_by_date_range_raises_for_invalid_range(tmp_path):
    store = _make_store(tmp_path)

    with pytest.raises(ValueError):
        store.query_by_date_range(date(2024, 12, 31), date(2024, 1, 1))


def test_query_by_category_is_case_insensitive(tmp_path):
    store = _make_store(tmp_path)
    store.insert([_sample_transaction(category="Groceries")])

    results_lower = store.query_by_category("groceries")
    results_upper = store.query_by_category("GROCERIES")
    results_exact = store.query_by_category("Groceries")

    assert len(results_lower) == 1
    assert len(results_upper) == 1
    assert len(results_exact) == 1


def test_query_by_category_returns_empty_list_for_no_match(tmp_path):
    store = _make_store(tmp_path)
    store.insert([_sample_transaction(category="Groceries")])

    results = store.query_by_category("Nonexistent")

    assert results == []


def test_get_all_returns_records_sorted_by_date_descending(tmp_path):
    store = _make_store(tmp_path)
    store.insert(
        [
            _sample_transaction(txn_date=date(2024, 1, 1), merchant="Oldest"),
            _sample_transaction(txn_date=date(2024, 6, 1), merchant="Newest"),
            _sample_transaction(txn_date=date(2024, 3, 1), merchant="Middle"),
        ]
    )

    results = store.get_all()

    assert [t.merchant for t in results] == ["Newest", "Middle", "Oldest"]


def test_get_all_returns_empty_list_when_store_is_empty(tmp_path):
    store = _make_store(tmp_path)

    results = store.get_all()

    assert results == []


def test_delete_removes_transaction(tmp_path):
    store = _make_store(tmp_path)
    store.insert([_sample_transaction()])
    txn_id = store.get_all()[0].id

    store.delete(txn_id)

    assert store.get_all() == []