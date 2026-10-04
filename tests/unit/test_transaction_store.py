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


# ---------------------------------------------------------------------------
# update_anomaly_scores — new public method tests
# ---------------------------------------------------------------------------

def test_update_anomaly_scores_sets_flags_correctly(tmp_path):
    """update_anomaly_scores writes is_anomaly and anomaly_score to the DB."""
    store = _make_store(tmp_path)
    store.insert([
        _sample_transaction(merchant="Store A"),
        _sample_transaction(merchant="Store B"),
    ])
    txns = store.get_all()
    id_a = next(t.id for t in txns if t.merchant == "Store A")
    id_b = next(t.id for t in txns if t.merchant == "Store B")

    store.update_anomaly_scores([
        (id_a, True, 0.87),
        (id_b, False, 0.12),
    ])

    updated = {t.merchant: t for t in store.get_all()}
    assert updated["Store A"].is_anomaly is True
    assert abs(updated["Store A"].anomaly_score - 0.87) < 1e-6
    assert updated["Store B"].is_anomaly is False
    assert abs(updated["Store B"].anomaly_score - 0.12) < 1e-6


def test_update_anomaly_scores_handles_empty_list(tmp_path):
    """An empty updates list must not raise and must not alter any row."""
    store = _make_store(tmp_path)
    store.insert([_sample_transaction()])
    before = store.get_all()[0]

    store.update_anomaly_scores([])   # must be a no-op

    after = store.get_all()[0]
    assert after.is_anomaly == before.is_anomaly
    assert after.anomaly_score == before.anomaly_score


def test_update_anomaly_scores_multiple_transactions(tmp_path):
    """All rows in the update list are written in a single call."""
    store = _make_store(tmp_path)
    merchants = [f"Merchant {i}" for i in range(5)]
    store.insert([_sample_transaction(merchant=m) for m in merchants])

    txns = store.get_all()
    updates = [(t.id, i % 2 == 0, float(i) * 0.1) for i, t in enumerate(txns)]
    store.update_anomaly_scores(updates)

    refreshed = {t.id: t for t in store.get_all()}
    for txn_id, expected_flag, expected_score in updates:
        assert refreshed[txn_id].is_anomaly is expected_flag
        assert abs(refreshed[txn_id].anomaly_score - expected_score) < 1e-6
