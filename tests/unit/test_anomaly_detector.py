import pytest

from anomaly.anomaly_detector import AnomalyDetector, MIN_TRANSACTIONS
from ingestion.synthetic_generator import SyntheticGenerator
from ingestion.transaction_store import TransactionStore


def _make_store(tmp_path, n: int = 50, seed: int = 42) -> TransactionStore:
    """Creates a TransactionStore populated with n synthetic transactions."""
    db_path = tmp_path / "test.db"
    store = TransactionStore(str(db_path))
    gen = SyntheticGenerator()
    txns = gen.generate(n=n, seed=seed)
    for t in txns:
        t.source_file = "test"
    store.insert(txns)
    return store


def test_fit_and_score_flags_anomalies_for_all_transactions(tmp_path):
    store = _make_store(tmp_path, n=50)
    detector = AnomalyDetector()

    detector.fit_and_score(store)

    all_txns = store.get_all()
    # Every transaction should have is_anomaly set (True or False)
    assert all(txn.is_anomaly is not None for txn in all_txns)


def test_anomaly_scores_are_in_valid_range(tmp_path):
    store = _make_store(tmp_path, n=50)
    detector = AnomalyDetector()

    detector.fit_and_score(store)

    all_txns = store.get_all()
    anomalies = [t for t in all_txns if t.is_anomaly]
    for txn in anomalies:
        assert txn.anomaly_score is not None
        assert 0.0 <= txn.anomaly_score <= 1.0


def test_fit_and_score_returns_correct_anomaly_count(tmp_path):
    store = _make_store(tmp_path, n=100)
    detector = AnomalyDetector(contamination=0.05)

    count = detector.fit_and_score(store)

    # With contamination=0.05 and 100 transactions, expect exactly 5 anomalies
    assert count == 5


def test_fit_raises_for_insufficient_transactions(tmp_path):
    store = _make_store(tmp_path, n=5)
    detector = AnomalyDetector()

    with pytest.raises(ValueError) as exc_info:
        detector.fit_and_score(store)

    assert str(MIN_TRANSACTIONS) in str(exc_info.value)


def test_fit_does_not_modify_flags_when_raising(tmp_path):
    store = _make_store(tmp_path, n=5)
    detector = AnomalyDetector()

    try:
        detector.fit_and_score(store)
    except ValueError:
        pass

    all_txns = store.get_all()
    # No flags should have been modified
    assert all(txn.is_anomaly is False for txn in all_txns)
    assert all(txn.anomaly_score is None for txn in all_txns)


def test_refit_updates_all_records_including_new_ones(tmp_path):
    store = _make_store(tmp_path, n=50)
    detector = AnomalyDetector()

    detector.fit_and_score(store)

    # Add more transactions then refit
    gen = SyntheticGenerator()
    new_txns = gen.generate(n=20, seed=99)
    for t in new_txns:
        t.source_file = "new_batch"
    store.insert(new_txns)

    detector.fit_and_score(store)

    all_txns = store.get_all()
    assert len(all_txns) == 70
    # All 70 records should have scores after refit
    scored = [t for t in all_txns if t.anomaly_score is not None]
    assert len(scored) == 70


def test_get_anomalies_returns_only_flagged_transactions(tmp_path):
    store = _make_store(tmp_path, n=50)
    detector = AnomalyDetector()

    detector.fit_and_score(store)
    anomalies = detector.get_anomalies(store)

    assert all(txn.is_anomaly for txn in anomalies)


def test_get_anomalies_sorted_by_score_descending(tmp_path):
    store = _make_store(tmp_path, n=100)
    detector = AnomalyDetector()

    detector.fit_and_score(store)
    anomalies = detector.get_anomalies(store)

    scores = [t.anomaly_score for t in anomalies]
    assert scores == sorted(scores, reverse=True)


def test_get_anomalies_returns_empty_list_when_none_flagged(tmp_path):
    store = _make_store(tmp_path, n=50)
    detector = AnomalyDetector()
    # Don't run fit_and_score — no anomalies flagged yet
    anomalies = detector.get_anomalies(store)
    assert anomalies == []