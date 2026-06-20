from datetime import date
from domain import Transaction, VectorStoreIndexError


def test_transaction_creation_with_required_fields_only():
    txn = Transaction(date=date(2024, 3, 15), merchant="Whole Foods", amount=87.43)

    assert txn.date == date(2024, 3, 15)
    assert txn.merchant == "Whole Foods"
    assert txn.amount == 87.43
    assert txn.category == ""
    assert txn.id is None
    assert txn.is_anomaly is False
    assert txn.anomaly_score is None
    assert txn.needs_review is False
    assert txn.source_file == ""


def test_transaction_creation_with_all_fields():
    txn = Transaction(
        date=date(2024, 3, 15),
        merchant="Whole Foods",
        amount=87.43,
        category="Groceries",
        id=1,
        is_anomaly=True,
        anomaly_score=0.92,
        needs_review=True,
        source_file="statement_march.csv",
    )

    assert txn.category == "Groceries"
    assert txn.is_anomaly is True
    assert txn.anomaly_score == 0.92


def test_vector_store_index_error_is_an_exception():
    assert issubclass(VectorStoreIndexError, Exception)
