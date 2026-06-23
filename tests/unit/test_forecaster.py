import pytest

from forecasting.forecaster import Forecaster, Forecast, ForecastPoint, MIN_HISTORY_DAYS
from ingestion.synthetic_generator import SyntheticGenerator
from ingestion.transaction_store import TransactionStore


def _make_store(tmp_path, n: int = 500, seed: int = 42) -> TransactionStore:
    """Creates a TransactionStore with enough data for forecasting."""
    db_path = tmp_path / "test.db"
    store = TransactionStore(str(db_path))
    gen = SyntheticGenerator()
    txns = gen.generate(n=n, seed=seed)
    for t in txns:
        t.source_file = "test"
    store.insert(txns)
    return store


def test_forecast_returns_correct_number_of_points(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    forecast = forecaster.forecast_category("Groceries", 30, store)

    assert len(forecast.points) == 30


def test_forecast_category_and_horizon_set_correctly(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    forecast = forecaster.forecast_category("Dining", 14, store)

    assert forecast.category == "Dining"
    assert forecast.horizon_days == 14


def test_all_yhat_values_are_non_negative(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    forecast = forecaster.forecast_category("Groceries", 30, store)

    assert all(p.yhat >= 0.0 for p in forecast.points)
    assert all(p.yhat_lower >= 0.0 for p in forecast.points)


def test_confidence_interval_structure_is_valid(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    forecast = forecaster.forecast_category("Groceries", 30, store)

    for point in forecast.points:
        assert point.yhat_lower <= point.yhat <= point.yhat_upper


def test_horizon_zero_raises_value_error(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    with pytest.raises(ValueError) as exc_info:
        forecaster.forecast_category("Groceries", 0, store)

    assert "horizon_days" in str(exc_info.value)


def test_horizon_too_large_raises_value_error(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    with pytest.raises(ValueError):
        forecaster.forecast_category("Groceries", 366, store)


def test_nonexistent_category_raises_value_error(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    with pytest.raises(ValueError) as exc_info:
        forecaster.forecast_category("NonExistent", 30, store)

    assert "NonExistent" in str(exc_info.value)


def test_insufficient_history_raises_value_error(tmp_path):
    """Category with fewer than 14 distinct days should raise ValueError."""
    from datetime import date
    from domain import Transaction

    db_path = tmp_path / "small.db"
    store = TransactionStore(str(db_path))

    # Manually insert exactly 5 transactions all on different days
    # but only 5 distinct dates — guaranteed < 14
    for i in range(5):
        txn = Transaction(
            date=date(2024, 1, i + 1),
            merchant="Whole Foods",
            amount=50.0 + i,
            category="Groceries",
            source_file="test",
        )
        store.insert([txn])

    forecaster = Forecaster()

    with pytest.raises(ValueError) as exc_info:
        forecaster.forecast_category("Groceries", 30, store)

    assert str(MIN_HISTORY_DAYS) in str(exc_info.value)


def test_forecast_all_returns_entry_for_every_category(tmp_path):
    store = _make_store(tmp_path)
    forecaster = Forecaster()

    results = forecaster.forecast_all(30, store)

    assert len(results) > 0
    # Every key should be either a Forecast or an error string
    for category, result in results.items():
        assert isinstance(result, (Forecast, str))


def test_forecast_all_never_raises(tmp_path):
    store = _make_store(tmp_path, n=10)  # Very small dataset
    forecaster = Forecaster()

    # Should not raise even with insufficient data
    try:
        results = forecaster.forecast_all(30, store)
        assert isinstance(results, dict)
    except Exception as e:
        pytest.fail(f"forecast_all raised unexpectedly: {e}")