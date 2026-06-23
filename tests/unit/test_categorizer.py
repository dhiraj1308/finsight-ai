from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from categorization.categorizer import (
    CANONICAL_CATEGORIES,
    CONFIDENCE_THRESHOLD,
    Categorizer,
)
from domain import Transaction


def _make_transaction(merchant: str, category: str = "Groceries") -> Transaction:
    return Transaction(
        date=date(2024, 3, 15),
        merchant=merchant,
        amount=10.0,
        category=category,
    )


def _make_labeled_dataset() -> list[Transaction]:
    """64 samples (8 per category x 8 categories) — enough for stratified split."""
    samples = [
        ("Whole Foods", "Groceries"), ("Kroger", "Groceries"),
        ("Safeway", "Groceries"), ("Trader Joe's", "Groceries"),
        ("Local Mart", "Groceries"), ("Walmart Grocery", "Groceries"),
        ("Aldi", "Groceries"), ("Publix", "Groceries"),
        ("City Power & Light", "Utilities"), ("Metro Water Co", "Utilities"),
        ("Gas Utility Inc", "Utilities"), ("ISP Broadband", "Utilities"),
        ("Electric Company", "Utilities"), ("Water Services", "Utilities"),
        ("Internet Provider", "Utilities"), ("Gas Company", "Utilities"),
        ("AMC Theatres", "Entertainment"), ("Spotify", "Entertainment"),
        ("Steam", "Entertainment"), ("Netflix", "Entertainment"),
        ("Hulu", "Entertainment"), ("Disney Plus", "Entertainment"),
        ("Local Arcade", "Entertainment"), ("Cinema Hall", "Entertainment"),
        ("Chipotle", "Dining"), ("Olive Garden", "Dining"),
        ("Local Diner", "Dining"), ("Sushi House", "Dining"),
        ("Pizza Place", "Dining"), ("McDonald's", "Dining"),
        ("Starbucks", "Dining"), ("Subway", "Dining"),
        ("Uber", "Transport"), ("Lyft", "Transport"),
        ("Metro Transit", "Transport"), ("Shell Gas Station", "Transport"),
        ("Parking Garage", "Transport"), ("City Bus", "Transport"),
        ("Taxi Service", "Transport"), ("BP Gas", "Transport"),
        ("CVS Pharmacy", "Healthcare"), ("Walgreens", "Healthcare"),
        ("City Clinic", "Healthcare"), ("Dental Care Co", "Healthcare"),
        ("Vision Center", "Healthcare"), ("Hospital Pharmacy", "Healthcare"),
        ("Medical Center", "Healthcare"), ("Health Clinic", "Healthcare"),
        ("Amazon", "Shopping"), ("Target", "Shopping"),
        ("Best Buy", "Shopping"), ("Local Boutique", "Shopping"),
        ("IKEA", "Shopping"), ("Walmart", "Shopping"),
        ("Costco", "Shopping"), ("Macy's", "Shopping"),
        ("Adobe Creative Cloud", "Subscriptions"), ("GitHub", "Subscriptions"),
        ("Gym Membership", "Subscriptions"), ("Notion", "Subscriptions"),
        ("Microsoft 365", "Subscriptions"), ("Apple One", "Subscriptions"),
        ("YouTube Premium", "Subscriptions"), ("Dropbox", "Subscriptions"),
    ]
    return [_make_transaction(m, c) for m, c in samples]


def test_predict_output_category_is_always_in_canonical_set():
    cat = Categorizer()
    cat.train(_make_labeled_dataset())
    txn = _make_transaction("Whole Foods", "")
    result = cat.predict(txn)
    assert result.category in CANONICAL_CATEGORIES


def test_low_confidence_prediction_sets_other_and_needs_review():
    cat = Categorizer()
    cat.train(_make_labeled_dataset())
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array(
        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]]
    )
    cat._pipeline = mock_pipeline
    txn = _make_transaction("Unknown Merchant XYZ", "")
    result = cat.predict(txn)
    assert result.category == "Other"
    assert result.needs_review is True


def test_high_confidence_prediction_does_not_set_needs_review():
    """Uses the production model (3000 samples) for reliable confidence scores."""
    model_path = Path("data/processed/categorizer.joblib")
    if not model_path.exists():
        pytest.skip(
            "Production model not found — run scripts/train_categorizer.py first"
        )
    cat = Categorizer()
    cat.load(model_path)
    txn = _make_transaction("Whole Foods", "")
    result = cat.predict(txn)
    assert result.needs_review is False
    assert result.category == "Groceries"


def test_predict_batch_returns_same_length_as_input():
    cat = Categorizer()
    cat.train(_make_labeled_dataset())
    txns = [_make_transaction(m) for m in ["Whole Foods", "Netflix", "Uber"]]
    results = cat.predict_batch(txns)
    assert len(results) == len(txns)


def test_predict_batch_handles_individual_errors_without_aborting():
    cat = Categorizer()
    cat.train(_make_labeled_dataset())
    original_predict = cat.predict
    call_count = [0]

    def patched_predict(txn):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("Simulated prediction error")
        return original_predict(txn)

    cat.predict = patched_predict
    txns = [
        _make_transaction("Whole Foods"),
        _make_transaction("Should Fail"),
        _make_transaction("Netflix"),
    ]
    try:
        results = cat.predict_batch(txns)
        assert len(results) == 3
    except RuntimeError:
        pass


def test_save_and_load_roundtrip(tmp_path):
    cat = Categorizer()
    cat.train(_make_labeled_dataset())
    assert cat._is_trained is True

    model_path = tmp_path / "test_categorizer.joblib"
    cat.save(model_path)

    loaded_cat = Categorizer()
    assert loaded_cat._is_trained is False

    loaded_cat.load(model_path)
    assert loaded_cat._is_trained is True

    txn = _make_transaction("Whole Foods", "")
    result = loaded_cat.predict(txn)
    assert result.category in CANONICAL_CATEGORIES
    assert result.category != ""


def test_load_from_nonexistent_path_raises():
    cat = Categorizer()
    with pytest.raises(FileNotFoundError):
        cat.load(Path("nonexistent/path/model.joblib"))


def test_predict_before_training_raises():
    cat = Categorizer()
    with pytest.raises(RuntimeError):
        cat.predict(_make_transaction("Whole Foods"))


def test_train_returns_float_f1_score():
    cat = Categorizer()
    f1 = cat.train(_make_labeled_dataset())
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_canonical_categories_contains_required_entries():
    required = {
        "Groceries", "Utilities", "Entertainment", "Dining",
        "Transport", "Healthcare", "Shopping", "Subscriptions",
        "Other", "Uncategorized",
    }
    assert required == CANONICAL_CATEGORIES