from datetime import date

import pytest

from ingestion.synthetic_generator import SyntheticGenerator


def test_generate_returns_required_fields_populated():
    gen = SyntheticGenerator()
    txns = gen.generate(n=1, seed=1)

    txn = txns[0]
    assert txn.date is not None
    assert txn.merchant != ""
    assert txn.amount != 0
    assert txn.category != ""


def test_generate_returns_exact_count():
    gen = SyntheticGenerator()
    txns = gen.generate(n=3000, seed=1)

    assert len(txns) == 3000


def test_generate_dates_within_range():
    gen = SyntheticGenerator()
    start = date(2024, 1, 1)
    end = date(2024, 3, 31)
    txns = gen.generate(n=500, start_date=start, end_date=end, seed=1)

    for txn in txns:
        assert start <= txn.date <= end


def test_generate_amounts_within_category_bounds():
    gen = SyntheticGenerator()
    txns = gen.generate(n=500, seed=1)

    for txn in txns:
        min_amt, max_amt = gen.CATEGORIES[txn.category]
        assert min_amt <= txn.amount <= max_amt


def test_same_seed_produces_identical_output():
    gen = SyntheticGenerator()
    txns_a = gen.generate(n=100, seed=42)
    txns_b = gen.generate(n=100, seed=42)

    assert len(txns_a) == len(txns_b)
    for a, b in zip(txns_a, txns_b):
        assert a.date == b.date
        assert a.merchant == b.merchant
        assert a.amount == b.amount
        assert a.category == b.category


def test_different_seeds_produce_different_output():
    gen = SyntheticGenerator()
    txns_a = gen.generate(n=100, seed=1)
    txns_b = gen.generate(n=100, seed=2)

    amounts_a = [t.amount for t in txns_a]
    amounts_b = [t.amount for t in txns_b]
    assert amounts_a != amounts_b


def test_n_zero_raises_value_error():
    gen = SyntheticGenerator()
    with pytest.raises(ValueError):
        gen.generate(n=0)


def test_n_too_large_raises_value_error():
    gen = SyntheticGenerator()
    with pytest.raises(ValueError):
        gen.generate(n=100_001)


def test_start_date_after_end_date_raises_value_error():
    gen = SyntheticGenerator()
    with pytest.raises(ValueError):
        gen.generate(start_date=date(2024, 12, 31), end_date=date(2024, 1, 1))


def test_covers_at_least_eight_categories():
    gen = SyntheticGenerator()
    assert len(gen.CATEGORIES) >= 8


def test_write_csv_creates_file_with_correct_header(tmp_path):
    gen = SyntheticGenerator()
    txns = gen.generate(n=10, seed=1)

    output_path = gen.write_csv(txns, tmp_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line == "date,merchant,amount,category"


def test_write_csv_row_count_matches_transaction_count(tmp_path):
    gen = SyntheticGenerator()
    txns = gen.generate(n=25, seed=1)

    output_path = gen.write_csv(txns, tmp_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    # header + 25 data rows
    assert len(lines) == 26