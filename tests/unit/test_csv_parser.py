from datetime import date
from pathlib import Path

from ingestion.csv_parser import CSVParser
from ingestion.pretty_printer import PrettyPrinter


def _write_csv(tmp_path: Path, content: str, filename: str = "test.csv") -> Path:
    file_path = tmp_path / filename
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_parses_canonical_headers_correctly(tmp_path):
    content = "date,merchant,amount,category\n2024-03-15,Whole Foods,87.43,Groceries\n"
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert len(txns) == 1
    assert txns[0].date == date(2024, 3, 15)
    assert txns[0].merchant == "Whole Foods"
    assert txns[0].amount == 87.43
    assert txns[0].category == "Groceries"
    assert summary.parsed == 1
    assert summary.skipped == 0


def test_maps_alias_headers_to_canonical_fields(tmp_path):
    content = (
        "Transaction Date,Description,Debit,Category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert len(txns) == 1
    assert txns[0].merchant == "Whole Foods"
    assert txns[0].amount == 87.43


def test_alias_mapping_is_case_insensitive(tmp_path):
    content = "TRANSACTION DATE,description,DEBIT,category\n2024-03-15,Netflix,15.99,Entertainment\n"
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert len(txns) == 1
    assert txns[0].merchant == "Netflix"


def test_row_missing_amount_is_skipped_with_warning(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
        "2024-03-16,BadRow,,Dining\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert summary.parsed == 1
    assert summary.skipped == 1
    assert any("Row 2" in w and "amount" in w for w in summary.warnings)


def test_row_with_unparseable_date_is_skipped_with_warning(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "not-a-date,Whole Foods,87.43,Groceries\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert summary.parsed == 0
    assert summary.skipped == 1
    assert any("not-a-date" in w for w in summary.warnings)


def test_row_with_unparseable_amount_is_skipped_with_warning(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "2024-03-15,Whole Foods,not-a-number,Groceries\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert summary.parsed == 0
    assert summary.skipped == 1
    assert any("not-a-number" in w for w in summary.warnings)


def test_empty_file_returns_zero_records_and_file_error(tmp_path):
    file_path = _write_csv(tmp_path, "")

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert txns == []
    assert summary.parsed == 0
    assert len(summary.file_errors) == 1


def test_parse_summary_counts_are_consistent(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
        "2024-03-16,BadRow,,Dining\n"
        "2024-03-17,Netflix,15.99,Entertainment\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    total_data_rows = 3
    assert summary.parsed + summary.skipped == total_data_rows


def test_file_with_bom_is_parsed_correctly(tmp_path):
    """Regression test: Excel/Windows-exported CSVs often include a UTF-8 BOM.
    Without utf-8-sig decoding, the BOM attaches to the first header and
    breaks column alias matching for the date field."""
    content = (
        "Transaction Date,Description,Debit,Category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
    )
    file_path = tmp_path / "bom_test.csv"
    file_path.write_bytes(content.encode("utf-8-sig"))

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert summary.parsed == 1
    assert summary.skipped == 0
    assert txns[0].date == date(2024, 3, 15)
    assert txns[0].merchant == "Whole Foods"


def test_mixed_date_formats_both_parse_correctly(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
        "03/18/2024,Shell Gas,45.20,Transport\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    txns, summary = parser.parse(file_path)

    assert summary.parsed == 2
    assert txns[0].date == date(2024, 3, 15)
    assert txns[1].date == date(2024, 3, 18)


def test_pretty_printer_writes_canonical_header(tmp_path):
    from domain import Transaction

    txns = [
        Transaction(date=date(2024, 3, 15), merchant="Whole Foods", amount=87.43, category="Groceries")
    ]
    printer = PrettyPrinter()
    csv_string = printer.to_csv_string(txns)

    first_line = csv_string.splitlines()[0]
    assert first_line == "date,merchant,amount,category"


def test_round_trip_preserves_all_fields(tmp_path):
    content = (
        "date,merchant,amount,category\n"
        "2024-03-15,Whole Foods,87.43,Groceries\n"
        "2024-03-16,Netflix,15.99,Entertainment\n"
    )
    file_path = _write_csv(tmp_path, content)

    parser = CSVParser()
    printer = PrettyPrinter()

    original_txns, _ = parser.parse(file_path)
    serialized = printer.to_csv_string(original_txns)

    reparsed_path = tmp_path / "reparsed.csv"
    reparsed_path.write_text(serialized, encoding="utf-8")
    reparsed_txns, _ = parser.parse(reparsed_path)

    assert len(original_txns) == len(reparsed_txns)
    for original, reparsed in zip(original_txns, reparsed_txns):
        assert original.date == reparsed.date
        assert original.merchant == reparsed.merchant
        assert original.amount == reparsed.amount
        assert original.category == reparsed.category