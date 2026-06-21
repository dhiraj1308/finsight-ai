from datetime import date
from pathlib import Path

from ingestion.pdf_parser import PDFParser

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_parses_valid_pdf_statement_correctly():
    parser = PDFParser()
    txns, summary = parser.parse(FIXTURES_DIR / "sample_bank_statement.pdf")

    assert summary.parsed == 4
    assert summary.skipped == 0
    assert summary.file_errors == []

    assert txns[0].date == date(2024, 3, 15)
    assert txns[0].merchant == "Whole Foods"
    assert txns[0].amount == 87.43
    assert txns[0].category == "Groceries"


def test_password_protected_pdf_returns_exact_error_message():
    parser = PDFParser()
    txns, summary = parser.parse(FIXTURES_DIR / "password_protected.pdf")

    assert txns == []
    assert summary.parsed == 0
    assert summary.file_errors == [
        "File is password-protected and cannot be read."
    ]


def test_no_transaction_table_returns_exact_error_message():
    parser = PDFParser()
    txns, summary = parser.parse(FIXTURES_DIR / "no_transaction_table.pdf")

    assert txns == []
    assert summary.parsed == 0
    assert summary.file_errors == ["No recognizable transaction table found."]


def test_nonexistent_file_returns_error_without_raising():
    parser = PDFParser()
    txns, summary = parser.parse(FIXTURES_DIR / "does_not_exist.pdf")

    assert txns == []
    assert summary.parsed == 0
    assert len(summary.file_errors) == 1


def test_parse_summary_counts_match_attempted_rows():
    parser = PDFParser()
    txns, summary = parser.parse(FIXTURES_DIR / "sample_bank_statement.pdf")

    # All 4 rows in our fixture are valid, so parsed should equal total rows
    assert summary.parsed == 4
    assert summary.skipped == 0