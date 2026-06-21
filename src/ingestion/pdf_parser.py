import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pathlib import Path

from domain import Transaction
from ingestion.csv_parser import ParseSummary, CSVParser

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 100


class PDFParser:
    """Parses bank/credit card statement PDF files into Transaction records."""

    def __init__(self):
        # Reuse CSVParser's field-mapping and value-parsing logic so both
        # parsers stay consistent — no duplicated alias/date/amount logic.
        self._field_mapper = CSVParser()

    def parse(self, file_path: Path) -> tuple[list[Transaction], ParseSummary]:
        file_path = Path(file_path)
        summary = ParseSummary()

        if not file_path.exists():
            summary.file_errors.append(f"File not found: {file_path}")
            return [], summary

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            summary.file_errors.append(
                f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES} bytes: {file_path}"
            )
            return [], summary

        try:
            pdf = pdfplumber.open(file_path)
        except PDFPasswordIncorrect:
            summary.file_errors.append(
                "File is password-protected and cannot be read."
            )
            return [], summary
        except Exception:
            summary.file_errors.append(f"Invalid or corrupt PDF file: {file_path}")
            return [], summary

        with pdf:
            if len(pdf.pages) > MAX_PAGES:
                summary.file_errors.append(
                    f"File exceeds maximum page count of {MAX_PAGES}: {file_path}"
                )
                return [], summary

            transactions: list[Transaction] = []
            total_rows_attempted = 0
            found_any_table = False

            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue  # need at least a header + 1 data row

                    header_row = table[0]
                    header_map = {
                        raw_header: self._field_mapper._canonical_field_name(
                            str(raw_header) if raw_header else ""
                        )
                        for raw_header in header_row
                    }

                    for row_index, raw_row in enumerate(table[1:], start=1):
                        total_rows_attempted += 1
                        canonical_row = {}
                        for col_index, raw_value in enumerate(raw_row):
                            raw_header = (
                                header_row[col_index]
                                if col_index < len(header_row)
                                else None
                            )
                            canonical_field = header_map.get(raw_header, "")
                            canonical_row[canonical_field] = (
                                str(raw_value).strip() if raw_value else ""
                            )

                        failing_fields = []

                        merchant = canonical_row.get("merchant", "").strip()
                        if not merchant:
                            failing_fields.append("merchant")

                        raw_date = canonical_row.get("date", "").strip()
                        parsed_date = None
                        if not raw_date:
                            failing_fields.append("date")
                        else:
                            try:
                                parsed_date = self._field_mapper._parse_date(raw_date)
                            except ValueError:
                                failing_fields.append("date")

                        raw_amount = canonical_row.get("amount", "").strip()
                        parsed_amount = None
                        if not raw_amount:
                            failing_fields.append("amount")
                        else:
                            try:
                                parsed_amount = self._field_mapper._parse_amount(
                                    raw_amount
                                )
                            except ValueError:
                                failing_fields.append("amount")

                        if failing_fields:
                            summary.skipped += 1
                            summary.warnings.append(
                                f"Page {page_number}, row {row_index}: "
                                f"failing field(s) {', '.join(failing_fields)}"
                            )
                            continue

                        found_any_table = True
                        category = canonical_row.get("category", "").strip()
                        transactions.append(
                            Transaction(
                                date=parsed_date,
                                merchant=merchant,
                                amount=parsed_amount,
                                category=category,
                            )
                        )
                        summary.parsed += 1

            if not found_any_table and summary.parsed == 0:
                summary.file_errors.append(
                    "No recognizable transaction table found."
                )
                return [], summary

            return transactions, summary