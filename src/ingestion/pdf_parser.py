import io
import re
import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pathlib import Path

from domain import Transaction
from ingestion.csv_parser import ParseSummary, CSVParser

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PAGES = 100

# Regex for a date token at the start of a transaction line.
# Matches: 08/07/2025  08-07-2025  08 Jul 2025  08-Jul-2025  08-Jul-25
_DATE_RE = re.compile(
    r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
    r"|\d{1,2}[\s\-\/][A-Za-z]{3}[\s\-\/]\d{2,4})"
)
# Regex for a standalone rupee amount (handles Indian commas, optional ₹/$)
_AMOUNT_RE = re.compile(r"[₹$]?\s*(\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?)")


class PDFParser:
    """Parses bank/credit card statement PDF files into Transaction records."""

    def __init__(self):
        # Reuse CSVParser's field-mapping and value-parsing logic so both
        # parsers stay consistent — no duplicated alias/date/amount logic.
        self._field_mapper = CSVParser()

    def parse_bytes(
        self,
        content: bytes,
        filename: str,
        password: str | None = None,
    ) -> tuple[list[Transaction], ParseSummary]:
        """Parse PDF from in-memory bytes, optionally decrypting with a password."""
        summary = ParseSummary()

        if len(content) > MAX_FILE_SIZE_BYTES:
            summary.file_errors.append(
                f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES} bytes: {filename}"
            )
            return [], summary

        try:
            pdf = pdfplumber.open(io.BytesIO(content), password=password)
        except PDFPasswordIncorrect:
            if password is None:
                summary.file_errors.append("PASSWORD_REQUIRED")
            else:
                summary.file_errors.append("PASSWORD_INCORRECT")
            return [], summary
        except Exception:
            summary.file_errors.append(f"Invalid or corrupt PDF file: {filename}")
            return [], summary

        with pdf:
            if len(pdf.pages) > MAX_PAGES:
                summary.file_errors.append(
                    f"File exceeds maximum page count of {MAX_PAGES}: {filename}"
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
                            # Skip credit/balance columns — they are not expenses
                            if canonical_field == "_ignore":
                                continue
                            # For duplicate canonical fields (e.g. two "amount"
                            # columns after mapping), keep the first non-empty
                            # value so Debit wins over Credit/Balance.
                            if canonical_field in canonical_row and canonical_row[canonical_field]:
                                continue
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

                        # Skip sentinel rows like Opening/Closing Balance
                        if any(kw in merchant.lower() for kw in (
                            "opening balance", "closing balance",
                            "brought forward", "carried forward",
                        )):
                            summary.skipped += 1
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
                # No structured tables found — try text-based extraction as fallback
                # (common in Axis Bank and other Indian bank PDFs)
                text_transactions, text_summary = self._parse_text_fallback(pdf)
                if text_transactions:
                    return text_transactions, text_summary
                summary.file_errors.append(
                    "No recognizable transaction table found."
                )
                return [], summary

            return transactions, summary

    def _parse_text_fallback(
        self, pdf: pdfplumber.PDF
    ) -> tuple[list[Transaction], ParseSummary]:
        """Text-based fallback for PDFs where extract_tables() finds nothing.

        Works by scanning each line of raw text for lines that start with a
        recognisable date token, then extracting the last currency amount on
        that line (or the next line) as the transaction amount, with everything
        in between treated as the merchant description.

        This handles Axis Bank credit card statements and similar Indian bank
        PDFs that use a text layout rather than embedded table structures.
        """
        summary = ParseSummary()
        transactions: list[Transaction] = []

        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            i = 0
            while i < len(lines):
                line = lines[i]
                date_match = _DATE_RE.match(line)
                if not date_match:
                    i += 1
                    continue

                raw_date_str = date_match.group(1).strip()
                try:
                    parsed_date = self._field_mapper._parse_date(raw_date_str)
                except ValueError:
                    i += 1
                    continue

                # Everything after the date token on the same line is the start
                # of the description.
                rest = line[date_match.end():].strip()

                # Find the rightmost amount token in `rest`
                amount_matches = list(_AMOUNT_RE.finditer(rest))

                if amount_matches:
                    # Indian bank statements often have: <merchant> <debit> <credit> <balance>
                    # We want the FIRST non-zero amount (debit), not the last (balance).
                    chosen_amt = None
                    for m in amount_matches:
                        val_str = m.group(1).replace(",", "")
                        try:
                            if float(val_str) != 0.0:
                                chosen_amt = m
                                break
                        except ValueError:
                            continue
                    if chosen_amt is None:
                        chosen_amt = amount_matches[0]
                    merchant = rest[: chosen_amt.start()].strip()
                    raw_amount = chosen_amt.group(1)
                else:
                    # Amount may be on the next line — look ahead one line
                    merchant = rest
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        amt_matches_next = list(_AMOUNT_RE.finditer(next_line))
                        if amt_matches_next:
                            raw_amount = amt_matches_next[-1].group(1)
                            i += 1  # consume the lookahead line
                        else:
                            i += 1
                            continue
                    else:
                        i += 1
                        continue

                if not merchant:
                    i += 1
                    continue

                # Skip balance/header sentinel lines
                merchant_lower = merchant.lower()
                if any(kw in merchant_lower for kw in ("opening balance", "closing balance", "brought forward", "carried forward")):
                    i += 1
                    continue

                # If merchant is purely numeric it's a column bleed — skip
                try:
                    float(merchant.replace(",", ""))
                    i += 1
                    continue
                except ValueError:
                    pass

                try:
                    parsed_amount = self._field_mapper._parse_amount(raw_amount)
                except ValueError:
                    i += 1
                    continue

                # Skip zero-amount lines (e.g. header rows that matched the regex)
                if parsed_amount == 0.0:
                    i += 1
                    continue

                transactions.append(
                    Transaction(
                        date=parsed_date,
                        merchant=merchant,
                        amount=parsed_amount,
                        category="",
                    )
                )
                summary.parsed += 1
                i += 1

        return transactions, summary

    def parse(
        self,
        file_path: Path,
        password: str | None = None,
    ) -> tuple[list[Transaction], ParseSummary]:
        """Parse a PDF from disk, optionally decrypting with a password.

        Handles file I/O and size/existence checks, then delegates all
        table-extraction logic to parse_bytes().
        """
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

        content = file_path.read_bytes()
        return self.parse_bytes(content, file_path.name, password=password)
