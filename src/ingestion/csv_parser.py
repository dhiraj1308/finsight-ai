import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from domain import Transaction


@dataclass
class ParseSummary:
    parsed: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    file_errors: list[str] = field(default_factory=list)


class CSVParser:
    """Parses bank/credit card statement CSV files into Transaction records."""

    COLUMN_ALIASES: dict[str, str] = {
        "description": "merchant",
        "narration": "merchant",
        "details": "merchant",
        "debit": "amount",
        "withdrawal": "amount",
        "charge": "amount",
        "transaction date": "date",
        "posted date": "date",
        "trans. date": "date",
    }

    def _canonical_field_name(self, raw_column_name: str) -> str:
        """Maps a raw CSV column header to its canonical field name."""
        normalized = raw_column_name.strip().lower()
        if normalized in ("date", "merchant", "amount", "category"):
            return normalized
        return self.COLUMN_ALIASES.get(normalized, normalized)

    def _parse_date(self, raw_value: str) -> date:
        """Tries common date formats; raises ValueError if none match."""
        raw_value = raw_value.strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(raw_value, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unparseable date: {raw_value}")

    def _parse_amount(self, raw_value: str) -> float:
        """Parses a numeric amount, stripping common currency symbols/commas."""
        cleaned = raw_value.strip().replace("$", "").replace(",", "")
        return float(cleaned)

    def parse(self, file_path: Path) -> tuple[list[Transaction], ParseSummary]:
        file_path = Path(file_path)
        summary = ParseSummary()

        try:
            raw_text = file_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            summary.file_errors.append(
                f"File could not be decoded as UTF-8: {file_path}"
            )
            return [], summary
        except FileNotFoundError:
            summary.file_errors.append(f"File not found: {file_path}")
            return [], summary

        if not raw_text.strip():
            summary.file_errors.append(f"File is empty: {file_path}")
            return [], summary

        reader = csv.DictReader(raw_text.splitlines())
        if reader.fieldnames is None:
            summary.file_errors.append(f"No header row found in file: {file_path}")
            return [], summary

        # Build a mapping from this file's actual headers -> canonical field names
        header_map = {
            raw_header: self._canonical_field_name(raw_header)
            for raw_header in reader.fieldnames
        }

        transactions: list[Transaction] = []

        for row_index, raw_row in enumerate(reader, start=1):
            canonical_row = {
                header_map[raw_header]: value for raw_header, value in raw_row.items()
            }

            merchant = canonical_row.get("merchant", "").strip()
            if not merchant:
                summary.skipped += 1
                summary.warnings.append(
                    f"Row {row_index}: missing field 'merchant'"
                )
                continue

            raw_date = canonical_row.get("date", "").strip()
            if not raw_date:
                summary.skipped += 1
                summary.warnings.append(f"Row {row_index}: missing field 'date'")
                continue
            try:
                parsed_date = self._parse_date(raw_date)
            except ValueError:
                summary.skipped += 1
                summary.warnings.append(
                    f"Row {row_index}: bad value '{raw_date}'"
                )
                continue

            raw_amount = canonical_row.get("amount", "").strip()
            if not raw_amount:
                summary.skipped += 1
                summary.warnings.append(f"Row {row_index}: missing field 'amount'")
                continue
            try:
                parsed_amount = self._parse_amount(raw_amount)
            except ValueError:
                summary.skipped += 1
                summary.warnings.append(
                    f"Row {row_index}: bad value '{raw_amount}'"
                )
                continue

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

        return transactions, summary