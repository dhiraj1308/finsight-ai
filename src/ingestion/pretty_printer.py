import csv
import io
from pathlib import Path

from domain import Transaction


class PrettyPrinter:
    """Serializes Transaction records back into canonical CSV format."""

    CANONICAL_FIELDS = ("date", "merchant", "amount", "category")

    def to_csv_string(self, transactions: list[Transaction]) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(self.CANONICAL_FIELDS)
        for txn in transactions:
            writer.writerow(
                [txn.date.isoformat(), txn.merchant, txn.amount, txn.category]
            )
        return output.getvalue()

    def to_csv(self, transactions: list[Transaction], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_csv_string(transactions), encoding="utf-8")