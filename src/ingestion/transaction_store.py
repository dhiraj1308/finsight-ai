import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from domain import Transaction


class TransactionStore:
    """Persists Transaction records in a SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        TEXT    NOT NULL,
                    merchant    TEXT    NOT NULL,
                    amount      REAL    NOT NULL,
                    category    TEXT    NOT NULL DEFAULT '',
                    is_anomaly  INTEGER NOT NULL DEFAULT 0,
                    anomaly_score REAL,
                    source_file TEXT    NOT NULL DEFAULT '',
                    needs_review INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_transaction
                ON transactions(date, merchant, amount, source_file)
                """
            )

    def _row_to_transaction(self, row: sqlite3.Row) -> Transaction:
        return Transaction(
            id=row["id"],
            date=date.fromisoformat(row["date"]),
            merchant=row["merchant"],
            amount=row["amount"],
            category=row["category"],
            is_anomaly=bool(row["is_anomaly"]),
            anomaly_score=row["anomaly_score"],
            source_file=row["source_file"],
            needs_review=bool(row["needs_review"]),
        )

    def insert(self, transactions: list[Transaction]) -> tuple[int, int]:
        inserted = 0
        skipped = 0
        with self._get_connection() as conn:
            for txn in transactions:
                try:
                    conn.execute(
                        """
                        INSERT INTO transactions
                            (date, merchant, amount, category, is_anomaly,
                             anomaly_score, source_file, needs_review)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            txn.date.isoformat(),
                            txn.merchant,
                            txn.amount,
                            txn.category,
                            int(txn.is_anomaly),
                            txn.anomaly_score,
                            txn.source_file,
                            int(txn.needs_review),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate (date, merchant, amount, source_file) combination
                    skipped += 1
        return inserted, skipped

    def query_by_date_range(
        self, start_date: date, end_date: date
    ) -> list[Transaction]:
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must not be after end_date ({end_date})"
            )
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM transactions
                WHERE date >= ? AND date <= ?
                ORDER BY date DESC
                """,
                (start_date.isoformat(), end_date.isoformat()),
            ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    def query_by_category(self, category: str) -> list[Transaction]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM transactions
                WHERE LOWER(category) = LOWER(?)
                ORDER BY date DESC
                """,
                (category,),
            ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    def get_all(self) -> list[Transaction]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY date DESC"
            ).fetchall()
        return [self._row_to_transaction(row) for row in rows]

    def delete(self, transaction_id: int) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM transactions WHERE id = ?", (transaction_id,)
            )