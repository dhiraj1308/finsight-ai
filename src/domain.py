from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Transaction:
    date: date
    merchant: str
    amount: float  # positive = debit/charge
    category: str = ""
    id: Optional[int] = None
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None  # [0.0, 1.0], higher = more anomalous
    needs_review: bool = False
    source_file: str = ""


class VectorStoreIndexError(Exception):
    """Raised when indexing a transaction into the vector store fails."""
    pass