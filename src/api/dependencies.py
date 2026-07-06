from __future__ import annotations

from dataclasses import dataclass

from anomaly.anomaly_detector import AnomalyDetector
from api.vector_store import VectorStore
from categorization.categorizer import Categorizer
from forecasting.forecaster import Forecaster
from ingestion.transaction_store import TransactionStore


@dataclass
class AppComponents:
    """Container for shared application components."""

    store: TransactionStore
    vector_store: VectorStore
    categorizer: Categorizer
    anomaly_detector: AnomalyDetector
    forecaster: Forecaster