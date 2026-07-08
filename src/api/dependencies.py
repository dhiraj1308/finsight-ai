from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.anomaly.anomaly_detector import AnomalyDetector
from .vector_store import VectorStore
from src.categorization.categorizer import Categorizer
from src.forecasting.forecaster import Forecaster
from src.ingestion.transaction_store import TransactionStore


@dataclass
class AppComponents:
    """Container for shared application components."""

    store: TransactionStore
    vector_store: VectorStore
    categorizer: Categorizer
    anomaly_detector: AnomalyDetector
    forecaster: Forecaster


def create_components(settings) -> AppComponents:
    """
    Create and initialize all shared backend components.
    """

    store = TransactionStore(settings.SQLITE_DB_PATH)

    vector_store = VectorStore(
        persist_dir=settings.CHROMA_PERSIST_DIR,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
    )

    categorizer = Categorizer()

    model_path = Path("data/processed/categorizer.joblib")

    if model_path.exists():
        categorizer.load(model_path)

    anomaly_detector = AnomalyDetector()

    forecaster = Forecaster()

    return AppComponents(
        store=store,
        vector_store=vector_store,
        categorizer=categorizer,
        anomaly_detector=anomaly_detector,
        forecaster=forecaster,
    )
