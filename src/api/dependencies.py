from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppComponents:
    """Container for shared application components."""

    store: object
    vector_store: object
    categorizer: object
    anomaly_detector: object
    forecaster: object


def create_components(settings) -> AppComponents:
    """Create and initialise all shared backend components.

    All heavy imports (torch, sentence_transformers, sklearn) are deferred
    to inside this function so they only load when the first request arrives,
    not at module import time. This prevents the Windows torch DLL crash that
    occurs when uvicorn's --reload spawns a worker subprocess before the
    process environment is fully initialised.

    IMPORT ORDER IS CRITICAL ON WINDOWS:
    sentence_transformers/torch MUST be imported before sklearn.
    sklearn loads BLAS/LAPACK native DLLs that mutate the Windows DLL loader
    state, preventing torch's c10.dll from initialising afterwards.
    Always keep torch initialisation first.
    """
    # 1. Force torch + sentence_transformers DLLs to initialise NOW, before
    #    sklearn is imported.  The VectorStore uses lazy model loading, so
    #    without this explicit eager import sklearn would win the DLL race and
    #    cause WinError 1114 when torch is loaded later.
    try:
        import torch  # noqa: F401 — side-effect: initialises c10.dll
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        pass  # non-Windows or torch not installed — skip gracefully

    from api.vector_store import VectorStore

    # 2. sklearn-dependent modules (AnomalyDetector, Categorizer)
    from anomaly.anomaly_detector import AnomalyDetector
    from categorization.categorizer import Categorizer

    # 3. Pure-Python / numpy modules — order-insensitive
    from forecasting.forecaster import Forecaster
    from ingestion.transaction_store import TransactionStore

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
