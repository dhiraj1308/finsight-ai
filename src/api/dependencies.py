from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class AppComponents:
    """Container for shared application components."""

    store: object
    vector_store: object
    categorizer: object
    anomaly_detector: object
    forecaster: object
    agent: object = None   # FinancialAgent — populated by the lifespan after construction


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


# ---------------------------------------------------------------------------
# Shared DTO helper
# ---------------------------------------------------------------------------

def _txn_to_dto(txn) -> "TransactionDTO":
    """Convert a domain Transaction to a TransactionDTO.

    Imported by transactions.py and anomalies.py routers.
    Defined here to avoid duplication across routers.
    The TYPE_CHECKING import keeps this module free of a hard api.models
    dependency at the module level while still allowing the type hint.
    """
    from api.models import TransactionDTO  # local import — avoids circular at load time

    return TransactionDTO(
        id=txn.id,
        date=txn.date,
        merchant=txn.merchant,
        amount=txn.amount,
        category=txn.category,
        is_anomaly=txn.is_anomaly,
        anomaly_score=txn.anomaly_score,
        needs_review=txn.needs_review,
        source_file=txn.source_file,
    )


# ---------------------------------------------------------------------------
# FastAPI component dependency
# ---------------------------------------------------------------------------

def get_components(request) -> Tuple:
    """Return the shared component tuple from app.state.

    Mirrors the behaviour of the former _get_components() in app.py:
      - Primary path: reads app.state.components (set by the lifespan).
      - Fallback path: constructs components on-the-fly for bare-script /
        test usage where the lifespan was bypassed and app.state.components
        was not set.

    Returns a 5-tuple: (store, vector_store, categorizer, anomaly_detector, forecaster)

    Usage in a router:
        from fastapi import Depends, Request
        from api.dependencies import get_components

        @router.get("/example")
        def handler(components=Depends(get_components)):
            store, vector_store, categorizer, anomaly_detector, forecaster = components
    """
    components = getattr(request.app.state, "components", None)
    if components is not None:
        return (
            components.store,
            components.vector_store,
            components.categorizer,
            components.anomaly_detector,
            components.forecaster,
        )

    # Fallback for tests / scripts that bypass the lifespan
    from config import get_settings

    settings = get_settings()
    built = create_components(settings)
    return (
        built.store,
        built.vector_store,
        built.categorizer,
        built.anomaly_detector,
        built.forecaster,
    )
