from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure src/ is on sys.path regardless of how uvicorn is invoked.
# Works for both:
#   uvicorn src.api.app:app          (project root, src/ not on path yet)
#   python -m uvicorn main:app ...   (main.py already adds src/)
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    ChatRequest,
    ChatResponse,
    ForecastDTO,
    ForecastPointDTO,
    IngestResponse,
    PasswordErrorResponse,
    TransactionDTO,
)
# CSVParser and PDFParser are imported lazily inside the /ingest handler
# to avoid loading pdfplumber/torch at module startup time.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan: load all heavy components ONCE at startup.
# Storing them in app.state means every request reuses the same objects —
# no re-loading the 11-second SentenceTransformer model per request.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared components before the server starts accepting requests."""
    from dotenv import load_dotenv
    load_dotenv()

    from agent.agent import FinancialAgent
    from api.dependencies import create_components
    from config import get_settings

    logger.info("FinSight AI startup: loading components (this may take ~10s)...")
    settings = get_settings()
    components = create_components(settings)
    app.state.components = components

    # Build the agent once so session history persists across requests
    # and the Groq client is not re-created on every chat call.
    app.state.agent = FinancialAgent(
        store=components.store,
        vector_store=components.vector_store,
        forecaster=components.forecaster,
        anomaly_detector=components.anomaly_detector,
    )
    logger.info("FinSight AI startup complete.")

    yield  # server is running — handle requests

    # Shutdown: nothing to clean up for now
    logger.info("FinSight AI shutdown.")


app = FastAPI(
    title="FinSight AI",
    description="Agentic Personal Finance Intelligence Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_components():
    """Return the shared AppComponents loaded at startup.

    Falls back to creating components on-the-fly if app.state is not
    yet populated (e.g. during testing without the lifespan context).
    """
    from fastapi import Request  # local import to avoid circular at module load

    # Access via app.state — populated once at startup
    components = getattr(app.state, "components", None)
    if components is not None:
        return (
            components.store,
            components.vector_store,
            components.categorizer,
            components.anomaly_detector,
            components.forecaster,
        )

    # Fallback for tests / scripts that call endpoints directly
    from api.dependencies import create_components
    from config import get_settings

    settings = get_settings()
    components = create_components(settings)
    return (
        components.store,
        components.vector_store,
        components.categorizer,
        components.anomaly_detector,
        components.forecaster,
    )


def _txn_to_dto(txn) -> TransactionDTO:
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


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), password: str | None = Form(None)):
    from ingestion.csv_parser import CSVParser
    from ingestion.pdf_parser import PDFParser

    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".pdf")):
        raise HTTPException(status_code=422, detail="Only PDF and CSV files are supported.")

    store, vector_store, categorizer, anomaly_detector, _ = _get_components()

    content = await file.read()

    if filename.endswith(".pdf"):
        transactions, summary = PDFParser().parse_bytes(content, filename, password=password)

        if summary.file_errors:
            if "PASSWORD_REQUIRED" in summary.file_errors:
                return JSONResponse(
                    status_code=422,
                    content=PasswordErrorResponse(
                        error_code="PASSWORD_REQUIRED",
                        detail="This PDF is password-protected. Please supply the decryption password.",
                    ).model_dump(),
                )
            if "PASSWORD_INCORRECT" in summary.file_errors:
                return JSONResponse(
                    status_code=422,
                    content=PasswordErrorResponse(
                        error_code="PASSWORD_INCORRECT",
                        detail="Incorrect password. Please try again.",
                    ).model_dump(),
                )
            raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

        for txn in transactions:
            txn.source_file = filename

        if categorizer._is_trained:
            transactions = categorizer.predict_batch(transactions)

        inserted, skipped = store.insert(transactions)

        # Index ONLY transactions that are not yet in the vector store.
        # Querying already-indexed IDs avoids re-embedding the entire database
        # on every upload — only genuinely new rows are embedded.
        already_indexed: set[int] = vector_store.indexed_ids
        all_txns = store.get_all()
        for txn in all_txns:
            if txn.id is not None and txn.id not in already_indexed:
                try:
                    vector_store.index(txn)
                except Exception as e:
                    logger.warning(f"Vector indexing failed for {txn.id}: {e}")

        try:
            if len(all_txns) >= 10:
                anomaly_detector.fit_and_score(store)
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        return IngestResponse(ingested=inserted, skipped=skipped, warnings=summary.warnings[:10])

    else:
        # CSV path: write to data/raw/, parse from disk, clean up
        tmp_path = Path(f"data/raw/{filename}")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(content)

        try:
            transactions, summary = CSVParser().parse(tmp_path)

            if summary.file_errors:
                raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

            for txn in transactions:
                txn.source_file = filename

            if categorizer._is_trained:
                transactions = categorizer.predict_batch(transactions)

            inserted, skipped = store.insert(transactions)

            # Index ONLY transactions that are not yet in the vector store.
            already_indexed = vector_store.indexed_ids
            all_txns = store.get_all()
            for txn in all_txns:
                if txn.id is not None and txn.id not in already_indexed:
                    try:
                        vector_store.index(txn)
                    except Exception as e:
                        logger.warning(f"Vector indexing failed for {txn.id}: {e}")

            try:
                if len(all_txns) >= 10:
                    anomaly_detector.fit_and_score(store)
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

            return IngestResponse(ingested=inserted, skipped=skipped, warnings=summary.warnings[:10])
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                logger.warning(f"Could not delete temp file (locked): {tmp_path}")


@app.get("/transactions", response_model=list[TransactionDTO])
async def get_transactions(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    store, _, _, _, _ = _get_components()

    try:
        if start_date and end_date:
            from datetime import date
            txns = store.query_by_date_range(date.fromisoformat(start_date), date.fromisoformat(end_date))
        elif category:
            txns = store.query_by_category(category)
        else:
            txns = store.get_all()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return [_txn_to_dto(t) for t in txns]


@app.get("/anomalies", response_model=list[TransactionDTO])
async def get_anomalies():
    store, _, _, anomaly_detector, _ = _get_components()
    anomalies = anomaly_detector.get_anomalies(store)
    return [_txn_to_dto(t) for t in anomalies]


@app.get("/forecast/{category}", response_model=ForecastDTO)
async def get_forecast(category: str, days: int = Query(default=30, ge=1, le=365)):
    store, _, _, _, forecaster = _get_components()
    try:
        forecast = forecaster.forecast_category(category, days, store)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ForecastDTO(
        category=forecast.category,
        horizon_days=forecast.horizon_days,
        points=[
            ForecastPointDTO(date=p.date, yhat=p.yhat, yhat_lower=p.yhat_lower, yhat_upper=p.yhat_upper)
            for p in forecast.points
        ],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Use the shared agent cached at startup — preserves session history
    # and avoids re-creating the Groq client on every request.
    agent = getattr(app.state, "agent", None)
    if agent is None:
        # Fallback: build agent on the fly (test/script usage without lifespan)
        from agent.agent import FinancialAgent
        store, vector_store, _, anomaly_detector, forecaster = _get_components()
        agent = FinancialAgent(
            store=store,
            vector_store=vector_store,
            forecaster=forecaster,
            anomaly_detector=anomaly_detector,
        )

    try:
        answer = agent.chat(message=request.message, session_id=request.session_id)
    except Exception as exc:
        logger.error("/chat error for session=%s: %s", request.session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinSight AI"}
