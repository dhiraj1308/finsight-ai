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

    # Build the agent once so session history persists across requests
    # and the Groq client is not re-created on every chat call.
    # The agent is stored inside AppComponents so all application state
    # lives in a single container at app.state.components.
    components.agent = FinancialAgent(
        store=components.store,
        vector_store=components.vector_store,
        forecaster=components.forecaster,
        anomaly_detector=components.anomaly_detector,
    )
    app.state.components = components
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
    from api.services.ingest_service import IngestService

    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".pdf")):
        raise HTTPException(status_code=422, detail="Only PDF and CSV files are supported.")

    store, vector_store, categorizer, anomaly_detector, _ = _get_components()
    service = IngestService(store, vector_store, categorizer, anomaly_detector)

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

        return service.process(transactions, filename, summary.warnings)

    else:
        # CSV path: write to data/raw/, parse from disk, clean up.
        # Derive a filesystem-safe basename from the uploaded filename so that
        # directory-traversal sequences such as "../../.env.csv" or
        # "subdir/evil.csv" cannot escape data/raw/.  The original `filename`
        # variable is intentionally left unchanged so that Transaction.source_file
        # and duplicate-detection behaviour are unaffected.
        safe_filename = Path(filename).name
        if not safe_filename:
            raise HTTPException(status_code=422, detail="Invalid filename.")
        tmp_path = Path("data/raw") / safe_filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(content)

        try:
            transactions, summary = CSVParser().parse(tmp_path)

            if summary.file_errors:
                raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

            return service.process(transactions, filename, summary.warnings)
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
    # Retrieve the agent from the unified AppComponents stored at startup.
    # Fallback: build an agent on the fly when app.state.components is absent
    # (e.g., integration tests that bypass the lifespan and inject components
    # directly — they set components.agent to a MagicMock or a real agent).
    components = getattr(app.state, "components", None)
    if components is not None:
        agent = components.agent
    else:
        agent = None

    if agent is None:
        # Last-resort fallback for scripts/tests without lifespan or injected agent
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
