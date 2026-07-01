from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from anomaly.anomaly_detector import AnomalyDetector
from api.models import (
    ChatRequest,
    ChatResponse,
    ForecastDTO,
    ForecastPointDTO,
    IngestResponse,
    TransactionDTO,
)
from api.vector_store import VectorStore
from categorization.categorizer import Categorizer
from forecasting.forecaster import Forecaster
from ingestion.csv_parser import CSVParser
from ingestion.pdf_parser import PDFParser
from ingestion.transaction_store import TransactionStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSight AI",
    description="Agentic Personal Finance Intelligence Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_components():
    from config import get_settings
    settings = get_settings()

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

    return store, vector_store, categorizer, anomaly_detector, forecaster


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
async def ingest(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".pdf")):
        raise HTTPException(status_code=422, detail="Only PDF and CSV files are supported.")

    store, vector_store, categorizer, anomaly_detector, _ = _get_components()

    content = await file.read()
    tmp_path = Path(f"data/raw/{filename}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(content)

    try:
        if filename.endswith(".csv"):
            parser = CSVParser()
        else:
            parser = PDFParser()

        transactions, summary = parser.parse(tmp_path)

        if summary.file_errors:
            raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

        for txn in transactions:
            txn.source_file = filename

        if categorizer._is_trained:
            transactions = categorizer.predict_batch(transactions)

        inserted, skipped = store.insert(transactions)

        all_txns = store.get_all()
        for txn in all_txns:
            if txn.id is not None:
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
    from dotenv import load_dotenv
    load_dotenv()

    from agent.agent import FinancialAgent
    from anomaly.anomaly_detector import AnomalyDetector
    from forecasting.forecaster import Forecaster

    store, vector_store, _, _, _ = _get_components()
    anomaly_detector = AnomalyDetector()
    forecaster = Forecaster()

    agent = FinancialAgent(
        store=store,
        vector_store=vector_store,
        forecaster=forecaster,
        anomaly_detector=anomaly_detector,
    )

    answer = agent.chat(message=request.message, session_id=request.session_id)
    return ChatResponse(answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FinSight AI"}
