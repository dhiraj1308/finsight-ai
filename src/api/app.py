from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on sys.path regardless of how uvicorn is invoked.
# Works for both:
#   uvicorn src.api.app:app          (project root, src/ not on path yet)
#   python -m uvicorn main:app ...   (main.py already adds src/)
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


# ---------------------------------------------------------------------------
# Router registration — one router per endpoint group.
# No prefixes: public paths remain /health, /ingest, /transactions, etc.
# ---------------------------------------------------------------------------

from api.routers import anomalies, chat, forecast, health, ingest, transactions  # noqa: E402

app.include_router(health.router)
app.include_router(transactions.router)
app.include_router(anomalies.router)
app.include_router(forecast.router)
app.include_router(chat.router)
app.include_router(ingest.router)
