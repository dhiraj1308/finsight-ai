from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class TransactionDTO(BaseModel):
    id: Optional[int]
    date: date
    merchant: str
    amount: float
    category: str
    is_anomaly: bool
    anomaly_score: Optional[float]
    needs_review: bool
    source_file: str


class ForecastPointDTO(BaseModel):
    date: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastDTO(BaseModel):
    category: str
    horizon_days: int
    points: list[ForecastPointDTO]


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    warnings: list[str] = []


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    session_id: str = Field(..., max_length=128)


class ChatResponse(BaseModel):
    answer: str