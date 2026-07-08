from __future__ import annotations

from datetime import date
from typing import Literal, Optional

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


class PasswordErrorResponse(BaseModel):
    error_code: Literal["PASSWORD_REQUIRED", "PASSWORD_INCORRECT"]
    detail: str


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    warnings: list[str] = []
    error_code: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    session_id: str = Field(..., max_length=128)


class ChatResponse(BaseModel):
    answer: str
