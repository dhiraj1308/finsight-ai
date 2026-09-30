"""Router: GET /anomalies"""
from __future__ import annotations

from fastapi import APIRouter, Request

from api.dependencies import _txn_to_dto, get_components
from api.models import TransactionDTO

router = APIRouter()


@router.get("/anomalies", response_model=list[TransactionDTO])
async def get_anomalies(request: Request):
    store, _, _, anomaly_detector, _ = get_components(request)
    anomalies = anomaly_detector.get_anomalies(store)
    return [_txn_to_dto(t) for t in anomalies]
