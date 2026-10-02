"""Router: GET /transactions"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.dependencies import _txn_to_dto, get_components
from api.models import TransactionDTO

router = APIRouter()


@router.get("/transactions", response_model=list[TransactionDTO])
async def get_transactions(
    request: Request,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    store, _, _, _, _ = get_components(request)

    try:
        if start_date and end_date:
            from datetime import date
            txns = store.query_by_date_range(
                date.fromisoformat(start_date), date.fromisoformat(end_date)
            )
        elif category:
            txns = store.query_by_category(category)
        else:
            txns = store.get_all()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return [_txn_to_dto(t) for t in txns]
