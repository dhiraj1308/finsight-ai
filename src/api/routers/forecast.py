"""Router: GET /forecast/{category} and GET /forecast/aggregate"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.models import ForecastDTO, ForecastPointDTO

router = APIRouter()


@router.get("/forecast/aggregate", response_model=ForecastDTO)
async def get_forecast_aggregate(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
):
    """Forecast total expense spending across all categories.

    Excludes income transactions (salary, credit, refunds, etc.) and
    aggregates all expense amounts by calendar date before forecasting.
    Returns the same ``ForecastDTO`` schema as the per-category endpoint,
    with ``category='Total Expenses'``.
    """
    from api.dependencies import get_components

    store, _, _, _, forecaster = get_components(request)
    try:
        forecast = forecaster.forecast_aggregate(days, store)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ForecastDTO(
        category=forecast.category,
        horizon_days=forecast.horizon_days,
        points=[
            ForecastPointDTO(
                date=p.date,
                yhat=p.yhat,
                yhat_lower=p.yhat_lower,
                yhat_upper=p.yhat_upper,
            )
            for p in forecast.points
        ],
    )


@router.get("/forecast/{category}", response_model=ForecastDTO)
async def get_forecast(
    request: Request,
    category: str,
    days: int = Query(default=30, ge=1, le=365),
):
    from api.dependencies import get_components

    store, _, _, _, forecaster = get_components(request)
    try:
        forecast = forecaster.forecast_category(category, days, store)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ForecastDTO(
        category=forecast.category,
        horizon_days=forecast.horizon_days,
        points=[
            ForecastPointDTO(
                date=p.date,
                yhat=p.yhat,
                yhat_lower=p.yhat_lower,
                yhat_upper=p.yhat_upper,
            )
            for p in forecast.points
        ],
    )
