"""Router: GET /forecast/{category}, GET /forecast/aggregate, and GET /forecast/aggregate/next-month"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from api.models import ForecastDTO, ForecastPointDTO

router = APIRouter()


@router.get("/forecast/aggregate/next-month", response_model=ForecastDTO)
async def get_forecast_aggregate_next_month(
    request: Request,
):
    """Forecast total expense spending for the next complete calendar month.

    The forecast period is exactly the calendar month following today's date
    (e.g. if today is 9 Oct 2026, the period is 1–30 Nov 2026).  The number
    of forecast points equals the number of days in that month (28–31).

    Applies the same eligibility guards as the rolling aggregate endpoint:
    income transactions are excluded, at least 14 distinct expense days and
    2 distinct expense calendar months are required.
    """
    from api.dependencies import get_components

    store, _, _, _, forecaster = get_components(request)
    try:
        forecast = forecaster.forecast_aggregate_next_month(store)
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
