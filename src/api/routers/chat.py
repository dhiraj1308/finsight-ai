"""Router: POST /chat"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from api.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    # Retrieve the agent from the unified AppComponents stored at startup.
    # Fallback: build an agent on the fly when app.state.components is absent
    # (e.g., integration tests that bypass the lifespan and inject components
    # directly — they set components.agent to a MagicMock or a real agent).
    components = getattr(request.app.state, "components", None)
    if components is not None:
        agent = components.agent
    else:
        agent = None

    if agent is None:
        # Last-resort fallback for scripts/tests without lifespan or injected agent
        from agent.agent import FinancialAgent
        from api.dependencies import get_components

        store, vector_store, _, anomaly_detector, forecaster = get_components(request)
        agent = FinancialAgent(
            store=store,
            vector_store=vector_store,
            forecaster=forecaster,
            anomaly_detector=anomaly_detector,
        )

    try:
        answer = agent.chat(message=body.message, session_id=body.session_id)
    except Exception as exc:
        logger.error(
            "/chat error for session=%s: %s", body.session_id, exc, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    return ChatResponse(answer=answer)
