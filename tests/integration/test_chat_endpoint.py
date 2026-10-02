"""Integration tests: POST /chat through the real HTTP API.

Exercises the /chat route end-to-end:

  HTTP POST /chat
  → FastAPI request validation (ChatRequest Pydantic model)
  → route
  → app.state.agent.chat(message, session_id)
  → ChatResponse JSON serialization

Two strategies used to prevent ALL real Groq/network calls:

1. Real FinancialAgent with Groq constructor patched (Test 1 only).
   - Uses OUT_OF_SCOPE_RESPONSE path which makes zero LLM calls.
   - patch("agent.agent.Groq") prevents any network initialisation.
   - patch.dict("os.environ", {"LLM_API_KEY": "test_key"}) satisfies the
     os.getenv() call inside __init__ without a real key.

2. MagicMock agent (Tests 2–8).
   - app.state.agent = mock_agent
   - agent.chat.return_value / .side_effect set per test.
   - Zero LLM or network calls possible.

Components:
  - FastAPI handler    — REAL (HTTP layer, Pydantic validation, routing)
  - FinancialAgent     — REAL in Test 1, MagicMock in Tests 2–8
  - TransactionStore   — isolated SQLite (required by AppComponents)
  - VectorStore        — mock (no embedding model)
  - Categorizer        — untrained
  - AnomalyDetector    — mock
  - Forecaster         — mock
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

from agent.agent import OUT_OF_SCOPE_RESPONSE


# ---------------------------------------------------------------------------
# No-op lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Client factory helpers
# ---------------------------------------------------------------------------

def _mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.indexed_ids = frozenset()
    return vs


def _make_client(tmp_path: Path) -> TestClient:
    """
    Return a TestClient with real isolated SQLite store + mocked ML.
    app.state.agent is left unset here; each test sets it explicitly
    before making requests.
    """
    from api.app import app
    from api.dependencies import AppComponents
    from ingestion.transaction_store import TransactionStore
    from categorization.categorizer import Categorizer

    store = TransactionStore(str(tmp_path / "chat_test.db"))

    cat = Categorizer()
    assert not cat._is_trained

    components = AppComponents(
        store=store,
        vector_store=_mock_vector_store(),
        categorizer=cat,
        anomaly_detector=MagicMock(),
        forecaster=MagicMock(),
    )

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        app.state.components = components
        # agent is intentionally NOT set on components here.
        # Each test sets app.state.components.agent to its own mock or real agent.
    finally:
        app.router.lifespan_context = original_lifespan

    return client


def _make_real_agent() -> object:
    """
    Construct a real FinancialAgent with the Groq client patched out.

    patch("agent.agent.Groq") replaces the Groq class before __init__ runs,
    so self._client becomes a MagicMock — no real network client is created.
    patch.dict("os.environ", {"LLM_API_KEY": "test_key"}) satisfies the
    os.getenv("LLM_API_KEY") call without requiring a real API key.

    The agent returned has a real _session_history dict, a real finance gate,
    and a real chat() method — but its LLM client is a harmless MagicMock.
    """
    from agent.agent import FinancialAgent

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("agent.agent.Groq") as mock_groq_cls:
            agent = FinancialAgent(
                store=MagicMock(),
                vector_store=MagicMock(),
                forecaster=MagicMock(),
                anomaly_detector=MagicMock(),
            )
    # Store a reference to the mock Groq instance so tests can assert on it
    agent._groq_mock = mock_groq_cls.return_value
    return agent


# ---------------------------------------------------------------------------
# TEST 1 — Real FinancialAgent, out-of-scope path, zero LLM calls
# ---------------------------------------------------------------------------

class TestChatOutOfScope:
    """
    Exercises real FinancialAgent.chat() through HTTP.
    The message is deliberately non-financial so the finance keyword gate
    returns OUT_OF_SCOPE_RESPONSE before any LLM call is made.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def test_out_of_scope_message_returns_canned_response(self, client):
        """
        POST a message that fails the finance keyword gate.
        The real FinancialAgent.chat() returns OUT_OF_SCOPE_RESPONSE
        without ever calling the (patched) Groq LLM client.
        """
        from api.app import app

        real_agent = _make_real_agent()
        app.state.components.agent = real_agent

        response = client.post(
            "/chat",
            json={"message": "What is the capital of France?", "session_id": "test-session-1"},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()

        assert "answer" in body, "'answer' key must be present in ChatResponse"
        assert isinstance(body["answer"], str)
        assert body["answer"] == OUT_OF_SCOPE_RESPONSE, (
            f"Expected OUT_OF_SCOPE_RESPONSE, got {body['answer']!r}"
        )

        # Verify no LLM call was made — the gate short-circuits before _call_llm
        real_agent._groq_mock.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests 2–8 use MagicMock agent — pure HTTP boundary tests
# ---------------------------------------------------------------------------

class TestChatValidation:
    """
    Pydantic / FastAPI validation on ChatRequest.
    All tests use app.state.agent = MagicMock() so no agent logic runs.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def _set_mock_agent(self, client) -> MagicMock:
        from api.app import app
        mock_agent = MagicMock()
        app.state.components.agent = mock_agent
        return mock_agent

    # ------------------------------------------------------------------
    # TEST 2 — missing message field
    # ------------------------------------------------------------------

    def test_chat_missing_message_returns_422(self, client):
        """POST without 'message' must return HTTP 422 (required field)."""
        mock_agent = self._set_mock_agent(client)

        response = client.post("/chat", json={"session_id": "s1"})

        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )
        assert "detail" in response.json()
        mock_agent.chat.assert_not_called()

    # ------------------------------------------------------------------
    # TEST 3 — missing session_id field
    # ------------------------------------------------------------------

    def test_chat_missing_session_id_returns_422(self, client):
        """POST without 'session_id' must return HTTP 422 (required field)."""
        mock_agent = self._set_mock_agent(client)

        response = client.post("/chat", json={"message": "How much did I spend?"})

        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )
        assert "detail" in response.json()
        mock_agent.chat.assert_not_called()

    # ------------------------------------------------------------------
    # TEST 4 — empty body
    # ------------------------------------------------------------------

    def test_chat_empty_body_returns_422(self, client):
        """POST with empty JSON body must return HTTP 422."""
        self._set_mock_agent(client)

        response = client.post("/chat", json={})

        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )
        assert "detail" in response.json()

    # ------------------------------------------------------------------
    # TEST 5 — message exceeds max_length=2000
    # ------------------------------------------------------------------

    def test_chat_message_over_max_length_returns_422(self, client):
        """
        POST with message length 2001 must return HTTP 422.
        Verifies ChatRequest.message max_length=2000 via the HTTP/Pydantic layer.
        """
        mock_agent = self._set_mock_agent(client)

        response = client.post(
            "/chat",
            json={"message": "x" * 2001, "session_id": "s1"},
        )

        assert response.status_code == 422, (
            f"Expected 422 for 2001-char message, got {response.status_code}"
        )
        assert "detail" in response.json()
        mock_agent.chat.assert_not_called()

    # ------------------------------------------------------------------
    # TEST 6 — session_id exceeds max_length=128
    # ------------------------------------------------------------------

    def test_chat_session_id_over_max_length_returns_422(self, client):
        """
        POST with session_id length 129 must return HTTP 422.
        Verifies ChatRequest.session_id max_length=128 via HTTP.
        """
        mock_agent = self._set_mock_agent(client)

        response = client.post(
            "/chat",
            json={"message": "test", "session_id": "x" * 129},
        )

        assert response.status_code == 422, (
            f"Expected 422 for 129-char session_id, got {response.status_code}"
        )
        assert "detail" in response.json()
        mock_agent.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Tests 7–8 — successful and error paths
# ---------------------------------------------------------------------------

class TestChatSuccessAndError:

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    # ------------------------------------------------------------------
    # TEST 7 — successful request + ChatResponse serialization
    # ------------------------------------------------------------------

    def test_chat_success_returns_answer(self, client):
        """
        POST a valid request with a mocked agent.
        Verifies:
          - HTTP 200
          - ChatResponse 'answer' field presence and value
          - 'session_id' is NOT echoed in the response
          - exact (message, session_id) arguments reach agent.chat()
        """
        from api.app import app

        mock_agent = MagicMock()
        mock_agent.chat.return_value = "You spent Rs.500 on Groceries."
        app.state.components.agent = mock_agent

        response = client.post(
            "/chat",
            json={
                "message": "How much did I spend on groceries?",
                "session_id": "s1",
            },
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()

        assert "answer" in body, "'answer' key must be present in ChatResponse"
        assert body["answer"] == "You spent Rs.500 on Groceries.", (
            f"Expected configured return value, got {body['answer']!r}"
        )
        assert "session_id" not in body, (
            "ChatResponse must not echo session_id; "
            f"got keys: {list(body.keys())}"
        )

        # Verify the exact arguments threaded through HTTP into agent.chat()
        mock_agent.chat.assert_called_once_with(
            message="How much did I spend on groceries?",
            session_id="s1",
        )

    # ------------------------------------------------------------------
    # TEST 8 — agent raises exception → HTTP 500
    # ------------------------------------------------------------------

    def test_chat_agent_error_returns_500(self, client):
        """
        When agent.chat() raises an exception the route must convert it
        to HTTP 500 with detail="Agent error: <message>".
        """
        from api.app import app

        mock_agent = MagicMock()
        mock_agent.chat.side_effect = RuntimeError("LLM unavailable")
        app.state.components.agent = mock_agent

        response = client.post(
            "/chat",
            json={"message": "How much did I spend?", "session_id": "s1"},
        )

        assert response.status_code == 500, (
            f"Expected 500, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert "detail" in body, "500 response must contain 'detail'"
        assert "LLM unavailable" in body["detail"], (
            f"Expected 'LLM unavailable' in detail, got: {body['detail']!r}"
        )
