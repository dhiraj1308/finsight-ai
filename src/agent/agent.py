"""FinSight AI financial agent.

Prompt budget design
--------------------
Target: stay below 4 500 tokens per LLM call (Groq TPM limit is 6 000).

Token budget breakdown (approximate):
  System prompt       ~120 tokens
  Tool schemas        ~400 tokens  (5 tools × ~80 tokens each)
  User message        ~50 tokens
  Tool results        ~300 tokens  (capped per tool)
  Agent scratchpad    ~200 tokens  (1 iteration max for simple queries)
  Safety margin       ~430 tokens
  ─────────────────────────────────
  Total               ~1 500 tokens  (well inside 4 500 limit)

Key constraints enforced here:
- max_iterations = 2  (limits scratchpad growth)
- return_intermediate_steps = False  (stops tool results re-entering context)
- Each tool output is hard-capped at 800 chars (see tools.py)
- Conversation history is NOT fed back to the LLM (stateless reasoning per turn)
"""
from __future__ import annotations

import logging
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from src.agent.tools import create_tools

logger = logging.getLogger(__name__)

FINANCE_KEYWORDS = {
    "spend", "spent", "spending", "transaction", "transactions",
    "category", "categories", "budget", "forecast", "predict",
    "anomaly", "anomalies", "unusual", "suspicious", "fraud",
    "total", "amount", "cost", "price", "purchase", "payment",
    "groceries", "dining", "transport", "utilities", "entertainment",
    "healthcare", "shopping", "subscriptions", "merchant", "bank",
    "credit", "debit", "statement", "balance", "expense", "income",
    "afford", "save", "saving", "money", "financial", "finance",
    "biggest", "largest", "most", "top", "highest", "increased",
}

OUT_OF_SCOPE_RESPONSE = (
    "I'm FinSight AI, a personal finance assistant. "
    "I can only help with questions about your transactions, spending patterns, "
    "budgets, forecasts, and anomalies. Please ask me something about your finances."
)

# Keep the system prompt short — every token here is repeated on every call.
_SYSTEM_PROMPT = (
    "You are FinSight AI, a personal finance assistant. "
    "Always call a tool to get real data before answering. "
    "Never invent numbers or transactions. "
    "Answer concisely in plain text. "
    "If a tool returns no data, say so honestly."
)

_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Maximum characters kept from a user message before sending to the LLM.
# Prevents a single very long question from spiking token usage.
_MAX_INPUT_CHARS = 400


class FinancialAgent:
    """Stateless-reasoning tool-calling agent for financial Q&A.

    Conversation display history (for the UI) is stored per session but is
    NOT injected into the LLM prompt. This keeps each call's token count
    predictable and avoids Groq's 6 000 TPM limit.
    """

    def __init__(self, store, vector_store, forecaster, anomaly_detector):
        self._store = store
        self._vector_store = vector_store
        self._forecaster = forecaster
        self._anomaly_detector = anomaly_detector
        # session_id -> list[(question, answer)] — display only, not fed to LLM
        self._session_history: dict[str, list[tuple[str, str]]] = {}

        api_key = os.getenv("LLM_API_KEY")
        self._llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
        )

        self._tools = create_tools(store, vector_store, forecaster, anomaly_detector)

        agent = create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=_AGENT_PROMPT,
        )

        self._executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=2,           # limits scratchpad / context growth
            return_intermediate_steps=False,  # don't re-inject tool results
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_finance_question(self, message: str) -> bool:
        """Return True if *message* contains at least one finance keyword."""
        lower = message.lower()
        return any(kw in lower for kw in FINANCE_KEYWORDS)

    def _trim_input(self, message: str) -> str:
        """Truncate *message* to ``_MAX_INPUT_CHARS`` if necessary."""
        if len(message) <= _MAX_INPUT_CHARS:
            return message
        return message[:_MAX_INPUT_CHARS] + "…"

    def _clean_response(self, text: str) -> str:
        """Strip common LLM self-correction artefacts from *text*."""
        markers = [
            "i made a mistake",
            "i should not have",
            "here is the correct response:",
            "let me correct that:",
        ]
        lower = text.lower()
        for marker in markers:
            idx = lower.find(marker)
            if idx != -1:
                remainder = text[idx:]
                colon = remainder.find(":")
                if 0 < colon < 100:
                    return remainder[colon + 1:].strip()
        return text.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        """Return the last 5 (question, answer) pairs for *session_id*."""
        return self._session_history.get(session_id, [])[-5:]

    def chat(self, message: str, session_id: str) -> str:
        """Process one user *message* and return the assistant reply.

        Parameters
        ----------
        message:
            User's question (up to 2 000 chars as enforced by the API layer;
            trimmed further to ``_MAX_INPUT_CHARS`` here for safety).
        session_id:
            Opaque identifier for the current browser session.
        """
        if not self._is_finance_question(message):
            return OUT_OF_SCOPE_RESPONSE

        trimmed = self._trim_input(message)

        try:
            result = self._executor.invoke({"input": trimmed})
            output: str = result.get("output", "")

            if not output or "stopped due to" in output.lower():
                answer = (
                    "I found relevant data but had trouble forming a complete "
                    "response. Please try rephrasing your question."
                )
            else:
                answer = self._clean_response(output)

        except Exception as exc:
            logger.error("Agent error for session %s: %s", session_id, exc)
            answer = (
                "I encountered an error while processing your question. "
                "Please try again or rephrase it."
            )

        # Store display history (never fed back to LLM)
        history = self._session_history.setdefault(session_id, [])
        history.append((message, answer))
        if len(history) > 5:
            self._session_history[session_id] = history[-5:]

        return answer
