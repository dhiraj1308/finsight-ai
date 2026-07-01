from __future__ import annotations

import logging
import os

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from agent.tools import create_tools

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
}

OUT_OF_SCOPE_RESPONSE = (
    "I'm FinSight AI, a personal finance assistant. I can only help with "
    "questions about your transactions, spending patterns, budgets, forecasts, "
    "and anomalies. Please ask me something about your financial data."
)

SYSTEM_PROMPT = """You are FinSight AI, a personal finance assistant.
You have access to the user's real transaction data through tools.
ALWAYS use tools to get real data before answering - never make up numbers or transactions.
If a tool returns no data, say so honestly. Never fabricate financial information.
As soon as a tool returns a result, immediately respond with that information
in plain text. Do not call the same tool twice for the same question."""

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class FinancialAgent:
    """
    Tool-calling agent. Each question is processed independently (stateless
    reasoning) to avoid a known Groq/LangChain compatibility issue where
    replaying tool-call history into the prompt causes malformed function
    call generation on later turns. Display history is kept separately
    per session for conversational context shown to the user, but is not
    fed back into the LLM's reasoning step.
    """

    def __init__(self, store, vector_store, forecaster, anomaly_detector):
        self._store = store
        self._vector_store = vector_store
        self._forecaster = forecaster
        self._anomaly_detector = anomaly_detector
        self._session_history = {}  # session_id -> list of (question, answer)

        api_key = os.getenv("LLM_API_KEY")
        self._llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0,
        )

        self._tools = create_tools(store, vector_store, forecaster, anomaly_detector)

        self._agent = create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=AGENT_PROMPT,
        )

        self._executor = AgentExecutor(
            agent=self._agent,
            tools=self._tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True,
        )

    def _is_finance_question(self, message):
        message_lower = message.lower()
        return any(kw in message_lower for kw in FINANCE_KEYWORDS)

    def _clean_response(self, text):
        markers = [
            "i made a mistake",
            "i should not have",
            "here is the correct response:",
            "let me correct that:",
        ]
        text_lower = text.lower()
        for marker in markers:
            idx = text_lower.find(marker)
            if idx != -1:
                remainder = text[idx:]
                colon_idx = remainder.find(":")
                if colon_idx != -1 and colon_idx < 100:
                    return remainder[colon_idx + 1:].strip()
        return text.strip()

    def get_history(self, session_id):
        """Returns the last 5 (question, answer) pairs for a session."""
        return self._session_history.get(session_id, [])[-5:]

    def chat(self, message, session_id):
        if not self._is_finance_question(message):
            return OUT_OF_SCOPE_RESPONSE

        try:
            result = self._executor.invoke({"input": message})
            output = result.get("output", "")

            if not output or "stopped due to" in output.lower():
                intermediate_steps = result.get("intermediate_steps", [])
                if intermediate_steps:
                    answer = self._clean_response(str(intermediate_steps[-1][1]))
                else:
                    answer = (
                        "I found relevant data but had trouble forming a "
                        "complete response. Please try rephrasing your question."
                    )
            else:
                answer = self._clean_response(output)

        except Exception as e:
            logger.error(f"Agent error for session {session_id}: {e}")
            answer = (
                "I encountered an error while processing your question. "
                "Please try again or rephrase your question."
            )

        if session_id not in self._session_history:
            self._session_history[session_id] = []
        self._session_history[session_id].append((message, answer))
        if len(self._session_history[session_id]) > 5:
            self._session_history[session_id] = self._session_history[session_id][-5:]

        return answer
