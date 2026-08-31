"""FinSight AI financial agent — direct-dispatch implementation.

Architecture
------------
We intentionally do NOT use LangChain's AgentExecutor / create_tool_calling_agent.
Reason: langchain-groq 0.1.x + llama-3.1-8b-instant produces
  'Failed to call a function. Please adjust your prompt.'
for multi-tool schemas with default arguments (a known upstream bug).

Instead we use a two-step loop:
  1. Send a structured prompt to the LLM asking it to PICK one tool and
     supply its arguments as a single JSON object on one line.
  2. Parse the JSON, call the Python function directly, inject the result
     back as a system message, and ask the LLM to produce a final answer.

This gives us:
- Zero dependency on LangChain function-calling (avoids the Groq 400 bug)
- Full control over what is sent to the model (prevents token overflow)
- ASCII-safe tool outputs (prevents Windows cp1252 UnicodeEncodeError)
- Meaningful fallback messages instead of generic errors
"""
from __future__ import annotations

import json
import logging
import os
import re

from groq import Groq

from .tools import build_tool_registry, TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finance keyword gate — keeps non-finance questions out of the LLM pipeline
# ---------------------------------------------------------------------------
FINANCE_KEYWORDS: frozenset[str] = frozenset({
    "spend", "spent", "spending", "paid", "pay", "payment",
    "bought", "buy", "purchase", "purchased", "charge", "charged", "cost",
    "transaction", "transactions", "expense", "expenses", "bill", "bills",
    "income", "salary", "balance", "amount", "total", "budget", "money",
    "cash", "finance", "financial", "fund",
    "dining", "food", "groceries", "grocery", "transport", "transportation",
    "fuel", "utilities", "utility", "entertainment", "healthcare", "health",
    "shopping", "subscriptions", "subscription", "travel", "insurance",
    "medical", "restaurant", "eating",
    "category", "categories", "merchant", "merchants", "forecast", "predict",
    "anomaly", "anomalies", "unusual", "suspicious", "fraud",
    "trend", "average", "summary", "breakdown", "statistics",
    "highest", "lowest", "biggest", "largest", "most", "top",
    "today", "yesterday", "week", "month", "year", "last", "this",
    "bank", "credit", "debit", "statement", "saving", "savings",
    "cashflow", "net", "gross", "invest", "investment",
})

OUT_OF_SCOPE_RESPONSE = (
    "I'm FinSight AI, a personal finance assistant. "
    "I can only help with questions about your transactions, spending, "
    "forecasts, budgets, and anomalies."
)

_MAX_INPUT_CHARS = 400

# Number of most-recent conversation turns (user + assistant pairs) to include
# in the prompt as context.  3 turns ≈ 6 messages keeps the context window
# small while still covering typical follow-up question chains.
_HISTORY_TURNS = 3

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DISPATCH_PROMPT = """\
You are FinSight AI. Given the user question, choose ONE tool to call.

Available tools:
{tool_descriptions}
{history_context}
Current user question: {question}

Reply with ONLY a single JSON object on one line. Nothing else. Example:
{{"tool": "get_spending_summary", "args": {{"category": "Dining"}}}}

If no tool is needed, reply:
{{"tool": "none", "args": {{}}}}
"""

_ANSWER_PROMPT = """\
You are FinSight AI, a concise personal finance assistant.
{history_context}
Current user question: {question}

Data retrieved by tool '{tool_name}':
{tool_result}

Write a clear, factual 1-3 sentence answer using ONLY the data above.
Do not invent numbers. If the data says no transactions were found, say so.
"""


class FinancialAgent:
    """Direct-dispatch financial agent — no LangChain AgentExecutor."""

    def __init__(self, store, vector_store, forecaster, anomaly_detector):
        self._store = store
        self._session_history: dict[str, list[tuple[str, str]]] = {}

        api_key = os.getenv("LLM_API_KEY")
        self._client = Groq(api_key=api_key)
        self._model = "openai/gpt-oss-20b"
        # Build tool registry: name -> callable
        self._tools = build_tool_registry(
            store, vector_store, forecaster, anomaly_detector
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_finance_question(self, message: str) -> bool:
        lower = message.lower()
        return any(kw in lower for kw in FINANCE_KEYWORDS)

    def _session_has_financial_context(self, session_id: str) -> bool:
        """Return True if this session already contains at least one exchange
        whose user message contained a finance keyword.

        Used to allow short contextual follow-ups (e.g. "What about that?" or
        "And last month?") that lack finance keywords themselves but clearly
        continue a financial conversation.

        Only the user side of prior turns is checked — not assistant answers —
        so that canned phrases in assistant responses never by themselves qualify
        a session as "financial".

        Session isolation is guaranteed: only this session's history is checked.
        """
        prior_pairs = self._session_history.get(session_id, [])
        return any(
            self._is_finance_question(user_msg)
            for user_msg, _assistant_msg in prior_pairs
        )

    # Interrogative / referential words that signal a follow-up question.
    # A message lacking finance keywords is only treated as a contextual
    # follow-up when it contains at least one of these markers AND is short
    # enough (≤ _MAX_FOLLOWUP_WORDS words) to be a plausible follow-up rather
    # than a fresh interrogative sentence about an unrelated subject.
    _FOLLOWUP_MARKERS: frozenset[str] = frozenset({
        "what", "which", "how", "when", "where", "who", "why",
        "that", "those", "this", "these", "it", "them", "its",
        "and", "but", "also", "too", "again", "more", "else",
        "previous", "prior", "before", "instead", "other", "another",
        "compare", "versus", "vs", "difference",
    })

    # Messages with more words than this threshold are unlikely to be simple
    # follow-ups — they are more likely to be independent new questions that
    # happen to use referential words (e.g. "What is the capital of France?").
    _MAX_FOLLOWUP_WORDS: int = 5

    def _looks_like_followup(self, message: str) -> bool:
        """Return True if the message looks like a short contextual follow-up.

        Two conditions must both hold:
          1. The message is short (≤ _MAX_FOLLOWUP_WORDS words) — longer
             messages are likely independent questions even if they start with
             "what" or "how".
          2. The message contains at least one follow-up/referential marker
             word — interrogative or referential words that indicate the
             message continues a prior conversation thread.

        Together these criteria allow "What about that?" (3 words) and
        "And last month?" (3 words) while blocking "What is the capital of
        France?" (7 words) even inside a session with financial history.
        """
        words = message.strip().split()
        if len(words) > self._MAX_FOLLOWUP_WORDS:
            return False
        lower = message.lower()
        return any(marker in lower for marker in self._FOLLOWUP_MARKERS)

    def _trim(self, text: str, limit: int = _MAX_INPUT_CHARS) -> str:
        return text if len(text) <= limit else text[:limit] + "..."

    def _format_history_context(self, session_id: str) -> str:
        """Return a compact, labelled conversation-history block for prompts.

        Fetches the most recent ``_HISTORY_TURNS`` (user, assistant) pairs for
        the session and formats them as a clearly labelled section so the LLM
        can use prior context (e.g. to resolve follow-up references like
        "what about last month?").

        Returns an empty string when there is no history yet, so first-message
        behaviour is completely unchanged.

        Session isolation is guaranteed because history is always looked up by
        ``session_id`` — different sessions never share a history bucket.
        """
        pairs = self._session_history.get(session_id, [])
        if not pairs:
            return ""
        recent = pairs[-_HISTORY_TURNS:]
        lines = ["\nConversation context (oldest to newest):"]
        for user_msg, assistant_msg in recent:
            # Trim individual turns to avoid runaway token growth from long
            # prior answers being recycled into every subsequent prompt.
            lines.append(f"User: {self._trim(user_msg, 200)}")
            lines.append(f"Assistant: {self._trim(assistant_msg, 200)}")
        lines.append("")  # trailing blank line for visual separation
        return "\n".join(lines)

    def _call_llm(self, prompt: str, max_tokens: int = 512) -> str:
        """Call the Groq LLM and return the content string."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _dispatch(self, question: str, history_context: str) -> tuple[str, str]:
        """Ask the LLM which tool to call and what args to pass.

        Returns (tool_name, tool_result).
        """
        prompt = _DISPATCH_PROMPT.format(
            tool_descriptions=TOOL_DESCRIPTIONS,
            history_context=history_context,
            question=question,
        )
        raw = self._call_llm(prompt, max_tokens=150).strip()
        logger.info("[dispatch] raw LLM reply: %r", raw)

        # Extract the first JSON object from the response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("[dispatch] No JSON found in LLM reply: %r", raw)
            return "none", ""

        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("[dispatch] JSON parse error: %s  raw=%r", exc, raw)
            return "none", ""

        tool_name: str = parsed.get("tool", "none")
        args: dict = parsed.get("args", {})

        if tool_name == "none" or tool_name not in self._tools:
            logger.info("[dispatch] tool_name=%r not in registry", tool_name)
            return "none", ""

        logger.info("[dispatch] calling tool=%r args=%r", tool_name, args)
        try:
            result: str = self._tools[tool_name](**args)
        except TypeError as exc:
            # Wrong arguments — call with no args as fallback
            logger.warning(
                "[dispatch] TypeError calling %s(%s): %s; retrying with no args",
                tool_name, args, exc,
            )
            try:
                result = self._tools[tool_name]()
            except Exception as exc2:
                logger.error("[dispatch] fallback also failed: %s", exc2)
                result = f"Error calling tool '{tool_name}'."
        except Exception as exc:
            logger.error("[dispatch] tool %s error: %s", tool_name, exc, exc_info=True)
            result = f"Error calling tool '{tool_name}'."

        logger.info("[dispatch] tool result: %r", result[:120])
        return tool_name, result

    def _synthesize(
        self, question: str, tool_name: str, tool_result: str, history_context: str
    ) -> str:
        """Ask the LLM to formulate a final answer given the tool result."""
        prompt = _ANSWER_PROMPT.format(
            history_context=history_context,
            question=question,
            tool_name=tool_name,
            tool_result=tool_result,
        )
        return self._call_llm(prompt, max_tokens=256).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[tuple[str, str]]:
        return self._session_history.get(session_id, [])[-5:]

    def chat(self, message: str, session_id: str) -> str:
        """Process one user message and return the assistant reply."""
        # Finance gate: allow if the message itself contains finance keywords,
        # OR if it is a contextual follow-up in an already-financial session.
        # A question is treated as a contextual follow-up only when ALL THREE hold:
        #   1. it lacks finance keywords itself,
        #   2. it contains at least one follow-up/referential marker word, AND
        #   3. at least one prior user message in this session had finance keywords.
        # This allows "What about that?" or "And last month?" mid-conversation
        # while still blocking unrelated declarative statements ("What is the
        # capital of France?") even after a financial conversation has started.
        is_followup_in_financial_session = (
            self._looks_like_followup(message)
            and self._session_has_financial_context(session_id)
        )
        if not self._is_finance_question(message) and not is_followup_in_financial_session:
            return OUT_OF_SCOPE_RESPONSE

        question = self._trim(message)
        logger.info("[chat] session=%s question=%r", session_id, question)

        # Build conversation context BEFORE adding the current message so that
        # the current turn is not included in its own context window.
        history_context = self._format_history_context(session_id)

        try:
            tool_name, tool_result = self._dispatch(question, history_context)

            if tool_name == "none" or not tool_result:
                # LLM decided no tool needed — answer directly, with context.
                ctx_section = history_context if history_context else ""
                prompt = (
                    f"You are FinSight AI, a concise personal finance assistant.\n"
                    f"{ctx_section}"
                    f"Current user question: {question}\n\n"
                    f"Answer concisely in 1-3 sentences."
                )
                answer = self._call_llm(prompt, max_tokens=200).strip()
            else:
                answer = self._synthesize(question, tool_name, tool_result, history_context)

            if not answer:
                answer = tool_result or "I could not retrieve relevant data. Please try rephrasing."

            logger.info("[chat] session=%s answer=%r", session_id, answer[:120])

        except Exception as exc:
            logger.error("[chat] error session=%s: %s", session_id, exc, exc_info=True)
            answer = (
                "I encountered an error processing your question. "
                "Please try again."
            )

        history = self._session_history.setdefault(session_id, [])
        history.append((message, answer))
        if len(history) > 10:
            self._session_history[session_id] = history[-10:]

        return answer
