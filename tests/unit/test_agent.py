from unittest.mock import MagicMock, patch

import pytest

from agent.agent import FinancialAgent, OUT_OF_SCOPE_RESPONSE


def _make_agent():
    """Build a FinancialAgent with the Groq client fully mocked out."""
    store = MagicMock()
    vector_store = MagicMock()
    forecaster = MagicMock()
    anomaly_detector = MagicMock()

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("agent.agent.Groq"):
            agent = FinancialAgent(store, vector_store, forecaster, anomaly_detector)
    return agent


def _stub_llm(agent, dispatch_reply: str, answer_reply: str = "Answer text."):
    """Configure agent._client to return controlled LLM responses.

    The agent calls the LLM twice for finance questions:
      1. dispatch call  → returns dispatch_reply (JSON tool selection)
      2. synthesise / direct-answer call → returns answer_reply
    """
    completion = MagicMock()
    completion.choices[0].message.content = dispatch_reply
    answer_completion = MagicMock()
    answer_completion.choices[0].message.content = answer_reply
    agent._client.chat.completions.create.side_effect = [
        completion, answer_completion
    ]


def test_out_of_scope_question_returns_canned_response():
    agent = _make_agent()
    result = agent.chat("What is the capital of France?", session_id="s1")
    assert result == OUT_OF_SCOPE_RESPONSE


def test_finance_question_is_not_blocked_by_scope_guard():
    agent = _make_agent()
    # Stub the LLM: dispatch returns "no tool", second call returns a direct answer
    _stub_llm(agent,
              dispatch_reply='{"tool": "none", "args": {}}',
              answer_reply="You spent $100.00 on groceries.")
    result = agent.chat("How much did I spend on groceries?", session_id="s2")
    assert result != OUT_OF_SCOPE_RESPONSE
    assert "100.00" in result


def test_session_history_tracks_question_answer_pairs():
    agent = _make_agent()
    _stub_llm(agent,
              dispatch_reply='{"tool": "none", "args": {}}',
              answer_reply="You spent $50.00 on dining.")
    agent.chat("How much did I spend on dining?", session_id="s3")
    history = agent.get_history("s3")
    assert len(history) == 1
    assert history[0][0] == "How much did I spend on dining?"
    assert "50.00" in history[0][1]


def test_session_history_retains_last_five_only():
    agent = _make_agent()
    # Each chat() needs two LLM calls; build enough stubs for 7 exchanges
    completions = []
    for _ in range(7):
        d = MagicMock()
        d.choices[0].message.content = '{"tool": "none", "args": {}}'
        a = MagicMock()
        a.choices[0].message.content = "Total: $1.00"
        completions.extend([d, a])
    agent._client.chat.completions.create.side_effect = completions

    for i in range(7):
        agent.chat(f"How much did I spend? Question {i}", session_id="s4")
    history = agent.get_history("s4")
    assert len(history) == 5


def test_new_session_starts_with_empty_history():
    agent = _make_agent()
    history = agent.get_history("brand-new-session")
    assert history == []


def test_tool_call_failure_falls_back_to_error_message():
    agent = _make_agent()
    agent._client.chat.completions.create.side_effect = Exception("Unexpected failure")
    result = agent.chat("How much did I spend on groceries?", session_id="s5")
    assert "error" in result.lower()


def test_finance_question_uses_tool_and_synthesises_answer():
    """When dispatch picks a real tool, chat() calls it and synthesises an answer."""
    agent = _make_agent()
    # Stub the tool registry so the tool call returns something concrete
    agent._tools["get_spending_summary"] = lambda **kw: "Spending on Groceries: Rs.87.43"
    _stub_llm(agent,
              dispatch_reply='{"tool": "get_spending_summary", "args": {"category": "Groceries"}}',
              answer_reply="You spent Rs.87.43 on Groceries.")
    result = agent.chat("How much on groceries?", session_id="s6")
    assert result != OUT_OF_SCOPE_RESPONSE
    assert "87.43" in result


def test_out_of_scope_never_calls_llm():
    """Non-finance questions must short-circuit before any LLM call."""
    agent = _make_agent()
    agent.chat("Tell me a joke", session_id="s7")
    agent._client.chat.completions.create.assert_not_called()
