from unittest.mock import MagicMock, patch

import pytest

from agent.agent import FinancialAgent, OUT_OF_SCOPE_RESPONSE


def _make_agent():
    store = MagicMock()
    vector_store = MagicMock()
    forecaster = MagicMock()
    anomaly_detector = MagicMock()

    with patch.dict("os.environ", {"LLM_API_KEY": "test_key"}):
        with patch("agent.agent.ChatGroq"):
            agent = FinancialAgent(store, vector_store, forecaster, anomaly_detector)
    return agent


def test_out_of_scope_question_returns_canned_response():
    agent = _make_agent()
    result = agent.chat("What is the capital of France?", session_id="s1")
    assert result == OUT_OF_SCOPE_RESPONSE


def test_finance_question_is_not_blocked_by_scope_guard():
    agent = _make_agent()
    agent._executor = MagicMock()
    agent._executor.invoke.return_value = {
        "output": "Total spending: $100.00",
        "intermediate_steps": [],
    }
    result = agent.chat("How much did I spend on groceries?", session_id="s2")
    assert result != OUT_OF_SCOPE_RESPONSE
    assert "100.00" in result


def test_session_history_tracks_question_answer_pairs():
    agent = _make_agent()
    agent._executor = MagicMock()
    agent._executor.invoke.return_value = {
        "output": "Total spending: $50.00",
        "intermediate_steps": [],
    }
    agent.chat("How much did I spend on dining?", session_id="s3")
    history = agent.get_history("s3")
    assert len(history) == 1
    assert history[0][0] == "How much did I spend on dining?"
    assert "50.00" in history[0][1]


def test_session_history_retains_last_five_only():
    agent = _make_agent()
    agent._executor = MagicMock()
    agent._executor.invoke.return_value = {
        "output": "Total: $1.00",
        "intermediate_steps": [],
    }
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
    agent._executor = MagicMock()
    agent._executor.invoke.side_effect = Exception("Unexpected failure")
    result = agent.chat("How much did I spend on groceries?", session_id="s5")
    assert "error" in result.lower()


def test_clean_response_strips_self_correction_preamble():
    agent = _make_agent()
    dirty = "I made a mistake. I should not have called the function again. Here is the correct response: Total spending: $42.00"
    cleaned = agent._clean_response(dirty)
    assert "I made a mistake" not in cleaned
    assert "$42.00" in cleaned


def test_clean_response_leaves_normal_text_unchanged():
    agent = _make_agent()
    normal = "Total spending for groceries: $87.43 across 5 transaction(s)."
    cleaned = agent._clean_response(normal)
    assert cleaned == normal