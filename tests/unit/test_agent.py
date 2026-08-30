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


# ---------------------------------------------------------------------------
# Conversation-history tests
# Verifies that prior turns are stored, bounded, isolated per session,
# and actually supplied to the LLM on subsequent requests.
# ---------------------------------------------------------------------------

from agent.agent import _HISTORY_TURNS


def _stub_llm_sequence(agent, *reply_pairs):
    """Configure agent._client to return a sequence of (dispatch, answer) pairs.

    Each element of reply_pairs is a (dispatch_reply, answer_reply) tuple.
    """
    completions = []
    for dispatch_reply, answer_reply in reply_pairs:
        d = MagicMock()
        d.choices[0].message.content = dispatch_reply
        a = MagicMock()
        a.choices[0].message.content = answer_reply
        completions.extend([d, a])
    agent._client.chat.completions.create.side_effect = completions


# TEST 1 — history is stored after each exchange
def test_history_stored_after_multiple_messages():
    """Each (user, assistant) pair is appended to the session history."""
    agent = _make_agent()
    _stub_llm_sequence(
        agent,
        ('{"tool": "none", "args": {}}', "You spent Rs.2450 on Dining."),
        ('{"tool": "none", "args": {}}', "You spent Rs.3200 on Groceries."),
    )
    agent.chat("How much on dining?", session_id="hist-1")
    agent.chat("How much on groceries?", session_id="hist-1")

    stored = agent._session_history["hist-1"]
    assert len(stored) == 2
    assert stored[0][0] == "How much on dining?"
    assert "2450" in stored[0][1]
    assert stored[1][0] == "How much on groceries?"
    assert "3200" in stored[1][1]


# TEST 2 — history reaches the LLM on the second request
def test_history_included_in_llm_prompt_on_second_request():
    """The dispatch prompt sent for the second question must contain the
    first exchange from conversation history."""
    agent = _make_agent()
    _stub_llm_sequence(
        agent,
        ('{"tool": "none", "args": {}}', "You spent Rs.2450 on Dining."),
        ('{"tool": "none", "args": {}}', "Last month you spent Rs.1800 on Dining."),
    )

    agent.chat("How much did I spend on dining?", session_id="hist-2")
    agent.chat("What about last month?", session_id="hist-2")

    # The second dispatch call is the third call overall (d1, a1, d2, a2).
    all_calls = agent._client.chat.completions.create.call_args_list
    assert len(all_calls) >= 3, "Expected at least 3 LLM calls by the second message"

    second_dispatch_call = all_calls[2]  # 0=dispatch1, 1=answer1, 2=dispatch2
    prompt_sent = second_dispatch_call[1]["messages"][0]["content"]

    # Prior user turn must appear in the prompt
    assert "How much did I spend on dining?" in prompt_sent, (
        "Previous user message must appear in the dispatch prompt for the follow-up"
    )
    # Prior assistant answer must appear in the prompt
    assert "2450" in prompt_sent, (
        "Previous assistant answer must appear in the dispatch prompt for the follow-up"
    )


# TEST 3 — session isolation
def test_session_history_is_isolated_per_session_id():
    """History from session A must never appear in prompts for session B."""
    agent = _make_agent()
    _stub_llm_sequence(
        agent,
        # session A first message
        ('{"tool": "none", "args": {}}', "Session A answer: Rs.5000 on Travel."),
        # session B first message
        ('{"tool": "none", "args": {}}', "Session B answer: Rs.100 on Dining."),
    )

    agent.chat("How much did I spend on travel?", session_id="session-A")
    agent.chat("How much on dining?", session_id="session-B")

    # session-B has its own empty history at the time of the call
    # (its first message was sent with no prior context)
    b_dispatch_call = agent._client.chat.completions.create.call_args_list[2]
    prompt_for_b = b_dispatch_call[1]["messages"][0]["content"]

    assert "session-A" not in prompt_for_b
    assert "Travel" not in prompt_for_b, (
        "Session A content ('Travel') must not appear in session B's prompt"
    )
    assert "5000" not in prompt_for_b, (
        "Session A answer amount must not appear in session B's prompt"
    )


# TEST 4 — bounded history (old turns are discarded)
def test_history_context_bounded_to_history_turns():
    """Only the most recent _HISTORY_TURNS pairs are included in the prompt.

    With _HISTORY_TURNS=3 and total_exchanges=5, the context window for the
    last question must contain rounds 1-3 but NOT round 0 (which has rolled off).
    """
    agent = _make_agent()

    # Use finance-keyword questions so the gate doesn't block them.
    total_exchanges = _HISTORY_TURNS + 2  # e.g. 5

    completions = []
    for i in range(total_exchanges):
        d = MagicMock()
        d.choices[0].message.content = '{"tool": "none", "args": {}}'
        a = MagicMock()
        a.choices[0].message.content = f"Total spending round {i}: Rs.{i * 100}."
        completions.extend([d, a])
    agent._client.chat.completions.create.side_effect = completions

    for i in range(total_exchanges):
        agent.chat(f"How much did I spend? Round {i}", session_id="bounded-sess")

    # Each exchange produces 2 LLM calls (dispatch + direct-answer).
    # Total calls = total_exchanges * 2.  The final dispatch is the
    # second-to-last call overall: index -2.
    all_calls = agent._client.chat.completions.create.call_args_list
    assert len(all_calls) == total_exchanges * 2, (
        f"Expected {total_exchanges * 2} LLM calls, got {len(all_calls)}"
    )

    last_dispatch_prompt = all_calls[-2][1]["messages"][0]["content"]

    # Round 0 must have rolled off the context window
    assert "Round 0" not in last_dispatch_prompt, (
        f"Oldest exchange (Round 0) must be outside the {_HISTORY_TURNS}-turn window"
    )
    # The most recent _HISTORY_TURNS rounds before the last must be present
    for i in range(1, _HISTORY_TURNS + 1):
        assert f"Round {i}" in last_dispatch_prompt, (
            f"Round {i} must appear in the {_HISTORY_TURNS}-turn context window"
        )


# TEST 5 — contextual follow-up (prior exchange available during dispatch)
def test_follow_up_question_has_prior_context_for_tool_dispatch():
    """Simulate: Q1='How much on dining?' → Q2='What about last month?'.

    The dispatch prompt for Q2 must contain the Q1 context so the LLM can
    infer the subject ('dining') from prior conversation.
    """
    agent = _make_agent()

    # Stub Q1 tool + synthesis
    agent._tools["get_spending_summary"] = lambda **kw: "Dining total: Rs.2450"
    _stub_llm_sequence(
        agent,
        (
            '{"tool": "get_spending_summary", "args": {"category": "Dining"}}',
            "You spent Rs.2450 on Dining.",
        ),
        (
            '{"tool": "get_spending_by_period", "args": {"period": "last_month"}}',
            "Last month dining: Rs.1800.",
        ),
    )

    agent.chat("How much did I spend on dining?", session_id="ctx-1")
    agent.chat("What about last month?", session_id="ctx-1")

    all_calls = agent._client.chat.completions.create.call_args_list
    # Q2 dispatch is call index 2 (Q1-dispatch=0, Q1-answer=1, Q2-dispatch=2)
    q2_dispatch_prompt = all_calls[2][1]["messages"][0]["content"]

    assert "How much did I spend on dining?" in q2_dispatch_prompt, (
        "Q1 user message must appear in Q2 dispatch context"
    )
    assert "2450" in q2_dispatch_prompt, (
        "Q1 assistant answer must appear in Q2 dispatch context"
    )


# TEST 6 — first message has no history context (empty context is transparent)
def test_first_message_has_no_history_in_prompt():
    """On the very first message of a session the history block must be absent."""
    agent = _make_agent()
    _stub_llm(
        agent,
        dispatch_reply='{"tool": "none", "args": {}}',
        answer_reply="No history yet.",
    )

    agent.chat("How much did I spend on groceries?", session_id="fresh-sess")

    first_dispatch = agent._client.chat.completions.create.call_args_list[0]
    prompt = first_dispatch[1]["messages"][0]["content"]

    assert "Conversation context" not in prompt, (
        "No history block should appear in the first message's prompt"
    )
