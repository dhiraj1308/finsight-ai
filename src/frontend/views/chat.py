"""Chat page — conversational AI assistant for personal finance queries."""
from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from frontend.services.api import APIClient
from frontend.utils import page_header

# Session state keys
_KEY_HISTORY = "chat_history"           # list[dict[str, str]]  role/content pairs
_KEY_SESSION_ID = "chat_session_id"     # stable UUID for this browser session
_KEY_PENDING_PROMPT = "chat_pending_prompt"  # suggestion carried across st.rerun()

# Backend constraint (mirrors ChatRequest field validation)
_MAX_MESSAGE_CHARS = 2000

# Suggested prompts shown when the conversation is empty
_SUGGESTED_PROMPTS: list[str] = [
    "What did I spend the most on last month?",
    "Which categories have the highest spending?",
    "Are there any unusual transactions I should know about?",
    "What is my average monthly spending?",
    "Can you forecast my spending for the next 30 days?",
]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _ensure_session_state() -> None:
    """Initialise chat-related session state keys if not already present."""
    if _KEY_HISTORY not in st.session_state:
        st.session_state[_KEY_HISTORY] = []
    if _KEY_SESSION_ID not in st.session_state:
        st.session_state[_KEY_SESSION_ID] = str(uuid.uuid4())[:128]
    if _KEY_PENDING_PROMPT not in st.session_state:
        st.session_state[_KEY_PENDING_PROMPT] = None


def _get_history() -> list[dict[str, str]]:
    """Return the current chat history list."""
    return st.session_state[_KEY_HISTORY]  # type: ignore[return-value]


def _append_message(role: str, content: str) -> None:
    """Append a message dict to the session history.

    Parameters
    ----------
    role:
        Either ``"user"`` or ``"assistant"``.
    content:
        The message text.
    """
    st.session_state[_KEY_HISTORY].append({"role": role, "content": content})


def _clear_history() -> None:
    """Erase conversation history and rotate the session ID."""
    st.session_state[_KEY_HISTORY] = []
    st.session_state[_KEY_SESSION_ID] = str(uuid.uuid4())[:128]
    st.session_state[_KEY_PENDING_PROMPT] = None


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------


def _send_message(client: APIClient, message: str) -> str | None:
    """Send *message* to the backend and return the assistant reply.

    Returns the answer string on success, or ``None`` on error (error is
    rendered inline before returning).
    """
    session_id: str = st.session_state[_KEY_SESSION_ID]
    try:
        result: dict[str, Any] = client.chat(
            message=message, session_id=session_id
        )
        return result.get("answer", "")
    except RuntimeError as exc:
        st.error(f"The assistant is unavailable: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {type(exc).__name__}. Please try again.")
        return None


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _render_history(history: list[dict[str, str]]) -> None:
    """Render all messages in *history* using ``st.chat_message``."""
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _render_suggested_prompts(prompts: list[str]) -> None:
    """Render clickable suggestion buttons when the conversation is empty.

    When a button is clicked, the prompt is stored in session state and
    ``st.rerun()`` is called immediately.  This ensures the widget tree is
    stable on every render pass — suggestions are either fully shown (empty
    history) or completely absent (non-empty history).  There is never a
    partial or non-functional state.

    The clicked prompt is picked up on the next run via
    ``_KEY_PENDING_PROMPT`` before this function is called again.
    """
    st.markdown("**Suggested questions:**")
    cols = st.columns(len(prompts))
    for i, (col, prompt) in enumerate(zip(cols, prompts)):
        # Stable, unique key: index + first 30 chars of prompt text.
        # Using an explicit key prevents Streamlit from matching buttons
        # by position when the column count or prompt list changes.
        key = f"suggestion_{i}_{prompt[:30]}"
        if col.button(prompt, key=key, use_container_width=True):
            st.session_state[_KEY_PENDING_PROMPT] = prompt
            st.rerun()


def _render_clear_button() -> None:
    """Render the Clear Conversation button with an inline confirmation."""
    with st.sidebar:
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state["_chat_confirm_clear"] = True

        if st.session_state.get("_chat_confirm_clear"):
            st.warning("This will erase all messages. Are you sure?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("Yes, clear", use_container_width=True):
                _clear_history()
                st.session_state.pop("_chat_confirm_clear", None)
                st.rerun()
            if col_no.button("Cancel", use_container_width=True):
                st.session_state.pop("_chat_confirm_clear", None)
                st.rerun()


def _render_message_count(history: list[dict[str, str]]) -> None:
    """Show a small caption with the current message count in the sidebar."""
    user_count = sum(1 for m in history if m["role"] == "user")
    with st.sidebar:
        st.caption(f"{user_count} message{'s' if user_count != 1 else ''} in this session")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(client: APIClient) -> None:
    """Render the Chat page."""
    page_header(
        "🤖 AI Financial Assistant",
        subtitle="Ask questions about your spending, forecasts and financial habits.",
    )

    _ensure_session_state()
    history = _get_history()

    # Sidebar: message count + clear button
    _render_message_count(history)
    _render_clear_button()

    # Replay existing conversation
    _render_history(history)

    # Suggested prompts — only rendered on an empty conversation.
    # Clicking a button stores the prompt in session state and triggers
    # st.rerun(), so the prompt is consumed on the *next* run below, after
    # the widget tree has been fully committed.
    if not history:
        _render_suggested_prompts(_SUGGESTED_PROMPTS)

    # Chat input — always at the bottom
    user_input: str | None = st.chat_input(
        "Ask about your finances…",
        max_chars=_MAX_MESSAGE_CHARS,
    )

    # Consume any pending suggestion (set by a button click + rerun).
    # Typed input always takes priority over a pending suggestion.
    pending: str | None = st.session_state.get(_KEY_PENDING_PROMPT)
    if pending:
        st.session_state[_KEY_PENDING_PROMPT] = None  # consume immediately

    active_prompt: str | None = user_input or pending

    if not active_prompt:
        return

    active_prompt = active_prompt.strip()
    if not active_prompt:
        return

    # Show the user message immediately
    with st.chat_message("user"):
        st.markdown(active_prompt)

    # Append to history before the backend call so it survives a failed call
    _append_message("user", active_prompt)

    # Call the backend with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = _send_message(client, active_prompt)

        if answer is not None:
            st.markdown(answer)
            _append_message("assistant", answer)
