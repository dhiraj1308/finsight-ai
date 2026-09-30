"""Shared UI utilities for the FinSight AI Streamlit frontend."""
from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str | None = None) -> None:
    """Render a standardised page header used by every page.

    Parameters
    ----------
    title:
        Main page heading, e.g. ``"💰 FinSight AI"``.
    subtitle:
        Optional caption rendered beneath the title.
    """
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()


def navigate_to(page: str) -> None:
    """Switch to *page* by updating the radio widget's stored state.

    ``app.py`` routes based on the value returned by ``st.radio(...,
    key="nav_radio")``.  Streamlit keyed-radio widgets return
    ``session_state["nav_radio"]`` on every rerun once state is stored, so
    setting that key before calling ``st.rerun()`` is the only reliable way
    to change the active page from code.  ``session_state["page"]`` is also
    updated for consistency (it mirrors the rendered page after each run).

    Parameters
    ----------
    page:
        The page label matching a key in ``app._PAGES``, e.g. ``"Upload"``.
    """
    st.session_state["nav_radio"] = page
    st.session_state["page"] = page
    st.rerun()
