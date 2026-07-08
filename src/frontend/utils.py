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
    """Switch to *page* using ``st.switch_page`` when available.

    Falls back to storing the selection in ``st.session_state["page"]`` so
    that ``app.py`` can pick it up on the next rerun when the Streamlit
    version pre-dates ``st.switch_page`` (added in 1.31).

    Parameters
    ----------
    page:
        The page label matching a key in ``app._PAGES``, e.g. ``"Upload"``.
    """
    if hasattr(st, "switch_page"):
        # st.switch_page expects a page file path relative to the app root
        # when using file-based MPA, but in our single-file routing pattern
        # we instead update session state and rerun so app.py re-dispatches.
        st.session_state["page"] = page
        st.rerun()
    else:
        st.session_state["page"] = page
        st.rerun()
