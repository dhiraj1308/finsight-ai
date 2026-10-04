"""Shared UI utilities for the FinSight AI Streamlit frontend."""
from __future__ import annotations

import streamlit as st

# Session-state key used to communicate a requested navigation destination
# from page code back to app.py's routing loop.  Using a dedicated "pending"
# key avoids writing to the nav_radio widget key after widget instantiation,
# which Streamlit forbids and raises StreamlitAPIException.
_PENDING_NAV_KEY = "_pending_page"


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
    """Request programmatic navigation to *page* on the next Streamlit rerun.

    Records the destination in ``st.session_state["_pending_page"]`` and
    triggers an immediate rerun.  ``app.py`` consumes this key *before*
    instantiating the ``nav_radio`` widget, so the radio is never written to
    after creation — avoiding ``StreamlitAPIException``.

    Parameters
    ----------
    page:
        The page label matching a key in ``app._PAGES``, e.g. ``"Upload"``.
    """
    st.session_state[_PENDING_NAV_KEY] = page
    st.rerun()
