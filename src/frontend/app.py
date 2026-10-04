"""Entry point for the FinSight AI Streamlit frontend."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path so `frontend`, `api`, etc. are importable
# regardless of which directory streamlit is launched from.
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st

from frontend.views import (
    analytics,
    anomalies,
    chat,
    dashboard,
    forecast,
    settings,
    transactions,
    upload,
)
from frontend.services.api import APIClient

st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide",
)

_PAGES = {
    "Dashboard": dashboard,
    "Upload": upload,
    "Transactions": transactions,
    "Analytics": analytics,
    "Forecast": forecast,
    "Anomalies": anomalies,
    "Chat": chat,
    "Settings": settings,
}

_PAGE_NAMES = list(_PAGES.keys())


def main() -> None:
    """Initialise session state, render sidebar, and dispatch to the active page."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()

    client: APIClient = st.session_state.api_client

    # Consume any pending programmatic navigation request BEFORE the radio
    # widget is instantiated.  navigate_to() (in frontend.utils) stores the
    # destination under "_pending_page" and calls st.rerun().  On the next
    # run we arrive here, pre-seed "nav_radio" with the target page (safe
    # because the widget does not yet exist in this run), then clear the
    # pending key so it is not consumed again.
    from frontend.utils import _PENDING_NAV_KEY
    pending = st.session_state.pop(_PENDING_NAV_KEY, None)
    if pending is not None and pending in _PAGE_NAMES:
        st.session_state["nav_radio"] = pending

    with st.sidebar:
        st.title("💰 FinSight AI")
        st.caption("Personal Finance Intelligence")
        st.divider()

        selection = st.radio(
            "Navigation",
            _PAGE_NAMES,
            index=0,
            key="nav_radio",
            label_visibility="collapsed",
        )

    st.session_state["page"] = selection

    try:
        _PAGES[selection].render(client)
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.exception(exc)


if __name__ == "__main__":
    main()
