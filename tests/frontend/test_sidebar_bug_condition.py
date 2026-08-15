"""
Bug Condition Exploration Test — Stale Index Overrides Radio Selection

Validates: Requirements 1.1, 1.2, 1.3

Property 1: Bug Condition — Stale Index Overrides Radio Selection
=================================================================
This test MUST FAIL on unfixed code. Failure confirms the bug exists.

Bug description (§1.3):
    When `st.session_state["page"]` contains the name of the previously active
    page (P_current) and `st.radio` is called with `index=default_index` derived
    from that stale value, the radio widget's selection is overridden back to
    P_current during the first rerun — discarding the user's click to P_new.

Expected behaviour (correct, §2.1 / §2.2):
    One click must navigate immediately; the return value of `st.radio` must be
    used directly as the active page without re-imposing the stale index.

Strategy:
    For each P_current in _PAGE_NAMES, for each P_new ≠ P_current:
      1. Set session_state["page"] = P_current (simulates state after prior render)
      2. Mock st.radio with a side_effect that reads the `index` kwarg and returns
         _PAGE_NAMES[index] — faithfully reproducing the broken Streamlit behaviour
         where passing index= overrides any user click and returns the stale page.
      3. Call main() via the app module.
      4. Capture which page module's .render() was called.
      5. Assert rendered page == P_new  →  FAILS on broken code.

Counterexamples found on unfixed code (all 56 pairs fail):
    session_state['page']='Dashboard', clicking 'Upload'    → renders 'Dashboard'
    session_state['page']='Dashboard', clicking 'Analytics' → renders 'Dashboard'
    session_state['page']='Upload',    clicking 'Dashboard' → renders 'Upload'
    ... (every P_current / P_new ≠ P_current combination fails)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable at collection time.
# conftest.py at the project root does this too, but we insert it here so the
# import below works whether or not pytest has already processed conftest.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# The 8 page names exactly as declared in app.py
# ---------------------------------------------------------------------------
_PAGE_NAMES = [
    "Dashboard",
    "Upload",
    "Transactions",
    "Analytics",
    "Forecast",
    "Anomalies",
    "Chat",
    "Settings",
]


# ---------------------------------------------------------------------------
# FakeSessionState: dict subclass supporting attribute-style access so that
# both ss["page"] and ss.api_client work in app.py.
# ---------------------------------------------------------------------------
class FakeSessionState(dict):
    """dict subclass that also supports attribute-style access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


# ---------------------------------------------------------------------------
# Install permanent stubs for streamlit and all view/service dependencies.
# These stay in sys.modules for the duration of the test session so that
# frontend.app can be imported and re-used without a real Streamlit server.
# ---------------------------------------------------------------------------
_MODULE_MOCK_ST = MagicMock(name="streamlit_mock")
_MODULE_MOCK_ST.set_page_config = MagicMock()

_PERMANENT_STUBS: dict[str, MagicMock] = {
    "streamlit": _MODULE_MOCK_ST,
    "frontend.views": MagicMock(),
    "frontend.views.analytics": MagicMock(),
    "frontend.views.anomalies": MagicMock(),
    "frontend.views.chat": MagicMock(),
    "frontend.views.dashboard": MagicMock(),
    "frontend.views.forecast": MagicMock(),
    "frontend.views.settings": MagicMock(),
    "frontend.views.transactions": MagicMock(),
    "frontend.views.upload": MagicMock(),
    "frontend.services": MagicMock(),
    "frontend.services.api": MagicMock(),
}

# Inject stubs permanently (only if not already present so we don't trample
# a real installation of streamlit for other tests in the suite)
for _stub_name, _stub_mod in _PERMANENT_STUBS.items():
    sys.modules.setdefault(_stub_name, _stub_mod)

# Evict any stale cached import so we get a fresh one with our stubs active
for _key in list(sys.modules):
    if _key == "frontend.app":
        del sys.modules[_key]

# Import the app module — at this point sys.modules has our stubs, so all
# `import streamlit as st` and view imports inside app.py resolve to mocks.
import frontend.app as _APP_MODULE  # type: ignore[import]

# Ensure app.py's module-level `st` variable points to our mock
_APP_MODULE.st = _MODULE_MOCK_ST  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Core simulation helper
# ---------------------------------------------------------------------------
def _run_main_with_stale_state(p_current: str, p_new: str) -> str:  # noqa: ARG001
    """
    Simulate one Streamlit rerun of main() under UNFIXED (broken) code conditions.

    The broken code:
      1. Reads session_state["page"] = p_current
      2. Computes default_index = _PAGE_NAMES.index(p_current)
      3. Calls st.radio(..., index=default_index)
      4. Streamlit re-imposes the index — radio returns p_current, NOT p_new

    We replicate step 4 via broken_radio_side_effect: inspect the `index` kwarg
    that the broken code passes and return _PAGE_NAMES[index].

    Parameters
    ----------
    p_current : str
        The page currently stored in session_state (stale value).
    p_new : str
        The page the user clicked (which the broken code discards).

    Returns
    -------
    str
        Name of the page that was actually rendered.
    """
    # Fresh session state for this rerun: page is stale (p_current)
    session_state = FakeSessionState({
        "page": p_current,
        "api_client": MagicMock(),
    })

    # Broken radio: returns _PAGE_NAMES[index] — i.e. p_current — ignoring the click
    def broken_radio_side_effect(*args, **kwargs):
        idx = kwargs.get("index", 0)
        return _PAGE_NAMES[idx]

    # Build per-test page mocks so we can detect which one's .render() gets called
    render_mocks: dict[str, MagicMock] = {}
    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        render_mocks[name] = mod.render
        mock_pages[name] = mod

    # Sidebar context manager
    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    # Wire up the module-level mock_st for this run
    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.radio.side_effect = broken_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx

    # Wire page registry
    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    rendered = [name for name, rm in render_mocks.items() if rm.called]
    if len(rendered) != 1:
        raise RuntimeError(
            f"Expected exactly 1 page rendered, got {rendered!r} "
            f"(p_current={p_current!r}, p_new={p_new!r})"
        )
    return rendered[0]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------
_page_name_st = st.sampled_from(_PAGE_NAMES)


def _page_pair_st():
    """Strategy producing (p_current, p_new) where p_new != p_current."""
    return st.tuples(_page_name_st, _page_name_st).filter(
        lambda pair: pair[0] != pair[1]
    )


# ---------------------------------------------------------------------------
# Property-Based Test — Bug Condition
# Validates: Requirements 1.1, 1.2, 1.3
# ---------------------------------------------------------------------------
@given(pair=_page_pair_st())
@settings(
    max_examples=56,  # aims to cover all 8×7=56 unique pairs
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property1_single_click_navigates_to_new_page(pair):
    """
    **Validates: Requirements 1.1, 1.2, 1.3**

    Property: For any P_current and any P_new ≠ P_current, after one simulated
    rerun where the user has clicked P_new, the rendered page MUST be P_new.

    On unfixed code this FAILS because the broken radio returns P_current
    (the stale session-state value re-imposed via default_index), not P_new.

    Counterexample pattern:
        session_state['page']='Dashboard', clicking 'Upload'
        → broken code renders 'Dashboard' instead of 'Upload'
    """
    p_current, p_new = pair
    rendered = _run_main_with_stale_state(p_current, p_new)
    assert rendered == p_new, (
        f"BUG CONFIRMED: session_state['page']='{p_current}', "
        f"user clicked '{p_new}', "
        f"but rendered page was '{rendered}' (stale index override)"
    )


# ---------------------------------------------------------------------------
# Parametric unit test — exhaustive coverage of all 56 pairs
# Validates: Requirements 1.1, 1.2, 1.3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "p_current,p_new",
    [
        (current, new)
        for current in _PAGE_NAMES
        for new in _PAGE_NAMES
        if current != new
    ],
)
def test_unit_single_click_navigates_to_new_page(p_current: str, p_new: str):
    """
    **Validates: Requirements 1.1, 1.2, 1.3**

    Exhaustive parametric test over all 56 (p_current, p_new) pairs.

    On unfixed code: FAILS for every pair with message:
        BUG CONFIRMED: session_state['page']='<X>', user clicked '<Y>',
        but rendered page was '<X>' (stale index override)
    """
    rendered = _run_main_with_stale_state(p_current, p_new)
    assert rendered == p_new, (
        f"BUG CONFIRMED: session_state['page']='{p_current}', "
        f"user clicked '{p_new}', "
        f"but rendered page was '{rendered}' (stale index override)"
    )
