"""
Preservation Property Tests — Sidebar, Session State, and Page View Behavior Unchanged

Validates: Requirements 2.3, 3.1, 3.2, 3.3, 3.4, 3.5

Property 2: Preservation — Sidebar, Session State, and Page View Behavior Unchanged
====================================================================================
These tests MUST PASS on unfixed code.

They capture the baseline behavior observed on unfixed code for non-buggy inputs
(cases where no navigation click occurs, or first-load defaults, or structural
sidebar behavior). This baseline must not regress after the fix is applied.

Properties tested (observation-first, unfixed code):
  2a. Rerender stability (req 3.1):
        When session_state["page"] == P and st.radio returns P (same page, no
        navigation), the rendered page is P.
  2b. session_state["page"] accuracy (req 3.2):
        After main() runs and renders page P, session_state["page"] == P.
  2c. APIClient pass-through (req 3.3):
        Each view's render(client) receives the exact same APIClient instance
        that is in session_state.api_client.
  2d. Sidebar structure (req 3.4):
        st.radio is called with options containing all 8 page names; sidebar
        shows st.title("💰 FinSight AI"), st.caption("Personal Finance
        Intelligence"), and st.divider().
  2e. Dashboard default on first load (req 2.3):
        When "page" is NOT in session_state, main() renders "Dashboard".
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable at collection time.
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
# Reuse the same pattern as test_sidebar_bug_condition.py.
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

# Inject stubs (only if not already present — shares session with bug condition test)
for _stub_name, _stub_mod in _PERMANENT_STUBS.items():
    sys.modules.setdefault(_stub_name, _stub_mod)

# Evict any stale cached import to get a fresh one with our stubs active
for _key in list(sys.modules):
    if _key == "frontend.app":
        del sys.modules[_key]

import frontend.app as _APP_MODULE  # type: ignore[import]

# Ensure app.py's module-level `st` variable points to our mock
_APP_MODULE.st = _MODULE_MOCK_ST  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Core simulation helper — non-buggy case (radio returns the same page)
# ---------------------------------------------------------------------------
def _run_main_stable(page: str, api_client=None) -> tuple[str, object]:
    """
    Simulate a rerun where st.radio returns `page` (same as session_state["page"]).
    This is the non-buggy path — no navigation is happening.

    Returns
    -------
    (rendered_page, api_client_received)
        rendered_page        : name of the page module whose .render() was called
        api_client_received  : the argument passed to render()
    """
    if api_client is None:
        api_client = MagicMock(name="api_client")

    session_state = FakeSessionState({
        "page": page,
        "api_client": api_client,
    })

    # Stable radio: always returns `page` (simulating no navigation click)
    def stable_radio_side_effect(*args, **kwargs):
        return page

    # Per-test page mocks
    render_mocks: dict[str, MagicMock] = {}
    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        render_mocks[name] = mod.render
        mock_pages[name] = mod

    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.radio.side_effect = stable_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx
    _MODULE_MOCK_ST.title.reset_mock()
    _MODULE_MOCK_ST.caption.reset_mock()
    _MODULE_MOCK_ST.divider.reset_mock()
    _MODULE_MOCK_ST.radio.reset_mock()
    _MODULE_MOCK_ST.radio.side_effect = stable_radio_side_effect  # re-set after reset

    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    rendered = [name for name, rm in render_mocks.items() if rm.called]
    if len(rendered) != 1:
        raise RuntimeError(
            f"Expected exactly 1 page rendered, got {rendered!r} (page={page!r})"
        )

    rendered_page = rendered[0]
    render_call_args = render_mocks[rendered_page].call_args
    api_client_received = render_call_args[0][0] if render_call_args else None

    return rendered_page, api_client_received


def _run_main_first_load() -> tuple[str, FakeSessionState]:
    """
    Simulate first load: session_state has NO "page" key.

    Returns
    -------
    (rendered_page, session_state)
    """
    api_client = MagicMock(name="api_client")
    session_state = FakeSessionState({
        "api_client": api_client,
        # "page" is intentionally absent
    })

    # On first load, app.py sets default_index=0, so radio returns index 0 → "Dashboard"
    def first_load_radio_side_effect(*args, **kwargs):
        idx = kwargs.get("index", 0)
        return _PAGE_NAMES[idx]

    render_mocks: dict[str, MagicMock] = {}
    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        render_mocks[name] = mod.render
        mock_pages[name] = mod

    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.radio.reset_mock()
    _MODULE_MOCK_ST.radio.side_effect = first_load_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx

    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    rendered = [name for name, rm in render_mocks.items() if rm.called]
    if len(rendered) != 1:
        raise RuntimeError(f"Expected exactly 1 page rendered on first load, got {rendered!r}")

    return rendered[0], session_state


def _run_main_sidebar_structure_check(page: str) -> MagicMock:
    """
    Run main() and return the mock_st so sidebar call args can be inspected.
    """
    api_client = MagicMock(name="api_client")
    session_state = FakeSessionState({
        "page": page,
        "api_client": api_client,
    })

    def stable_radio_side_effect(*args, **kwargs):
        return page

    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        mock_pages[name] = mod

    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.title.reset_mock()
    _MODULE_MOCK_ST.caption.reset_mock()
    _MODULE_MOCK_ST.divider.reset_mock()
    _MODULE_MOCK_ST.radio.reset_mock()
    _MODULE_MOCK_ST.radio.side_effect = stable_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx

    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    return _MODULE_MOCK_ST


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------
_page_name_st = st.sampled_from(_PAGE_NAMES)


# ===========================================================================
# Property 2a: Rerender stability (req 3.1)
# ===========================================================================

@given(page=_page_name_st)
@settings(
    max_examples=8,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2a_rerender_stability(page: str):
    """
    **Validates: Requirements 3.1**

    Property: For any valid page P, when session_state["page"] == P and
    st.radio returns P (same page, no navigation click), main() renders P.

    This is the non-buggy path — unfixed code handles it correctly.
    Must pass on unfixed code AND continue to pass after fix.
    """
    rendered, _ = _run_main_stable(page)
    assert rendered == page, (
        f"REGRESSION: session_state['page']='{page}', radio returns '{page}', "
        f"but rendered page was '{rendered}'"
    )


@pytest.mark.parametrize("page", _PAGE_NAMES)
def test_unit_rerender_stability(page: str):
    """
    **Validates: Requirements 3.1**

    Parametric unit test: stable rerender for each of the 8 pages.
    """
    rendered, _ = _run_main_stable(page)
    assert rendered == page, (
        f"REGRESSION: page='{page}' should render itself on stable rerender, "
        f"but rendered '{rendered}'"
    )


# ===========================================================================
# Property 2b: session_state["page"] accuracy (req 3.2)
# ===========================================================================

@given(page=_page_name_st)
@settings(
    max_examples=8,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2b_session_state_page_accuracy(page: str):
    """
    **Validates: Requirements 3.2**

    Property: After main() renders page P, session_state["page"] == P.

    Checks that the `st.session_state["page"] = selection` write is correct
    and that session state accurately reflects what was rendered.
    """
    api_client = MagicMock(name="api_client")
    session_state = FakeSessionState({
        "page": page,
        "api_client": api_client,
    })

    def stable_radio_side_effect(*args, **kwargs):
        return page

    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        mock_pages[name] = mod

    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.radio.reset_mock()
    _MODULE_MOCK_ST.radio.side_effect = stable_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx

    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    assert session_state["page"] == page, (
        f"REGRESSION: after rendering page='{page}', "
        f"session_state['page']={session_state['page']!r} (expected '{page}')"
    )


@pytest.mark.parametrize("page", _PAGE_NAMES)
def test_unit_session_state_page_accuracy(page: str):
    """
    **Validates: Requirements 3.2**

    After main() runs for page P, session_state["page"] must equal P.
    """
    api_client = MagicMock(name="api_client")
    session_state = FakeSessionState({
        "page": page,
        "api_client": api_client,
    })

    def stable_radio_side_effect(*args, **kwargs):
        return page

    mock_pages: dict[str, MagicMock] = {}
    for name in _PAGE_NAMES:
        mod = MagicMock()
        mod.render = MagicMock()
        mock_pages[name] = mod

    sidebar_ctx = MagicMock()
    sidebar_ctx.__enter__ = MagicMock(return_value=sidebar_ctx)
    sidebar_ctx.__exit__ = MagicMock(return_value=False)

    _MODULE_MOCK_ST.session_state = session_state
    _MODULE_MOCK_ST.radio.reset_mock()
    _MODULE_MOCK_ST.radio.side_effect = stable_radio_side_effect
    _MODULE_MOCK_ST.sidebar = sidebar_ctx

    _APP_MODULE._PAGES = mock_pages  # type: ignore[attr-defined]
    _APP_MODULE._PAGE_NAMES = _PAGE_NAMES.copy()  # type: ignore[attr-defined]

    _APP_MODULE.main()

    assert session_state["page"] == page, (
        f"session_state['page'] should be '{page}' but got {session_state['page']!r}"
    )


# ===========================================================================
# Property 2c: APIClient pass-through (req 3.3)
# ===========================================================================

@given(page=_page_name_st)
@settings(
    max_examples=8,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2c_api_client_passthrough(page: str):
    """
    **Validates: Requirements 3.3**

    Property: For any valid page P, the render(client) call receives the exact
    same APIClient instance that is stored in session_state.api_client.

    Identity check (is), not equality — the same object must be passed through.
    """
    api_client = MagicMock(name="api_client_singleton")
    rendered, received_client = _run_main_stable(page, api_client=api_client)

    assert received_client is api_client, (
        f"REGRESSION: render() for page='{page}' received a different APIClient "
        f"instance than session_state.api_client"
    )


@pytest.mark.parametrize("page", _PAGE_NAMES)
def test_unit_api_client_passthrough(page: str):
    """
    **Validates: Requirements 3.3**

    render(client) must receive the exact APIClient instance from session_state.
    """
    api_client = MagicMock(name="api_client_singleton")
    rendered, received_client = _run_main_stable(page, api_client=api_client)

    assert received_client is api_client, (
        f"render() for page='{page}' should receive session_state.api_client "
        f"but received a different object"
    )


# ===========================================================================
# Property 2d: Sidebar structure (req 3.4)
# ===========================================================================

@given(page=_page_name_st)
@settings(
    max_examples=8,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2d_sidebar_structure(page: str):
    """
    **Validates: Requirements 3.4**

    Property: For any valid page P, when main() runs:
      - st.title("💰 FinSight AI") is called
      - st.caption("Personal Finance Intelligence") is called
      - st.divider() is called
      - st.radio is called with all 8 page names in its positional args
    """
    mock_st = _run_main_sidebar_structure_check(page)

    # Check title
    mock_st.title.assert_called_once_with("💰 FinSight AI")

    # Check caption
    mock_st.caption.assert_called_once_with("Personal Finance Intelligence")

    # Check divider was called
    mock_st.divider.assert_called_once()

    # Check radio was called with all 8 page names
    radio_call_args = mock_st.radio.call_args
    assert radio_call_args is not None, "st.radio was not called"

    # The page names are the second positional argument (_PAGE_NAMES)
    positional_args = radio_call_args[0]
    assert len(positional_args) >= 2, (
        f"st.radio expected at least 2 positional args (label + options), "
        f"got {positional_args!r}"
    )
    radio_options = positional_args[1]
    for name in _PAGE_NAMES:
        assert name in radio_options, (
            f"REGRESSION: '{name}' not found in st.radio options {radio_options!r}"
        )


@pytest.mark.parametrize("page", _PAGE_NAMES)
def test_unit_sidebar_structure(page: str):
    """
    **Validates: Requirements 3.4**

    Sidebar must show title, caption, divider, and radio with all 8 page names.
    """
    mock_st = _run_main_sidebar_structure_check(page)

    mock_st.title.assert_called_once_with("💰 FinSight AI")
    mock_st.caption.assert_called_once_with("Personal Finance Intelligence")
    mock_st.divider.assert_called_once()

    radio_call_args = mock_st.radio.call_args
    assert radio_call_args is not None, "st.radio was not called"
    positional_args = radio_call_args[0]
    radio_options = positional_args[1]
    assert list(radio_options) == _PAGE_NAMES, (
        f"st.radio options mismatch: expected {_PAGE_NAMES}, got {list(radio_options)}"
    )


# ===========================================================================
# Property 2e: Dashboard default on first load (req 2.3)
# ===========================================================================

def test_property2e_dashboard_default_on_first_load():
    """
    **Validates: Requirements 2.3**

    When "page" is NOT in session_state (first load), main() must render
    "Dashboard" (index 0).

    The unfixed code handles this correctly: when "page" is absent,
    default_index = 0, so radio returns _PAGE_NAMES[0] = "Dashboard".
    Must continue to work after the fix.
    """
    rendered, session_state = _run_main_first_load()
    assert rendered == "Dashboard", (
        f"REGRESSION: first load (no 'page' in session_state) should render "
        f"'Dashboard' but rendered '{rendered}'"
    )


def test_unit_first_load_sets_session_state_page():
    """
    **Validates: Requirements 2.3**

    On first load, after main() runs, session_state["page"] must be "Dashboard".
    """
    rendered, session_state = _run_main_first_load()
    assert session_state["page"] == "Dashboard", (
        f"First load should set session_state['page']='Dashboard', "
        f"got {session_state['page']!r}"
    )


def test_unit_first_load_no_page_key_renders_index_zero():
    """
    **Validates: Requirements 2.3**

    Verify that with an empty session_state (no 'page' key at all), the
    rendered page is the first entry in _PAGE_NAMES (Dashboard).
    """
    rendered, _ = _run_main_first_load()
    assert rendered == _PAGE_NAMES[0], (
        f"Expected index-0 page ('{_PAGE_NAMES[0]}') on first load, "
        f"got '{rendered}'"
    )
