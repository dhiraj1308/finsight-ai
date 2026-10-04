"""Unit tests for the Analytics page KPI calculations.

These tests verify that _kpi_cards() passes correct labels and values to
st.metric() for the four spending-based KPIs:

  1. Total Spending
  2. Transaction Count
  3. Avg Transaction
  4. Largest Transaction

Streamlit is fully mocked so no Streamlit server is required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helper: stub out streamlit and import the module under test
# ---------------------------------------------------------------------------

def _make_mock_st():
    """Return a MagicMock that stands in for the streamlit module."""
    mock_st = MagicMock(name="streamlit_mock")
    # st.columns(4) must return a sequence of 4 column context managers
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col, col, col, col]
    return mock_st


def _kpi_cards_with_mock_st(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Call analytics._kpi_cards(df) with Streamlit mocked out.

    Returns a list of (label, value) tuples collected from every
    col.metric() call in the order they were made.
    """
    mock_st = _make_mock_st()

    # Patch streamlit inside the analytics module's namespace
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        # Re-import to pick up the patched streamlit
        if "frontend.views.analytics" in sys.modules:
            del sys.modules["frontend.views.analytics"]
        # Also patch any dependencies that import streamlit at module level
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]

        # Provide stub modules for frontend dependencies
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())

        import frontend.views.analytics as analytics_mod
        analytics_mod.st = mock_st

        analytics_mod._kpi_cards(df)

    # Collect metric calls: col.metric(label, value)
    col = mock_st.columns.return_value[0]
    calls = col.metric.call_args_list
    return [(c.args[0], c.args[1]) for c in calls]


# ---------------------------------------------------------------------------
# TEST 1 — correct KPI values for a normal dataset
# ---------------------------------------------------------------------------

def test_kpi_values_for_positive_amounts():
    """
    Given amounts [100, 200, 300]:
      Total Spending    = 600
      Transaction Count = 3
      Avg Transaction   = 200
      Largest Transaction = 300
    """
    df = pd.DataFrame({
        "amount": [100.0, 200.0, 300.0],
        "category": ["Dining", "Groceries", "Shopping"],
        "merchant": ["A", "B", "C"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })

    mock_st = _make_mock_st()

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())

        import frontend.views.analytics as analytics_mod
        analytics_mod.st = mock_st
        analytics_mod._kpi_cards(df)

    col = mock_st.columns.return_value[0]
    calls = col.metric.call_args_list
    assert len(calls) == 4

    labels  = [c.args[0] for c in calls]
    values  = [c.args[1] for c in calls]

    assert labels[0] == "Total Spending"
    assert "600" in values[0], f"Expected 600 in Total Spending, got {values[0]!r}"

    assert labels[1] == "Transaction Count"
    assert "3" in values[1], f"Expected count 3, got {values[1]!r}"

    assert labels[2] == "Avg Transaction"
    assert "200" in values[2], f"Expected avg 200, got {values[2]!r}"

    assert labels[3] == "Largest Transaction"
    assert "300" in values[3], f"Expected largest 300, got {values[3]!r}"


# ---------------------------------------------------------------------------
# TEST 2 — empty dataset does not raise and shows zero values
# ---------------------------------------------------------------------------

def test_kpi_empty_dataset_does_not_raise():
    """Empty DataFrame must not raise an exception or produce NaN values."""
    df = pd.DataFrame(columns=["amount", "category", "merchant", "date"])

    mock_st = _make_mock_st()

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())

        import frontend.views.analytics as analytics_mod
        analytics_mod.st = mock_st
        # Should not raise
        analytics_mod._kpi_cards(df)

    col = mock_st.columns.return_value[0]
    calls = col.metric.call_args_list
    assert len(calls) == 4

    values = [c.args[1] for c in calls]
    # All numeric values should contain "0" and must not contain "NaN" or "nan"
    for v in values:
        assert "nan" not in v.lower(), f"NaN appeared in KPI value: {v!r}"
        assert "0" in v or v.endswith(",")  # "₹0.00" or "0" for count


# ---------------------------------------------------------------------------
# TEST 3 — no negative-amount logic is present
# ---------------------------------------------------------------------------

def test_kpi_cards_does_not_use_negative_income_logic():
    """_kpi_cards() must not filter for negative amounts or compute net_flow.

    This test reads the source of _kpi_cards to confirm the old
    income/net_flow calculations are gone.
    """
    import inspect
    for dep in list(sys.modules):
        if dep.startswith("frontend."):
            del sys.modules[dep]
    sys.modules.setdefault("frontend.services.api", MagicMock())
    sys.modules.setdefault("frontend.utils", MagicMock())

    mock_st = MagicMock()
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        import frontend.views.analytics as analytics_mod

    source = inspect.getsource(analytics_mod._kpi_cards)

    assert "net_flow" not in source, \
        "net_flow calculation must not appear in _kpi_cards"
    assert "income" not in source, \
        "income calculation must not appear in _kpi_cards"
    assert 'amount"] < 0' not in source and "amount < 0" not in source, \
        "Negative-amount filter must not appear in _kpi_cards"

    # Verify expected positive labels are present
    assert "Total Spending" in source
    assert "Transaction Count" in source
    assert "Avg Transaction" in source
    assert "Largest Transaction" in source
