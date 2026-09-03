"""Unit tests for upload.py _show_result() status banner logic.

Verifies that the correct Streamlit status function (success vs warning)
is called based on the ingested/skipped counts in the IngestResponse payload.

Streamlit is fully mocked — no Streamlit server or real API is required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helper: load upload module with Streamlit mocked out
# ---------------------------------------------------------------------------

def _make_mock_st() -> MagicMock:
    """Return a MagicMock standing in for the streamlit module."""
    mock_st = MagicMock(name="streamlit_mock")
    # st.columns(N) must return N mock column objects
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)
    mock_st.columns.return_value = [col, col, col, col]
    # st.expander must work as a context manager
    expander = MagicMock()
    expander.__enter__ = MagicMock(return_value=expander)
    expander.__exit__ = MagicMock(return_value=False)
    mock_st.expander.return_value = expander
    return mock_st


def _call_show_result(result: dict, mock_st: MagicMock) -> None:
    """Import upload module fresh with mock_st, then call _show_result(result)."""
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())

        import frontend.views.upload as upload_mod
        upload_mod.st = mock_st
        upload_mod._show_result(result)


# ---------------------------------------------------------------------------
# TEST 1 — successful ingestion: st.success, not st.warning
# ---------------------------------------------------------------------------

def test_show_result_success_when_ingested_gt_zero():
    """ingested=10, skipped=0 → st.success called, st.warning not called."""
    mock_st = _make_mock_st()
    _call_show_result({"ingested": 10, "skipped": 0, "warnings": []}, mock_st)

    mock_st.success.assert_called_once()
    success_msg = mock_st.success.call_args.args[0]
    assert "successfully" in success_msg.lower(), (
        f"Expected 'successfully' in success message, got: {success_msg!r}"
    )
    # The upload-level status must not show a warning
    # (warnings for individual rows come from st.warning inside the expander,
    # so we check that the top-level st.warning was NOT called at all)
    mock_st.warning.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 2 — duplicate upload: st.warning, not st.success
# ---------------------------------------------------------------------------

def test_show_result_warning_when_all_duplicates():
    """ingested=0, skipped=10 → st.warning called, st.success not called."""
    mock_st = _make_mock_st()
    _call_show_result({"ingested": 0, "skipped": 10, "warnings": []}, mock_st)

    mock_st.warning.assert_called_once()
    warning_msg = mock_st.warning.call_args.args[0]
    assert "already exist" in warning_msg.lower() or "no new" in warning_msg.lower(), (
        f"Expected duplicate-explanation in warning message, got: {warning_msg!r}"
    )
    mock_st.success.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 3 — partial success: st.success even when some are skipped
# ---------------------------------------------------------------------------

def test_show_result_success_when_partial_ingestion():
    """ingested=8, skipped=2 → st.success called (partial success is still success)."""
    mock_st = _make_mock_st()
    _call_show_result(
        {"ingested": 8, "skipped": 2, "warnings": ["Row 3: bad value 'x'"]},
        mock_st,
    )

    mock_st.success.assert_called_once()
    # Top-level warning must NOT be called for the banner
    # (per-row warnings appear inside the expander via a separate call)
    # We verify by checking that warning was only called inside the expander,
    # i.e. the direct mock_st.warning on the module was not used for the banner.
    # The per-row warning appears via: mock_st.warning(warning_text) inside
    # the expander context — which is the same mock, so we check the first call.
    if mock_st.warning.called:
        first_warning = mock_st.warning.call_args_list[0].args[0]
        # If warning was called, it must be for a per-row issue, not the banner
        assert "already exist" not in first_warning.lower(), (
            "st.warning must not be called with the 'already exist' banner "
            "when ingested > 0"
        )
        assert "no transactions found" not in first_warning.lower(), (
            "st.warning must not be called with 'no transactions found' "
            "when ingested > 0"
        )


# ---------------------------------------------------------------------------
# TEST 4 — empty result: st.warning with "no transactions" message
# ---------------------------------------------------------------------------

def test_show_result_warning_when_nothing_ingested_and_nothing_skipped():
    """ingested=0, skipped=0, warnings=[] → st.warning with 'no transactions' message."""
    mock_st = _make_mock_st()
    _call_show_result({"ingested": 0, "skipped": 0, "warnings": []}, mock_st)

    mock_st.warning.assert_called_once()
    warning_msg = mock_st.warning.call_args.args[0]
    assert "no transactions" in warning_msg.lower(), (
        f"Expected 'no transactions' in warning message, got: {warning_msg!r}"
    )
    mock_st.success.assert_not_called()


# ---------------------------------------------------------------------------
# TEST 5 — metric cards always rendered regardless of banner type
# ---------------------------------------------------------------------------

def test_show_result_always_renders_ingested_and_skipped_metrics():
    """Both 'Transactions Ingested' and 'Transactions Skipped' metric cards
    must be rendered regardless of which status banner is shown."""
    for result in [
        {"ingested": 10, "skipped": 0, "warnings": []},
        {"ingested": 0, "skipped": 10, "warnings": []},
        {"ingested": 0, "skipped": 0, "warnings": []},
        {"ingested": 5, "skipped": 3, "warnings": ["Row 1: bad value"]},
    ]:
        mock_st = _make_mock_st()
        _call_show_result(result, mock_st)

        # st.columns must be called to lay out metric cards
        mock_st.columns.assert_called()

        col = mock_st.columns.return_value[0]
        metric_calls = col.metric.call_args_list
        assert len(metric_calls) >= 2, (
            f"Expected at least 2 metric cards for result {result}, "
            f"got {len(metric_calls)}"
        )
        labels = [c.args[0] for c in metric_calls]
        assert "Transactions Ingested" in labels
        assert "Transactions Skipped" in labels


# ---------------------------------------------------------------------------
# TEST 6 — per-row warnings expander still works after the banner change
# ---------------------------------------------------------------------------

def test_show_result_warnings_expander_shown_when_warnings_present():
    """When warnings are present the expander must be opened."""
    mock_st = _make_mock_st()
    _call_show_result(
        {"ingested": 5, "skipped": 0, "warnings": ["Row 2: bad date 'xyz'"]},
        mock_st,
    )

    mock_st.expander.assert_called_once()
    expander_label = mock_st.expander.call_args.args[0]
    assert "Warning" in expander_label or "warning" in expander_label, (
        f"Expander label should mention warnings, got: {expander_label!r}"
    )
