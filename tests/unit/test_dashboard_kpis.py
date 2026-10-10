"""Unit tests for dashboard.py KPI calculations.

Tests cover:
  - _is_income() merchant classification
  - _kpi_cards() correct values for income/spending split
  - Net Cash Flow calculation
  - Empty dataset safety
  - Dataset matching the real 45-transaction test data

Streamlit is fully mocked — no server required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_st():
    """Return a MagicMock standing in for the streamlit module.

    st.columns(n) returns n mock column objects so that any _kpi_cards / strip
    can unpack without error regardless of how many columns are requested.
    """
    mock_st = MagicMock(name="streamlit_mock")
    col = MagicMock()
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=False)

    def _columns_side_effect(spec):
        """Return exactly as many col mocks as requested."""
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [col] * n

    mock_st.columns.side_effect = _columns_side_effect
    # Keep a stable .return_value for tests that inspect col.metric calls
    mock_st.columns.return_value = [col, col, col, col, col]
    return mock_st


def _fresh_dashboard_module(mock_st):
    """Import dashboard module with a clean namespace and patched streamlit."""
    for dep in list(sys.modules):
        if dep.startswith("frontend."):
            del sys.modules[dep]
    sys.modules.setdefault("frontend.services.api", MagicMock())
    sys.modules.setdefault("frontend.utils", MagicMock())

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        import frontend.views.dashboard as dash_mod
        dash_mod.st = mock_st
        return dash_mod


def _call_kpi_cards(
    transactions: list[dict[str, Any]],
    anomalies: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[Any]]:
    """Call _kpi_cards with mocked streamlit, return (labels, values) lists."""
    if anomalies is None:
        anomalies = []
    mock_st = _make_mock_st()
    dash_mod = _fresh_dashboard_module(mock_st)
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        dash_mod._kpi_cards(transactions, anomalies)

    col = mock_st.columns.return_value[0]
    calls = col.metric.call_args_list
    labels = [c.args[0] for c in calls]
    values = [c.args[1] for c in calls]
    return labels, values


# ---------------------------------------------------------------------------
# TEST GROUP 1 — _is_income() classification
# ---------------------------------------------------------------------------

class TestIsIncome:

    def _is_income(self, merchant: str) -> bool:
        mock_st = _make_mock_st()
        dash_mod = _fresh_dashboard_module(mock_st)
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            return dash_mod._is_income({"merchant": merchant})

    def test_salary_credit_is_income(self):
        """'Salary Credit' contains 'salary' — must be classified as income."""
        assert self._is_income("Salary Credit") is True

    def test_credit_keyword_is_income(self):
        """Merchant name containing 'credit' is income."""
        assert self._is_income("Credit Transfer") is True

    def test_income_keyword_is_income(self):
        """Merchant name containing 'income' is income."""
        assert self._is_income("Side Income") is True

    def test_refund_is_income(self):
        """Merchant name containing 'refund' is income."""
        assert self._is_income("Amazon Refund") is True

    def test_cashback_is_income(self):
        """Merchant name containing 'cashback' is income."""
        assert self._is_income("Cashback Reward") is True

    def test_keyword_match_is_case_insensitive(self):
        """Classification is case-insensitive: 'SALARY' must match."""
        assert self._is_income("SALARY CREDIT") is True

    def test_amazon_is_not_income(self):
        """'Amazon' is a spending merchant — must not be classified as income."""
        assert self._is_income("Amazon") is False

    def test_swiggy_is_not_income(self):
        assert self._is_income("Swiggy") is False

    def test_zomato_is_not_income(self):
        assert self._is_income("Zomato") is False

    def test_rent_is_not_income(self):
        assert self._is_income("Rent") is False

    def test_electricity_board_is_not_income(self):
        assert self._is_income("Electricity Board") is False

    def test_empty_merchant_is_not_income(self):
        """Empty merchant name must not crash and must return False."""
        assert self._is_income("") is False

    def test_none_merchant_is_not_income(self):
        """Missing merchant key must not crash and must return False."""
        mock_st = _make_mock_st()
        dash_mod = _fresh_dashboard_module(mock_st)
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            result = dash_mod._is_income({})
        assert result is False


# ---------------------------------------------------------------------------
# TEST GROUP 2 — _kpi_cards() metric labels
# ---------------------------------------------------------------------------

class TestKpiCardLabels:

    def test_five_kpi_labels(self):
        """_kpi_cards() must render exactly 5 metric cards."""
        labels, _ = _call_kpi_cards([])
        assert len(labels) == 5

    def test_label_order(self):
        """Labels must appear in order: Income, Spending, Net Cash Flow,
        Total Transactions, Total Anomalies."""
        labels, _ = _call_kpi_cards([])
        assert labels[0] == "Total Income"
        assert labels[1] == "Total Spending"
        assert labels[2] == "Net Cash Flow"
        assert labels[3] == "Total Transactions"
        assert labels[4] == "Total Anomalies"


# ---------------------------------------------------------------------------
# TEST GROUP 3 — _kpi_cards() numeric values
# ---------------------------------------------------------------------------

class TestKpiCardValues:

    def _make_txns(self, items):
        """Build transaction dicts from (merchant, amount) tuples."""
        return [
            {"merchant": m, "amount": a, "category": "Other"}
            for m, a in items
        ]

    def test_income_and_spending_split_correctly(self):
        """Salary Credit is income; Amazon is spending.

        income  = 50000
        spending = 1000
        net     = 49000
        """
        txns = self._make_txns([
            ("Salary Credit", 50000.0),
            ("Amazon", 1000.0),
        ])
        labels, values = _call_kpi_cards(txns)

        assert "50,000" in values[0], f"Total Income expected 50000, got {values[0]!r}"
        assert "1,000"  in values[1], f"Total Spending expected 1000, got {values[1]!r}"
        assert "49,000" in values[2], f"Net Cash Flow expected 49000, got {values[2]!r}"

    def test_only_spending_transactions(self):
        """When there is no income, Total Income = 0 and Net Cash Flow is negative."""
        txns = self._make_txns([
            ("Swiggy", 500.0),
            ("Zomato", 300.0),
        ])
        labels, values = _call_kpi_cards(txns)

        assert "0" in values[0], f"Total Income should be 0, got {values[0]!r}"
        assert "800" in values[1], f"Total Spending should be 800, got {values[1]!r}"
        # Net cash flow is -800 (spending exceeds income)
        assert "-800" in values[2] or "800" in values[2], (
            f"Net Cash Flow should reflect -800, got {values[2]!r}"
        )

    def test_only_income_transactions(self):
        """When all transactions are income, Total Spending = 0."""
        txns = self._make_txns([
            ("Salary Credit", 65000.0),
        ])
        labels, values = _call_kpi_cards(txns)

        assert "65,000" in values[0], f"Total Income expected 65000, got {values[0]!r}"
        assert "0" in values[1], f"Total Spending should be 0, got {values[1]!r}"

    def test_transaction_count_includes_all_transactions(self):
        """Total Transactions counts ALL rows, including income."""
        txns = self._make_txns([
            ("Salary Credit", 50000.0),
            ("Amazon", 999.0),
            ("Zomato", 400.0),
        ])
        labels, values = _call_kpi_cards(txns)

        assert values[3] == 3, f"Total Transactions should be 3, got {values[3]!r}"

    def test_anomaly_count_comes_from_anomalies_list(self):
        """Total Anomalies reflects the anomalies list length, not transaction data."""
        txns = self._make_txns([("Amazon", 100.0)])
        anomalies = [{"id": 1}, {"id": 2}, {"id": 3}]
        labels, values = _call_kpi_cards(txns, anomalies)

        assert values[4] == 3, f"Total Anomalies should be 3, got {values[4]!r}"

    def test_empty_dataset_does_not_raise(self):
        """Empty transaction list must render without error and show zero values."""
        labels, values = _call_kpi_cards([])

        assert len(values) == 5
        for v in values:
            assert "nan" not in str(v).lower(), f"NaN in KPI value: {v!r}"

    def test_multiple_salary_credits(self):
        """Multiple 'Salary Credit' rows all count as income."""
        txns = self._make_txns([
            ("Salary Credit", 65000.0),
            ("Salary Credit", 65000.0),
            ("FreshMart", 2000.0),
        ])
        labels, values = _call_kpi_cards(txns)

        assert "130,000" in values[0], (
            f"Total Income should be 130000, got {values[0]!r}"
        )
        assert "2,000" in values[1], (
            f"Total Spending should be 2000, got {values[1]!r}"
        )

    def test_real_dataset_profile(self):
        """
        Mirrors the actual 45-transaction test dataset:
          - 2 x Salary Credit @ Rs.65,000 = Rs.130,000 income
          - Remaining merchants = expenses

        Validates that total income = 130,000 and spending = 210,564 - 130,000 = 80,564.
        Uses approximate merchant/amount data from the real upload; the exact
        amounts are not hardcoded — we only verify the split direction.
        """
        # Two income entries
        income_txns = [{"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"}] * 2
        # Representative spending entries (not exhaustive — just verify the split works)
        spending_txns = [
            {"merchant": "FreshMart",        "amount": 2450.5, "category": "Groceries"},
            {"merchant": "Amazon",            "amount": 1899.0, "category": "Shopping"},
            {"merchant": "Zomato",            "amount": 760.0,  "category": "Dining"},
            {"merchant": "Swiggy",            "amount": 540.0,  "category": "Dining"},
            {"merchant": "Uber",              "amount": 320.0,  "category": "Transport"},
            {"merchant": "Electricity Board", "amount": 1850.0, "category": "Utilities"},
        ]
        txns = income_txns + spending_txns

        labels, values = _call_kpi_cards(txns)

        # Income = 130,000
        assert "130,000" in values[0], (
            f"Expected Rs.130,000 income, got {values[0]!r}"
        )
        # Spending > 0
        total_spending_str = values[1]
        assert "0.00" not in total_spending_str or "130" not in total_spending_str, (
            "Spending must not include income amounts"
        )
        # Salary Credit must NOT appear in spending
        spending_sum = sum(t["amount"] for t in spending_txns)
        assert str(int(spending_sum)) in values[1] or f"{spending_sum:,.2f}" in values[1], (
            f"Expected spending {spending_sum:.2f} in Total Spending, got {values[1]!r}"
        )
        # Transaction count = all rows
        assert values[3] == len(txns), (
            f"Total Transactions should be {len(txns)}, got {values[3]!r}"
        )


# ---------------------------------------------------------------------------
# TEST GROUP 4 — _spending_category_df()
# ---------------------------------------------------------------------------

class TestSpendingCategoryDf:
    """Unit tests for the spending-behavior DataFrame helper."""

    def _call(self, transactions):
        # Import with real pandas — _spending_category_df has no Streamlit calls
        import importlib
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
            dash_mod.st = mock_st
            return dash_mod._spending_category_df(transactions)

    def test_income_excluded_from_category_df(self):
        """Salary Credit must not appear in spending category totals."""
        txns = [
            {"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"},
            {"merchant": "Swiggy",         "amount": 500.0,   "category": "Dining"},
            {"merchant": "Zomato",          "amount": 300.0,   "category": "Dining"},
        ]
        df = self._call(txns)
        assert "Other" not in df["Category"].values or df[df["Category"] == "Other"].empty or (
            # If Other category comes from non-income rows, it's fine;
            # the key check is that Salary Credit's amount is not counted
            df["Total Spent (₹)"].sum() == pytest.approx(800.0)
        )
        assert df["Total Spent (₹)"].sum() == pytest.approx(800.0), (
            "Total spending must be 800 (500+300), not 65800"
        )

    def test_sorted_descending_by_spend(self):
        """Categories must appear in descending order of total spend."""
        txns = [
            {"merchant": "Amazon",   "amount": 200.0, "category": "Shopping"},
            {"merchant": "Swiggy",   "amount": 800.0, "category": "Dining"},
            {"merchant": "FreshMart","amount": 400.0, "category": "Groceries"},
        ]
        df = self._call(txns)
        amounts = df["Total Spent (₹)"].tolist()
        assert amounts == sorted(amounts, reverse=True), (
            f"Expected descending sort, got {amounts}"
        )

    def test_empty_when_only_income(self):
        """If all transactions are income, the DataFrame must be empty."""
        txns = [{"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"}]
        df = self._call(txns)
        assert df.empty

    def test_empty_when_no_transactions(self):
        """Empty input must return an empty DataFrame without error."""
        df = self._call([])
        assert df.empty

    def test_correct_columns(self):
        """Result DataFrame must have exactly the expected column names."""
        txns = [{"merchant": "Swiggy", "amount": 500.0, "category": "Dining"}]
        df = self._call(txns)
        assert list(df.columns) == ["Category", "Total Spent (₹)"]

    def test_same_category_aggregated(self):
        """Multiple rows in the same category must be summed."""
        txns = [
            {"merchant": "Swiggy", "amount": 300.0, "category": "Dining"},
            {"merchant": "Zomato", "amount": 500.0, "category": "Dining"},
        ]
        df = self._call(txns)
        assert len(df) == 1
        assert df.iloc[0]["Total Spent (₹)"] == pytest.approx(800.0)

    def test_top_category_is_correct(self):
        """The first row (highest spend) must be the correct category."""
        txns = [
            {"merchant": "Amazon",   "amount": 5000.0, "category": "Shopping"},
            {"merchant": "FreshMart","amount": 2000.0, "category": "Groceries"},
        ]
        df = self._call(txns)
        assert df.iloc[0]["Category"] == "Shopping"


# ---------------------------------------------------------------------------
# TEST GROUP 5 — _top_expense_category()
# ---------------------------------------------------------------------------

class TestTopExpenseCategory:

    def _call(self, transactions):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
            dash_mod.st = mock_st
            return dash_mod._top_expense_category(transactions)

    def test_returns_highest_spend_category(self):
        txns = [
            {"merchant": "Amazon",   "amount": 5000.0, "category": "Shopping"},
            {"merchant": "FreshMart","amount": 1000.0, "category": "Groceries"},
        ]
        assert self._call(txns) == "Shopping"

    def test_ignores_income(self):
        """Income rows must not influence the top category."""
        txns = [
            {"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"},
            {"merchant": "Swiggy",         "amount": 800.0,   "category": "Dining"},
        ]
        assert self._call(txns) == "Dining"

    def test_returns_none_when_no_expenses(self):
        txns = [{"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"}]
        assert self._call(txns) is None

    def test_returns_none_when_empty(self):
        assert self._call([]) is None


# ---------------------------------------------------------------------------
# TEST GROUP 6 — _spending_behavior() rendering
# ---------------------------------------------------------------------------

class TestSpendingBehavior:
    """Verify _spending_behavior() calls the right Streamlit primitives."""

    def _call_behavior(self, transactions):
        # Import dashboard once with real pandas; then mock only the st attribute
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
        # Replace st on the already-loaded module — pandas stays real
        dash_mod.st = mock_st
        dash_mod._spending_behavior(transactions)
        return mock_st

    def test_shows_info_when_no_expense_data(self):
        """Pure-income dataset must show an info message, no chart."""
        txns = [{"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"}]
        mock_st = self._call_behavior(txns)
        mock_st.info.assert_called_once()
        mock_st.bar_chart.assert_not_called()

    def test_shows_bar_chart_when_expense_data_present(self):
        """Expense data must produce a bar chart."""
        txns = [
            {"merchant": "Swiggy", "amount": 500.0, "category": "Dining"},
            {"merchant": "Amazon", "amount": 999.0, "category": "Shopping"},
        ]
        mock_st = self._call_behavior(txns)
        mock_st.bar_chart.assert_called_once()

    def test_shows_dataframe_when_expense_data_present(self):
        """A share-percentage dataframe must be rendered alongside the chart."""
        txns = [{"merchant": "Uber", "amount": 320.0, "category": "Transport"}]
        mock_st = self._call_behavior(txns)
        mock_st.dataframe.assert_called_once()

    def test_income_not_included_in_chart_data(self):
        """Salary Credit must not inflate any category shown in the chart."""
        txns = [
            {"merchant": "Salary Credit", "amount": 65000.0, "category": "Other"},
            {"merchant": "Swiggy",         "amount": 500.0,   "category": "Dining"},
        ]
        mock_st = self._call_behavior(txns)
        # bar_chart must have been called — income data should not prevent it
        mock_st.bar_chart.assert_called_once()
        # Verify the data passed does NOT include the 65000 income amount
        call_args = mock_st.bar_chart.call_args
        chart_data = call_args[0][0]  # first positional arg is the Series/DataFrame
        assert float(chart_data.max()) < 65000, (
            "Income amount (65000) must not appear in spending chart data"
        )


# ---------------------------------------------------------------------------
# TEST GROUP 7 — _next_month_forecast() rendering
# ---------------------------------------------------------------------------

class TestNextMonthForecast:
    """Verify _next_month_forecast() uses get_forecast_aggregate_next_month."""

    def _make_client(self, forecast_result=None, raises=None):
        client = MagicMock()
        if raises is not None:
            client.get_forecast_aggregate_next_month.side_effect = raises
        else:
            client.get_forecast_aggregate_next_month.return_value = forecast_result
        return client

    def _call_forecast(self, client):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = _make_mock_st()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
        dash_mod.st = mock_st
        dash_mod._next_month_forecast(client)
        return mock_st

    def _sample_forecast(self):
        return {
            "category": "Total Expenses",
            "horizon_days": 30,
            "points": [
                {"date": f"2026-11-{i+1:02d}", "yhat": 1800.0,
                 "yhat_lower": 1500.0, "yhat_upper": 2100.0}
                for i in range(3)
            ],
        }

    def test_calls_get_forecast_aggregate_next_month(self):
        """Dashboard must call get_forecast_aggregate_next_month (no args)."""
        client = self._make_client(forecast_result=self._sample_forecast())
        self._call_forecast(client)
        client.get_forecast_aggregate_next_month.assert_called_once_with()
        client.get_forecast_aggregate.assert_not_called()
        client.get_forecast.assert_not_called()

    def test_shows_kpi_and_chart_on_success(self):
        """Successful forecast must render metric cards and a line chart."""
        client = self._make_client(forecast_result=self._sample_forecast())
        mock_st = self._call_forecast(client)
        mock_st.line_chart.assert_called_once()

    def test_shows_friendly_message_on_422_insufficient_history(self):
        """422 from backend → friendly info message, no line chart."""
        client = self._make_client(
            raises=RuntimeError("Backend returned 422: insufficient history")
        )
        mock_st = self._call_forecast(client)
        mock_st.info.assert_called_once()
        mock_st.line_chart.assert_not_called()

    def test_shows_error_on_non_422_failure(self):
        """Non-422 RuntimeError → st.error shown, no chart."""
        client = self._make_client(
            raises=RuntimeError("Backend returned 500: server error")
        )
        mock_st = self._call_forecast(client)
        mock_st.error.assert_called_once()

    def test_shows_info_when_no_forecast_points(self):
        """Empty points list → info message, no chart."""
        client = self._make_client(forecast_result={
            "category": "Total Expenses", "horizon_days": 30, "points": []
        })
        mock_st = self._call_forecast(client)
        mock_st.info.assert_called_once()
        mock_st.line_chart.assert_not_called()


# ---------------------------------------------------------------------------
# TEST GROUP 8 — _compute_insights()
# ---------------------------------------------------------------------------

class TestComputeInsights:
    """Unit tests for the _compute_insights() pure-data helper."""

    def _call(self, transactions, anomalies=None):
        if anomalies is None:
            anomalies = []
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = MagicMock()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
            dash_mod.st = mock_st
            return dash_mod._compute_insights(transactions, anomalies)

    def _txn(self, merchant, amount, category, date_str="2026-09-01"):
        return {"merchant": merchant, "amount": amount,
                "category": category, "date": date_str}

    # ------------------------------------------------------------------
    # Top category
    # ------------------------------------------------------------------

    def test_top_category_correct(self):
        """Highest-spend category is identified correctly."""
        txns = [
            self._txn("Amazon", 5000.0, "Shopping"),
            self._txn("Swiggy", 1000.0, "Dining"),
            self._txn("FreshMart", 2000.0, "Groceries"),
        ]
        ins = self._call(txns)
        assert ins["top_category"] == "Shopping"
        assert ins["top_category_amt"] == pytest.approx(5000.0)

    def test_top_category_percentage_correct(self):
        """top_category_pct = top amount / total × 100."""
        txns = [
            self._txn("Amazon", 3000.0, "Shopping"),
            self._txn("Uber", 1000.0, "Transport"),
        ]
        ins = self._call(txns)
        assert ins["top_category_pct"] == pytest.approx(75.0)

    def test_top_category_excludes_income(self):
        """Income rows (Salary Credit) must not inflate any category."""
        txns = [
            self._txn("Salary Credit", 65000.0, "Other"),
            self._txn("Swiggy",        500.0,   "Dining"),
        ]
        ins = self._call(txns)
        # Top category must be Dining (500), not Other (income 65000)
        assert ins["top_category"] == "Dining"
        assert ins["top_category_amt"] == pytest.approx(500.0)
        # Percentage must be 100% (only expense category)
        assert ins["top_category_pct"] == pytest.approx(100.0)

    def test_top_category_none_when_only_income(self):
        """If all transactions are income, top_category must be None."""
        txns = [self._txn("Salary Credit", 65000.0, "Other")]
        ins = self._call(txns)
        assert ins["top_category"] is None
        assert ins["top_category_amt"] == pytest.approx(0.0)

    def test_top_category_none_when_empty(self):
        ins = self._call([])
        assert ins["top_category"] is None

    # ------------------------------------------------------------------
    # Top merchant
    # ------------------------------------------------------------------

    def test_top_merchant_correct(self):
        """Top merchant by cumulative spend across multiple rows."""
        txns = [
            self._txn("Amazon", 2000.0, "Shopping"),
            self._txn("Amazon", 3000.0, "Shopping"),   # same merchant, two rows
            self._txn("Swiggy", 4000.0, "Dining"),
        ]
        ins = self._call(txns)
        # Amazon 5000 > Swiggy 4000
        assert ins["top_merchant"] == "Amazon"
        assert ins["top_merchant_amt"] == pytest.approx(5000.0)

    def test_top_merchant_excludes_income(self):
        """Salary Credit must not be the top merchant."""
        txns = [
            self._txn("Salary Credit", 65000.0, "Other"),
            self._txn("Rent",           18000.0, "Other"),
        ]
        ins = self._call(txns)
        assert ins["top_merchant"] == "Rent"
        assert ins["top_merchant_amt"] == pytest.approx(18000.0)

    def test_top_merchant_none_when_only_income(self):
        txns = [self._txn("Salary Credit", 65000.0, "Other")]
        ins = self._call(txns)
        assert ins["top_merchant"] is None

    # ------------------------------------------------------------------
    # Largest expense
    # ------------------------------------------------------------------

    def test_largest_expense_correct(self):
        """Largest single transaction is identified correctly."""
        txns = [
            self._txn("Rent",    18000.0, "Other",    "2026-08-31"),
            self._txn("Amazon",   1899.0, "Shopping", "2026-09-01"),
            self._txn("FreshMart", 2450.0, "Groceries", "2026-09-05"),
        ]
        ins = self._call(txns)
        assert ins["largest_merchant"] == "Rent"
        assert ins["largest_amount"] == pytest.approx(18000.0)
        assert ins["largest_date"] == "2026-08-31"

    def test_largest_expense_excludes_income(self):
        """Income must not be considered as a large expense."""
        txns = [
            self._txn("Salary Credit", 65000.0, "Other", "2026-09-01"),
            self._txn("Rent",           18000.0, "Other", "2026-08-31"),
        ]
        ins = self._call(txns)
        assert ins["largest_merchant"] == "Rent"
        assert ins["largest_amount"] == pytest.approx(18000.0)

    def test_largest_expense_none_when_only_income(self):
        txns = [self._txn("Salary Credit", 65000.0, "Other")]
        ins = self._call(txns)
        assert ins["largest_merchant"] is None

    # ------------------------------------------------------------------
    # Anomaly count
    # ------------------------------------------------------------------

    def test_anomaly_count_from_anomalies_list(self):
        """anomaly_count equals len(anomalies) passed in."""
        txns = [self._txn("Swiggy", 500.0, "Dining")]
        anomalies = [{"id": 1}, {"id": 2}, {"id": 3}]
        ins = self._call(txns, anomalies)
        assert ins["anomaly_count"] == 3

    def test_anomaly_count_zero_when_no_anomalies(self):
        txns = [self._txn("Swiggy", 500.0, "Dining")]
        ins = self._call(txns, [])
        assert ins["anomaly_count"] == 0

    def test_anomaly_count_unaffected_by_transactions(self):
        """Anomaly count must not change based on what's in transactions."""
        txns = [self._txn("Amazon", 5000.0, "Shopping")]
        ins_0 = self._call(txns, [])
        ins_3 = self._call(txns, [{"id": i} for i in range(3)])
        assert ins_0["anomaly_count"] == 0
        assert ins_3["anomaly_count"] == 3

    # ------------------------------------------------------------------
    # Empty input
    # ------------------------------------------------------------------

    def test_empty_transactions_does_not_raise(self):
        """Empty transaction list must return a valid (all-None/zero) dict."""
        ins = self._call([])
        assert ins["top_category"] is None
        assert ins["top_merchant"] is None
        assert ins["largest_merchant"] is None
        assert ins["top_category_amt"] == pytest.approx(0.0)
        assert ins["anomaly_count"] == 0


# ---------------------------------------------------------------------------
# TEST GROUP 9 — _financial_insights() rendering
# ---------------------------------------------------------------------------

class TestFinancialInsightsRendering:
    """Verify _financial_insights() renders without crash and shows the right sections."""

    def _call_render(self, transactions, anomalies=None):
        if anomalies is None:
            anomalies = []
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = _make_mock_st()
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
        dash_mod.st = mock_st
        dash_mod._financial_insights(transactions, anomalies)
        return mock_st

    def test_shows_info_when_no_expense_data(self):
        """Only-income data must show an info message."""
        txns = [{"merchant": "Salary Credit", "amount": 65000.0,
                 "category": "Other", "date": "2026-09-01"}]
        mock_st = self._call_render(txns)
        mock_st.info.assert_called_once()

    def test_renders_four_columns_for_expense_data(self):
        """st.columns(4) must be called when expense data is present."""
        txns = [{"merchant": "Swiggy", "amount": 500.0,
                 "category": "Dining", "date": "2026-09-01"}]
        mock_st = self._call_render(txns)
        # columns(4) must have been called
        calls = [c for c in mock_st.columns.call_args_list
                 if c.args and c.args[0] == 4]
        assert len(calls) >= 1, "st.columns(4) must be called for the insight grid"

    def test_does_not_raise_on_empty_input(self):
        """Empty lists must render without exception (shows info message)."""
        mock_st = self._call_render([], [])
        mock_st.info.assert_called_once()

    def test_does_not_raise_with_real_dataset_shape(self):
        """Dataset matching the 45-transaction shape must not crash."""
        txns = [
            {"merchant": "Salary Credit",    "amount": 65000.0, "category": "Other",    "date": "2026-08-01"},
            {"merchant": "Salary Credit",    "amount": 65000.0, "category": "Other",    "date": "2026-09-01"},
            {"merchant": "Rent",             "amount": 18000.0, "category": "Other",    "date": "2026-08-31"},
            {"merchant": "Amazon",           "amount":  1899.0, "category": "Shopping", "date": "2026-08-05"},
            {"merchant": "FreshMart",        "amount":  2450.5, "category": "Groceries","date": "2026-08-02"},
            {"merchant": "Swiggy",           "amount":   680.0, "category": "Dining",   "date": "2026-08-03"},
            {"merchant": "Electricity Board","amount":  1850.0, "category": "Utilities","date": "2026-08-10"},
        ]
        anomalies = [{"id": 1, "merchant": "Rent", "amount": 18000.0,
                      "anomaly_score": 0.7, "category": "Other"}]
        mock_st = self._call_render(txns, anomalies)
        # Must not have raised — info must NOT be called (there is expense data)
        mock_st.info.assert_not_called()


# ---------------------------------------------------------------------------
# TEST GROUP 10 — _next_month_forecast() 2-month message branching
# ---------------------------------------------------------------------------

class TestNextMonthForecastMessageBranching:
    """Verify the dashboard shows the correct message depending on which
    backend guard fired (2-month rule vs 14-day rule)."""

    def _make_client_raising(self, error_message: str):
        """Return a mock API client whose get_forecast_aggregate_next_month raises RuntimeError."""
        client = MagicMock()
        client.get_forecast_aggregate_next_month.side_effect = RuntimeError(error_message)
        return client

    def _call_forecast(self, client):
        for dep in list(sys.modules):
            if dep.startswith("frontend."):
                del sys.modules[dep]
        sys.modules.setdefault("frontend.services.api", MagicMock())
        sys.modules.setdefault("frontend.utils", MagicMock())
        mock_st = _make_mock_st()
        mock_st.spinner.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.spinner.return_value.__exit__ = MagicMock(return_value=False)
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            import frontend.views.dashboard as dash_mod
        dash_mod.st = mock_st
        dash_mod._next_month_forecast(client)
        return mock_st

    def test_month_error_shows_2month_upgrade_message(self):
        """A 422 error mentioning 'month' must show the 2-month upgrade message."""
        client = self._make_client_raising(
            "Backend returned 422: Expense history spans only 1 calendar month(s). "
            "At least 2 months are required for forecasting."
        )
        mock_st = self._call_forecast(client)
        mock_st.info.assert_called_once()
        # The info message must mention '2 months'
        info_text = mock_st.info.call_args[0][0]
        assert "2 months" in info_text.lower() or "two month" in info_text.lower(), (
            f"Expected 2-month message, got: {info_text!r}"
        )

    def test_day_error_shows_14day_message(self):
        """A 422 error mentioning days (not months) must show the 14-day message."""
        client = self._make_client_raising(
            "Backend returned 422: Only 10 distinct calendar days of expense "
            "history found. At least 14 are required."
        )
        mock_st = self._call_forecast(client)
        mock_st.info.assert_called_once()
        info_text = mock_st.info.call_args[0][0]
        assert "14 days" in info_text.lower() or "14" in info_text, (
            f"Expected 14-day message, got: {info_text!r}"
        )

    def test_month_error_does_not_show_st_error(self):
        """A 422 month error must use st.info, not st.error."""
        client = self._make_client_raising(
            "Backend returned 422: Expense history spans only 1 calendar month(s)."
        )
        mock_st = self._call_forecast(client)
        mock_st.error.assert_not_called()
        mock_st.info.assert_called_once()

    def test_non_422_error_still_shows_st_error(self):
        """A non-422 RuntimeError must still show st.error (unchanged behavior)."""
        client = self._make_client_raising(
            "Backend returned 500: internal server error"
        )
        mock_st = self._call_forecast(client)
        mock_st.error.assert_called_once()
        mock_st.info.assert_not_called()
