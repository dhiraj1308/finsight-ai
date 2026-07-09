"""LangChain tools for the FinSight AI financial agent.

Each tool is designed to return compact, factual summaries — never raw
transaction lists — so that the combined prompt stays well below the
Groq TPM limit of 6 000 tokens.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from langchain.tools import tool

logger = logging.getLogger(__name__)

# Hard caps on how many rows each tool may include in its output.
# Keep these small: every character returned by a tool ends up in the
# next LLM reasoning step as part of the prompt context.
_MAX_RETRIEVE = 6        # vector-search results
_MAX_ANOMALIES = 5       # anomaly rows
_MAX_TOP_MERCHANTS = 8   # rows in the top-merchants list
_MAX_OUTPUT_CHARS = 800  # safety truncation for any single tool output


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Truncate *text* to *limit* characters, appending an ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def create_tools(store, vector_store, forecaster, anomaly_detector):
    """Create and return the list of LangChain tools bound to the given components."""

    @tool
    def retrieve_transactions(query: str) -> str:
        """Find transactions semantically relevant to a natural-language query.

        Use this when the user asks about specific merchants, purchases, or
        spending in a particular context. Returns up to 6 matching transactions
        with date, merchant, amount and category — nothing else.

        Args:
            query: Plain-language description, e.g. 'dining out last month'.
        """
        try:
            results = vector_store.search(query, k=_MAX_RETRIEVE)
            if not results:
                return "No transactions found matching that query."
            lines = [f"{len(results)} relevant transaction(s):"]
            for t in results:
                lines.append(
                    f"  {t.date}  {t.merchant:<22}  ₹{t.amount:>9.2f}  {t.category}"
                )
            return _truncate("\n".join(lines))
        except Exception as exc:
            logger.error("retrieve_transactions error: %s", exc)
            return "Error retrieving transactions."

    @tool
    def get_spending_summary(category: str = "all") -> str:
        """Return total and transaction count for a category (or all categories).

        Use this to answer 'how much did I spend on X', 'what is my total
        spending', 'which category costs the most', etc.
        Never returns individual transactions — only aggregated totals.

        Args:
            category: A category name such as 'Groceries', or 'all' for a
                      breakdown across every category.
        """
        try:
            category = category.strip().strip('"').strip("'")
            if not category or category.lower() == "all":
                txns = store.get_all()
                if not txns:
                    return "No transactions stored yet."
                totals: dict[str, float] = defaultdict(float)
                counts: dict[str, int] = defaultdict(int)
                for t in txns:
                    totals[t.category] += t.amount
                    counts[t.category] += 1
                grand = sum(totals.values())
                lines = [
                    f"Total spending: ₹{grand:.2f} across {len(txns)} transactions.",
                    "By category:",
                ]
                for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
                    lines.append(
                        f"  {cat:<25}  ₹{amt:>10.2f}  ({counts[cat]} txns)"
                    )
                return _truncate("\n".join(lines))
            else:
                txns = store.query_by_category(category)
                if not txns:
                    return f"No transactions found for category '{category}'."
                total = sum(t.amount for t in txns)
                return (
                    f"Category '{category}': "
                    f"₹{total:.2f} total across {len(txns)} transaction(s)."
                )
        except Exception as exc:
            logger.error("get_spending_summary error: %s", exc)
            return "Error calculating spending summary."

    @tool
    def get_top_merchants(limit: int = 8) -> str:
        """Return the top merchants by total spending.

        Use this when the user asks 'where did I spend the most',
        'biggest purchases', 'top merchants', etc.

        Args:
            limit: Number of merchants to return (default 8, max 10).
        """
        try:
            limit = min(max(1, limit), 10)
            txns = store.get_all()
            if not txns:
                return "No transactions stored yet."
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for t in txns:
                totals[t.merchant] += t.amount
                counts[t.merchant] += 1
            ranked = sorted(totals.items(), key=lambda x: -x[1])[:limit]
            lines = [f"Top {len(ranked)} merchants by total spending:"]
            for rank, (merchant, amt) in enumerate(ranked, start=1):
                lines.append(
                    f"  {rank}. {merchant:<25}  ₹{amt:>10.2f}  ({counts[merchant]} txns)"
                )
            return _truncate("\n".join(lines))
        except Exception as exc:
            logger.error("get_top_merchants error: %s", exc)
            return "Error retrieving top merchants."

    @tool
    def run_forecast(category: str, horizon_days: int = 30) -> str:
        """Forecast future spending for a category over a given number of days.

        Use this to answer questions about future spending predictions or budgeting.

        Args:
            category: Spending category, e.g. 'Groceries'.
            horizon_days: Days to forecast ahead (1-90, default 30).
        """
        try:
            horizon_days = min(max(1, horizon_days), 90)
            forecast = forecaster.forecast_category(category, horizon_days, store)
            total = sum(p.yhat for p in forecast.points)
            avg_daily = total / horizon_days if horizon_days > 0 else 0.0
            first = forecast.points[0]
            last = forecast.points[-1]
            return _truncate(
                f"Forecast for '{category}' over next {horizon_days} days:\n"
                f"  Projected total: ₹{total:.2f}\n"
                f"  Average daily:   ₹{avg_daily:.2f}\n"
                f"  Period: {first.date} → {last.date}\n"
                f"  Day 1 estimate: ₹{first.yhat:.2f} "
                f"(₹{first.yhat_lower:.2f}–₹{first.yhat_upper:.2f})"
            )
        except ValueError as exc:
            return f"Cannot forecast '{category}': {exc}"
        except Exception as exc:
            logger.error("run_forecast error: %s", exc)
            return "Error running forecast."

    @tool
    def get_anomalies(limit: int = 5) -> str:
        """Return the top anomalous transactions detected by the AI model.

        Use this when the user asks about unusual charges, suspicious
        transactions, potential fraud, or spending anomalies.

        Args:
            limit: Maximum number of anomalies to return (default 5, max 10).
        """
        try:
            limit = min(max(1, limit), 10)
            anomalies = anomaly_detector.get_anomalies(store)
            if not anomalies:
                return "No anomalous transactions detected."
            top = anomalies[:limit]
            lines = [
                f"{len(anomalies)} anomalies detected. "
                f"Top {len(top)}:"
            ]
            for t in top:
                score = f"{t.anomaly_score:.2f}" if t.anomaly_score is not None else "?"
                lines.append(
                    f"  {t.date}  {t.merchant:<22}  ₹{t.amount:>9.2f}"
                    f"  {t.category}  score={score}"
                )
            return _truncate("\n".join(lines))
        except Exception as exc:
            logger.error("get_anomalies error: %s", exc)
            return "Error retrieving anomalies."

    return [
        retrieve_transactions,
        get_spending_summary,
        get_top_merchants,
        run_forecast,
        get_anomalies,
    ]
