from __future__ import annotations

import logging
from typing import Optional

from langchain.tools import tool

logger = logging.getLogger(__name__)


def create_tools(store, vector_store, forecaster, anomaly_detector):
    """
    Creates LangChain tools bound to the given data components.
    Returns a list of tools the agent can invoke.
    """

    @tool
    def retrieve_transactions(query: str, k: int = 5) -> str:
        """
        Retrieve transactions semantically similar to a natural language query.
        Use this to find relevant transactions when answering questions about
        specific spending, merchants, or categories.
        Args:
            query: Natural language description of transactions to find
            k: Number of results to return (default 5, max 20)
        """
        try:
            k = min(max(1, k), 20)
            results = vector_store.search(query, k=k)
            if not results:
                return "No transactions found matching that query."
            lines = [f"Found {len(results)} transaction(s):"]
            for txn in results:
                lines.append(
                    f"- {txn.date}: {txn.merchant} | "
                    f"${txn.amount:.2f} | {txn.category}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"retrieve_transactions error: {e}")
            return f"Error retrieving transactions: {str(e)}"

    @tool
    def calculate_total(query: str) -> str:
        """
        Calculate total spending. Use this to answer questions like
        'how much did I spend on X' or 'what was my total spending'.
        Args:
            query: Either a category name (e.g. 'Groceries') or 'all' for
                   total spending across all categories. Pass just the
                   category name as plain text, no extra formatting.
        """
        try:
            cleaned = query.strip()
            if "=" in cleaned:
                cleaned = cleaned.split("=", 1)[1]
            cleaned = cleaned.strip().strip('"').strip("'").strip()

            if not cleaned or cleaned.lower() == "all":
                txns = store.get_all()
                filter_desc = "all transactions"
            else:
                txns = store.query_by_category(cleaned)
                filter_desc = cleaned

            if not txns:
                return f"No transactions found for {filter_desc}."

            total = sum(t.amount for t in txns)
            return (
                f"Total spending for {filter_desc}: "
                f"${total:.2f} across {len(txns)} transaction(s)."
            )
        except Exception as e:
            logger.error(f"calculate_total error: {e}")
            return f"Error calculating total: {str(e)}"

    @tool
    def run_forecast(category: str, horizon_days: int = 30) -> str:
        """
        Forecast future spending for a category over a given number of days.
        Use this to answer questions about future spending predictions or budgeting.
        Args:
            category: The spending category to forecast (e.g. 'Groceries')
            horizon_days: Number of days to forecast ahead (1-365, default 30)
        """
        try:
            forecast = forecaster.forecast_category(category, horizon_days, store)
            total_forecast = sum(p.yhat for p in forecast.points)
            avg_daily = total_forecast / horizon_days if horizon_days > 0 else 0
            first = forecast.points[0]
            last = forecast.points[-1]
            return (
                f"Forecast for {category} over next {horizon_days} days:\n"
                f"- Projected total: ${total_forecast:.2f}\n"
                f"- Average daily: ${avg_daily:.2f}\n"
                f"- From {first.date} to {last.date}\n"
                f"- First day estimate: ${first.yhat:.2f} "
                f"(range: ${first.yhat_lower:.2f} - ${first.yhat_upper:.2f})"
            )
        except ValueError as e:
            return f"Cannot forecast {category}: {str(e)}"
        except Exception as e:
            logger.error(f"run_forecast error: {e}")
            return f"Error running forecast: {str(e)}"

    @tool
    def get_anomalies(limit: int = 10) -> str:
        """
        Get transactions flagged as anomalies (unusual spending patterns).
        Use this to answer questions about suspicious charges, unusual spending,
        or potential fraud.
        Args:
            limit: Maximum number of anomalies to return (default 10)
        """
        try:
            anomalies = anomaly_detector.get_anomalies(store)
            if not anomalies:
                return "No anomalous transactions detected."
            top = anomalies[:limit]
            lines = [f"Found {len(anomalies)} anomalous transaction(s). Top {len(top)}:"]
            for txn in top:
                score = f"{txn.anomaly_score:.3f}" if txn.anomaly_score else "N/A"
                lines.append(
                    f"- {txn.date}: {txn.merchant} | "
                    f"${txn.amount:.2f} | {txn.category} | score: {score}"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"get_anomalies error: {e}")
            return f"Error getting anomalies: {str(e)}"

    return [retrieve_transactions, calculate_total, run_forecast, get_anomalies]
