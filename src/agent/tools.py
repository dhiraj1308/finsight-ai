"""Plain Python tool functions for the FinSight AI direct-dispatch agent.

No LangChain decorators are used here — each function is a regular callable
that the agent dispatches to directly after parsing the LLM's JSON reply.

All outputs use ASCII-safe currency notation (Rs.) instead of the Unicode
rupee sign (U+20B9) which causes UnicodeEncodeError on Windows cp1252.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta

logger = logging.getLogger(__name__)

_MAX_OUTPUT_CHARS = 800

# ---------------------------------------------------------------------------
# Tool descriptions sent to the LLM in the dispatch prompt
# ---------------------------------------------------------------------------
TOOL_DESCRIPTIONS = """\
1. get_spending_summary(category)
   Total spending for one category, or all categories if category="all".
   Use for: "how much on dining", "total groceries", "spending on food", etc.
   category values: Dining, Groceries, Transport, Healthcare, Shopping,
                    Utilities, Entertainment, Subscriptions, Other, all

2. get_top_merchants(limit)
   Top merchants ranked by spending. limit=1..10, default 8.
   Use for: "biggest merchants", "where did I spend most", "top transactions"

3. get_spending_by_period(period)
   Spending totals for a time window.
   period values: today, yesterday, this_week, last_week,
                  this_month, last_month, this_year, last_year
   Use for: "last month spending", "this week total", "spending today"

4. retrieve_transactions(query)
   Semantic search for specific transactions by merchant/context.
   Use for: "Swiggy orders", "Amazon purchases", "salary credit"

5. run_forecast(category, horizon_days)
   Predict future spending. horizon_days default 30.
   Use for: "forecast groceries", "predict next month dining spend"

6. get_anomalies(limit)
   Unusual/suspicious transactions. limit default 5.
   Use for: "unusual charges", "suspicious transactions", "anomalies"
"""

# ---------------------------------------------------------------------------
# Category synonym resolution
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, str] = {
    "dining": "Dining", "food": "Dining", "restaurant": "Dining",
    "restaurants": "Dining", "eating": "Dining", "eat": "Dining",
    "meal": "Dining", "meals": "Dining", "lunch": "Dining",
    "dinner": "Dining", "breakfast": "Dining", "cafe": "Dining",
    "coffee": "Dining", "swiggy": "Dining", "zomato": "Dining",
    "dine": "Dining", "eatery": "Dining",
    "groceries": "Groceries", "grocery": "Groceries",
    "supermarket": "Groceries", "vegetables": "Groceries",
    "fruits": "Groceries", "mart": "Groceries", "kirana": "Groceries",
    "transport": "Transport", "transportation": "Transport",
    "travel": "Transport", "fuel": "Transport", "petrol": "Transport",
    "cab": "Transport", "taxi": "Transport", "uber": "Transport",
    "ola": "Transport", "metro": "Transport", "bus": "Transport",
    "train": "Transport", "flight": "Transport", "auto": "Transport",
    "utilities": "Utilities", "utility": "Utilities",
    "electricity": "Utilities", "water": "Utilities", "gas": "Utilities",
    "internet": "Utilities", "broadband": "Utilities", "wifi": "Utilities",
    "bills": "Utilities", "bill": "Utilities", "recharge": "Utilities",
    "healthcare": "Healthcare", "health": "Healthcare",
    "medical": "Healthcare", "medicine": "Healthcare",
    "doctor": "Healthcare", "hospital": "Healthcare",
    "pharmacy": "Healthcare", "insurance": "Healthcare",
    "dental": "Healthcare", "clinic": "Healthcare",
    "entertainment": "Entertainment", "movies": "Entertainment",
    "movie": "Entertainment", "netflix": "Entertainment",
    "spotify": "Entertainment", "gaming": "Entertainment",
    "game": "Entertainment", "games": "Entertainment",
    "sports": "Entertainment", "gym": "Entertainment",
    "fitness": "Entertainment", "concert": "Entertainment",
    "shopping": "Shopping", "clothes": "Shopping", "clothing": "Shopping",
    "amazon": "Shopping", "flipkart": "Shopping", "fashion": "Shopping",
    "shoes": "Shopping", "accessories": "Shopping", "apparel": "Shopping",
    "subscriptions": "Subscriptions", "subscription": "Subscriptions",
    "streaming": "Subscriptions", "membership": "Subscriptions",
    "other": "Other", "misc": "Other", "miscellaneous": "Other",
}


def _resolve_category(raw: str, store) -> str | None:
    """Resolve user input to the exact DB category. Returns None for 'all'."""
    raw = raw.strip().strip('"').strip("'")
    if not raw or raw.lower() == "all":
        return None

    lower = raw.lower()
    all_txns = store.get_all()
    db_cats = {t.category.lower(): t.category for t in all_txns}

    # 1. Exact match
    if lower in db_cats:
        return db_cats[lower]

    # 2. Synonym lookup
    canonical = _SYNONYMS.get(lower)
    if canonical:
        if canonical.lower() in db_cats:
            return db_cats[canonical.lower()]
        return canonical  # will produce "0 results" message with clear category name

    # 3. Substring match
    for db_lower, db_orig in db_cats.items():
        if lower in db_lower or db_lower in lower:
            return db_orig

    return raw  # fall through — query will return 0 results with clear message


def _truncate(text: str) -> str:
    """Encode to ASCII (replacing non-ASCII) and truncate to _MAX_OUTPUT_CHARS."""
    safe = text.encode("ascii", errors="replace").decode("ascii")
    return safe if len(safe) <= _MAX_OUTPUT_CHARS else safe[:_MAX_OUTPUT_CHARS] + "..."


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_spending_summary(store, category: str = "all") -> str:
    """Return spending total(s) for a category or all categories."""
    try:
        resolved = _resolve_category(category, store)
        if resolved is None:
            txns = store.get_all()
            if not txns:
                return "No transactions found in the database."
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for t in txns:
                totals[t.category] += t.amount
                counts[t.category] += 1
            grand = sum(totals.values())
            lines = [f"Total: Rs.{grand:.2f} across {len(txns)} transactions. By category:"]
            for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
                lines.append(f"  {cat}: Rs.{amt:.2f} ({counts[cat]} txns)")
            return _truncate("\n".join(lines))
        else:
            txns = store.query_by_category(resolved)
            if not txns:
                all_txns = store.get_all()
                available = sorted({t.category for t in all_txns})
                return (
                    f"No transactions found for '{resolved}'. "
                    f"Available: {', '.join(available)}."
                )
            total = sum(t.amount for t in txns)
            return _truncate(
                f"Spending on '{resolved}': Rs.{total:.2f} "
                f"across {len(txns)} transaction(s)."
            )
    except Exception as exc:
        logger.error("get_spending_summary error: %s", exc, exc_info=True)
        return f"Error retrieving spending for '{category}'."


def get_top_merchants(store, limit: int = 8) -> str:
    """Return top merchants by total spending."""
    try:
        limit = min(max(1, int(limit)), 10)
        txns = store.get_all()
        if not txns:
            return "No transactions in the database."
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for t in txns:
            totals[t.merchant] += t.amount
            counts[t.merchant] += 1
        ranked = sorted(totals.items(), key=lambda x: -x[1])[:limit]
        lines = [f"Top {len(ranked)} merchants:"]
        for i, (m, amt) in enumerate(ranked, 1):
            lines.append(f"  {i}. {m}: Rs.{amt:.2f} ({counts[m]} txns)")
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.error("get_top_merchants error: %s", exc, exc_info=True)
        return "Error retrieving top merchants."


def get_spending_by_period(store, period: str = "last_month") -> str:
    """Return spending totals for a named time window."""
    try:
        today = date.today()
        p = period.lower().replace(" ", "_")
        if p == "today":
            start = end = today
        elif p == "yesterday":
            start = end = today - timedelta(days=1)
        elif p == "this_week":
            start = today - timedelta(days=today.weekday())
            end = today
        elif p == "last_week":
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
        elif p == "this_month":
            start = today.replace(day=1)
            end = today
        elif p == "last_month":
            first_this = today.replace(day=1)
            end = first_this - timedelta(days=1)
            start = end.replace(day=1)
        elif p == "this_year":
            start = today.replace(month=1, day=1)
            end = today
        elif p == "last_year":
            start = today.replace(year=today.year - 1, month=1, day=1)
            end = today.replace(year=today.year - 1, month=12, day=31)
        else:
            return (
                f"Unknown period '{period}'. Use: today, yesterday, "
                "this_week, last_week, this_month, last_month, "
                "this_year, last_year."
            )
        txns = store.query_by_date_range(start, end)
        if not txns:
            return f"No transactions for {p} ({start} to {end})."
        total = sum(t.amount for t in txns)
        totals: dict[str, float] = defaultdict(float)
        for t in txns:
            totals[t.category] += t.amount
        lines = [
            f"Spending for {p} ({start} to {end}):",
            f"  Total: Rs.{total:.2f} ({len(txns)} txns)",
        ]
        for cat, amt in sorted(totals.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: Rs.{amt:.2f}")
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.error("get_spending_by_period error: %s", exc, exc_info=True)
        return f"Error retrieving spending for period '{period}'."


def retrieve_transactions(vector_store, query: str) -> str:
    """Semantic search for relevant transactions."""
    try:
        results = vector_store.search(query, k=6)
        if not results:
            return f"No transactions found matching '{query}'."
        lines = [f"{len(results)} transaction(s) found:"]
        for t in results:
            lines.append(
                f"  {t.date}  {t.merchant[:22]}  Rs.{t.amount:.2f}  {t.category}"
            )
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.error("retrieve_transactions error: %s", exc, exc_info=True)
        return "Error retrieving transactions."


def run_forecast(store, forecaster, category: str, horizon_days: int = 30) -> str:
    """Forecast future spending for a category."""
    try:
        horizon_days = min(max(1, int(horizon_days)), 90)
        resolved = _resolve_category(category, store) or category
        forecast = forecaster.forecast_category(resolved, horizon_days, store)
        total = sum(p.yhat for p in forecast.points)
        avg = total / horizon_days if horizon_days else 0
        first, last = forecast.points[0], forecast.points[-1]
        return _truncate(
            f"Forecast for '{resolved}' over {horizon_days} days: "
            f"Rs.{total:.2f} total (avg Rs.{avg:.2f}/day). "
            f"Period: {first.date} to {last.date}."
        )
    except ValueError as exc:
        return f"Cannot forecast '{category}': {exc}"
    except Exception as exc:
        logger.error("run_forecast error: %s", exc, exc_info=True)
        return f"Error forecasting '{category}'."


def get_anomalies(store, anomaly_detector, limit: int = 5) -> str:
    """Return top anomalous transactions."""
    try:
        limit = min(max(1, int(limit)), 10)
        anomalies = anomaly_detector.get_anomalies(store)
        if not anomalies:
            return "No anomalous transactions detected."
        top = anomalies[:limit]
        lines = [f"{len(anomalies)} anomaly/anomalies. Top {len(top)}:"]
        for t in top:
            score = f"{t.anomaly_score:.2f}" if t.anomaly_score is not None else "N/A"
            lines.append(
                f"  {t.date}  {t.merchant[:20]}  Rs.{t.amount:.2f}"
                f"  {t.category}  score={score}"
            )
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.error("get_anomalies error: %s", exc, exc_info=True)
        return "Error retrieving anomalies."


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_tool_registry(store, vector_store, forecaster, anomaly_detector) -> dict:
    """Return a dict of tool_name -> zero-or-more-args callable."""
    return {
        "get_spending_summary": lambda category="all": get_spending_summary(store, category),
        "get_top_merchants": lambda limit=8: get_top_merchants(store, limit),
        "get_spending_by_period": lambda period="last_month": get_spending_by_period(store, period),
        "retrieve_transactions": lambda query="": retrieve_transactions(vector_store, query),
        "run_forecast": lambda category="Other", horizon_days=30: run_forecast(store, forecaster, category, horizon_days),
        "get_anomalies": lambda limit=5: get_anomalies(store, anomaly_detector, limit),
    }
