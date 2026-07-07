"""
API client for the FinSight AI FastAPI backend.

All public methods raise ``RuntimeError`` on non-2xx responses or network
failures so callers can handle errors uniformly without inspecting raw HTTP
objects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "http://127.0.0.1:8000"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / timeout defaults
# ---------------------------------------------------------------------------
_DEFAULT_TIMEOUT: int = 30  # seconds
_UPLOAD_TIMEOUT: int = 120  # larger budget for file uploads

_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)


class APIClient:
    """
    Thin HTTP client that wraps every FinSight AI REST endpoint.

    A single :class:`requests.Session` is reused across calls for connection
    pooling.  The session is configured with an automatic retry strategy for
    transient failures.

    Example
    -------
    >>> client = APIClient()
    >>> client.health_check()
    {'status': 'ok', 'service': 'FinSight AI'}
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        """Perform a GET request and return the decoded JSON body."""
        url = self._url(path)
        try:
            response = self._session.get(url, params=params, timeout=_DEFAULT_TIMEOUT)
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to the FinSight AI backend at {self._base_url}. "
                "Make sure the server is running."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Request to {url} timed out after {_DEFAULT_TIMEOUT}s."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Unexpected network error: {exc}") from exc

        self._raise_for_status(response)
        return response.json()

    def _post(
        self,
        path: str,
        json: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> Any:
        """Perform a POST request and return the decoded JSON body."""
        url = self._url(path)
        try:
            response = self._session.post(
                url, json=json, files=files, timeout=timeout
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Could not connect to the FinSight AI backend at {self._base_url}. "
                "Make sure the server is running."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError(
                f"Request to {url} timed out after {timeout}s."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Unexpected network error: {exc}") from exc

        self._raise_for_status(response)
        return response.json()

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Translate non-2xx responses into a :class:`RuntimeError`."""
        if response.ok:
            return

        detail: str
        try:
            body = response.json()
            detail = body.get("detail") or response.text
        except ValueError:
            detail = response.text or response.reason or "unknown error"

        raise RuntimeError(
            f"Backend returned {response.status_code}: {detail}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, str]:
        """
        Ping the ``/health`` endpoint.

        Returns
        -------
        dict
            ``{"status": "ok", "service": "FinSight AI"}`` on success.

        Raises
        ------
        RuntimeError
            If the server is unreachable or returns a non-2xx status.
        """
        return self._get("/health")

    def upload_statement(self, file_path: str | Path) -> dict[str, Any]:
        """
        Upload a bank statement (CSV or PDF) to the ``/ingest`` endpoint.

        The file is streamed as ``multipart/form-data``.

        Parameters
        ----------
        file_path:
            Absolute or relative path to the statement file.

        Returns
        -------
        dict
            Parsed :class:`IngestResponse` payload with keys
            ``ingested``, ``skipped``, and ``warnings``.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist on disk.
        RuntimeError
            On network or server errors.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Statement file not found: {path}")

        with path.open("rb") as fh:
            files = {"file": (path.name, fh, _mime_type(path))}
            result = self._post("/ingest", files=files, timeout=_UPLOAD_TIMEOUT)

        logger.info(
            "Ingested %s: %d inserted, %d skipped.",
            path.name,
            result.get("ingested", 0),
            result.get("skipped", 0),
        )
        return result

    def get_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve transactions from the ``/transactions`` endpoint.

        Supply *start_date* and *end_date* together to filter by date range,
        or *category* to filter by spending category.  With no arguments all
        stored transactions are returned.

        Parameters
        ----------
        start_date:
            ISO-8601 date string (``"YYYY-MM-DD"``), inclusive lower bound.
        end_date:
            ISO-8601 date string (``"YYYY-MM-DD"``), inclusive upper bound.
        category:
            Spending category label, e.g. ``"Food & Drink"``.

        Returns
        -------
        list[dict]
            List of transaction objects matching the :class:`TransactionDTO`
            schema.

        Raises
        ------
        RuntimeError
            On network or server errors.
        """
        params: dict[str, Any] = {}
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date
        if category is not None:
            params["category"] = category

        return self._get("/transactions", params=params or None)

    def get_anomalies(self) -> list[dict[str, Any]]:
        """
        Retrieve anomalous transactions from the ``/anomalies`` endpoint.

        Returns
        -------
        list[dict]
            List of transaction objects flagged as anomalies.

        Raises
        ------
        RuntimeError
            On network or server errors.
        """
        return self._get("/anomalies")

    def get_forecast(
        self, category: str, days: int = 30
    ) -> dict[str, Any]:
        """
        Retrieve a spending forecast from ``/forecast/{category}``.

        Parameters
        ----------
        category:
            Spending category to forecast, e.g. ``"Groceries"``.
        days:
            Forecast horizon in days (1–365, default 30).

        Returns
        -------
        dict
            Parsed :class:`ForecastDTO` payload with keys ``category``,
            ``horizon_days``, and ``points``.

        Raises
        ------
        ValueError
            If *days* is outside the accepted range [1, 365].
        RuntimeError
            On network or server errors.
        """
        if not 1 <= days <= 365:
            raise ValueError(f"'days' must be between 1 and 365, got {days}.")

        return self._get(f"/forecast/{category}", params={"days": days})

    def chat(self, message: str, session_id: str) -> dict[str, str]:
        """
        Send a natural-language message to the AI agent via ``/chat``.

        Parameters
        ----------
        message:
            User's question or instruction (max 2 000 characters).
        session_id:
            Opaque string that links messages in the same conversation
            (max 128 characters).

        Returns
        -------
        dict
            Parsed :class:`ChatResponse` payload with key ``answer``.

        Raises
        ------
        ValueError
            If *message* exceeds 2 000 characters or *session_id* exceeds
            128 characters.
        RuntimeError
            On network or server errors.
        """
        if len(message) > 2000:
            raise ValueError(
                f"'message' must be ≤ 2 000 characters (got {len(message)})."
            )
        if len(session_id) > 128:
            raise ValueError(
                f"'session_id' must be ≤ 128 characters (got {len(session_id)})."
            )

        return self._post("/chat", json={"message": message, "session_id": session_id})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _mime_type(path: Path) -> str:
    """Return the MIME type for supported statement file formats."""
    suffix = path.suffix.lower()
    mime_map = {
        ".csv": "text/csv",
        ".pdf": "application/pdf",
    }
    return mime_map.get(suffix, "application/octet-stream")
