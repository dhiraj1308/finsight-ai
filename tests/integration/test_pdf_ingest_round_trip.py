"""Integration tests: PDF ingest through the real HTTP API.

Exercises the PDF path of POST /ingest end-to-end:

  PDF bytes in multipart POST
  → real PDFParser.parse_bytes()   (called inside the handler)
  → real TransactionStore (isolated SQLite)
  → IngestResponse  / PasswordErrorResponse / HTTPException(422)

Also verifies GET /transactions after a successful PDF ingest.

Uses existing committed fixtures — no PDF generation at test time:
  tests/fixtures/sample_bank_statement.pdf  (4 valid rows)
  tests/fixtures/password_protected.pdf     (encrypted, password="test123")
  tests/fixtures/no_transaction_table.pdf   (prose-only, no tables)

Components that are REAL:
  - PDFParser       — called internally by the /ingest handler (not mocked)
  - TransactionStore — SQLite under pytest tmp_path (isolated)
  - FastAPI handler — HTTP request/response, Pydantic serialization

Components that are MOCKED (not the subject of these tests):
  - VectorStore    — no SentenceTransformer/torch loading
  - Categorizer    — untrained (_is_trained=False)
  - AnomalyDetector — 4 transactions < MIN_TRANSACTIONS=10, never called
  - Forecaster     — not involved in /ingest

Fixture data in sample_bank_statement.pdf
(confirmed by generate_pdf_fixtures.py + passing unit tests):
  Row 1: 2024-03-15  Whole Foods  87.43  Groceries
  Row 2: 2024-03-16  Netflix      15.99  Entertainment
  Row 3: 2024-03-18  Shell Gas    45.20  Transport
  Row 4: 2024-03-20  Chipotle     12.50  Dining
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixture file locations
# ---------------------------------------------------------------------------
_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
_VALID_PDF     = _FIXTURES_DIR / "sample_bank_statement.pdf"
_PASSWORD_PDF  = _FIXTURES_DIR / "password_protected.pdf"
_NO_TABLE_PDF  = _FIXTURES_DIR / "no_transaction_table.pdf"

# Known data from the valid fixture (confirmed by unit tests)
_EXPECTED_MERCHANTS = {"Whole Foods", "Netflix", "Shell Gas", "Chipotle"}
_WHOLE_FOODS_ROW = {
    "merchant": "Whole Foods",
    "amount": 87.43,
    "category": "Groceries",
    "date": "2024-03-15",
}
_N = 4   # rows in the valid fixture


# ---------------------------------------------------------------------------
# No-op lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _noop_lifespan(app):
    yield


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _make_client(tmp_path: Path) -> TestClient:
    """Return a TestClient with a real isolated store and mocked ML."""
    from api.app import app
    from api.dependencies import AppComponents
    from ingestion.transaction_store import TransactionStore
    from categorization.categorizer import Categorizer

    store = TransactionStore(str(tmp_path / "pdf_ingest_test.db"))

    mock_vs = MagicMock()
    mock_vs.indexed_ids = frozenset()

    cat = Categorizer()
    assert not cat._is_trained

    components = AppComponents(
        store=store,
        vector_store=mock_vs,
        categorizer=cat,
        anomaly_detector=MagicMock(),
        forecaster=MagicMock(),
    )

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        app.state.components = components
        app.state.agent = MagicMock()
    finally:
        app.router.lifespan_context = original_lifespan

    return client


# ---------------------------------------------------------------------------
# Helper: POST a PDF file
# ---------------------------------------------------------------------------

def _post_pdf(client: TestClient, path: Path, password: str | None = None):
    """Upload a PDF fixture via POST /ingest."""
    files: dict = {
        "file": (path.name, path.read_bytes(), "application/pdf"),
    }
    if password is not None:
        files["password"] = (None, password)
    return client.post("/ingest", files=files)


# ---------------------------------------------------------------------------
# TEST 1 — Successful PDF ingest
# ---------------------------------------------------------------------------

class TestPdfIngestSuccess:
    """POST a valid PDF → parse_bytes → store.insert → GET /transactions."""

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def test_pdf_ingest_success_returns_200(self, client):
        """POST valid PDF must return HTTP 200."""
        response = _post_pdf(client, _VALID_PDF)
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_pdf_ingest_success_ingested_count(self, client):
        """ingested must equal 4 (rows in the fixture)."""
        body = _post_pdf(client, _VALID_PDF).json()
        assert body["ingested"] == _N, (
            f"Expected ingested={_N}, got {body['ingested']}"
        )

    def test_pdf_ingest_success_skipped_and_warnings(self, client):
        """skipped must be 0 and warnings empty for the valid fixture."""
        body = _post_pdf(client, _VALID_PDF).json()
        assert body["skipped"] == 0
        assert body["warnings"] == []

    def test_pdf_ingest_success_anomaly_and_review_none(self, client):
        """
        4 transactions < MIN_TRANSACTIONS=10 → anomaly detection skipped.
        Categorizer untrained → needs_review_count=None.
        """
        body = _post_pdf(client, _VALID_PDF).json()
        assert body["anomalies_detected"] is None
        assert body["needs_review_count"] is None

    def test_pdf_ingest_get_transactions_count(self, client):
        """After ingest, GET /transactions must return exactly 4 records."""
        _post_pdf(client, _VALID_PDF)
        response = client.get("/transactions")
        assert response.status_code == 200
        txns = response.json()
        assert len(txns) == _N, (
            f"Expected {_N} transactions, got {len(txns)}"
        )

    def test_pdf_ingest_get_transactions_merchants(self, client):
        """Merchants in GET /transactions must match the PDF fixture rows."""
        _post_pdf(client, _VALID_PDF)
        txns = client.get("/transactions").json()
        merchants = {t["merchant"] for t in txns}
        assert merchants == _EXPECTED_MERCHANTS, (
            f"Expected {_EXPECTED_MERCHANTS}, got {merchants}"
        )

    def test_pdf_ingest_source_file_is_original_pdf_filename(self, client):
        """Every persisted transaction must have source_file == fixture filename."""
        _post_pdf(client, _VALID_PDF)
        txns = client.get("/transactions").json()
        for txn in txns:
            assert txn["source_file"] == _VALID_PDF.name, (
                f"Expected source_file={_VALID_PDF.name!r}, "
                f"got {txn['source_file']!r}"
            )

    def test_pdf_ingest_known_row_values_persisted(self, client):
        """The Whole Foods row (first in fixture) must appear with exact values."""
        _post_pdf(client, _VALID_PDF)
        txns = client.get("/transactions").json()
        wf = next((t for t in txns if t["merchant"] == "Whole Foods"), None)
        assert wf is not None, "Whole Foods transaction must be present"
        assert round(wf["amount"], 2) == _WHOLE_FOODS_ROW["amount"]
        assert wf["category"] == _WHOLE_FOODS_ROW["category"]
        assert wf["date"] == _WHOLE_FOODS_ROW["date"]

    def test_pdf_ingest_all_transactiondto_fields_present(self, client):
        """Every transaction returned by GET /transactions must have all 9 DTO fields."""
        _post_pdf(client, _VALID_PDF)
        txns = client.get("/transactions").json()
        required = {
            "id", "date", "merchant", "amount", "category",
            "is_anomaly", "anomaly_score", "needs_review", "source_file",
        }
        for txn in txns:
            missing = required - txn.keys()
            assert not missing, f"Transaction missing fields: {missing}"


# ---------------------------------------------------------------------------
# TEST 2 — Duplicate PDF ingest
# ---------------------------------------------------------------------------

class TestDuplicatePdfIngest:
    """Upload the same PDF twice → second upload must be fully skipped."""

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def test_duplicate_pdf_ingest_is_skipped(self, client):
        """
        First upload: ingested=4, skipped=0.
        Second upload of the exact same PDF: ingested=0, skipped=4.
        Verifies source_file participates in the (date, merchant, amount, source_file)
        deduplication key.
        """
        first = _post_pdf(client, _VALID_PDF).json()
        assert first["ingested"] == _N
        assert first["skipped"] == 0

        second = _post_pdf(client, _VALID_PDF).json()
        assert second["ingested"] == 0, (
            f"Expected ingested=0 on duplicate, got {second['ingested']}"
        )
        assert second["skipped"] == _N, (
            f"Expected skipped={_N} on duplicate, got {second['skipped']}"
        )

    def test_duplicate_pdf_ingest_does_not_create_extra_transactions(self, client):
        """After two identical uploads, GET /transactions must return exactly 4 rows."""
        _post_pdf(client, _VALID_PDF)
        _post_pdf(client, _VALID_PDF)

        txns = client.get("/transactions").json()
        assert len(txns) == _N, (
            f"Expected {_N} transactions after duplicate PDF, got {len(txns)}"
        )


# ---------------------------------------------------------------------------
# TEST 3 — Password-protected PDF → PasswordErrorResponse
# ---------------------------------------------------------------------------

class TestPasswordProtectedPdf:
    """
    POST a password-protected PDF without a password.
    The handler returns JSONResponse(422, PasswordErrorResponse(...)) —
    a structured response with error_code, not a standard HTTPException.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def test_password_protected_pdf_returns_422(self, client):
        """POST password-protected PDF without password → HTTP 422."""
        response = _post_pdf(client, _PASSWORD_PDF)
        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )

    def test_password_protected_pdf_returns_structured_error_code(self, client):
        """
        Response must contain error_code == 'PASSWORD_REQUIRED'.
        This is the PasswordErrorResponse shape {error_code, detail},
        distinct from the standard HTTPException {detail} shape.
        """
        body = _post_pdf(client, _PASSWORD_PDF).json()
        assert "error_code" in body, (
            f"Expected 'error_code' key in response, got: {list(body.keys())}"
        )
        assert body["error_code"] == "PASSWORD_REQUIRED", (
            f"Expected error_code='PASSWORD_REQUIRED', got {body['error_code']!r}"
        )
        assert "detail" in body, "Response must also contain 'detail'"

    def test_password_protected_pdf_does_not_contain_ingested_field(self, client):
        """
        The password error response is PasswordErrorResponse, not IngestResponse.
        The 'ingested' key must not appear.
        """
        body = _post_pdf(client, _PASSWORD_PDF).json()
        assert "ingested" not in body, (
            f"'ingested' must not appear in PasswordErrorResponse, got: {body}"
        )

    def test_password_protected_pdf_does_not_store_transactions(self, client):
        """No transactions must be persisted when the PDF requires a password."""
        _post_pdf(client, _PASSWORD_PDF)
        txns = client.get("/transactions").json()
        assert txns == [], (
            f"Expected no transactions after password-protected upload, got {txns}"
        )


# ---------------------------------------------------------------------------
# TEST 4 — PDF with no transaction table → standard HTTPException 422
# ---------------------------------------------------------------------------

class TestNoTablePdf:
    """
    POST a PDF with prose only — no embedded tables.
    PDFParser falls through to text fallback which also finds nothing,
    then returns file_errors=["No recognizable transaction table found."].
    Handler raises HTTPException(422) — standard {detail: str} shape.
    """

    @pytest.fixture()
    def client(self, tmp_path):
        return _make_client(tmp_path)

    def test_pdf_without_table_returns_422(self, client):
        """POST a prose-only PDF must return HTTP 422."""
        response = _post_pdf(client, _NO_TABLE_PDF)
        assert response.status_code == 422, (
            f"Expected 422, got {response.status_code}: {response.text}"
        )

    def test_pdf_without_table_returns_standard_detail_shape(self, client):
        """
        This error is HTTPException(422), not JSONResponse with error_code.
        Response must have 'detail' but must NOT have 'error_code'.
        """
        body = _post_pdf(client, _NO_TABLE_PDF).json()
        assert "detail" in body, "422 response must contain 'detail'"
        assert "error_code" not in body, (
            f"'error_code' must not appear in standard 422, got: {body}"
        )

    def test_pdf_without_table_detail_message(self, client):
        """Detail must mention the exact parser error message."""
        body = _post_pdf(client, _NO_TABLE_PDF).json()
        assert "No recognizable transaction table found." in body["detail"], (
            f"Expected the table-not-found message in detail, got: {body['detail']!r}"
        )

    def test_pdf_without_table_does_not_store_transactions(self, client):
        """No transactions must be persisted after a failed PDF parse."""
        _post_pdf(client, _NO_TABLE_PDF)
        txns = client.get("/transactions").json()
        assert txns == [], (
            f"Expected no transactions after no-table PDF, got {txns}"
        )
