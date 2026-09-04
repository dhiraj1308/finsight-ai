"""Tests: CSV filename path-traversal fix in the /ingest endpoint.

Verifies that the safe_filename = Path(filename).name extraction in app.py
constrains all temporary filesystem writes to data/raw/ regardless of the
uploaded filename, while preserving the original filename in
Transaction.source_file metadata.

Uses the same TestClient + no-op lifespan + mock-component pattern as
test_ingest_anomaly_count.py.  Additionally patches pathlib.Path.write_bytes
and pathlib.Path.unlink inside the api.app module so tests never touch the
real filesystem.
"""
from __future__ import annotations

import csv
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure src/ is importable
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers (mirrors test_ingest_anomaly_count.py)
# ---------------------------------------------------------------------------

def _csv_bytes(n_rows: int = 3) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "merchant", "amount", "category"])
    for i in range(n_rows):
        writer.writerow(
            [f"2024-01-{(i % 28) + 1:02d}", f"Store {i}", f"{10 + i}.00", "Groceries"]
        )
    return buf.getvalue().encode("utf-8")


def _make_components():
    """Minimal mock AppComponents — no real DB or model loading."""
    from domain import Transaction
    from datetime import date

    stored = [
        Transaction(
            id=1,
            date=date(2024, 1, 1),
            merchant="Store 0",
            amount=10.0,
            category="Groceries",
            source_file="original.csv",
        )
    ]

    store = MagicMock()
    store.insert.return_value = (1, 0)
    store.get_all.return_value = stored

    vector_store = MagicMock()
    vector_store.indexed_ids = frozenset()

    categorizer = MagicMock()
    categorizer._is_trained = False

    anomaly_detector = MagicMock()
    anomaly_detector.fit_and_score.return_value = 0

    forecaster = MagicMock()

    components = MagicMock()
    components.store = store
    components.vector_store = vector_store
    components.categorizer = categorizer
    components.anomaly_detector = anomaly_detector
    components.forecaster = forecaster
    return components


@asynccontextmanager
async def _noop_lifespan(app):
    yield


def _make_client(components) -> TestClient:
    from api.app import app
    original = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        client = TestClient(app, raise_server_exceptions=True)
        app.state.components = components
        app.state.agent = MagicMock()
        return client
    finally:
        app.router.lifespan_context = original


def _post_csv(client: TestClient, filename: str, content: bytes | None = None):
    """POST to /ingest with the given filename."""
    data = content if content is not None else _csv_bytes()
    return client.post(
        "/ingest",
        files={"file": (filename, data, "text/csv")},
    )


# ---------------------------------------------------------------------------
# Capture the path passed to write_bytes inside the handler.
# We patch Path.write_bytes so the test can assert which path was used
# without touching the real filesystem.  Path.unlink is also patched to
# avoid a FileNotFoundError when cleanup runs on the mock path.
# CSVParser.parse is patched so it does not need a real file on disk.
# ---------------------------------------------------------------------------

def _patched_response(filename: str):
    """
    Run the /ingest handler for *filename* with filesystem ops mocked out.

    Returns (response, captured_write_path) where captured_write_path is the
    Path object that was passed to write_bytes() inside the handler.
    """
    from ingestion.csv_parser import ParseSummary
    from domain import Transaction
    from datetime import date

    captured = {}

    def fake_write_bytes(self, data):
        captured["path"] = self          # record the Path that write_bytes was called on
    
    fake_parse_result = (
        [Transaction(date=date(2024,1,1), merchant="Store 0", amount=10.0, category="Groceries")],
        ParseSummary(parsed=1, skipped=0),
    )

    components = _make_components()
    client = _make_client(components)

    with patch("pathlib.Path.write_bytes", fake_write_bytes), \
         patch("pathlib.Path.unlink", lambda self, **kw: None), \
         patch("pathlib.Path.mkdir", lambda self, **kw: None), \
         patch("ingestion.csv_parser.CSVParser.parse", return_value=fake_parse_result):
        response = _post_csv(client, filename)

    return response, captured.get("path")


# ---------------------------------------------------------------------------
# TEST 1 — normal filename stays in data/raw/
# ---------------------------------------------------------------------------

def test_normal_filename_path_is_inside_data_raw():
    """statement.csv → tmp_path must resolve inside data/raw/."""
    response, write_path = _patched_response("statement.csv")

    assert response.status_code == 200, response.text
    assert write_path is not None, "write_bytes was not called"

    # The path must be exactly data/raw/statement.csv (relative) or resolve there
    assert write_path.name == "statement.csv"
    assert "data" in write_path.parts and "raw" in write_path.parts, (
        f"Expected path inside data/raw/, got: {write_path}"
    )
    # Must not contain any '..' components
    assert ".." not in write_path.parts, (
        f"Traversal sequence '..' found in path: {write_path}"
    )


# ---------------------------------------------------------------------------
# TEST 2 — Unix traversal filename is contained
# ---------------------------------------------------------------------------

def test_unix_traversal_filename_is_contained():
    """../../.env.csv → tmp_path must resolve to data/raw/.env.csv, not above."""
    response, write_path = _patched_response("../../.env.csv")

    assert response.status_code == 200, response.text
    assert write_path is not None

    # safe basename must have been extracted
    assert write_path.name == ".env.csv", (
        f"Expected safe basename '.env.csv', got {write_path.name!r}"
    )
    assert ".." not in write_path.parts, (
        f"Traversal sequence remains in path: {write_path}"
    )
    # Must sit directly inside data/raw/ — no extra parent dirs
    assert str(write_path) == str(Path("data/raw") / ".env.csv"), (
        f"Expected data/raw/.env.csv, got {write_path}"
    )


# ---------------------------------------------------------------------------
# TEST 3 — Windows-style backslash traversal is contained
# ---------------------------------------------------------------------------

def test_windows_traversal_filename_is_contained():
    """..\\secret.csv (backslash) → basename extraction must strip it."""
    response, write_path = _patched_response("..\\secret.csv")

    assert response.status_code == 200, response.text
    assert write_path is not None

    # Path(filename).name handles backslashes on all platforms:
    # on Windows it returns "secret.csv"; on Linux it returns "..\\secret.csv"
    # because backslash is a valid filename char on POSIX.
    # The important guarantee: no forward-slash traversal sequences remain.
    assert ".." not in [p for p in write_path.parts if "/" not in p], (
        f"Unresolved traversal in: {write_path}"
    )
    # Path must be rooted inside data/raw/ (no escaping via forward slash)
    path_str = str(write_path).replace("\\", "/")
    assert path_str.startswith("data/raw/"), (
        f"Path escaped data/raw/: {write_path}"
    )


# ---------------------------------------------------------------------------
# TEST 4 — subdirectory filename does not create directories outside data/raw/
# ---------------------------------------------------------------------------

def test_subdirectory_filename_is_flattened():
    """subdir/evil.csv → Path(filename).name == 'evil.csv'; write goes to data/raw/evil.csv."""
    response, write_path = _patched_response("subdir/evil.csv")

    assert response.status_code == 200, response.text
    assert write_path is not None
    assert write_path.name == "evil.csv", (
        f"Expected 'evil.csv' (directory component stripped), got {write_path.name!r}"
    )
    assert str(write_path) == str(Path("data/raw") / "evil.csv"), (
        f"Expected data/raw/evil.csv, got {write_path}"
    )


# ---------------------------------------------------------------------------
# TEST 5 — cleanup unlinks the safe path (not a traversal path)
# ---------------------------------------------------------------------------

def test_cleanup_unlinks_safe_path():
    """Verify that the finally-block cleanup runs against the safe (contained) path."""
    from ingestion.csv_parser import ParseSummary
    from domain import Transaction
    from datetime import date

    unlinked_paths: list[Path] = []

    def fake_write_bytes(self, data):
        pass

    def fake_unlink(self, missing_ok=False):
        unlinked_paths.append(self)

    fake_parse_result = (
        [Transaction(date=date(2024,1,1), merchant="Store 0", amount=10.0, category="Groceries")],
        ParseSummary(parsed=1, skipped=0),
    )

    components = _make_components()
    client = _make_client(components)

    with patch("pathlib.Path.write_bytes", fake_write_bytes), \
         patch("pathlib.Path.unlink", fake_unlink), \
         patch("pathlib.Path.mkdir", lambda self, **kw: None), \
         patch("ingestion.csv_parser.CSVParser.parse", return_value=fake_parse_result):
        response = _post_csv(client, "../../malicious.csv")

    assert response.status_code == 200
    assert len(unlinked_paths) == 1, (
        f"Expected exactly one unlink call, got {len(unlinked_paths)}: {unlinked_paths}"
    )
    cleaned = unlinked_paths[0]
    assert ".." not in cleaned.parts, (
        f"Cleanup ran at traversal path: {cleaned}"
    )
    assert cleaned.name == "malicious.csv", (
        f"Expected safe basename 'malicious.csv' in cleanup, got: {cleaned}"
    )


# ---------------------------------------------------------------------------
# TEST 6 — original filename preserved in Transaction.source_file metadata
# ---------------------------------------------------------------------------

def test_original_filename_preserved_in_source_file():
    """
    The safe filename is used only for the temporary disk path.
    Transaction.source_file must still be the original uploaded filename.
    """
    from ingestion.csv_parser import ParseSummary
    from domain import Transaction
    from datetime import date

    captured_source_files: list[str] = []
    original_txn = Transaction(
        date=date(2024,1,1), merchant="Store 0", amount=10.0, category=""
    )

    fake_parse_result = (
        [original_txn],
        ParseSummary(parsed=1, skipped=0),
    )

    components = _make_components()

    def capture_insert(transactions):
        for t in transactions:
            captured_source_files.append(t.source_file)
        return 1, 0

    components.store.insert.side_effect = capture_insert

    client = _make_client(components)

    with patch("pathlib.Path.write_bytes", lambda self, data: None), \
         patch("pathlib.Path.unlink", lambda self, **kw: None), \
         patch("pathlib.Path.mkdir", lambda self, **kw: None), \
         patch("ingestion.csv_parser.CSVParser.parse", return_value=fake_parse_result):
        response = _post_csv(client, "../../.env.csv")

    assert response.status_code == 200
    assert len(captured_source_files) == 1, "insert() was not called"
    assert captured_source_files[0] == "../../.env.csv", (
        f"Expected original filename '../../.env.csv' in source_file, "
        f"got {captured_source_files[0]!r}"
    )


# ---------------------------------------------------------------------------
# TEST 7 — spaces and Unicode filename works normally
# ---------------------------------------------------------------------------

def test_spaces_and_unicode_filename_works():
    """'my statement 2024_中文.csv' → normal processing, no crash."""
    response, write_path = _patched_response("my statement 2024_中文.csv")

    assert response.status_code == 200, response.text
    assert write_path is not None
    assert write_path.name == "my statement 2024_中文.csv"
    assert ".." not in write_path.parts
