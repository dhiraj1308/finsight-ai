# Implementation Plan: Password-Protected PDF Upload

## Overview

Add end-to-end support for encrypted PDF uploads. Work proceeds backend-first (parser →
models → endpoint) then frontend (API client → upload view), finishing with tests and
wiring. Each step is independently runnable and builds on the previous one.

Implementation language: **Python**.

---

## Tasks

- [x] 1. Extend `PDFParser` with in-memory decryption support
  - Add `parse_bytes(self, content: bytes, filename: str, password: str | None = None) -> tuple[list[Transaction], ParseSummary]` to `src/ingestion/pdf_parser.py`
    - Open the PDF via `pdfplumber.open(io.BytesIO(content), password=password)` — no filesystem write
    - On `PDFPasswordIncorrect` with `password is None` → `summary.file_errors.append("PASSWORD_REQUIRED")`; return `[], summary`
    - On `PDFPasswordIncorrect` with `password is not None` → `summary.file_errors.append("PASSWORD_INCORRECT")`; return `[], summary`
    - Port the existing size-check, page-count-check, and table-extraction logic from `parse()` into the shared core used by both entry points
    - Add `import io` at the top of the file
  - Modify `parse(self, file_path: Path, password: str | None = None)` to accept the optional `password` kwarg and delegate to `parse_bytes` after reading the file bytes
  - Verify that the `password` parameter never touches `self`, `summary.file_errors` strings, `summary.warnings` strings, or any `Transaction` field
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.3_

  - [ ]* 1.1 Write unit tests for `PDFParser.parse_bytes`
    - Use `scripts/generate_pdf_fixtures.py` (or `pikepdf`) to produce a small fixture encrypted PDF in `tests/fixtures/`
    - Test: `parse_bytes(encrypted, "f.pdf", password=None)` → `file_errors == ["PASSWORD_REQUIRED"]`, empty transaction list
    - Test: `parse_bytes(encrypted, "f.pdf", password="wrong")` → `file_errors == ["PASSWORD_INCORRECT"]`, empty transaction list
    - Test: `parse_bytes(encrypted, "f.pdf", password="correct")` → transactions match the unencrypted fixture
    - Test: `parse_bytes` with unencrypted PDF and `password=None` → normal parse succeeds
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 1.2 Write property test — password never leaks into ParseSummary
    - **Property 1: Password never appears in ParseSummary output**
    - Use `hypothesis` to generate arbitrary non-empty `password` strings
    - Call `parse_bytes(encrypted_fixture_bytes, "f.pdf", password=pw)` for each
    - Assert `pw` does not appear in any element of `summary.file_errors`, `summary.warnings`, or any `Transaction` field
    - **Validates: Requirements 2.5, 6.1, 6.2**

- [x] 2. Add `PasswordErrorResponse` model and update `IngestResponse`
  - In `src/api/models.py`:
    - Add `from typing import Literal` import
    - Add `PasswordErrorResponse(BaseModel)` with fields `error_code: Literal["PASSWORD_REQUIRED", "PASSWORD_INCORRECT"]` and `detail: str`
    - Add `error_code: str | None = None` field to `IngestResponse`
  - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 2.1 Write unit tests for new Pydantic models
    - Test `PasswordErrorResponse` validates correctly for both literal values and rejects arbitrary strings
    - Test `IngestResponse` serialises with `error_code=None` on success
    - _Requirements: 3.1, 3.3_

- [x] 3. Update the `Ingest_Endpoint` to handle password form field and return typed errors
  - In `src/api/app.py`:
    - Add `from fastapi import Form` to the existing FastAPI import line
    - Add `from fastapi.responses import JSONResponse` import
    - Add `from api.models import PasswordErrorResponse` to the models import block
    - Change the `ingest` signature to: `async def ingest(file: UploadFile = File(...), password: str | None = Form(None))`
    - Read file content with `content = await file.read()` (already done; keep this)
    - For `.pdf` files: call `PDFParser().parse_bytes(content, filename, password=password)` instead of writing to `data/raw/` and calling `parse(tmp_path)`
    - After calling `parse_bytes`, check `summary.file_errors`:
      - If `"PASSWORD_REQUIRED"` in `file_errors` → `return JSONResponse(status_code=422, content=PasswordErrorResponse(error_code="PASSWORD_REQUIRED", detail="This PDF is password-protected. Please supply the decryption password.").model_dump())`
      - If `"PASSWORD_INCORRECT"` in `file_errors` → return similar 422 with `error_code="PASSWORD_INCORRECT"`, `detail="Incorrect password. Please try again."`
    - For `.pdf` files with no password error, skip the `data/raw/` write entirely (process entirely in memory)
    - For `.csv` files keep the existing `data/raw/` write and `CSVParser` path unchanged
    - Ensure `password` is never passed to `logger.*`, never included in any `HTTPException(detail=...)` string
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.2, 6.1, 6.2, 6.3_

  - [ ]* 3.1 Write integration tests for `POST /ingest` with FastAPI `TestClient`
    - Use encrypted PDF fixture bytes (from Task 1 fixtures)
    - Test: POST without password → 422, body `{"error_code": "PASSWORD_REQUIRED", ...}`
    - Test: POST with wrong password → 422, body `{"error_code": "PASSWORD_INCORRECT", ...}`
    - Test: POST with correct password → 200, `IngestResponse.ingested > 0`
    - Test: POST CSV with password field present → 200, standard `IngestResponse` with no `error_code`
    - Test: POST unencrypted PDF → 200, no `error_code`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 8.1_

- [x] 4. Checkpoint — ensure all backend tests pass
  - Run `pytest tests/ -x -q` and confirm no failures before proceeding to frontend.

- [x] 5. Add typed exception classes and update `APIClient` in `src/frontend/services/api.py`
  - Add `PasswordRequiredError` exception class at module level:
    ```python
    class PasswordRequiredError(Exception):
        def __init__(self, filename: str) -> None:
            self.filename = filename
            super().__init__(f"PDF is password-protected: {filename}")
    ```
  - Add `PasswordIncorrectError` exception class at module level (same pattern)
  - Modify `_raise_for_status` to accept an optional `filename: str = ""` parameter and inspect the 422 body:
    - Parse JSON body; if `error_code == "PASSWORD_REQUIRED"` → raise `PasswordRequiredError(filename)`
    - If `error_code == "PASSWORD_INCORRECT"` → raise `PasswordIncorrectError(filename)`
    - Fall through to existing `RuntimeError` for all other errors
  - Modify `upload_statement` to support two calling conventions:
    - Existing: `upload_statement(file_path: str | Path)` — reads from disk, no password
    - New: `upload_statement(file_path=None, *, file_bytes: bytes, filename: str, password: str | None = None)` — in-memory bytes
    - When `password` is non-empty, add it to the `files` dict as `files["password"] = (None, password)` — multipart text field, never in the URL
    - Pass `filename` to `_raise_for_status` so typed exceptions carry the name
  - _Requirements: 4.1, 4.2, 5.5, 9.1_

  - [ ]* 5.1 Write unit tests for `_raise_for_status` typed exception dispatch
    - Build mock `requests.Response` objects with various 422 bodies
    - Test: body `{"error_code": "PASSWORD_REQUIRED"}` → raises `PasswordRequiredError`, not `RuntimeError`
    - Test: body `{"error_code": "PASSWORD_INCORRECT"}` → raises `PasswordIncorrectError`
    - Test: body `{"detail": "some other 422"}` → raises `RuntimeError`
    - Test: 200 response → no exception
    - _Requirements: 4.1, 4.2_

  - [ ]* 5.2 Write property test — password never appears in request URL
    - **Property 4: Password is transmitted as a multipart body field, never in the URL**
    - Use `hypothesis` to generate arbitrary non-empty password strings and filenames
    - Call `upload_statement(file_bytes=b"%PDF-1.4...", filename=fn, password=pw)` with a mocked session
    - Inspect the `PreparedRequest`: assert `pw` is not a substring of `request.url`
    - Assert the multipart body contains a `password` field equal to `pw`
    - **Validates: Requirements 5.5, 9.1**

- [x] 6. Update `Upload_Page` in `src/frontend/views/upload.py`
  - Add imports: `from frontend.services.api import PasswordRequiredError, PasswordIncorrectError`
  - Add `_MAX_PASSWORD_ATTEMPTS = 5` constant
  - Add `_clear_password_state()` helper that deletes `pdf_bytes`, `pdf_name`, `pw_prompt`, `pw_attempts` from `st.session_state` using `pop(..., None)`
  - Refactor `render()` into two branches:

    **Branch A — no active password prompt** (`not st.session_state.get("pw_prompt")`):
    - Show file uploader (enabled, same as current)
    - On Upload button click: read `uploaded_file.getbuffer()` into `raw_bytes` (not a temp file)
    - Call `client.upload_statement(file_bytes=raw_bytes, filename=uploaded_file.name)`
    - On `PasswordRequiredError`: store `st.session_state["pdf_bytes"] = raw_bytes`, `st.session_state["pdf_name"] = uploaded_file.name`, `st.session_state["pw_prompt"] = True`, `st.session_state["pw_attempts"] = 0`; rerun
    - On `RuntimeError`: `st.error(f"Upload failed: {exc}")`
    - On success: `_show_result(result)`

    **Branch B — password prompt active** (`st.session_state.get("pw_prompt")`):
    - Show file uploader with `disabled=True`
    - Show `st.text_input("PDF Password", type="password", key="pw_input")`
    - Show "Unlock & Upload" button
    - On button click:
      - If `password == ""` or not password: `st.warning("Password cannot be empty")`; return
      - Call `client.upload_statement(file_bytes=st.session_state["pdf_bytes"], filename=st.session_state["pdf_name"], password=password)`
      - On `PasswordIncorrectError`:
        - `st.session_state["pw_attempts"] += 1`
        - If `pw_attempts >= _MAX_PASSWORD_ATTEMPTS`: `st.error("Too many failed attempts. Please re-upload the file.")`, `_clear_password_state()`
        - Else: `st.error("Incorrect password — please try again.")`
      - On success: `_show_result(result)`, `_clear_password_state()`
      - On other `Exception`: `st.error(...)`, `_clear_password_state()`
  - Remove `_save_to_temp` and `_do_upload` helpers (they are replaced by the new logic); only remove if they are no longer referenced
  - _Requirements: 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 8.2, 8.3_

  - [ ]* 6.1 Write unit tests for `Upload_Page` session state transitions
    - Mock `APIClient.upload_statement` to raise `PasswordRequiredError`, `PasswordIncorrectError`, or return a success dict
    - Test: `PasswordRequiredError` → session state set correctly (`pw_prompt=True`, `pw_attempts=0`, bytes stored)
    - Test: success → `_clear_password_state` called, result rendered
    - Test: empty password → no API call made, warning shown
    - _Requirements: 4.3, 5.1, 5.2, 5.3_

  - [ ]* 6.2 Write property test — attempt counter terminates prompt after exactly 5 failures
    - **Property 5: Attempt counter terminates the prompt after exactly 5 wrong passwords**
    - Simulate 1–4 wrong password responses: assert `pw_prompt` remains `True` and `pw_attempts` equals the count
    - Simulate the 5th wrong password response: assert `pw_prompt` not in `session_state` (cleared), error message shown
    - **Validates: Requirements 7.1, 7.2, 7.3**

  - [ ]* 6.3 Write property test — session state is fully cleared after any terminal outcome
    - **Property 6: Session state is fully cleared after any terminal outcome**
    - For each terminal outcome (success, RuntimeError, 5th wrong password), assert that `pdf_bytes`, `pdf_name`, `pw_prompt`, `pw_attempts` are absent from `session_state`
    - **Validates: Requirements 5.3, 6.5, 7.3**

- [x] 7. Final checkpoint — ensure all tests pass
  - Run `pytest tests/ -x -q` and confirm clean pass
  - Manually verify no `password` value appears in `data/raw/` after an encrypted PDF upload
  - Manually verify the Streamlit upload page shows the password prompt for an encrypted PDF

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP build
- The `_save_to_temp` / `_do_upload` helpers in `upload.py` are only removed in Task 6 once the new in-memory flow is proven; keep them until then to avoid breaking the existing non-PDF path
- `scripts/generate_pdf_fixtures.py` already exists; check if it already produces an encrypted fixture before writing a new one
- All property tests use `hypothesis`; add it to `requirements-lock.txt` if not already present
- The `pikepdf` package is only needed for fixture generation in tests, not at runtime
