"""Router: POST /ingest"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from api.models import IngestResponse, PasswordErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    password: str | None = Form(None),
):
    from ingestion.csv_parser import CSVParser
    from ingestion.pdf_parser import PDFParser
    from api.services.ingest_service import IngestService
    from api.dependencies import get_components
    from config import get_settings

    filename = file.filename or ""
    if not (filename.endswith(".csv") or filename.endswith(".pdf")):
        raise HTTPException(status_code=422, detail="Only PDF and CSV files are supported.")

    store, vector_store, categorizer, anomaly_detector, _ = get_components(request)
    service = IngestService(store, vector_store, categorizer, anomaly_detector)

    content = await file.read()

    if filename.endswith(".pdf"):
        transactions, summary = PDFParser().parse_bytes(content, filename, password=password)

        if summary.file_errors:
            if "PASSWORD_REQUIRED" in summary.file_errors:
                return JSONResponse(
                    status_code=422,
                    content=PasswordErrorResponse(
                        error_code="PASSWORD_REQUIRED",
                        detail="This PDF is password-protected. Please supply the decryption password.",
                    ).model_dump(),
                )
            if "PASSWORD_INCORRECT" in summary.file_errors:
                return JSONResponse(
                    status_code=422,
                    content=PasswordErrorResponse(
                        error_code="PASSWORD_INCORRECT",
                        detail="Incorrect password. Please try again.",
                    ).model_dump(),
                )
            raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

        return service.process(transactions, filename, summary.warnings)

    else:
        # CSV path: write to data/raw/, parse from disk, clean up.
        # Derive a filesystem-safe basename from the uploaded filename so that
        # directory-traversal sequences such as "../../.env.csv" or
        # "subdir/evil.csv" cannot escape data/raw/.  The original `filename`
        # variable is intentionally left unchanged so that Transaction.source_file
        # and duplicate-detection behaviour are unaffected.
        safe_filename = Path(filename).name
        if not safe_filename:
            raise HTTPException(status_code=422, detail="Invalid filename.")
        settings = get_settings()
        tmp_path = Path(settings.CSV_UPLOAD_DIR) / safe_filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(content)

        try:
            transactions, summary = CSVParser().parse(tmp_path)

            if summary.file_errors:
                raise HTTPException(status_code=422, detail=f"File error: {summary.file_errors[0]}")

            return service.process(transactions, filename, summary.warnings)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except PermissionError:
                logger.warning(f"Could not delete temp file (locked): {tmp_path}")
