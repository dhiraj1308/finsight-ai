# Implementation Plan: FinSight AI Platform

## Overview

Incremental 30-day build organized across four weeks. Each week targets a self-contained
layer of the system so each layer can be tested in isolation before the next layer builds
on top of it. The design's 29 Correctness Properties are validated through hand-written
pytest example tests — not property-based testing frameworks.

All code is Python. Storage: SQLite + ChromaDB. ML: scikit-learn, Prophet, sentence-transformers.
Agent: LangChain. API: FastAPI. Frontend: Streamlit.

---

## Tasks

## Week 1 — Days 1–7: Project Setup + Ingestion Layer

- [ ] 1. Bootstrap project structure and shared domain model
  - Create `src/__init__.py`, `src/domain.py` with the canonical `Transaction` dataclass
    (fields: `date`, `merchant`, `amount`, `category`, `id`, `is_anomaly`, `anomaly_score`,
    `needs_review`, `source_file`) exactly as specified in the design.
  - Add `__init__.py` files to each existing `src/` sub-package so imports work.
  - Update `requirements.txt` with the full pinned dependency list:
    `fastapi`, `uvicorn`, `pydantic`, `langchain`, `langchain-openai`, `chromadb`,
    `sentence-transformers`, `prophet`, `scikit-learn`, `pdfplumber`, `python-dotenv`,
    `streamlit`, `pytest`, `pytest-mock` (keep existing pandas, numpy, faker, matplotlib,
    seaborn, python-dotenv; add missing packages with pinned versions).
  - Create `conftest.py` at the repo root that adds `src/` to `sys.path`.
  - Create `tests/unit/` and `tests/integration/` directories with `__init__.py` and
    `tests/fixtures/` directory with a `.gitkeep`.
  - _Requirements: 12.1_

- [ ] 2. Implement `SyntheticGenerator`
  - [ ] 2.1 Write `src/ingestion/synthetic_generator.py` with the `SyntheticGenerator` class
    - Implement `generate(n, start_date, end_date, seed)` producing exactly N `Transaction`
      records with per-category merchant names and amount ranges as specified in Req 1.3.
    - Implement `write_csv(transactions, output_dir)` writing to `data/synthetic/`.
    - Raise `ValueError` with the parameter name for `n < 1`, `n > 100_000`,
      or `start_date > end_date`.
    - Default `n=3000`; use `seed` to seed `random` / `numpy.random` for determinism.
    - Add `src/ingestion/__init__.py` exporting `SyntheticGenerator`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [ ]* 2.2 Write unit tests for `SyntheticGenerator` in `tests/unit/test_synthetic_generator.py`
    - Test cases derived from Properties 1–5:
      - Generate 1 record → all required fields non-null/non-empty/non-zero.
      - Generate 3000 records → exactly 3000 returned, all dates within range.
      - `seed=42` twice → lists are identical field-for-field (Property 4).
      - `n=0` raises `ValueError`; `n=100_001` raises `ValueError` (Property 5).
      - `start_date > end_date` raises `ValueError` (Property 5).
      - Sample of 50 records → all amounts within declared per-category bounds (Property 3).
      - No output file is written when `ValueError` is raised.
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.7_

- [ ] 3. Implement `CSVParser` and `PrettyPrinter`
  - [ ] 3.1 Write `src/ingestion/csv_parser.py` with `ParseSummary` dataclass and `CSVParser` class
    - Implement `parse(file_path)` returning `(list[Transaction], ParseSummary)`.
    - Apply `COLUMN_ALIASES` (case-insensitive): "Description"/"Narration"/"Details" → `merchant`;
      "Debit"/"Withdrawal"/"Charge" → `amount`; "Transaction Date"/"Posted Date"/"Trans. Date" → `date`.
    - Skip rows with missing required fields or unparseable date/amount; append warning with
      1-based row number and field name or raw value.
    - Return 0 records + file-level error for empty or non-UTF-8 files (no unhandled exception).
    - Summary satisfies `parsed + skipped == total_data_rows`.
    - Export `CSVParser`, `ParseSummary` from `src/ingestion/__init__.py`.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [ ] 3.2 Write `src/ingestion/pretty_printer.py` with `PrettyPrinter` class
    - Implement `to_csv(transactions, output_path)` and `to_csv_string(transactions)`.
    - Header must be exactly `date,merchant,amount,category`; date formatted `YYYY-MM-DD`.
    - Export `PrettyPrinter` from `src/ingestion/__init__.py`.
    - _Requirements: 2.8, 2.9_

  - [ ]* 2.3 Write unit tests in `tests/unit/test_csv_parser.py` (and round-trip tests)
    - Test cases derived from Properties 6–10:
      - Parse canonical CSV → correct `Transaction` fields (Property 6).
      - Parse alias-header CSV → fields correctly mapped (Property 7).
      - Row missing `amount` → skipped, warning contains row number and field name (Property 8).
      - Row with non-numeric `amount` → skipped with raw value in warning (Property 8).
      - Row with unparseable date → skipped with raw value in warning (Property 8).
      - Empty file → 0 records + file error, no exception (Req 2.6).
      - `parsed + skipped == total_data_rows` for a mixed file (Property 9).
      - Round-trip: `to_csv_string()` → `parse()` preserves all fields character-for-character
        (Property 10).
    - Create fixture files `tests/fixtures/sample_bank_statement.csv`,
      `tests/fixtures/sample_alt_headers.csv`, `tests/fixtures/sample_mixed.csv`.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

- [ ] 4. Implement `PDFParser`
  - [ ] 4.1 Write `src/ingestion/pdf_parser.py` with `PDFParser` class
    - Use `pdfplumber` to extract text/tables from each page.
    - Reject password-protected PDFs with exact message:
      `"File is password-protected and cannot be read."`.
    - Reject files > 10 MB or > 100 pages, or invalid PDFs, with descriptive messages.
    - Return exact message `"No recognizable transaction table found."` when no page yields
      a row with a parseable date and numeric amount.
    - Skip individual unparseable rows and record warnings with page number, row index,
      and failing field name(s).
    - Summary includes total rows attempted, successfully parsed count, and skipped count.
    - Apply same `COLUMN_ALIASES` mapping as `CSVParser`.
    - Export `PDFParser` from `src/ingestion/__init__.py`.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 4.2 Write unit tests in `tests/unit/test_pdf_parser.py`
    - Test cases derived from Req 3.3–3.7:
      - Password-protected PDF returns exact error message, 0 records, no exception.
      - Valid PDF with transaction table returns correct records.
      - PDF with no recognizable table returns exact error message.
      - Bad row on a page is skipped; warning contains page number, row index, field name.
      - Parse summary counts are consistent.
    - Create fixture files: `tests/fixtures/sample_bank_statement.pdf`,
      `tests/fixtures/password_protected.pdf`, `tests/fixtures/no_transaction_table.pdf`.
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.7_

- [ ] 5. Implement `TransactionStore`
  - [ ] 5.1 Write `src/ingestion/transaction_store.py` with `TransactionStore` class
    - Initialize SQLite database at `SQLITE_DB_PATH`; create `transactions` table and
      `uq_transaction` unique index on `(date, merchant, amount, source_file)` on first run.
    - Implement `insert(transactions) -> (inserted: int, skipped: int)` with deduplication.
    - Implement `query_by_date_range(start_date, end_date) -> list[Transaction]`; raise
      `ValueError` if `start_date > end_date`.
    - Implement `query_by_category(category) -> list[Transaction]` (case-insensitive match).
    - Implement `get_all() -> list[Transaction]` sorted by date descending.
    - Implement `delete(transaction_id)` for use by Vector Store atomicity pattern.
    - Export `TransactionStore` from `src/ingestion/__init__.py`.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 5.2 Write unit tests in `tests/unit/test_transaction_store.py`
    - Test cases derived from Properties 11–14:
      - Insert 3 distinct records → each gets a unique id (Property 11).
      - Insert duplicate `(date, merchant, amount, source_file)` → skipped count = 1,
        no second row in DB (Property 11).
      - `query_by_date_range` with boundary dates returns only in-range records (Property 12).
      - `query_by_date_range(start > end)` raises `ValueError` (Req 4.4).
      - `query_by_category` is case-insensitive, returns empty list for no match (Property 13).
      - `get_all()` returns all records newest-first (Property 14).
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 6. Week 1 checkpoint — Ensure all tests pass
  - Run `pytest tests/unit/test_synthetic_generator.py tests/unit/test_csv_parser.py tests/unit/test_pdf_parser.py tests/unit/test_transaction_store.py -v`.
  - Ensure all tests pass; ask the user if questions arise.

---

## Week 2 — Days 8–14: ML Pipeline

- [ ] 7. Implement `Categorizer`
  - [ ] 7.1 Write `src/categorization/categorizer.py` with `Categorizer` class
    - Define `CANONICAL_CATEGORIES` frozenset and `CONFIDENCE_THRESHOLD = 0.60`.
    - Implement `train(labeled_transactions) -> float` building a sklearn `Pipeline`
      (TF-IDF vectorizer on `merchant` → `LogisticRegression` or `LinearSVC`).
      Split 80/20 for validation; return weighted F1 on validation set (must be ≥ 0.80
      on ≥ 200 labeled transactions per Req 5.4).
    - Implement `predict(transaction) -> Transaction` — if confidence < 0.60, set
      `category = "Other"` and `needs_review = True`.
    - Implement `predict_batch(transactions) -> list[Transaction]` — same length as input;
      per-record errors → `category="Other"`, `needs_review=True`, continue.
    - Implement `save(path)` and `load(path)` using `joblib`.
    - Add `src/categorization/__init__.py` exporting `Categorizer`, `CANONICAL_CATEGORIES`.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 Write unit tests in `tests/unit/test_categorizer.py`
    - Test cases derived from Properties 15–17:
      - `predict()` output `category` is always in `CANONICAL_CATEGORIES` (Property 15).
      - Mock classifier returning confidence < 0.60 → `category="Other"`, `needs_review=True`
        (Property 16).
      - `predict_batch()` on N records returns list of length N (Property 17).
      - `predict_batch()` where one record raises internally → that record gets `"Other"` +
        `needs_review=True`, rest are processed (Property 17).
    - _Requirements: 5.1, 5.5, 5.6_

- [ ] 8. Implement `AnomalyDetector`
  - [ ] 8.1 Write `src/anomaly/anomaly_detector.py` with `AnomalyDetector` class
    - Implement `fit_and_score(store) -> int` using `IsolationForest` from scikit-learn.
    - Feature matrix: `[amount, label_encoded_category]` (two numeric columns).
    - Normalize score: `score = (raw_decision * -1).clip(0, 1)`.
    - Update `is_anomaly` and `anomaly_score` for every record in the store.
    - Return count of flagged anomalies.
    - Raise `ValueError` if store has < 10 transactions (do not modify any flags).
    - Implement `get_anomalies(store) -> list[Transaction]` returning `is_anomaly=True`
      records ordered by `anomaly_score` descending.
    - Add `src/anomaly/__init__.py` exporting `AnomalyDetector`.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 8.2 Write unit tests in `tests/unit/test_anomaly_detector.py`
    - Test cases derived from Properties 18–20:
      - Fit on 10+ records → every record has `is_anomaly` set (Property 18).
      - All anomalous records have `anomaly_score` in `[0.0, 1.0]` (Property 18).
      - Fit on < 10 records raises `ValueError` without modifying any flags (Req 6.7).
      - After inserting more records and re-fitting, all records (old + new) have updated
        scores (Property 19).
      - `get_anomalies()` returns only `is_anomaly=True` records, sorted by score desc
        (Property 20).
    - _Requirements: 6.1, 6.4, 6.5, 6.6, 6.7_

- [ ] 9. Implement `Forecaster`
  - [ ] 9.1 Write `src/forecasting/forecaster.py` with `ForecastPoint`, `Forecast`, `Forecaster`
    - Define `ForecastPoint` and `Forecast` dataclasses.
    - Implement `forecast_category(category, horizon_days, store) -> Forecast`:
      - Raise `ValueError` if `horizon_days` outside `[1, 365]`.
      - Raise `ValueError` if category has no transactions.
      - Raise `ValueError` if category has < 14 distinct calendar days of history.
      - Use Prophet with `daily_seasonality=False`, `interval_width=0.95`;
        read `yearly_seasonality` and `weekly_seasonality` from environment.
      - Floor `yhat`, `yhat_lower` at `0.0` in every `ForecastPoint`.
    - Implement `forecast_all(horizon_days, store) -> dict[str, Forecast | str]` — never
      raises; per-category error strings for insufficient data.
    - Add `src/forecasting/__init__.py` exporting `Forecaster`, `Forecast`, `ForecastPoint`.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ]* 9.2 Write unit tests in `tests/unit/test_forecaster.py`
    - Test cases derived from Properties 21–22:
      - `horizon_days=0` raises `ValueError`; `horizon_days=366` raises `ValueError`
        (Req 7.7).
      - Category with no transactions raises `ValueError` (Req 7.5).
      - Category with < 14 distinct days raises `ValueError` (Req 7.3).
      - Valid forecast (mocked Prophet output) → all `ForecastPoint` have `yhat >= 0`,
        `yhat_lower >= 0`, `yhat_lower <= yhat <= yhat_upper` (Property 21).
      - `forecast_all()` returns dict entries for all categories with data; does not raise
        even when some categories have insufficient data (Property 22).
    - _Requirements: 7.1, 7.3, 7.4, 7.5, 7.6, 7.7_

- [ ] 10. Implement `VectorStore`
  - [ ] 10.1 Write `src/api/vector_store.py` with `VectorStore` class
    - Initialize persistent ChromaDB collection named `"transactions"` at `CHROMA_PERSIST_DIR`.
    - Load `SentenceTransformer(embedding_model_name)` on construction (CPU-only).
    - Implement `index(transaction)` — upsert document with text
      `"{merchant} {category} {amount} {date}"` (exact order per Property 23 / Req 8.1)
      and metadata dict.
    - Implement `search(query, k) -> list[Transaction]` returning `min(k, total_indexed)`
      results reconstructed from metadata.
    - Implement `delete(transaction_id)` — no-op if id not found.
    - Add `src/api/__init__.py` exporting `VectorStore`.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 10.2 Write unit tests in `tests/unit/test_vector_store.py`
    - Test cases derived from Properties 23–27:
      - Indexed document text is `"{merchant} {category} {amount} {date}"` (Property 23).
        Use a mock ChromaDB client to capture the `document` argument passed to `add`.
      - Search with `k=5` on 3 indexed records returns 3 results (Property 24).
      - Search with `k=2` on 10 records returns 2 results (Property 24).
      - Delete an indexed ID → subsequent search does not return that transaction (Property 25).
      - Delete an ID never indexed → no exception raised (Property 25).
      - Index same transaction twice → collection contains exactly one document for that id
        (Property 26 — upsert idempotency).
    - _Requirements: 8.1, 8.3, 8.5, 8.6_

  - [ ] 10.3 Implement atomic insert pattern in `TransactionStore`
    - Update `TransactionStore.insert()` to wrap the SQL insert and `VectorStore.index()` call
      in a single SQLite transaction context so that a `VectorStore.index()` failure triggers
      a rollback.
    - Raise a `VectorStoreIndexError` (define in `src/domain.py`) on indexing failure.
    - _Requirements: 8.7_

  - [ ]* 10.4 Write unit tests for atomic insert in `tests/unit/test_vector_store.py`
    - Test cases derived from Property 27:
      - Simulate `VectorStore.index()` raising an exception during insert → `TransactionStore`
        is rolled back and the record is absent from the DB (Property 27).
    - _Requirements: 8.7_

- [ ] 11. Week 2 checkpoint — Ensure all tests pass
  - Run `pytest tests/unit/ -v`.
  - Ensure all tests pass; ask the user if questions arise.

---

## Week 3 — Days 15–21: API + Agent

- [ ] 12. Implement configuration management
  - [ ] 12.1 Write `src/config.py` with a `Settings` class (using `pydantic-settings` or
    manual `python-dotenv` loading)
    - Required variables: `SQLITE_DB_PATH`, `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL_NAME`,
      `LLM_API_KEY`.
    - Optional variables with defaults: `PROPHET_YEARLY_SEASONALITY=true`,
      `PROPHET_WEEKLY_SEASONALITY=true`, `LOG_LEVEL=INFO`.
    - On startup, if any required variable is missing, log an error naming the missing
      variable and raise `SystemExit(1)`.
    - System environment variables take precedence over `.env` file.
    - Export `get_settings()` factory function from `src/config.py`.
    - _Requirements: 12.1, 12.2, 12.3_

  - [ ]* 12.2 Write unit tests in `tests/unit/test_config.py`
    - Test cases derived from Property 29 + Req 12.3:
      - Each of the four required vars missing individually → `SystemExit` raised, error
        message contains the variable name (Property 29).
      - All required vars present → `Settings` loads without error.
      - Optional vars absent → documented defaults are used (Req 12.2).
      - System env var overrides `.env` file value for the same key (Req 12.3).
    - _Requirements: 12.1, 12.2, 12.3_

- [ ] 13. Implement Pydantic DTOs and FastAPI app skeleton
  - [ ] 13.1 Write `src/api/models.py` with all Pydantic v2 response DTOs
    - Define `TransactionDTO`, `ForecastPointDTO`, `ForecastDTO`, `IngestResponse`,
      `ChatResponse` exactly matching the design's data model section.
    - Define `ChatRequest(message: str, session_id: str)` with max-length validators
      (message ≤ 2000 chars, session_id ≤ 128 chars).
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ] 13.2 Write `src/api/app.py` with the FastAPI application and all six endpoints
    - Wire up `POST /ingest`, `GET /transactions`, `GET /anomalies`,
      `GET /forecast/{category}`, `POST /chat`, `GET /docs` (auto-generated).
    - `POST /ingest`: detect PDF vs CSV by content-type / extension; delegate to parser;
      run `Categorizer.predict_batch()`; call `TransactionStore.insert()` + `VectorStore.index()`;
      run `AnomalyDetector.fit_and_score()` if store has ≥ 10 records; return `IngestResponse`.
    - `GET /transactions`: accept optional `start_date`, `end_date`, `category` query params;
      return HTTP 200 with `list[TransactionDTO]` (empty list when no matches).
    - `GET /anomalies`: return `list[TransactionDTO]` ordered by anomaly score descending.
    - `GET /forecast/{category}`: accept `days` param (1–365, default 30); return `ForecastDTO`;
      return HTTP 422 with descriptive message if insufficient data.
    - `POST /chat`: accept `ChatRequest`; delegate to `FinancialAgent.chat()`; return `ChatResponse`.
    - Non-PDF/CSV upload → HTTP 422; internal errors → HTTP 500 + log full stack trace.
    - Add dependency injection for `TransactionStore`, `VectorStore`, `Categorizer`,
      `AnomalyDetector`, `Forecaster`, `FinancialAgent` via FastAPI `Depends`.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

- [ ] 14. Implement `FinancialAgent`
  - [ ] 14.1 Write `src/agent/tools.py` defining the four LangChain tools
    - `retrieve_transactions`: calls `VectorStore.search(query, k)` and returns serialized
      transaction text.
    - `calculate_total`: calls `TransactionStore.query_by_category()` or
      `query_by_date_range()` and sums amounts.
    - `run_forecast`: calls `Forecaster.forecast_category()` and returns a text summary.
    - `get_anomalies`: calls `AnomalyDetector.get_anomalies()` and returns serialized list.
    - Each tool has typed input schema; no free-text parameter values as structured args.
    - Add `src/agent/__init__.py` exporting `TOOL_REGISTRY`.
    - _Requirements: 9.2, 9.3_

  - [ ] 14.2 Write `src/agent/agent.py` with `FinancialAgent` class
    - `__init__` accepts `store`, `vector_store`, `forecaster`, and `session_memory` dict.
    - Implement out-of-scope guard: keyword check on message before passing to LangChain
      executor; return canned response if non-finance question.
    - Implement `chat(message, session_id) -> str` using LangChain agent with `TOOL_REGISTRY`.
    - Session memory: keep last 5 `HumanMessage` + `AIMessage` pairs per `session_id`
      in `session_memory` dict; trim on each call.
    - When tool raises exception: inform user of tool error; do not fabricate data.
    - When all tools return empty: state no relevant data found; do not fabricate.
    - Export `FinancialAgent` from `src/agent/__init__.py`.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ]* 14.3 Write unit tests in `tests/unit/test_agent.py`
    - Test cases derived from Property 28 + Req 9.4–9.7:
      - After 6 exchanges in a session, memory contains exactly the last 5 pairs
        (10 messages); earlier messages are absent (Property 28).
      - New `session_id` starts with empty memory.
      - When all tools return empty → response string does not assert fabricated data
        (Req 9.4).
      - Out-of-scope question (e.g., "What is the weather today?") → canned scoped
        response without tool invocations (Req 9.6).
      - Tool raises exception → response contains error acknowledgment, no fabricated
        amounts (Req 9.7).
    - Use `pytest-mock` to patch LangChain executor and tool functions.
    - _Requirements: 9.4, 9.5, 9.6, 9.7_

- [ ] 15. Week 3 checkpoint — Ensure all unit tests pass
  - Run `pytest tests/unit/ -v`.
  - Ensure all tests pass; ask the user if questions arise.

---

## Week 4 — Days 22–30: Frontend + Polish

- [ ] 16. Write API integration tests
  - [ ] 16.1 Write `tests/integration/test_api.py` using `fastapi.testclient.TestClient`
    - `POST /ingest` with a valid CSV fixture → HTTP 200, `{"ingested": N}` where N > 0.
    - `POST /ingest` with a `.txt` file → HTTP 422.
    - `POST /ingest` with a valid PDF fixture → HTTP 200, `{"ingested": N}`.
    - `GET /transactions` with no data → HTTP 200, `[]`.
    - `GET /transactions` with `start_date` after `end_date` → HTTP 422.
    - `GET /transactions` with `category` filter → HTTP 200, only matching records returned.
    - `GET /anomalies` with no data → HTTP 200, `[]`.
    - `GET /forecast/{category}` with a category that has insufficient data → HTTP 422
      with descriptive message.
    - `POST /chat` with valid `message` and `session_id` → HTTP 200, response has `answer`
      field (mock the agent).
    - `POST /chat` with `message` > 2000 characters → HTTP 422.
    - `GET /docs` → HTTP 200.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

  - [ ] 16.2 Write `tests/integration/test_ingestion_pipeline.py`
    - End-to-end test using in-memory SQLite and mocked ChromaDB:
      - Upload sample CSV → parse → `predict_batch` → `TransactionStore.insert()` →
        `VectorStore.index()` for each record → `AnomalyDetector.fit_and_score()`.
      - Assert returned `ingested` count equals number of valid rows in the CSV.
      - Assert inserted records appear in `TransactionStore.get_all()`.
      - Assert duplicate upload of the same CSV returns `ingested=0`.
    - _Requirements: 10.1, 4.2, 4.3, 8.7_

- [ ] 17. Implement Streamlit Dashboard
  - [ ] 17.1 Write `src/dashboard/app.py` with the Streamlit frontend
    - Page layout: sidebar for file upload + API base URL; main area with tabs for
      Overview, Anomalies, Forecast, Chat.
    - **Overview tab**: fetch `GET /transactions`, compute spending by category, display
      bar or pie chart; fetch monthly totals and display trend line chart; show empty-state
      message if no data.
    - **Anomalies tab**: fetch `GET /anomalies`; display table of merchant, amount, date,
      anomaly score; show empty-state message if empty.
    - **Forecast tab**: fetch `GET /forecast/{category}` for each available category;
      display line chart with confidence interval bands; show per-category "insufficient
      data" message where applicable.
    - **File upload widget**: `st.file_uploader` for PDF/CSV; on submit call `POST /ingest`;
      display success message with ingested count or error message.
    - **Chat tab**: `st.text_input` for message; call `POST /chat` with `session_id` from
      `st.session_state`; display conversation history; maintain history across reruns via
      `st.session_state`.
    - If API is unreachable on any call, display an error banner (do not render broken UI).
    - Add `src/dashboard/__init__.py`.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9_

- [ ] 18. Wire everything together and finalize environment setup
  - [ ] 18.1 Write `src/main.py` as the FastAPI application entry point
    - Call `get_settings()` on import — if any required variable is missing, log the
      variable name and call `sys.exit(1)` before creating the FastAPI app.
    - Instantiate `TransactionStore`, `VectorStore`, `Categorizer`, `AnomalyDetector`,
      `Forecaster`, `FinancialAgent` using settings values.
    - Expose the `app` object from `src/api/app.py` for `uvicorn src.main:app`.
    - Configure Python logging to respect `LOG_LEVEL` from settings.
    - _Requirements: 12.1, 12.2, 12.3_

  - [ ] 18.2 Create `.env.example` in the repo root
    - List all required variables (`SQLITE_DB_PATH`, `CHROMA_PERSIST_DIR`,
      `EMBEDDING_MODEL_NAME`, `LLM_API_KEY`) with placeholder values and inline comments
      describing purpose and valid value ranges.
    - List optional variables with their default values and comments.
    - _Requirements: 12.4_

  - [ ] 18.3 Train `Categorizer` on synthetic data and save model artifact
    - Write `scripts/train_categorizer.py` that:
      - Generates 3000 labeled transactions via `SyntheticGenerator`.
      - Trains `Categorizer` and asserts returned F1 ≥ 0.80.
      - Saves the trained model to `data/processed/categorizer.joblib`.
    - This script is run once during setup; the saved model is loaded by the API at startup.
    - _Requirements: 5.2, 5.3, 5.4_

- [ ] 19. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ -v`.
  - Ensure all unit and integration tests pass; ask the user if questions arise.


---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Core
  implementation tasks are never marked optional.
- Each task references specific requirements for traceability.
- Checkpoints after each week ensure incremental validation before moving to the next layer.
- Test cases are hand-written `def test_...` functions using `pytest.raises`, `assert`, and
  `pytest-mock`. No Hypothesis or property-based testing framework is used.
- The 29 Correctness Properties in `design.md` serve as specification/interview artifacts;
  the test tasks above describe which test cases are derived from each property.
- The Categorizer must be trained before the API starts. Run
  `python scripts/train_categorizer.py` once after Week 2 to produce the model artifact.
- CPU-only constraint: `all-MiniLM-L6-v2` (22 MB) for embeddings; Prophet fits are fast
  at 3000 transactions; Isolation Forest is CPU-native. No GPU required anywhere.
- `pytest-cov` can be added optionally in Week 4 to identify meaningful gaps, but coverage
  percentage is aspirational and not a blocking gate.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["2.2", "3.1", "3.2"] },
    { "id": 2, "tasks": ["2.3", "4.1", "5.1"] },
    { "id": 3, "tasks": ["4.2", "5.2", "7.1"] },
    { "id": 4, "tasks": ["7.2", "8.1", "9.1", "10.1"] },
    { "id": 5, "tasks": ["8.2", "9.2", "10.2", "10.3"] },
    { "id": 6, "tasks": ["10.4", "12.1", "13.1"] },
    { "id": 7, "tasks": ["12.2", "13.2", "14.1"] },
    { "id": 8, "tasks": ["14.2"] },
    { "id": 9, "tasks": ["14.3", "16.1", "16.2", "18.1", "18.2", "18.3"] },
    { "id": 10, "tasks": ["17.1"] }
  ]
}
```
