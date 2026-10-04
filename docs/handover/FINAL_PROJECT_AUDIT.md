# FINAL_PROJECT_AUDIT.md
# FinSight AI — Final Project Audit for ChatGPT Continuation
# All answers based on direct codebase inspection. Inferences explicitly labeled.
# Date: 2026-07-05

---

## SECTION 1 — Design Decisions

### NumPy Vector Store (`src/api/vector_store.py`)

**Current implementation:** Custom NumPy cosine similarity search. Embeddings stored as
`(N, 384)` float32 array in `vector_store.npy`. Metadata in `vector_store_metadata.json`.
Both loaded into RAM on every `VectorStore()` instantiation.

**Evidence of choice:** Commit `a2e167b` — "Implement VectorStore with sentence-transformers
embeddings and numpy similarity search." ChromaDB is installed (v1.5.9 in venv) but never
imported in any source file. The change was deliberate.

**Alternatives:** ChromaDB (spec choice), FAISS, SQLite-vss, Annoy.

**Advantages of NumPy:**
- Zero test isolation issues (each test gets clean `tmp_path`)
- Full implementation visibility (150 lines, no black box)
- No ChromaDB version upgrade surprises
- Sub-millisecond search for ≤10,000 vectors

**Disadvantages:**
- O(n) `_find_index()` scan on every index/delete operation
- Entire matrix loaded into RAM on every instantiation
- No metadata filtering during search
- Full `.npy` rewrite on every `index()` call (O(n²) disk writes during batch ingest)

**Recommendation: Keep for v1. Replace with ChromaDB at >10,000 transactions.**
The migration path is documented — it's a `vector_store.py` rewrite that doesn't touch
any other module. The interface contract (`index`, `search`, `delete`) stays the same.

---

### Logistic Regression (`src/categorization/categorizer.py`)

**Current implementation:** `sklearn.Pipeline` with `TfidfVectorizer(analyzer='char_wb',
ngram_range=(2,4), max_features=10000, sublinear_tf=True)` + `LogisticRegression(C=5.0,
class_weight='balanced', max_iter=1000)`.

**Evidence of choice:** `categorizer.py` line 73–90. Requirements doc Requirement 5.3
explicitly lists "TF-IDF or sentence-transformer embeddings" as valid options.

**Alternatives:** LinearSVC (faster, no probability calibration), sentence-transformers
(better generalization), zero-shot LLM classification.

**Advantages:** 75KB model artifact, 5-second training, character n-gram features capture
merchant-specific substrings, `predict_proba()` enables confidence thresholding.

**Disadvantages:** Memorizes exact training-set merchant names. F1=1.0 on synthetic data
but ~35% on unseen real merchants (verified by live execution). No semantic understanding.

**Recommendation: Keep for v1 but expand training data. Long-term migrate to sentence-transformer
embeddings using the same `all-MiniLM-L6-v2` model already loaded by VectorStore.**

---

### Isolation Forest (`src/anomaly/anomaly_detector.py`)

**Current implementation:** `IsolationForest(contamination=0.05, random_state=42)` with
feature matrix `[amount, label_encoded_category]`.

**Evidence of choice:** `anomaly_detector.py` lines 44–58. Design doc explicitly specifies
Isolation Forest from scikit-learn.

**Alternatives:** Local Outlier Factor, One-Class SVM, per-category z-score thresholding.

**Advantages:** Unsupervised (no labels needed), handles multi-dimensional features,
sklearn native, no hyperparameter tuning required for demo.

**Disadvantages:** Fixed contamination=0.05 always flags exactly 5% as anomalies.
Score normalization produces near-zero values (max=0.068 confirmed in live DB).
Re-fits from scratch on every ingest (no incremental training). Doesn't reason about
whether an amount is unusual FOR that specific category.

**Recommendation: Keep Isolation Forest but fix score normalization (min-max rescaling
per batch). Consider per-category z-score as a v2 addition.**

---

### EWMA Forecasting (`src/forecasting/forecaster.py`)

**Current implementation:** Exponentially weighted moving average (α=0.3) + linear trend
via `numpy.polyfit` on second half of history. 95% CI from residual std × 1.96.

**Evidence of choice:** Commit `1737418` — "Remove unused prophet and statsmodels
dependencies." `cmdstanpy==1.3.0` is in the venv (Prophet's backend) but Prophet is
not imported anywhere.

**Alternatives:** Prophet (spec choice), statsmodels ARIMA, simple linear regression,
Holt-Winters exponential smoothing.

**Advantages:** Zero external dependencies beyond numpy/pandas, sub-millisecond per
category, produces correct output structure (yhat/yhat_lower/yhat_upper), satisfies
all spec acceptance criteria except the specific library name.

**Disadvantages:** No seasonal adjustment (no weekly/monthly patterns), linear trend
assumption breaks for exponential growth/decline, CI from residual std is approximate.

**Recommendation: Keep EWMA. Prophet's 8-second-per-category fit time on CPU makes it
impractical for the forecast_all endpoint. If seasonality becomes important, add Prophet
as an optional backend with EWMA fallback.**

---

### SQLite (`src/ingestion/transaction_store.py`)

**Current implementation:** Python stdlib `sqlite3` with manual schema initialization.
`uq_transaction` unique index on `(date, merchant, amount, source_file)`.

**Alternatives:** PostgreSQL, DuckDB, flat CSV files.

**Advantages:** No server process, portable single file, stdlib (zero dependency),
transactions support atomic operations, row_factory enables named column access.

**Disadvantages:** No concurrent write access (single writer at a time), limited
indexing options for analytics queries (e.g., `GROUP BY category` has no covering index).

**Recommendation: Keep. SQLite is correct for single-user local use. PostgreSQL would
add operational complexity (server management, connection pooling) with no benefit at
this scale.**

---

### FastAPI (`src/api/app.py`)

**Current implementation:** FastAPI 0.110.0 with Pydantic v2 DTOs, CORSMiddleware,
manual component instantiation in `_get_components()`.

**Alternatives:** Flask, Django REST Framework, Starlette directly.

**Advantages:** Native OpenAPI docs, Pydantic v2 validation, async support, type hints
drive documentation, `TestClient` for integration tests.

**Disadvantages:** Current `_get_components()` defeats async benefits (synchronous
component loading on every request), no `Depends` injection pattern used.

**Recommendation: Keep FastAPI. Fix `_get_components()` → `lifespan` pattern.**

---

### Streamlit (planned, not implemented)

**Current implementation:** Does not exist. No source files. Not installed.

**Evidence of choice:** `tasks.md` Task 17.1 specifies Streamlit. `requirements.md`
Requirement 11 specifies Streamlit-like behavior.

**Alternatives:** React + Vite, Gradio, Panel, Dash.

**Advantages:** Same language as backend, `st.session_state` handles chat state, direct
matplotlib integration, no JavaScript/build step.

**Recommendation: Implement Streamlit as specified. It's the right choice for a Python-first
data project with a chat interface.**

---

### LangChain (`src/agent/agent.py`, `src/agent/tools.py`)

**Current implementation:** `langchain==0.1.20`, `create_tool_calling_agent` +
`AgentExecutor`, `ChatGroq(model="llama-3.1-8b-instant")`, 4 tools.

**Alternatives:** Raw Groq API + manual tool parsing, LangGraph, semantic-kernel.

**Advantages:** `@tool` decorator makes tool definition clean, `AgentExecutor` handles
iteration and error recovery, `handle_parsing_errors=True` is essential for small LLMs.

**Disadvantages:** LangChain 0.1.x is several major versions behind current (0.3+),
breaking changes in upgrade path, Groq/LangChain compatibility issue with tool-call
history (documented in DEVELOPER_MEMORY_DUMP.md).

**Recommendation: Keep LangChain 0.1.20 for now. Upgrading to 0.3+ would require
significant agent refactoring. Only upgrade when new LangChain features are needed.**

---

### Groq + Llama 3.1 8B (`src/agent/agent.py`)

**Current implementation:** `ChatGroq(model="llama-3.1-8b-instant", temperature=0)`.
Free tier. Model name hardcoded. No fallback.

**Alternatives:** OpenAI GPT-4o (paid), Anthropic Claude (paid), local Ollama (free, no API).

**Advantages:** Free tier (14,400 req/day), fast inference (~500ms), no billing risk.

**Disadvantages:** Smaller model quality, known multi-turn tool-call bug, model name
hardcoded, no fallback on rate limit.

**Recommendation: Keep for development. Add `LLM_MODEL_NAME` env var so the model can
be swapped without code changes.**

---

### sentence-transformers (`src/api/vector_store.py`)

**Current implementation:** `SentenceTransformer("all-MiniLM-L6-v2")`, 384-dim CPU
embeddings, loaded on every `VectorStore()` instantiation.

**Alternatives:** OpenAI `text-embedding-ada-002`, TF-IDF document vectors, BM25.

**Advantages:** Free, CPU-only, 22MB model, good semantic quality for short merchant texts.

**Disadvantages:** Model loaded on every API request (via `_get_components()`), requires
internet on first use to download from HuggingFace, PyTorch DLL issue prevents pytest
collection on Windows.

**Recommendation: Keep. Fix the per-request reload issue with `lifespan` startup.**


---

## SECTION 2 — Divergence From Original Plan

| Spec Said | Actual Implementation | File(s) | Beneficial? | Revisit? |
|---|---|---|---|---|
| ChromaDB vector store | NumPy + `.npy` files | `src/api/vector_store.py` | **Yes** — simpler, no test isolation issues | At >10k transactions |
| Prophet forecasting | EWMA + linear trend | `src/forecasting/forecaster.py` | **Yes** — 64x faster on CPU | If seasonality needed |
| `pydantic-settings` for config | Manual `python-dotenv` + custom Settings class | `src/config.py` | **Yes** — `sys.exit(1)` on missing vars is cleaner | No |
| Atomic VectorStore+SQLite insert | Non-atomic, VectorStore failure logged as warning | `src/api/app.py` | **No** — design violation | Yes — implement atomicity |
| FastAPI `Depends` DI | `_get_components()` per-request helper | `src/api/app.py` | **No** — breaks session memory, causes latency | Yes — fix with `lifespan` |
| Session memory via LangChain `ConversationBufferWindowMemory` | In-memory dict of tuples, stateless LLM reasoning | `src/agent/agent.py` | **Yes** — avoids Groq tool-replay bug | Partial — add DB persistence |
| `IngestResponse(ingested: int)` only | Added `skipped: int` and `warnings: list[str]` | `src/api/models.py` | **Yes** — more informative | No |
| `GET /docs` as explicit endpoint | Auto-generated by FastAPI | `src/api/app.py` | **Yes** — FastAPI handles this natively | No |
| `langchain-openai` in task requirements | `langchain-groq` instead | `src/agent/agent.py` | **Yes** — Groq is free, OpenAI is paid | Only if OpenAI quality needed |
| Streamlit dashboard | Not implemented | N/A | N/A — outstanding task | Must be built |
| Integration tests | Not implemented | `tests/integration/` (empty) | N/A — outstanding task | Must be written |
| `predict_batch` errors → individual `try/except` | Correct — but the test validates it poorly | `src/categorization/categorizer.py` | Neutral | Fix the test |


---

## SECTION 3 — Module Health Review

### Ingestion (SyntheticGenerator, CSVParser, PDFParser, PrettyPrinter)
**Rating: 8/10**
- Strongest: CSVParser — column alias mapping, BOM handling, row-level error recovery, round-trip guarantee. Clean, testable, defensive.
- Weakest: PDFParser — works only on table-structured PDFs generated by reportlab. Real bank PDFs (coordinate-based text) return 0 records.
- Biggest risk: PDFParser fails silently on most real PDFs. No error is raised — it returns 0 records with the "No recognizable transaction table found" message. Users will think the parser is broken.
- Next improvement: Add `page.extract_words()` fallback with column detection when `page.extract_tables()` returns nothing.
- Files: `src/ingestion/csv_parser.py`, `src/ingestion/pdf_parser.py`, `src/ingestion/synthetic_generator.py`, `src/ingestion/pretty_printer.py`

---

### Transaction Storage (TransactionStore)
**Rating: 9/10**
- Strongest: Unique index `uq_transaction(date, merchant, amount, source_file)` handles deduplication correctly. All CRUD methods are clean and parameterized (no SQL injection risk). Named row factory.
- Weakest: No `update_anomaly_scores()` public method — `AnomalyDetector` accesses `_get_connection()` directly.
- Biggest risk: The `_init_schema()` creates schema with `CREATE TABLE IF NOT EXISTS`. If a schema migration is ever needed (new column), it won't run automatically. No migration framework.
- Next improvement: Add `update_anomaly_scores(dict[int, tuple[bool, float]]) -> None` public method. Add `query_by_needs_review() -> list[Transaction]`.
- File: `src/ingestion/transaction_store.py`

---

### Categorization (Categorizer)
**Rating: 6/10**
- Strongest: Confidence threshold + `needs_review` flag is exactly right. `save()`/`load()` roundtrip works. Stratified split logic handles small datasets.
- Weakest: Model trained on 44 synthetic merchants. F1=1.0 on synthetic data, ~35% on real data.
- Biggest risk: Users uploading real bank statements will see most transactions categorized as "Other" with `needs_review=True`. This makes the categorizer appear broken.
- Next improvement: Expand `SyntheticGenerator.MERCHANTS` with real-world names and retrain. Long-term: switch to sentence-transformer embeddings.
- File: `src/categorization/categorizer.py`

---

### Anomaly Detection (AnomalyDetector)
**Rating: 6/10**
- Strongest: Unsupervised (no labels needed), correctly implements Isolation Forest, `is_anomaly` boolean is accurate, `MIN_TRANSACTIONS=10` guard prevents meaningless fits.
- Weakest: Score normalization produces near-zero values (max=0.068 in live DB). Scores are meaningless for ranking. Directly accesses `store._get_connection()`.
- Biggest risk: Dashboard visualizations relying on `anomaly_score` magnitude will be flat/meaningless.
- Next improvement: Fix normalization to min-max per batch. Add `TransactionStore.update_anomaly_scores()` public method.
- File: `src/anomaly/anomaly_detector.py`

---

### Forecasting (Forecaster)
**Rating: 8/10**
- Strongest: EWMA + linear trend produces structurally correct output. `forecast_all()` never raises. Error messages are specific. Floor at 0.0 is correct.
- Weakest: No seasonal adjustment. `alpha=0.3` is hardcoded. Linear trend breaks for non-linear spending patterns.
- Biggest risk: Users with weekly spending patterns (e.g., grocery shopping every Saturday) will see flat forecasts that miss obvious peaks.
- Next improvement: Make `alpha` configurable via env var. Add `EWMA_ALPHA` to Settings.
- File: `src/forecasting/forecaster.py`

---

### Vector Store (VectorStore)
**Rating: 7/10**
- Strongest: Clean interface (index/search/delete), upsert idempotency, correct cosine similarity, persists to disk.
- Weakest: O(n) `_find_index()`, full matrix in RAM, full `.npy` rewrite per `index()`, re-indexes ALL transactions on every ingest.
- Biggest risk: PyTorch DLL issue means vector store tests cannot run in pytest on this Windows machine. Test suite is incomplete.
- Next improvement: Add `transaction_id → index` dict for O(1) lookup. Batch index saves. Only index new transactions during ingest.
- File: `src/api/vector_store.py`

---

### Agent (FinancialAgent, tools)
**Rating: 7/10**
- Strongest: Scope guard is effective. `_clean_response()` handles known LLM artifacts. `max_iterations=3` prevents runaway. Tool docstrings are well-written.
- Weakest: Session memory broken across HTTP requests. No date-range tool. `calculate_total` only accepts category name.
- Biggest risk: Chat context is stateless — the second question in a conversation has no knowledge of the first.
- Next improvement: Fix session memory via `lifespan`. Add `calculate_total_by_date_range` tool.
- Files: `src/agent/agent.py`, `src/agent/tools.py`

---

### API (FastAPI app)
**Rating: 7/10**
- Strongest: All 6 endpoints functional, Pydantic validation, CORS configured, file type validation, correct HTTP status codes.
- Weakest: `_get_components()` anti-pattern, no `lifespan` startup, no rate limiting, no HTTP 500 middleware, unsanitized upload filename.
- Biggest risk: Every request reloads SentenceTransformer (~4 second overhead). In practice the API is unusably slow until this is fixed.
- Next improvement: Fix `_get_components()` → FastAPI `lifespan`. Sanitize upload filename.
- File: `src/api/app.py`

---

### Database
**Rating: 9/10**
- Strongest: Schema is clean and minimal. Unique index handles deduplication. `row_factory = sqlite3.Row` enables named access. `_init_schema()` is idempotent.
- Weakest: No migration framework. No indexes on `category`, `date`, `is_anomaly` columns for fast filtering queries.
- Biggest risk: A schema change (new column) requires manual database migration — no tooling exists.
- Next improvement: Add `CREATE INDEX IF NOT EXISTS idx_category ON transactions(category)` and `idx_date ON transactions(date)` for faster query performance at scale.
- File: `src/ingestion/transaction_store.py`

---

### Configuration
**Rating: 9/10**
- Strongest: `sys.exit(1)` on missing required vars is exactly what the spec requires. Singleton pattern is correct. `reset_settings()` enables clean testing.
- Weakest: `PROPHET_YEARLY_SEASONALITY` and `PROPHET_WEEKLY_SEASONALITY` settings are unused. `configure_logging()` is never called from `main.py`.
- Biggest risk: If someone removes `.env` and restarts the API, the first request (not startup) triggers `sys.exit(1)` — which kills the uvicorn worker in an unexpected way.
- Next improvement: Call `get_settings()` inside `lifespan` startup so startup failures happen cleanly before the server accepts connections.
- File: `src/config.py`

---

### Testing
**Rating: 7/10**
- Strongest: 86 tests pass. Every public method has at least one test. Edge cases (BOM, empty files, duplicate inserts, boundary dates) are covered. `tmp_path` used correctly.
- Weakest: `test_vector_store.py` uncollectable on Windows (PyTorch DLL). Integration tests absent. `test_predict_batch_handles_individual_errors_without_aborting` is a false positive.
- Biggest risk: No API contract tests. Any `app.py` refactoring could break endpoints without any test catching it.
- Next improvement: Write `tests/integration/test_api.py` with `TestClient`. Fix the broken batch test.
- Files: `tests/unit/` (all 12 files), `tests/integration/` (empty)


---

## SECTION 4 — Code Review

### Excellent Code — Clean, leave mostly unchanged

**`src/domain.py`**
Perfect minimal domain model. Single responsibility. All downstream modules depend on it
and it has never needed to change. The `VectorStoreIndexError` exception class is correctly
placed here. Do not add business logic to this file.

**`src/ingestion/csv_parser.py`**
Textbook parser design. `ParseSummary` dataclass captures all outcomes. Column aliases
are data-driven (dict) not logic-driven (if/elif). `_parse_date()` and `_parse_amount()`
are pure functions with single responsibilities. BOM handled transparently. 12 tests all
pass. This is the cleanest non-trivial file in the project.

**`src/ingestion/pretty_printer.py`**
43 lines, two methods, zero side effects. Does exactly what it says. Roundtrip tested.

**`src/ingestion/synthetic_generator.py`**
Clean data generation with isolated `random.Random(seed)` for determinism. Category bounds
and merchant names are data-driven. Validation errors are specific. 12 tests.

**`src/ingestion/transaction_store.py`**
Clean SQLite wrapper. Parameterized queries throughout. Named row factory. Idempotent
schema init. One flaw (`AnomalyDetector` uses `_get_connection()`) is external to this file.

**`src/forecasting/forecaster.py`**
Well-structured with clear validation hierarchy: check horizon → check data exists →
check history length → compute. `forecast_all()` never-raises contract is implemented
correctly. EWMA math is correctly documented in comments.

**`src/config.py`**
Clean settings class with fast-fail validation, singleton pattern, and test reset support.
`configure_logging()` exists but isn't called — minor issue only.

---

### Acceptable Code — Works, could be improved

**`src/categorization/categorizer.py`**
The code is structurally clean. The problem is the training data quality, not the code.
The stratified split logic is clever. The `predict()` in-place mutation pattern is a
smell but consistent. The `save()`/`load()` pattern via joblib is correct.
Improvement: `is_trained` property (not `_is_trained` attribute) for better encapsulation.

**`src/anomaly/anomaly_detector.py`**
Logic is correct. The `_get_connection()` violation is the only real issue. Score
normalization formula is mathematically valid but produces poor output range.
Improvement: Public `update_anomaly_scores()` method in TransactionStore + fix normalization.

**`src/api/vector_store.py`**
Functionally correct. Clean interface. Issues: O(n) lookup, full-matrix reload per request,
silent exception swallowing in `_load()`. Each issue is fixable independently.
Improvement: Dict-based O(1) lookup, explicit error logging in `_load()`.

**`src/agent/tools.py`**
Tool definitions are clean and well-documented. The `calculate_total` manual input cleaning
(`if "=" in cleaned`) is a pragmatic hack. The tool closure captures variables correctly.
Improvement: Add `calculate_total_by_date_range` tool.

**`src/api/models.py`**
Clean Pydantic v2 models. `max_length` validators on `ChatRequest`. No issues with current
models. Will need updates when new endpoints are added.

**`tests/unit/test_categorizer.py`**
Mostly good. `test_predict_batch_handles_individual_errors_without_aborting` is the only
problem. `test_high_confidence_prediction_does_not_set_needs_review` correctly skips when
model is absent.

---

### Weak Code — Should eventually be redesigned

**`src/api/app.py`**
`_get_components()` is the central problem. Per-request component instantiation is
fundamentally wrong for a production API. No DI, no lifespan, no filename sanitization,
no rate limiting. CORS wildcard. The endpoint logic itself is correct but the infrastructure
around it needs a rewrite. This is the highest-priority refactor target.

**`src/ingestion/pdf_parser.py`**
`PDFParser` holds a `CSVParser` instance and calls its private methods. This is an
inheritance relationship implemented incorrectly as composition with private access.
The real fix is a `FieldMapper` utility class. Also: the parser only works on
table-structured PDFs — a fundamental limitation not visible in the code.

**`tests/unit/test_categorizer.py` (one specific test)**
`test_predict_batch_handles_individual_errors_without_aborting` wraps assertions in
`try/except RuntimeError`. This test can never fail, making it worse than useless.
Needs to be rewritten with a mock that raises inside `predict()`.

**`src/main.py`**
3 lines. Imports `app` from `api.app`. That's it. `configure_logging()` is never called.
The intended `lifespan`-based startup belongs here or in `app.py`. Currently this file
is a placeholder.


---

## SECTION 5 — Hidden Dependencies

| Dependency | Direction | Mechanism | Break Scenario |
|---|---|---|---|
| `PDFParser` → `CSVParser` private methods | `pdf_parser.py` calls `_canonical_field_name`, `_parse_date`, `_parse_amount` from `CSVParser` | `self._field_mapper = CSVParser()` | Rename any private method in `csv_parser.py` → runtime `AttributeError` in `pdf_parser.py` |
| `AnomalyDetector` → `TransactionStore._get_connection()` | anomaly_detector.py calls store's private DB method | Direct internal access | Changing SQLite connection management in `transaction_store.py` breaks `anomaly_detector.py` |
| `VectorStore._transaction_text()` format → Agent retrieval quality | Embedding space depends on text format `"{merchant} {category} {amount} {date}"` | Semantic search quality | Changing the format invalidates existing `.npy` embeddings — old and new entries would be in different embedding spaces |
| `conftest.py` path → ALL test imports | Every test file uses `from domain import Transaction` etc. without `src.` prefix | `sys.path.insert(0, "src")` in conftest | Deleting or moving `conftest.py` → every test fails with `ModuleNotFoundError` |
| `categorizer.joblib` label encoder → category string format | LabelEncoder is trained on exact `SyntheticGenerator` category strings | Joblib persistence | Changing category names in `SyntheticGenerator` without retraining → decoder produces wrong labels |
| `app.py` hardcoded model path → working directory | `Path("data/processed/categorizer.joblib")` is relative | Relative `Path()` call | Running uvicorn from any directory other than project root → model not found, categorization silently skipped |
| `test_pdf_parser.py` → fixture files on disk | Tests read `tests/fixtures/*.pdf` | Direct `Path(__file__)` resolution | Running tests without fixture files (fresh clone without running generate script) → all 5 tests fail |
| `Transaction` field order → `PrettyPrinter` CSV output | `CANONICAL_FIELDS = ("date", "merchant", "amount", "category")` | Hard-coded tuple | Adding a field to `Transaction` without updating `PrettyPrinter.CANONICAL_FIELDS` → field not serialized |
| `Transaction` SQLite schema → `Transaction` dataclass fields | `_init_schema()` column list must match dataclass | Manual sync | Adding field to `Transaction` without adding column to schema → `OperationalError` on insert |
| `ChatRequest` max_length → Pydantic validation | `message: str = Field(..., max_length=2000)` | Pydantic v2 constraint | If max_length is removed → no protection against large payloads reaching Groq API |
| `IsolationForest` feature matrix order → score interpretation | `[amount, encoded_category]` column order is implicit | numpy `column_stack` | Swapping column order doesn't raise an error but changes which dimension drives anomaly detection |

---

## SECTION 6 — Untested Areas

| Untested Functionality | Why Not Tested | Testing Needed |
|---|---|---|
| All 6 API endpoints via HTTP | `tests/integration/` is empty | `TestClient`-based integration tests (tasks.md Task 16.1) |
| Full ingest pipeline end-to-end | No integration test exists | `test_ingestion_pipeline.py` with real CSV file, verify counts |
| `POST /chat` with real Groq API | Network call — avoided in unit tests | Manual test only; mock in integration tests |
| VectorStore with real data (200 entries) | PyTorch DLL issue prevents collection on Windows | Fix DLL issue or skip via marker; tests are written but can't run |
| Atomic rollback (VectorStore failure → SQLite rollback) | Not implemented, so no test exists | Add implementation first, then property test |
| `AnomalyDetector` with `contamination="auto"` | Tests only use float contamination | Add test with `contamination="auto"` |
| `GET /transactions` with both `start_date` and `category` simultaneously | Endpoint handler uses if/elif — only one filter applied | Integration test with both params |
| `Forecaster.forecast_all()` on empty store | `forecast_all()` calls `store.get_all()` on empty DB — returns empty dict | Unit test exists for `forecast_all` but only with data |
| `VectorStore._save()` partial failure (write interruption) | Hard to test; platform-specific | Not worth testing in unit tests |
| PDF parser on real bank PDFs | No real PDFs available | Manual testing with actual HDFC/SBI/Chase exports |
| `app.py` HTTP 500 responses | No test verifies internal error → HTTP 500 | Integration test with mocked component raising exception |
| `needs_review=True` transactions in API responses | `TransactionDTO` includes `needs_review` but no test verifies its value | Integration test: upload ambiguous merchants, verify `needs_review=True` in response |

---

## SECTION 7 — Performance Review

| Bottleneck | Rank | Current Cost | Fix | Expected Improvement |
|---|---|---|---|---|
| `SentenceTransformer` loaded per request | 1 | ~4 seconds per any API call | FastAPI `lifespan` singleton | 4000ms → 0ms overhead per request |
| VectorStore re-indexes ALL transactions per ingest | 2 | O(n) encode calls per ingest | Only index newly inserted txns | 200 encodes → N_new encodes (e.g., 10 for typical upload) |
| `categorizer.joblib` loaded per request | 3 | ~50ms per request | `lifespan` singleton | 50ms → 0ms |
| `VectorStore._find_index()` O(n) scan | 4 | ~0.1ms at 200 txns, ~50ms at 100k | Dict-based O(1) lookup | Negligible now, critical at scale |
| Full `.npy` rewrite per `index()` call | 5 | ~1ms at 200 txns, ~20ms at 100k | Batch saves | Negligible now, important at scale |
| IsolationForest full refit per ingest | 6 | ~100ms at 200 txns, ~5s at 100k | Periodic refit (every 100 new txns) | At 100k: 5s → ~50ms amortized |
| Groq API latency for chat | 7 | ~500ms-2s (network) | None (external) | Cannot improve without changing provider |

**Priority fix: Item 1 (lifespan startup) eliminates 80% of observed latency.**


---

## SECTION 8 — Scalability Review (200 → 100,000 Transactions)

| Component | Problem at 100k | Severity | Fix |
|---|---|---|---|
| `VectorStore` in-RAM matrix | 100k × 384 × 4 bytes = 153 MB in RAM on every request | Critical | Switch to ChromaDB or FAISS; or at minimum keep matrix loaded as singleton |
| `VectorStore._find_index()` O(n) | 100k iterations per `index()` call | High | Dict `{transaction_id: matrix_index}` for O(1) lookup |
| `VectorStore` full `.npy` rewrite | 153 MB written to disk per `index()` call | High | Batch writes with dirty flag |
| Re-index all on ingest | 100k encode calls per ingest | Critical | Only index new transactions |
| IsolationForest full refit | `np.column_stack([amounts, categories])` creates 100k × 2 array in RAM + IsolationForest fit = ~5 seconds | High | Periodic refit, not per-ingest |
| `store.get_all()` in agent tools | `calculate_total("all")` loads all 100k transactions into Python memory | Medium | Add SQL aggregation: `SELECT SUM(amount) FROM transactions` |
| `GET /transactions` without pagination | Returns all 100k rows as JSON | High | Add `limit`/`offset` or `cursor` pagination to the endpoint |
| `TransactionStore.get_all()` sort | SQLite sorts 100k rows without an index on `date` | Medium | Add `CREATE INDEX idx_date ON transactions(date)` |
| SQLite write lock | Concurrent ingest requests would queue behind the SQLite write lock | Low (single-user) | Not a concern for single-user; matters for multi-user |

**Critical architectural change needed at 50k+:** Replace in-memory NumPy VectorStore
with ChromaDB persistent collection. The interface contract (`index/search/delete`)
stays identical — it's a `vector_store.py` rewrite only.

**Database additions needed at 10k+:**
```sql
CREATE INDEX IF NOT EXISTS idx_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_is_anomaly ON transactions(is_anomaly);
```

---

## SECTION 9 — Security Review

The following concerns are NOT documented in previous handover files:

**NEW: The `tmp_path` for uploaded files uses the upload filename directly.**
```python
tmp_path = Path(f"data/raw/{filename}")
```
On Windows, `filename` from `UploadFile.filename` can contain path separators like
`..\\..\\config.py`. `Path("data/raw/../../src/config.py")` resolves to
`src/config.py`. The file would be WRITTEN to `src/config.py`, overwriting the source.
**This is a path traversal vulnerability.** Fix:
```python
safe_filename = Path(file.filename).name  # strips all path components
tmp_path = Path("data/raw") / safe_filename
```

**NEW: No Content-Type validation on file upload.**
FastAPI's `UploadFile` provides `content_type` from the HTTP header. But the current
code only checks the filename extension: `filename.endswith(".csv")`. A client can
send `Content-Type: application/pdf` with a CSV body and it will be processed as a CSV
(correctly), or send `Content-Type: text/plain` with a `.pdf` extension and it will be
processed by PDFParser (likely returning 0 records, not a security issue but confusing).
No MIME validation is performed.

**NEW: Exception messages may leak internal paths.**
`summary.file_errors.append(f"File not found: {file_path}")` in `CSVParser` and
`PDFParser` includes the full filesystem path in error messages. These errors are returned
in the HTTP response body. A user who uploads a nonexistent path (if they could craft one)
would see the server's filesystem path structure. In practice this only occurs with the
`file_not_found` edge case which the API prevents (it always writes the file first).

**All other security concerns were already documented in HANDOVER_DEEP_DIVE.md Section 14.**

---

## SECTION 10 — API Contract Review

### `POST /ingest`
- Production ready? **No** — path traversal vulnerability, per-request model reload
- Missing validation: file size limit on CSV (spec says ≤100MB, not enforced), MIME type check
- Missing error handling: HTTP 500 for unexpected errors (currently logs warning, may silently succeed with 0 ingested)
- Missing edge case: what happens if filename is empty string? `file.filename or ""` returns `""`, then `"".endswith(".csv")` is False → HTTP 422 (correct, but confusing error message)

### `GET /transactions`
- Production ready? **Almost** — functional but per-request model reload
- Missing validation: None beyond date parsing
- Missing error handling: No error if only `start_date` is provided without `end_date` (currently returns all transactions — potentially unexpected behavior)
- Missing edge case: `start_date` and `category` both provided → `start_date and end_date` check fails (only start_date is truthy), falls through to `category` branch → date ignored silently

### `GET /anomalies`
- Production ready? **Yes** — simple passthrough, no validation needed
- Missing: No pagination (returns all anomalies)

### `GET /forecast/{category}`
- Production ready? **Almost** — correct HTTP 422 on insufficient data
- Missing validation: Category path parameter not validated against known categories — unknown category returns 422 with a reasonable error message (correct behavior)
- Missing edge case: `days=0` — FastAPI Query `ge=1` catches this correctly with HTTP 422

### `POST /chat`
- Production ready? **No** — session memory broken, model hardcoded, no rate limiting
- Missing validation: `session_id` accepts empty string — produces valid but broken session
- Missing error handling: Groq rate limit (HTTP 429) surfaces as generic "error" message, not a retryable error indication
- Missing edge case: What if `message` is all whitespace? Passes `max_length=2000` check but the scope guard (`_is_finance_question`) likely returns False → out-of-scope response

### `GET /health`
- Production ready? **Yes** — trivial, no issues

---

## SECTION 11 — Database Review

**Schema (from `transaction_store.py` `_init_schema()`):**
```sql
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount REAL NOT NULL,
    category TEXT NOT NULL DEFAULT '',
    is_anomaly INTEGER NOT NULL DEFAULT 0,
    anomaly_score REAL,
    source_file TEXT NOT NULL DEFAULT '',
    needs_review INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX uq_transaction ON transactions(date, merchant, amount, source_file);
```

**Normalization issues:**
- `category` is stored as a text string per transaction — there is no `categories` lookup table. If a category name changes (e.g., "Dining" → "Food"), all rows need an `UPDATE`. Acceptable for this project scope.
- `merchant` is stored raw per transaction — no merchant normalization. "WHOLE FOODS #123" and "WHOLE FOODS #456" are different merchants. Acceptable.

**Indexing opportunities:**
- `category` — no index. `query_by_category()` does a full table scan with `LOWER(category)`. At 100k rows this degrades. Add: `CREATE INDEX IF NOT EXISTS idx_category ON transactions(LOWER(category))`
- `date` — no index. `query_by_date_range()` uses `WHERE date >= ? AND date <= ?` on a TEXT column (ISO 8601 format sorts lexicographically correctly, so this works). But without an index it's a full scan. Add: `CREATE INDEX IF NOT EXISTS idx_date ON transactions(date)`
- `is_anomaly` — no index. `get_anomalies()` does `WHERE is_anomaly=1` via Python filter on `get_all()`, not SQL. At 100k rows, loading all to filter in Python is wasteful. Add SQL filter: `WHERE is_anomaly = 1`

**Missing constraints:**
- `amount` has no `CHECK (amount > 0)` constraint — negative amounts are valid in SQLite but semantically wrong for this domain
- `category` has no `CHECK` constraint against `CANONICAL_CATEGORIES` — any string is accepted

**Migration concerns:**
- No migration framework (Alembic, Flask-Migrate). Schema changes require manual SQL.
- `CREATE TABLE IF NOT EXISTS` means schema additions (new columns) are silent no-ops — they don't get added to existing tables.
- If a schema migration is ever needed: `ALTER TABLE transactions ADD COLUMN <name> <type> DEFAULT <value>` is SQLite's only safe in-place migration option. Dropping/renaming columns requires creating a new table and copying data.


---

## SECTION 12 — ML Review

### Categorizer

**Strengths:**
- TF-IDF on character n-grams is well-suited for short merchant name strings
- `class_weight='balanced'` handles category imbalance in training data
- Confidence threshold (0.60) + `needs_review` flag is the right safety net design
- Save/load via joblib is correct and tested
- Stratified split logic handles small datasets without crashing

**Weaknesses:**
- Training vocabulary: 44 merchants × 8 categories = exactly the merchants in `SyntheticGenerator.MERCHANTS`
- Zero semantic understanding: `"Starbucks"` and `"Coffee Shop"` share no character n-grams with `"Chipotle"` or `"Pizza Place"`
- F1=1.0 on synthetic data is a false benchmark
- ~35% of real-world merchant names score below the 0.60 confidence threshold (verified: Starbucks=0.352, DoorDash=0.301, Instacart=0.498)

**Expected real-world performance:**
- Merchants that appear in training vocabulary (IKEA, Walgreens, Netflix, etc.): 98%+ accuracy
- Common merchants NOT in training vocabulary: 30-40% accuracy
- For a real bank statement from an Indian user: likely <20% confident predictions (Indian merchants like Zomato, Swiggy, BigBasket, Reliance Fresh are not in the training set)

**Recommended improvements (in order):**
1. Add 100+ real-world merchant names to `SyntheticGenerator.MERCHANTS` and retrain
2. Switch to sentence-transformer embeddings (same `all-MiniLM-L6-v2` model already in use) for zero-shot generalization
3. Add a user-correction workflow: when `needs_review=True`, let users set the correct category, store as training data, retrain periodically

---

### Anomaly Detector

**Strengths:**
- Fully unsupervised — no labeled data required
- `is_anomaly` boolean flag is accurate (10/200 transactions flagged at 5% contamination)
- `MIN_TRANSACTIONS=10` guard prevents meaningless fits
- Re-fits on every ingest, so scores stay current with new data

**Expected false positives:**
High. Any legitimate large purchase (e.g., annual insurance premium, big IKEA haul) that is
statistically unusual compared to historical spending will be flagged. With contamination=0.05,
exactly 5% of transactions are always flagged regardless of whether they're genuinely anomalous.
In a 200-transaction database with uniform synthetic data, the 10 "anomalies" are simply
the highest-amount transactions in each category — not true anomalies.

**Expected false negatives:**
Low for truly unusual amounts. But the model uses only `[amount, category]` features —
it cannot detect:
- Unusual merchant (a transaction from a merchant you've never used)
- Unusual time pattern (3 AM transaction)
- Geographic anomaly (transaction from a different country)
- Duplicate transactions

**Weaknesses:**
- Score normalization broken (max=0.068 in live DB, should be approaching 1.0)
- Fixed contamination=0.05 always flags exactly 5% regardless of actual data
- Feature set too limited: amount + category only
- Directly accesses `store._get_connection()` — encapsulation violation

---

### Forecaster

**Strengths:**
- EWMA correctly weights recent data more heavily
- Linear trend component handles stable growth/decline patterns
- `forecast_all()` never-raises contract is correctly implemented
- Floor at 0.0 is correct for spending data
- 95% CI from residual std is reasonable for smooth data

**When it will fail:**
- **Weekly seasonal spending** (e.g., grocery shopping every weekend): EWMA produces a flat average, completely missing the weekly peaks
- **Monthly cyclic spending** (e.g., rent on the 1st, salary on the 15th): no seasonal component
- **Step changes** (e.g., user starts spending much more on dining from March onwards): EWMA will lag by weeks at α=0.3
- **Categories with very few transactions but >14 days**: may forecast with large CI bands that include negative values (floored at 0)
- **Categories with highly irregular amounts**: CI based on residual std will be very wide, making the forecast visually useless

---

## SECTION 13 — Agent Review

### Questions the agent answers best
- "How much did I spend on [category]?" → `calculate_total` tool → exact SQL sum
- "What anomalies are in my transactions?" → `get_anomalies` tool → list from DB
- "Forecast my [category] spending for 30 days" → `run_forecast` tool → EWMA result
- "Show me my recent [category] transactions" → `retrieve_transactions` → semantic search

### Questions that will fail
- "How much did I spend last month?" — no date-range tool; `calculate_total` only accepts category names
- "Compare my January vs February spending" — no comparison tool
- "What was my biggest expense?" — no sort/max tool; `retrieve_transactions` uses semantic similarity not amount ranking
- "How many times did I go to Chipotle?" — tools return text, not counts; LLM must parse
- Any multi-step financial question requiring >3 tool calls (blocked by `max_iterations=3`)
- Any follow-up question relying on previous answer (session memory lost between HTTP requests)

### Additional tools that would improve the agent
1. `calculate_total_by_date_range(category: str, start_date: str, end_date: str)` — most impactful missing tool
2. `get_monthly_summary(month: str)` — returns spending by category for a month
3. `get_top_merchants(category: str, limit: int = 5)` — top merchants by total spend
4. `get_transaction_count(category: str)` — count of transactions per category

### Prompt improvements that would help
1. Add explicit instruction: "When the user asks about time periods like 'last month', use today's date to calculate the date range." (Current prompt has no date context)
2. Add: "If you cannot answer with the available tools, say so clearly rather than using `retrieve_transactions` as a fallback for everything."
3. Add the current date to the system prompt so the agent can calculate relative dates

---

## SECTION 14 — Production Readiness

### Critical (must fix before any public deployment)
- [ ] Path traversal vulnerability in `/ingest` filename handling
- [ ] No authentication on any endpoint
- [ ] Session memory broken (each chat turn is stateless)
- [ ] `requirements.txt` incomplete — fresh install fails
- [ ] `_get_components()` per-request model reload makes API unusably slow (~4s overhead)
- [ ] Streamlit dashboard does not exist

### Important (should fix before real users)
- [ ] Anomaly score normalization produces near-zero values (scores meaningless)
- [ ] Categorizer trained only on synthetic merchants (~35% accuracy on real data)
- [ ] No rate limiting on any endpoint
- [ ] No pagination on `/transactions` (returns all rows)
- [ ] `configure_logging()` never called — `LOG_LEVEL` setting has no effect
- [ ] PyTorch DLL issue prevents full test suite from running on Windows
- [ ] Integration tests completely absent
- [ ] CORS wildcard (`allow_origins=["*"]`)
- [ ] No Docker/deployment configuration
- [ ] Model name hardcoded (`llama-3.1-8b-instant`) — no config option

### Nice-to-have (polish and robustness)
- [ ] `DELETE /transactions/{id}` endpoint
- [ ] `GET /transactions?needs_review=true` filter
- [ ] `AnomalyDetector.update_anomaly_scores()` public method (remove `_get_connection()` usage)
- [ ] `FieldMapper` utility class to clean up `PDFParser`→`CSVParser` private method coupling
- [ ] `PROPHET_*` settings removed or wired up
- [ ] README updated to reflect current status
- [ ] `docs/` directory populated with architecture diagram
- [ ] `data/raw/roundtrip_test.csv` documented or committed to fixtures

---

## SECTION 15 — Development Priorities: Next 20 Tasks

| # | Task | Effort | Impact | Files |
|---|---|---|---|---|
| 1 | Fix `requirements.txt` — add all missing packages | 30min | Critical: enables fresh install | `requirements.txt` |
| 2 | FastAPI `lifespan` startup — replace `_get_components()` | 2h | Critical: fixes latency + session memory | `src/api/app.py` |
| 3 | Fix upload filename path traversal | 30min | Critical: security | `src/api/app.py` |
| 4 | Write `tests/integration/test_api.py` (11 test cases) | 3h | High: API contract confidence | `tests/integration/test_api.py` |
| 5 | Write `tests/integration/test_ingestion_pipeline.py` | 2h | High: end-to-end confidence | `tests/integration/test_ingestion_pipeline.py` |
| 6 | Create `src/dashboard/app.py` — Overview + Anomalies tabs | 4h | High: visible product | `src/dashboard/app.py`, `src/dashboard/__init__.py` |
| 7 | Add Forecast + Chat tabs to dashboard | 4h | High: completes product | `src/dashboard/app.py` |
| 8 | Fix anomaly score normalization (min-max per batch) | 1h | High: scores become meaningful | `src/anomaly/anomaly_detector.py`, `tests/unit/test_anomaly_detector.py` |
| 9 | Fix `test_predict_batch_handles_individual_errors_without_aborting` | 30min | Medium: test correctness | `tests/unit/test_categorizer.py` |
| 10 | Wire `configure_logging()` in `lifespan` startup | 15min | Medium: LOG_LEVEL works | `src/api/app.py` |
| 11 | Expand `SyntheticGenerator.MERCHANTS` with real-world names | 1h | Medium: categorizer real-world accuracy | `src/ingestion/synthetic_generator.py` |
| 12 | Retrain categorizer after merchant expansion | 15min | Medium: model update | `scripts/train_categorizer.py` |
| 13 | Add `calculate_total_by_date_range` agent tool | 2h | Medium: answers "last month" queries | `src/agent/tools.py`, `tests/unit/test_agent.py` |
| 14 | Add indexes to SQLite schema (category, date, is_anomaly) | 30min | Medium: scalability | `src/ingestion/transaction_store.py` |
| 15 | Add `TransactionStore.update_anomaly_scores()` public method | 1h | Low-Medium: fixes encapsulation | `src/ingestion/transaction_store.py`, `src/anomaly/anomaly_detector.py` |
| 16 | Add `GET /transactions?needs_review=true` filter | 1h | Medium: product feature | `src/api/app.py`, `src/ingestion/transaction_store.py` |
| 17 | Add pagination to `GET /transactions` | 1h | Medium: scalability | `src/api/app.py`, `src/api/models.py` |
| 18 | Remove unused `PROPHET_*` settings from config | 15min | Low: clarity | `src/config.py`, `.env.example` |
| 19 | Update `README.md` — mark Week 3 complete, add setup guide | 30min | Low: documentation | `README.md` |
| 20 | Commit all handover documents | 5min | Low: preserves knowledge | `PROJECT_STATUS.md`, `HANDOVER_DEEP_DIVE.md`, `DEVELOPER_MEMORY_DUMP.md`, `FINAL_PROJECT_AUDIT.md` |


---

## SECTION 16 — Risk Assessment

| Change | Regression Risk | Tests to Run After |
|---|---|---|
| Modify `src/domain.py` Transaction fields | **CRITICAL** — breaks every module | All unit tests + `pytest tests/unit/ --ignore=tests/unit/test_vector_store.py` |
| Modify `src/ingestion/csv_parser.py` COLUMN_ALIASES | High — alias mapping tests will catch it | `pytest tests/unit/test_csv_parser.py -v` |
| Modify `src/ingestion/csv_parser.py` private method names | High — silently breaks PDFParser | `pytest tests/unit/test_pdf_parser.py -v` |
| Modify `src/api/app.py` (add lifespan) | High — all endpoints change initialization | `pytest tests/unit/test_config.py -v` + full integration test suite |
| Modify `src/api/models.py` field names | Medium — changes JSON response contract | `pytest tests/unit/test_domain.py -v` + integration tests |
| Modify `src/agent/agent.py` FINANCE_KEYWORDS | Medium — scope guard behavior changes | `pytest tests/unit/test_agent.py -v` |
| Modify `src/agent/agent.py` OUT_OF_SCOPE_RESPONSE | Medium — exact string test will fail | `pytest tests/unit/test_agent.py::test_out_of_scope_question_returns_canned_response` |
| Modify `src/api/vector_store.py` `_transaction_text()` | High — invalidates existing `.npy` file | Delete `data/processed/vector_store/` and re-ingest all data |
| Modify `src/ingestion/transaction_store.py` schema | High — existing DB incompatible | Manual schema migration + all transaction_store tests |
| Modify `src/forecasting/forecaster.py` MIN_HISTORY_DAYS | Medium — forecaster tests reference constant by name | `pytest tests/unit/test_forecaster.py -v` |
| Modify `src/anomaly/anomaly_detector.py` MIN_TRANSACTIONS | Medium — test references constant by name | `pytest tests/unit/test_anomaly_detector.py -v` |
| Retrain `categorizer.joblib` | Low-Medium — `test_high_confidence_prediction_does_not_set_needs_review` may fail | `pytest tests/unit/test_categorizer.py::test_high_confidence_prediction_does_not_set_needs_review` |

---

## SECTION 17 — Repository Audit

### Outdated Documentation
- `README.md` — says "Status: In active development (Day 1 of 30)" and "Week 3–4: RAG + agentic layer" as not done. Week 3 is complete. Needs updating.
- `README.md` — tech stack lists "Prophet" which was removed. Remove from stack listing.
- `.env.example` line 1 — has `?` encoding artifact: `# FinSight AI ? Environment Configuration`
- Design doc mentions ChromaDB which was replaced. Not critical since design.md is a spec artifact.

### Outdated Comments
- `src/forecasting/forecaster.py` docstring says "Pure numpy — no external ML dependencies" — accurate but could mention Prophet was considered.
- `src/agent/agent.py` class docstring mentions "a known Groq/LangChain compatibility issue" — accurate, explains the stateless design.

### Unused Files
- `data/raw/roundtrip_test.csv` — development artifact, no test uses it. Can be deleted.
- `.kombai/canvas/` and `.kombai/design-systems/` directories — empty, from a UI design tool. Can be deleted.
- `notebooks/` — empty directory. Can be deleted or populated.
- `docs/` — only `.gitkeep`. Can populate with architecture diagram or delete.

### Unused Dependencies in `requirements.txt`
- `faker==24.0.0` — `SyntheticGenerator` uses `random`, not Faker. Faker may be entirely unused. (Inference: no `from faker import Faker` in any source file)
- `jupyter==1.0.0` — notebooks dir is empty. Dev-only dependency.
- `matplotlib==3.8.3`, `seaborn==0.13.2` — not used in source code yet (dashboard not built). Keep for dashboard.

### Unused Packages in venv (not in `requirements.txt`, likely unused)
- `chromadb==1.5.9` — installed, never imported
- `pydantic-settings==2.14.2` — installed, never imported
- `prophet` — removed from requirements, likely uninstalled but `cmdstanpy` and `stanio` remain

### Duplicate Logic
- Date parsing: `CSVParser._parse_date()` is called by both `CSVParser` and `PDFParser` (via `self._field_mapper._parse_date()`). Logic isn't duplicated — reused correctly — but the architecture is awkward.
- Amount parsing: Same as above for `_parse_amount()`.
- `VectorStore._load()` exception handler silently swallows the error — this is a copy of a common "be resilient on startup" pattern but should at least log.

### Dead Code
- `src/api/__init__.py` re-exports `TransactionDTO`, `ForecastDTO`, etc. — these re-exports are never used by any importer. `app.py` imports from `api.models` directly, not from `api`.
- `from typing import Optional` in `src/agent/tools.py` — `Optional` is imported but never used in the file.

### TODO/FIXME Comments
None found in any source file (verified by searching in HANDOVER_DEEP_DIVE.md Section 11).

---

## SECTION 18 — Final Recommendations

Based entirely on the current codebase, in priority order:

**1. Fix `src/api/app.py` first — everything else follows from it.**
The `_get_components()` anti-pattern is the root cause of: slow API, broken session memory,
untestable endpoints. Replace with FastAPI `lifespan`. This single change unblocks
integration testing, fixes the chat product, and makes the API production-viable.
_Evidence: `app.py` `_get_components()` function; session memory in `agent.py` that requires
a singleton agent instance._

**2. Write integration tests before adding features.**
`tests/integration/` is empty. Any refactoring of `app.py` (needed for the lifespan fix)
has zero test coverage at the HTTP contract level. Write the tests while the behavior is
known-correct, then refactor safely.
_Evidence: `tests/integration/__init__.py` only; `tasks.md` Task 16 specified these tests._

**3. Build the Streamlit dashboard in one focused session.**
The backend is stable enough to build against. All required API endpoints exist and return
correct data. Building Overview → Anomalies → Forecast → Chat tabs in order gives
incremental verification. The chat tab has one known limitation (stateless sessions) that
will be fixed by recommendation #1.
_Evidence: `tasks.md` Task 17; `requirements.md` Requirement 11._

**4. Fix the anomaly score normalization before the dashboard launch.**
Any visualization of `anomaly_score` with current values (max=0.068) will look broken.
This is a one-line change in `anomaly_detector.py` that requires re-running anomaly
detection on existing data.
_Evidence: Live DB query showing all scores near zero (HANDOVER_DEEP_DIVE.md Section 1)._

**5. Expand `SyntheticGenerator.MERCHANTS` before demoing with real data.**
The categorizer's poor performance on real merchants is the most visible quality issue.
Adding Starbucks, McDonald's, DoorDash, Uber Eats, Apple Pay, Google Pay, Zomato, Swiggy
to the merchant vocabulary and retraining would immediately improve the demo experience.
_Evidence: Live categorizer test showing Starbucks→Transport (0.352), DoorDash→Dining (0.301)._

**6. Do NOT migrate the VectorStore to ChromaDB at this stage.**
The NumPy implementation handles 200 transactions in 0.29MB. It will handle 3,000 in 4.4MB.
ChromaDB migration adds risk with no user-visible benefit until the transaction count
grows significantly. Keep NumPy; add the O(1) dict lookup fix (30 minutes) instead.
_Evidence: `vector_store.npy` shape (200, 384); `vector_store.py` `_find_index()` implementation._

**7. Do NOT upgrade LangChain from 0.1.20.**
LangChain 0.2.x and 0.3.x have breaking changes to the agent API. The current
`create_tool_calling_agent` + `AgentExecutor` pattern would need significant rewriting.
The Groq compatibility workaround in `agent.py` is specific to 0.1.x behavior. Upgrading
risks breaking working functionality for no user-visible benefit.
_Evidence: `requirements.txt` `langchain==0.1.20`; `agent.py` stateless design comment._

**8. Commit all handover documents to git.**
`PROJECT_STATUS.md`, `HANDOVER_DEEP_DIVE.md`, `DEVELOPER_MEMORY_DUMP.md`, and
`FINAL_PROJECT_AUDIT.md` are currently untracked. They represent significant knowledge
transfer value and should be version-controlled.
_Evidence: `git status` showing only `PROJECT_STATUS.md` as untracked._

---

*End of FINAL_PROJECT_AUDIT.md*
