# SOURCE_CODE_INSPECTION.md
# FinSight AI — Final Source Code Inspection
# Every answer based on direct reading of source files.
# Inferences explicitly labeled. Date: 2026-07-05

---

## SECTION 1 — Hidden Invariants

Every invariant the code silently depends on. If any of these breaks, the project
fails — sometimes silently.

**INV-01: `Transaction.id` is always None until after SQLite insert**
`domain.py`: `id: Optional[int] = None`. Parsers create transactions with `id=None`.
`VectorStore.index()` calls `self._find_index(transaction.id)` where `transaction.id`
may be `None`. `_find_index(None)` iterates all metadata checking `meta.get("transaction_id") == None` — this will match any metadata entry that has `transaction_id: null`. If a corrupt metadata entry has null transaction_id, indexing logic breaks silently.
The invariant: **`VectorStore.index()` must only be called on transactions that have
been inserted into SQLite (so `id` is an integer, not None).** `app.py` enforces this
with `if txn.id is not None: vector_store.index(txn)` — but only in the ingest flow.

**INV-02: SQLite date strings are always ISO 8601 format (YYYY-MM-DD)**
`transaction_store.py` stores `txn.date.isoformat()` which always produces `YYYY-MM-DD`.
`query_by_date_range` uses `WHERE date >= ?` as a TEXT comparison. This works only
because ISO 8601 dates sort lexicographically correctly. If any date is stored in a
non-ISO format (e.g., `03/15/2024`), it will not sort or filter correctly.

**INV-03: `Transaction.category` strings match exactly what the LabelEncoder was trained on**
`categorizer.py`: `self._label_encoder.inverse_transform([predicted_index])` returns the
category string exactly as it appeared in training data (from `SyntheticGenerator`). If
any category string in training data has a trailing space, different capitalization, or
Unicode variation, the returned string won't match `CANONICAL_CATEGORIES` entries.
Current training data uses exact strings like `"Groceries"`, `"Dining"` — no spaces.

**INV-04: The `.npy` file and `metadata.json` have the same number of entries**
`vector_store.py` assumes `len(self._embeddings) == len(self._metadata)`. The `index()`,
`delete()`, and `search()` methods all rely on this alignment — `top_indices` from
`np.argsort` are used as direct indices into both `_metadata` and `_embeddings`. If a
write is interrupted between `np.save()` and `json.dump()` in `_save()`, the two files
diverge silently. `_load()` does NOT validate this alignment.

**INV-05: `VectorStore._transaction_text()` format must never change while embeddings exist on disk**
`f"{transaction.merchant} {transaction.category} {transaction.amount} {transaction.date}"`.
Old embeddings are stored with this exact text format. New embeddings use the same format.
If the format changes, old and new embeddings are in incompatible semantic spaces. Cosine
similarity between an old embedding (format A) and a new query (format B) is meaningless.

**INV-06: `LabelEncoder` classes order is deterministic from training data order**
`categorizer.py`: `self._label_encoder.fit_transform(labels)` fits the encoder with
classes sorted alphabetically by sklearn. The saved `categorizer.joblib` contains the
fitted encoder. If the same model file is loaded on a different sklearn version where
`LabelEncoder.fit_transform` produces a different class ordering, all predictions are
wrong. Currently sklearn 1.4.0 is pinned, so this is stable.

**INV-07: `TransactionStore.get_all()` always returns newest-first ordering**
`transaction_store.py`: `ORDER BY date DESC`. Several callers depend on this ordering:
- `AnomalyDetector.fit_and_score()` iterates transactions in this order and zips with scores
- `anomaly_detector.get_anomalies()` filters `all_txns` and re-sorts by score (safe)
- `Forecaster._get_daily_totals()` uses `query_by_category()` which also orders by date DESC,
  then re-sorts by `ds` ascending for EWMA. This means a `sort_values("ds")` call in the
  forecaster effectively reverses the order — safe but redundant.

**INV-08: Anomaly score update order matches transaction list order**
`anomaly_detector.py`:
```python
for txn, score, prediction in zip(transactions, normalized_scores, predictions):
```
`transactions = store.get_all()` → `feature_matrix = np.column_stack(...)` built from
`transactions` → `model.fit(feature_matrix)` → `raw_scores = model.decision_function(feature_matrix)`.
The score at index `i` corresponds to `transactions[i]`. The zip must use the same list
in the same order. If any code path reorders `transactions` before the zip, scores are
assigned to wrong transactions. Currently safe because the list is never reordered between
`get_all()` and the zip.

**INV-09: `SyntheticGenerator.MERCHANTS` and `SyntheticGenerator.CATEGORIES` keys are identical**
`generate()` does `category = rng.choice(categories)` then `self.MERCHANTS[category]`.
If `MERCHANTS` has a key not in `CATEGORIES`, the category is never selected (only
`CATEGORIES.keys()` is used). If `CATEGORIES` has a key not in `MERCHANTS`, `self.MERCHANTS[category]`
raises `KeyError` at runtime. Current state: both dicts have exactly 8 identical keys.

**INV-10: `PrettyPrinter.CANONICAL_FIELDS` matches exactly what `CSVParser` expects**
`CANONICAL_FIELDS = ("date", "merchant", "amount", "category")`. `CSVParser.parse()`
reads these exact field names from headers. The round-trip test verifies this. If
`CANONICAL_FIELDS` tuple order changes, `csv.writer.writerow(self.CANONICAL_FIELDS)`
writes headers in a new order — but CSVParser reads columns by name (not position via
`DictReader`), so parsing still works. However, the spec says canonical header must be
`date,merchant,amount,category` in that exact order. The tuple order is load-bearing
for spec compliance, not for CSVParser correctness.

**INV-11: `date_span_days` in `SyntheticGenerator.generate()` can be zero**
If `start_date == end_date`, `date_span_days = 0`. Then `rng.randint(0, 0)` always
returns 0. All transactions get `start_date`. This is technically valid (not an error)
but produces an unrealistic dataset where every transaction is on the same day. No
validation prevents this.

**INV-12: The `half = max(n // 2, 2)` in `_ewma_with_trend()` requires history ≥ 2 points**
`forecaster.py`: `half = max(n // 2, 2)` then `np.polyfit(x, y[-half:], 1)`. `polyfit`
with degree 1 requires at least 2 points. `max(n // 2, 2)` ensures `half >= 2` always.
But `y[-2:]` from a dataset with `n=14` gives the last 2 points. `polyfit` on 2 points
always produces a perfect linear fit (not a trend estimate). This means the "trend"
for minimum-history forecasts is just the slope between the last two observations —
which could be very noisy. Not a code bug but a statistical limitation.

**INV-13: `FinancialAgent.__init__()` reads `LLM_API_KEY` from environment at construction time**
`agent.py`: `api_key = os.getenv("LLM_API_KEY")`. If the environment variable is not set
at construction time, `api_key = None`. `ChatGroq(api_key=None, ...)` does not immediately
raise — it defers the error until the first API call. This means `FinancialAgent()` can
be constructed without a valid API key, and the error only surfaces during `chat()`.

**INV-14: `_get_components()` uses relative path for `categorizer.joblib`**
`Path("data/processed/categorizer.joblib")` is relative to the current working directory,
not `__file__`. If the process is started from any directory other than the project root,
the model is silently not loaded (`if model_path.exists()` returns False). No error is
raised — categorization is just skipped.

**INV-15: `load_dotenv(override=False)` is called at module import time**
`config.py` line 13: `load_dotenv(override=False)`. This runs when `config.py` is imported.
If `.env` does not exist at import time, `load_dotenv` silently does nothing. If tests
later set environment variables with `patch.dict(os.environ, ...)` AFTER this line runs,
the patched values work correctly because `override=False` means system env takes precedence.
But if any test relies on `.env` file values specifically, the timing matters.


---

## SECTION 2 — Refactoring Danger Map

Ten most dangerous files to modify, ranked by blast radius.

### #1 — `src/domain.py` ★★★★★ MOST DANGEROUS
**Why dangerous:** Every single module imports `Transaction` from this file. There are
zero indirections — no DTO conversion layer, no mapping functions. The dataclass fields
are directly read in `transaction_store.py` (`txn.date.isoformat()`, `txn.merchant`, etc.),
`anomaly_detector.py` (`txn.amount`, `txn.category`, `txn.id`, `txn.is_anomaly`),
`vector_store.py` (`transaction.merchant`, `transaction.category`, etc.), `categorizer.py`
(`transaction.merchant`, `transaction.category`, `transaction.needs_review`), and all
test files.

**What breaks:** Adding a required field breaks every constructor call site.
Renaming a field breaks every attribute access. Changing `id: Optional[int]` to required
breaks parsers (they create transactions without IDs). Changing `is_anomaly: bool = False`
to `Optional[bool]` breaks `bool(row["is_anomaly"])` in `_row_to_transaction`.

**Tests to run after:** `pytest tests/unit/ --ignore=tests/unit/test_vector_store.py -v`
plus manual verify of `data/processed/finsight.db` schema compatibility.

---

### #2 — `src/api/app.py` ★★★★★
**Why dangerous:** All endpoints are defined here. `_get_components()` is called by every
endpoint. Any change to component initialization affects all endpoints simultaneously.
The ingest pipeline (parse → categorize → store → index → anomaly) is entirely in the
`ingest()` function — if any step breaks, the whole pipeline breaks.

**What breaks:** Changing `_get_components()` can break all 6 endpoints. Moving component
creation to `lifespan` changes the initialization order and test setup. Changing the ingest
pipeline order (e.g., categorizing before setting `source_file`) changes behavior.

**Tests to run after:** Integration test suite (when written). Currently: start server
manually and test `POST /ingest` with `test_statement.csv`.

---

### #3 — `src/ingestion/csv_parser.py` ★★★★
**Why dangerous:** Both `PDFParser` and the test suite depend on its private methods.
`PDFParser.__init__()` stores `self._field_mapper = CSVParser()` and calls
`._canonical_field_name()`, `._parse_date()`, `._parse_amount()`. If any private method
is renamed or refactored, `PDFParser` breaks with `AttributeError` at runtime — no
static analysis catches this. Also: `COLUMN_ALIASES` keys are tested implicitly
by `test_csv_parser.py` tests that parse alias-header CSVs.

**What breaks:** Renaming private methods → PDFParser runtime failure. Changing
`COLUMN_ALIASES` values → alias mapping tests fail. Changing date format list order →
different formats parsed for ambiguous dates.

**Tests to run after:** `pytest tests/unit/test_csv_parser.py tests/unit/test_pdf_parser.py -v`

---

### #4 — `src/api/vector_store.py` ★★★★
**Why dangerous:** `_transaction_text()` format is load-bearing for all existing stored
embeddings. `_find_index()` O(n) scan is used by both `index()` and `delete()`. The
`_save()`/`_load()` cycle is the only persistence. Changing any of these invalidates
the live `vector_store.npy` and `vector_store_metadata.json` files.

**What breaks:** Changing text format → existing embeddings incompatible with new queries.
Changing `COLLECTION_FILE` or `METADATA_FILE` constant names → can't find existing files.
Changing `np.save` to a different format → `np.load` fails on existing files.

**Tests to run after:** `pytest tests/unit/test_vector_store.py` (if DLL issue resolved)
+ delete and re-ingest all data to regenerate embeddings.

---

### #5 — `src/ingestion/transaction_store.py` ★★★★
**Why dangerous:** SQLite schema is defined here. The `_row_to_transaction()` method must
stay in sync with both the schema and the `Transaction` dataclass. `_get_connection()` is
called by `AnomalyDetector` directly — changing connection management breaks anomaly detection.
The `uq_transaction` unique index defines what counts as a duplicate — changing it changes
data deduplication behavior globally.

**What breaks:** Adding a column without `DEFAULT` → existing rows fail to query.
Removing `row_factory = sqlite3.Row` → `row["id"]` becomes `row[0]` everywhere.
Changing `_get_connection` signature → `AnomalyDetector.fit_and_score()` breaks.

**Tests to run after:** `pytest tests/unit/test_transaction_store.py tests/unit/test_anomaly_detector.py -v`

---

### #6 — `src/agent/agent.py` ★★★
**Why dangerous:** `OUT_OF_SCOPE_RESPONSE` string is tested by exact equality.
`FINANCE_KEYWORDS` set determines what questions reach the LLM — adding/removing keywords
changes user-facing behavior. `_clean_response()` marker list was built from observed
LLM behavior — changing markers affects response quality.

**What breaks:** Changing `OUT_OF_SCOPE_RESPONSE` string → `test_out_of_scope_question_returns_canned_response` fails.
Removing any keyword from `FINANCE_KEYWORDS` → questions about that topic get the out-of-scope response.
Changing `max_iterations=3` → agent may never terminate or produce incomplete answers.

**Tests to run after:** `pytest tests/unit/test_agent.py -v`

---

### #7 — `src/categorization/categorizer.py` ★★★
**Why dangerous:** `CONFIDENCE_THRESHOLD = 0.60` and `CANONICAL_CATEGORIES` are tested
by value. `predict()` mutates the input `Transaction` object in-place. `train()` uses
`random_state=42` — changing this produces a different model even on identical data.

**What breaks:** Changing `CONFIDENCE_THRESHOLD` → `test_low_confidence_prediction_sets_other_and_needs_review`
needs threshold-specific mock data. Removing `class_weight='balanced'` → imbalanced
category predictions. Changing `_is_trained` to a property → `app.py`'s `if categorizer._is_trained` fails.

**Tests to run after:** `pytest tests/unit/test_categorizer.py -v` + retrain model + manual verification.

---

### #8 — `src/anomaly/anomaly_detector.py` ★★★
**Why dangerous:** `MIN_TRANSACTIONS = 10` appears verbatim in the error message tested
by `test_anomaly_detector.py`. The score normalization formula is wrong but produces
values that tests currently pass on. The `store._get_connection()` access is a private
interface dependency.

**What breaks:** Changing `MIN_TRANSACTIONS` value → test assertion `str(MIN_TRANSACTIONS) in str(exc_info.value)` fails.
Fixing score normalization → tests that check scores are `<= 1.0` still pass, but
the live database values change dramatically (requires re-running anomaly detection).
Removing the `with store._get_connection() as conn:` pattern → anomaly updates don't persist.

**Tests to run after:** `pytest tests/unit/test_anomaly_detector.py -v` + verify live DB scores.

---

### #9 — `conftest.py` ★★★★
**Why dangerous:** This 2-line file is the foundation of all test imports. It inserts
`src/` into `sys.path` at index 0. Without it, every test file fails with `ModuleNotFoundError`
because they all use absolute imports like `from domain import Transaction`.

**What breaks:** Deleting it → all 86 tests fail immediately. Moving it to a subdirectory →
tests in parent directories lose the path. Changing the path inserted → any module not in
`src/` is not findable.

**Tests to run after:** All tests — `pytest tests/unit/ --ignore=tests/unit/test_vector_store.py`

---

### #10 — `src/forecasting/forecaster.py` ★★★
**Why dangerous:** `MIN_HISTORY_DAYS = 14` appears in the error message tested by exact
substring: `assert str(MIN_HISTORY_DAYS) in str(exc_info.value)`. The EWMA formula and
confidence interval computation are tested indirectly through the output constraints
(`yhat >= 0`, `yhat_lower <= yhat <= yhat_upper`).

**What breaks:** Changing `MIN_HISTORY_DAYS` → test assertion fails. Changing CI calculation
(e.g., using 1.645 for 90% CI) → `yhat_lower <= yhat` might fail if margin becomes large.
Changing `half = max(n // 2, 2)` → `polyfit` might receive < 2 points on edge cases.

**Tests to run after:** `pytest tests/unit/test_forecaster.py -v`


---

## SECTION 3 — Implicit Contracts

Every implicit contract between modules — never enforced by code, always relied upon.

**IC-01: `CSVParser.parse()` returns transactions with `source_file=""`**
No code enforces that `source_file` is empty from parsers. `app.py` sets it after parsing
(`for txn in transactions: txn.source_file = filename`). Any caller that forgets this step
will insert transactions with `source_file=""`, making the deduplication key
`(date, merchant, amount, "")`. Two imports of different files with the same transaction
data would then incorrectly deduplicate as if they came from the same source.
File: `src/api/app.py` line ~67, `src/ingestion/csv_parser.py`

**IC-02: `Categorizer.predict()` receives a Transaction whose `merchant` is non-empty**
`_pipeline.predict_proba([transaction.merchant])` — if `merchant` is `""`, TF-IDF produces
a zero vector. LogisticRegression still returns a probability distribution (uniform or
close to it). The `except Exception` handler catches this and sets `category="Other"`.
But the implicit contract is: only call `predict()` on transactions with meaningful merchant names.

**IC-03: `AnomalyDetector.fit_and_score()` assumes all transactions have non-empty `category`**
`label_encoder.fit_transform(categories)` where `categories = [txn.category for txn in transactions]`.
If any transaction has `category=""` (uncategorized), the label encoder creates a label for `""`.
The feature matrix `[amount, encoded_category]` still works — but `""` gets its own
encoded integer, and anomaly detection treats "uncategorized" as a distinct category.
Transactions categorized as `"Other"` and those as `""` will be treated differently.

**IC-04: `VectorStore.search()` returns Transaction objects with ONLY these fields populated: `id, date, merchant, amount, category`**
The returned `Transaction` objects are constructed from metadata only. They have:
- `is_anomaly = False` (default)
- `anomaly_score = None` (default)
- `needs_review = False` (default)
- `source_file = ""` (default)

Any caller that uses these objects and accesses `is_anomaly` or `anomaly_score` will get
wrong values. The agent tool `retrieve_transactions` converts these to strings — safe.
But if `VectorStore.search()` results were ever used for anomaly display, they'd show wrong flags.

**IC-05: `Forecaster.forecast_category()` expects `store.query_by_category()` to return transactions sorted by date**
`_get_daily_totals()` calls `store.query_by_category(category)` which returns `ORDER BY date DESC`.
Then `df.sort_values("ds")` re-sorts ascending. So the sort order from SQLite is immediately
overridden. The implicit contract (transactions are date-sorted) is technically unneeded
here because of the re-sort. But if `query_by_category` ever returned an unordered result,
the behavior would still be correct because of `sort_values`.

**IC-06: `app.py` calls `store.get_all()` between `store.insert()` and `vector_store.index()` to get IDs**
```python
inserted, skipped = store.insert(transactions)
all_txns = store.get_all()
for txn in all_txns:
    if txn.id is not None:
        vector_store.index(txn)
```
The implicit contract: after `insert()`, `get_all()` returns all transactions including
newly inserted ones WITH their auto-assigned IDs. This is correct for SQLite. But note:
`get_all()` returns ALL transactions, not just newly inserted ones. So all existing
transactions are re-indexed on every ingest.

**IC-07: `FinancialAgent` tools capture `store` from closure — the store must be connected to the live database**
`create_tools(store, vector_store, forecaster, anomaly_detector)` creates closures.
The `store` captured is whatever `_get_components()` returned. If the SQLite file is deleted
or moved after the store is created, subsequent tool calls fail with SQLite errors.
The agent tools have no reconnection logic.

**IC-08: `TransactionStore.insert()` is called with transactions in the order they should be stored**
Insert order determines auto-increment IDs. While no code relies on specific ID values,
the fact that IDs are sequential in insertion order means the first-seen transactions
get lower IDs. If insert order ever matters (e.g., for display), this implicit ordering
is relied upon.

**IC-09: `Categorizer.predict_batch()` preserves the input list order**
`return [self.predict(txn) for txn in transactions]` — list comprehension preserves order.
`app.py` does `transactions = categorizer.predict_batch(transactions)` then
`inserted, skipped = store.insert(transactions)`. The insert order matches the parse order.
No code explicitly verifies this, but any reordering in `predict_batch` would be invisible
until a deduplication or ordering issue surfaced.

**IC-10: `get_anomalies()` in tools.py calls `anomaly_detector.get_anomalies(store)` which calls `store.get_all()`**
The tool's limit parameter `[:limit]` slices AFTER loading all anomalies into memory.
With 5000 anomalous transactions (100k total × 5% contamination = 5000), all 5000 are
loaded to return 10. The implicit contract: `limit` is a post-load slice, not a SQL LIMIT.


---

## SECTION 4 — Missing Abstractions

**MISS-01: `FieldMapper` utility class**
`csv_parser.py` defines `_canonical_field_name()`, `_parse_date()`, `_parse_amount()`.
`pdf_parser.py` instantiates `CSVParser` just to call these three methods:
`self._field_mapper = CSVParser()`. Both parsers share these as a cross-cutting concern.
**Wanted:** `class FieldMapper` in `src/ingestion/field_mapper.py` with public methods
`canonical_field_name()`, `parse_date()`, `parse_amount()`. Both parsers receive it
via constructor injection or inherit from it. This eliminates the private-method access.

**MISS-02: `ParserBase` abstract base class**
`CSVParser.parse()` and `PDFParser.parse()` both return `tuple[list[Transaction], ParseSummary]`.
Both produce the same summary structure. Neither implements a common interface.
`app.py` distinguishes them by string: `if filename.endswith(".csv"): parser = CSVParser() else: parser = PDFParser()`.
**Wanted:** `from abc import ABC, abstractmethod` → `class StatementParser(ABC):` with
`@abstractmethod def parse(self, file_path: Path) -> tuple[list[Transaction], ParseSummary]: ...`.
Then `app.py` can use a factory: `parser_map = {".csv": CSVParser, ".pdf": PDFParser}`.

**MISS-03: Service layer / orchestration class for the ingest pipeline**
`app.py`'s `ingest()` function does: parse → set source_file → categorize → insert → re-index
all → anomaly detect. This is business logic inside an HTTP handler. It's 35 lines of
pipeline orchestration in an async endpoint function.
**Wanted:** `class IngestionService` with `ingest(file_path, filename) -> IngestResult`.
This makes the pipeline testable without HTTP (just call `service.ingest(path, name)`).

**MISS-04: A `ScoringNormalizer` or at minimum a `normalize_scores()` function**
`anomaly_detector.py`: `normalized_scores = np.clip(-raw_scores, 0, 1).tolist()`.
This one-liner embeds a normalization strategy that needs to change. If normalization
is extracted to a function, it can be unit-tested in isolation and swapped without
touching `fit_and_score()`.
**Wanted:** `def normalize_anomaly_scores(raw_scores: np.ndarray) -> list[float]`.

**MISS-05: A response builder / DTO factory**
`app.py` has `_txn_to_dto(txn) -> TransactionDTO`. This is a good start — it's already
extracted. But `ForecastDTO` is built inline in `get_forecast()`:
```python
ForecastDTO(
    category=forecast.category,
    horizon_days=forecast.horizon_days,
    points=[ForecastPointDTO(...) for p in forecast.points],
)
```
This conversion is duplicated if another endpoint or the agent ever returns a `ForecastDTO`.
**Wanted:** `_forecast_to_dto(forecast: Forecast) -> ForecastDTO` alongside `_txn_to_dto`.

**MISS-06: A `SessionStore` abstraction for agent history**
`agent.py`: `self._session_history = {}  # session_id -> list of (question, answer)`.
This is an in-memory dict. When the lifespan fix is applied, one `FinancialAgent` instance
will persist. But if the server restarts, history is lost. The `SessionStore` abstraction
would allow swapping in a SQLite-backed implementation later without touching `FinancialAgent`.
**Wanted:** `class SessionStore(Protocol)` with `get(session_id) -> list`, `add(session_id, pair)`, `trim(session_id, max_size)`.


---

## SECTION 5 — Duplicate Logic

**DUP-01: Field extraction logic in `CSVParser.parse()` and `PDFParser.parse()`**
`csv_parser.py` extracts fields in this order: merchant → date → amount → category, with
skip+warn per missing/bad field. `pdf_parser.py` does the exact same field extraction and
validation in `parse()`, but using a `failing_fields` list pattern instead of early continue.
The validation logic (check empty, try parse, append to error list) is semantically
identical but structurally different. `FieldMapper` (see MISS-01) would centralize this.
Files: `csv_parser.py` lines 74–113, `pdf_parser.py` lines 54–100

**DUP-02: Date parsing in `CSVParser._parse_date()` and called identically via PDFParser**
Both parsers support the same 4 date formats: `%Y-%m-%d`, `%m/%d/%Y`, `%d/%m/%Y`, `%m-%d-%Y`.
This list lives in `_parse_date()` and is used by PDFParser via `self._field_mapper._parse_date()`.
Not truly duplicated code — reused via the `_field_mapper` reference. But the 4 formats
are a hidden constant that should be explicit: `SUPPORTED_DATE_FORMATS = [...]` at module
level in `csv_parser.py`.

**DUP-03: `AnomalyDetector` imported twice in `app.py`**
At module level: `from anomaly.anomaly_detector import AnomalyDetector`
Inside `chat()` endpoint: `from anomaly.anomaly_detector import AnomalyDetector`
This is literal duplicate import of the same class. The local import inside `chat()` is
completely unnecessary since it's already imported at module level.
File: `src/api/app.py` lines 9, 113

**DUP-04: `Forecaster` imported twice in `app.py`**
Same pattern as DUP-03:
Module level: `from forecasting.forecaster import Forecaster`
Inside `chat()`: `from forecasting.forecaster import Forecaster`
File: `src/api/app.py` lines 14, 114

**DUP-05: `load_dotenv()` called in two places**
`config.py` line 13: `load_dotenv(override=False)` — runs at module import.
`app.py` `chat()` endpoint line 110: `from dotenv import load_dotenv; load_dotenv()` — runs on every chat request.
The second call is redundant — the dotenv is already loaded. The second call uses default
`override=True` semantics (no argument), while config.py uses `override=False`. This means
the second call could OVERRIDE system environment variables with `.env` values for any
request that hits the `/chat` endpoint. This is a subtle behavioral difference.
Files: `src/config.py` line 13, `src/api/app.py` inside `chat()` function

**DUP-06: `FinancialAgent.chat()` session trim logic is written out twice conceptually**
The session history management in `chat()`:
```python
self._session_history[session_id].append((message, answer))
if len(self._session_history[session_id]) > 5:
    self._session_history[session_id] = self._session_history[session_id][-5:]
```
This 3-line block with an if/slice is a pattern that belongs in a helper method
`_update_session(session_id, message, answer)`. Not critical but reduces cognitive load.


---

## SECTION 6 — Exception Flow

### Swallowed exceptions

**SW-01: `VectorStore._load()` swallows ALL exceptions silently**
```python
except Exception as e:
    self._embeddings = None
    self._metadata = []
```
The exception `e` is captured but never used (no `logger.error`, no `logger.warning`, nothing).
If `vector_store.npy` is corrupted, zero-byte, or in the wrong format, the VectorStore
silently starts empty. The user sees empty search results without any error indication.
**Fix:** `logger.error(f"VectorStore failed to load from {self._persist_dir}: {e}")`

**SW-02: `PDFParser.parse()` swallows all `pdfplumber.open()` exceptions**
```python
except Exception:
    summary.file_errors.append(f"Invalid or corrupt PDF file: {file_path}")
    return [], summary
```
This catches `MemoryError`, `SystemExit`, `KeyboardInterrupt` (though Python 3 doesn't
catch those with bare `except Exception`), and any pdfplumber internal errors. The error
message is generic. The original exception is completely discarded.
**Fix:** Log the original exception. Use `except Exception as e: logger.debug(str(e))`.

**SW-03: `Categorizer.predict()` swallows all exceptions**
```python
except Exception:
    transaction.category = "Other"
    transaction.needs_review = True
```
If `sklearn` raises a `MemoryError` (on a very large merchant string), or a `ValueError`
from internal pipeline state, or even a programming error (typo in code), this silently
sets the category to "Other". The caller has no way to know whether "Other" means
"low confidence" or "prediction crashed". The exception is discarded entirely.
**Fix:** At minimum, log the exception type and message at DEBUG level.

**SW-04: `app.py` ingest swallows VectorStore and AnomalyDetector failures**
```python
except Exception as e:
    logger.warning(f"Vector indexing failed for {txn.id}: {e}")
```
and
```python
except Exception as e:
    logger.warning(f"Anomaly detection failed: {e}")
```
These are logged as warnings, not errors. The HTTP response returns 200 with `ingested=N`
even when indexing or anomaly detection partially or completely failed. The caller has
no way to know that the VectorStore is now out of sync.

---

### Inconsistent exception handling

**INCON-01: `ValueError` vs `RuntimeError` for "not initialized" errors**
`Categorizer.predict()` raises `RuntimeError` when not trained.
`Forecaster.forecast_category()` raises `ValueError` for bad parameters.
Both are "caller did something wrong" errors. The choice of exception type is inconsistent:
`ValueError` is for invalid values, `RuntimeError` is for invalid state. Using `RuntimeError`
for "untrained model" is arguably more correct (it's a state error, not a value error).
But mixing them means callers must catch both.

**INCON-02: `CSVParser` returns errors vs `TransactionStore` raises errors**
`CSVParser.parse()` returns `([], summary_with_file_errors)` for bad files — never raises.
`TransactionStore.query_by_date_range()` raises `ValueError` for `start_date > end_date`.
There's no consistent policy: some errors are returned, some are raised.
`app.py` handles this by: catching `ValueError` from store queries with `try/except`,
and checking `summary.file_errors` after parsing.

**INCON-03: Generic `except Exception` in tool functions**
All four tools in `tools.py` have:
```python
except Exception as e:
    logger.error(f"<tool>_error: {e}")
    return f"Error ...: {str(e)}"
```
This returns error strings to the LLM, which then tells the user "there was an error."
This is functional but hides the cause. If `store.get_all()` raises an `OperationalError`
(SQLite issue), the LLM gets `"Error calculating total: no such table: transactions"` —
which it then paraphrases into an unhelpful chat response.

---

### Poor error messages

**ERR-01: `VectorStore._load()` — no message at all** (see SW-01)

**ERR-02: `PDFParser` — file path leaked in error messages**
`f"File not found: {file_path}"` includes the full server filesystem path.
This is returned in the HTTP response body. On a server, this leaks internal path structure.

**ERR-03: `Categorizer.load()` error message could be more specific**
`raise RuntimeError("Loaded model file is missing pipeline or label encoder.")`
This tells you WHAT is missing but not WHERE. Add the file path:
`f"Loaded model file at {path} is missing pipeline or label encoder."`


---

## SECTION 7 — Type Safety

### Missing type hints

**TY-01: `FinancialAgent.__init__()` parameters are untyped**
```python
def __init__(self, store, vector_store, forecaster, anomaly_detector):
```
No type hints. Should be:
```python
def __init__(
    self,
    store: TransactionStore,
    vector_store: VectorStore,
    forecaster: Forecaster,
    anomaly_detector: AnomalyDetector,
) -> None:
```
File: `src/agent/agent.py` line 46

**TY-02: `FinancialAgent.chat()` and helper methods lack return type hints**
```python
def _is_finance_question(self, message):    # missing -> bool
def _clean_response(self, text):            # missing -> str
def get_history(self, session_id):          # missing -> list[tuple[str, str]]
def chat(self, message, session_id):        # missing -> str, also params untyped
```
File: `src/agent/agent.py`

**TY-03: `_txn_to_dto(txn)` parameter is untyped**
```python
def _txn_to_dto(txn) -> TransactionDTO:
```
Should be `def _txn_to_dto(txn: Transaction) -> TransactionDTO:`.
File: `src/api/app.py`

**TY-04: `create_tools()` parameters and return type are untyped**
```python
def create_tools(store, vector_store, forecaster, anomaly_detector):
```
Should be fully typed. File: `src/agent/tools.py`

---

### Optional misuse

**TY-05: `TransactionDTO.id` is `Optional[int]` but used as `int` in VectorStore**
`models.py`: `id: Optional[int]`. But `vector_store.py` uses `transaction.id` as a key
in `_find_index(transaction.id)`. If `id` is `None` (which it is for newly parsed
transactions before insert), `_find_index(None)` searches for `None` in metadata — a
valid but meaningless search that returns -1, then inserts a new entry keyed to `None`.
If multiple None-id transactions are indexed, they all get separate entries that can
never be found again by ID (since `_find_index(None)` returns the first one).

**TY-06: `anomaly_score: Optional[float]` in Transaction leads to guarded sort**
`anomaly_detector.py`:
```python
return sorted(anomalies, key=lambda t: t.anomaly_score or 0.0, reverse=True)
```
The `or 0.0` guard handles `None`. But if `anomaly_score=0.0` legitimately, `0.0 or 0.0 == 0.0`
— safe. If `anomaly_score=None`, `None or 0.0 == 0.0` — treats missing scores as 0.
Correct behavior, but the type allows `None` for scored transactions, which is wrong.
A scored transaction should always have a float score. Using `Optional[float]` for an
"assigned score" field is imprecise.

---

### Typing inconsistencies

**TY-07: `Forecaster.forecast_all()` return type uses old-style Union**
```python
def forecast_all(...) -> dict[str, Union[Forecast, str]]:
```
Python 3.10+ allows `dict[str, Forecast | str]`. The file has `from __future__ import annotations`
which enables the `|` syntax even on Python 3.9. But `Union` is used from `typing`. Minor
inconsistency — the `from __future__ import annotations` at top of the file should allow
`Forecast | str` instead of `Union[Forecast, str]`.

**TY-08: `Settings` class has class-level type annotations that are never enforced**
```python
class Settings:
    SQLITE_DB_PATH: str
    CHROMA_PERSIST_DIR: str
    ...
```
These are class-level annotations without values — they don't create attributes. The
actual attributes are created in `_load()` via `self.SQLITE_DB_PATH = os.getenv(...)`.
`os.getenv()` returns `Optional[str]` (can return `None`). But the annotation says `str`.
After `_validate_required()` confirms the variable exists, `os.getenv()` is safe to
cast as `str` — but the type system doesn't know this. A mypy check would flag this.


---

## SECTION 8 — Resource Lifecycle

### SQLite connections

**Created:** On every `TransactionStore._get_connection()` call — a new `sqlite3.connect()` per call.
**Used:** Inside `with self._get_connection() as conn:` blocks. The `with` statement calls
`conn.__exit__()` which commits (on success) or rolls back (on exception) and closes the connection.
**Destroyed:** Automatically at end of `with` block. SQLite connections are not pooled.
**Owner:** `TransactionStore` — each method opens and closes its own connection.
**Possible leaks:** No connection leaks. Every connection is opened in a `with` block.
However, `AnomalyDetector` directly opens a connection via `with store._get_connection() as conn:`
— the connection is properly closed but bypasses `TransactionStore`'s public interface.

---

### SentenceTransformer model

**Created:** In `VectorStore.__init__()`: `self._model = SentenceTransformer(embedding_model_name)`.
Downloads ~22MB model from HuggingFace on first use. Cached in `~/.cache/huggingface/`.
**Current lifecycle (broken):** Created on every `VectorStore()` instantiation. `VectorStore()`
is called inside `_get_components()` which is called on every HTTP request. Model is loaded
fresh per request, then garbage-collected when `_get_components()` returns and the local
variable goes out of scope. Cost: ~2-4 seconds per request.
**Desired lifecycle (after lifespan fix):** Loaded once at startup, shared for entire server lifetime.
**Owner:** `VectorStore` instance owns the model.
**Possible leaks:** No memory leak per se — Python GC handles it. But the model is a large
object (~200MB+ for PyTorch weights) being created and destroyed repeatedly under current architecture.

---

### Temporary upload files

**Created:** `tmp_path = Path(f"data/raw/{filename}")` then `tmp_path.write_bytes(content)`.
**Destroyed:** `finally: tmp_path.unlink(missing_ok=True)`.
**Owner:** The `ingest()` endpoint. The `finally` block guarantees deletion.
**Possible leak:** If `tmp_path.write_bytes(content)` fails (disk full, permission error), the
`finally` block still runs `unlink(missing_ok=True)` — safe. But if the write succeeds and
a `PermissionError` occurs on `unlink` (file locked by antivirus scanner on Windows), a warning
is logged but the file remains. The code handles this: `except PermissionError: logger.warning(...)`.
So orphaned temp files can accumulate in `data/raw/` on Windows.

---

### VectorStore `.npy` and `.json` files

**Created:** First `index()` call creates both files via `_save()`.
**Updated:** On every `index()` and `delete()` call via `_save()`.
**Loaded:** In `VectorStore.__init__()` via `_load()`.
**Owner:** `VectorStore` instance. Files persist on disk between restarts.
**Possible corruption:** If the process is killed between `np.save(npy)` and `json.dump(metadata)`,
the two files are out of sync. No transaction-like atomicity between the two writes.

---

### LLM / Groq API connection

**Created:** `ChatGroq(api_key=..., model=..., temperature=0)` in `FinancialAgent.__init__()`.
**Current lifecycle:** Created on every `/chat` HTTP request (new `FinancialAgent` per request).
**Desired lifecycle:** Created once at `lifespan` startup.
**Owner:** `FinancialAgent`. The Groq client is an HTTP client — stateless, no persistent connection.
**Possible leaks:** No leaks. HTTP connections are managed by the underlying `httpx` library.

---

### Categorizer joblib model

**Created:** `categorizer.load(model_path)` in `_get_components()`.
**Current lifecycle:** Loaded on every request that calls `_get_components()`. The loaded
pipeline (TF-IDF matrix + LogReg weights, ~75KB) is GC'd after each request.
**Desired lifecycle:** Loaded once at startup.
**Owner:** `Categorizer` instance.
**Possible leaks:** None. Small object, properly GC'd.


---

## SECTION 9 — State Management

| State | Location | Why | Correct? |
|---|---|---|---|
| Parsed transactions (in-flight) | Python `list[Transaction]` in memory | Returned from parsers, passed to categorizer and store | Yes — transient, short-lived |
| All transactions | SQLite `data/processed/finsight.db` | Persistent cross-session storage | Yes |
| Transaction embeddings | `data/processed/vector_store/vector_store.npy` | Persistent vector index | Yes, but partial corruption risk |
| Embedding metadata | `data/processed/vector_store/vector_store_metadata.json` | Maps embedding rows to transaction IDs | Yes, paired with .npy |
| In-memory embedding matrix | `VectorStore._embeddings` (np.ndarray) | Loaded from .npy on init for search | Yes — RAM cache of disk state |
| In-memory metadata list | `VectorStore._metadata` (list[dict]) | Loaded from JSON on init | Yes — RAM cache of disk state |
| Trained categorizer | `data/processed/categorizer.joblib` | Persistent model artifact | Yes |
| Session chat history | `FinancialAgent._session_history` (dict) | Per-session conversation context | **No** — lost on request end due to per-request instantiation |
| Configuration singleton | `config._settings` (module-level global) | Prevent re-reading env vars | Yes — singleton pattern is correct |
| Anomaly flags + scores | SQLite `transactions.is_anomaly`, `transactions.anomaly_score` | Persistent anomaly state | Yes — but scores near-zero (normalization bug) |
| `needs_review` flag | SQLite `transactions.needs_review` | Persistent low-confidence flag | Yes — stored but never surfaced via API filter |
| EWMA smoothing factor | `Forecaster._alpha = 0.3` | Algorithm parameter | Yes — instance state, correct |
| IsolationForest contamination | `AnomalyDetector._contamination = 0.05` | Algorithm parameter | Yes — but hardcoded, no config option |
| LabelEncoder state (categorizer) | Inside `categorizer.joblib` | Maps encoded ints back to category strings | Yes — part of saved artifact |
| LabelEncoder state (anomaly) | Local variable in `fit_and_score()` | Maps category strings to ints for features | Yes — recreated per fit, not persisted (correct) |

---

## SECTION 10 — Long Methods

**LM-01: `CSVParser.parse()` — 55 lines**
Natural boundaries:
1. File reading and validation (lines 57–76) → `_read_file(path) -> str`
2. Header mapping (lines 78–83) → already implicitly separate via `_canonical_field_name`
3. Row parsing loop (lines 85–116) → `_parse_rows(reader, header_map) -> tuple[list[Transaction], ParseSummary]`
The loop itself has 6 distinct validation steps per row — this is acceptable for a parser.
The method is long but not complex. Splitting is optional.

**LM-02: `PDFParser.parse()` — 78 lines**
Natural boundaries:
1. File validation (lines 16–40) → `_validate_file(path) -> Optional[str]`
2. Page/table iteration (lines 42–112) → `_extract_from_pages(pdf, summary) -> list[Transaction]`
3. No-table detection (lines 113–116) → already at end of method
The nested `for page / for table / for row` structure creates 3 levels of nesting — hard
to read. Split at the "for row_index, raw_row" level into a `_parse_row()` method.

**LM-03: `VectorStore.search()` — 24 lines of math**
The cosine similarity computation (normalize matrix → normalize query → dot product →
argsort) is one logical unit. Could be extracted to `_cosine_search(query_embedding, k)`.
The result reconstruction (metadata → Transaction objects) is a separate concern.
Natural split: search → reconstruct.

**LM-04: `FinancialAgent.chat()` — 37 lines**
Natural boundaries:
1. Scope guard check (lines 107–108) → already extracted as `_is_finance_question()`
2. LLM invocation + output extraction (lines 110–128) → `_invoke_agent(message) -> str`
3. Session history update (lines 130–134) → `_update_session(session_id, message, answer)`
These three steps are conceptually distinct and each has its own error handling.

**LM-05: `Categorizer.train()` — 55 lines**
Natural boundaries:
1. Input validation (lines 48–57)
2. Stratified split logic (lines 59–70) → `_build_train_val_split(texts, labels)`
3. Pipeline construction and fitting (lines 72–92)
4. Validation scoring (lines 94–97)
The method does 4 distinct things. Splitting would make testing each concern easier.


---

## SECTION 11 — Code Smells

### Magic numbers

**MN-01: `10 * 1024 * 1024` in `pdf_parser.py` line 8**
`MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024` — this is documented as a constant. Fine.
But the error message says `f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES} bytes"`.
The human-readable error shows bytes (10485760) not megabytes. Should show "10 MB".

**MN-02: `1.96` in `forecaster.py`**
`z95 = 1.96  # 95% confidence interval z-score`. Has a comment — acceptable. But it's
a magic number that controls the CI width. Should be a named constant at module level:
`Z_SCORE_95_CI = 1.96`.

**MN-03: `0.3` in `Forecaster.__init__(self, alpha: float = 0.3)`**
EWMA smoothing factor hardcoded as default. Has no corresponding constant. Should be
`DEFAULT_EWMA_ALPHA = 0.3` at module level.

**MN-04: `0.2` in `Categorizer.train()` appears twice**
`test_size=0.2` and in `min_samples_for_stratify = int(n_classes / 0.2) + 1`. The `0.2`
should be `TEST_SIZE = 0.2` to avoid the two usages drifting.

**MN-05: `5` in `FinancialAgent`**
`if len(self._session_history[session_id]) > 5:` and `self._session_history[session_id][-5:]`
and `return self._session_history.get(session_id, [])[-5:]`. The number `5` appears three
times with no named constant. Should be `MAX_SESSION_HISTORY = 5`.

**MN-06: `3` in `AgentExecutor(max_iterations=3)`**
`max_iterations=3` in `agent.py`. No constant, no comment explaining why 3.

---

### Magic strings

**MS-01: `"all-MiniLM-L6-v2"` appears in two files**
`vector_store.py`: `EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"` — correctly declared as
a module constant.
`.env.example`: `EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2` — correct, this is configuration.
But the constant in `vector_store.py` is used as a DEFAULT parameter, meaning if the env
var is not set but a VectorStore is constructed directly, it uses this hardcoded default.
The single source of truth for the model name is the env var — the in-code default is
a fallback that could drift from the env var.

**MS-02: `"llama-3.1-8b-instant"` in `agent.py` is hardcoded**
No constant, no env var. If Groq deprecates this model, the code must be changed. Should
be `LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")`.

**MS-03: `"data/processed/categorizer.joblib"` in `app.py`**
`model_path = Path("data/processed/categorizer.joblib")` — relative path hardcoded as string.
No constant, no env var. If the model path changes, must update `app.py`.

**MS-04: Tool name strings in `FINANCE_KEYWORDS`**
Not strictly magic strings, but `"groceries"`, `"dining"`, etc. in `FINANCE_KEYWORDS` must
match category names from `SyntheticGenerator.CATEGORIES`. If a category is renamed, the
keyword list won't prevent out-of-scope false positives for that category name.

---

### Feature envy

**FE-01: `AnomalyDetector.fit_and_score()` accesses TransactionStore internals**
`with store._get_connection() as conn:` — `AnomalyDetector` is doing database operations
that `TransactionStore` should provide. The anomaly detector is envying the store's
database capabilities.

**FE-02: `PDFParser._parse_row()` logic uses `CSVParser` private methods**
`PDFParser` is envying `CSVParser`'s field parsing capabilities. It wants to use the
parsing logic without being a `CSVParser`.

---

### God functions

**GF-01: `app.py`'s `ingest()` endpoint**
Does: file type validation → temp file creation → parsing → source_file assignment →
categorization → storage → vector indexing → anomaly detection → temp file cleanup →
response building. Eight distinct responsibilities in one async function.

---

### Tight coupling / cyclic dependencies

**TC-01: `anomaly_detector.py` → `transaction_store.py` (accesses private method)**
Direct coupling to `store._get_connection()`.

**TC-02: `pdf_parser.py` → `csv_parser.py` (accesses private methods)**
`self._field_mapper._canonical_field_name()`, `._parse_date()`, `._parse_amount()`.

**TC-03: `app.py` → almost all modules**
`app.py` imports from `anomaly`, `api.models`, `api.vector_store`, `categorization`,
`forecasting`, `ingestion.csv_parser`, `ingestion.pdf_parser`, `ingestion.transaction_store`.
9 module-level imports. This is a god module in terms of imports — it touches everything.
No cyclic dependencies in the traditional sense, but `app.py` has maximum fan-in.

---

### Temporary fields

**TF-01: `total_rows_attempted` in `PDFParser.parse()` is computed but never returned**
`total_rows_attempted = 0` is incremented in the loop but the `ParseSummary` returned
does NOT include it as a field — `ParseSummary` has `parsed`, `skipped`, `warnings`, `file_errors`.
The spec says "return a summary containing the total number of rows attempted" (Requirement 3.7).
The count is computed but silently discarded — `ParseSummary` has no `attempted` field.
This is a spec compliance gap hidden in a local variable.


---

## SECTION 12 — Dead Paths

**DP-01: `from typing import Optional` in `tools.py` — never used**
`tools.py` line 3: `from typing import Optional`. No `Optional` type annotation appears
anywhere in `tools.py`. The `@tool` decorated functions use plain type hints.
Dead import.

**DP-02: `src/api/__init__.py` re-exports — never imported by anyone**
```python
from .vector_store import VectorStore
from .models import TransactionDTO, ForecastDTO, IngestResponse, ChatRequest, ChatResponse
```
`app.py` imports `from api.models import (ChatRequest, ChatResponse, ...)` directly.
No file in the project imports `from api import TransactionDTO` or `from api import VectorStore`.
These re-exports are dead code.

**DP-03: `Settings` class-level annotations are never read as annotations**
```python
class Settings:
    SQLITE_DB_PATH: str
    CHROMA_PERSIST_DIR: str
    ...
```
These are class-level variable annotations with no value. They appear as documentation
but `Settings.__annotations__` would expose them — however, nothing reads `Settings.__annotations__`
in the project. The actual instance attributes are set in `_load()`. These annotations
are effectively dead documentation.

**DP-04: `conftest.py` `sys.path.insert` is redundant with `src/main.py`'s insert**
`src/main.py` line 5: `sys.path.insert(0, str(Path(__file__).parent))` — inserts `src/` for uvicorn.
`conftest.py` line 2: `sys.path.insert(0, str(Path(__file__).parent / "src"))` — inserts `src/` for pytest.
Both do the same thing in different contexts. Neither is dead, but their existence implies
the path setup is decentralized. If `main.py`'s path insert were removed, uvicorn would fail.
If `conftest.py`'s insert were removed, all tests would fail.

**DP-05: `PROPHET_YEARLY_SEASONALITY` and `PROPHET_WEEKLY_SEASONALITY` in `config.py`**
Both settings are loaded in `Settings._load()`:
```python
self.PROPHET_YEARLY_SEASONALITY = (os.getenv("PROPHET_YEARLY_SEASONALITY", "true").lower() == "true")
self.PROPHET_WEEKLY_SEASONALITY = (os.getenv("PROPHET_WEEKLY_SEASONALITY", "true").lower() == "true")
```
Neither is ever read by any other module. `Forecaster` doesn't import or use `get_settings()`.
These are dead configuration values.

**DP-06: `VectorStore` `_load()` exception handler captures `e` but never uses it**
```python
except Exception as e:
    self._embeddings = None
    self._metadata = []
```
The variable `e` is bound but not logged or re-raised. Python may even emit a warning
in some linters: "local variable 'e' is assigned to but never used."

**DP-07: `data/raw/roundtrip_test.csv` — no test or code references this file**
This file exists in `data/raw/` (gitignored directory). No test file imports it. No script
generates it. It appears to be a development artifact from manual testing. Dead file.

**DP-08: `notebooks/` directory — completely empty**
No notebooks. The directory exists with no files. Dead directory.


---

## SECTION 13 — Future Merge Conflicts

### If three engineers work on Dashboard / API / ML simultaneously:

**Files that will constantly conflict:**

**`src/api/app.py` — highest conflict risk**
- Dashboard engineer: adding new endpoints (e.g., `GET /categories`, `DELETE /transactions/{id}`)
- API engineer: refactoring `_get_components()` to `lifespan`, changing component references
- ML engineer: adding categorizer retraining endpoint, changing ingest pipeline to return more ML metadata
All three are touching the same file. **Every PR will conflict here.**

Mitigation: Split `app.py` into `src/api/routes/ingest.py`, `src/api/routes/transactions.py`,
`src/api/routes/forecast.py`, `src/api/routes/chat.py`. Use FastAPI `APIRouter` to include
them in a central `app.py` that only handles app creation and middleware. Each engineer
works in their own router file.

**`src/api/models.py` — medium conflict risk**
- Dashboard engineer: adding new response fields (e.g., `page_count` to `IngestResponse`)
- ML engineer: adding `confidence_score` to `TransactionDTO`
Both touch the same Pydantic models. **Will conflict on model fields.**

Mitigation: Split into `src/api/models/transaction.py`, `src/api/models/forecast.py`,
`src/api/models/chat.py`. Each engineer modifies their own models file.

**`requirements.txt` — guaranteed conflict**
Every engineer adds packages. Single flat file. **Every PR that adds a dependency conflicts.**

Mitigation: Use `pyproject.toml` with dependency groups (`[tool.poetry.dependencies]`,
`[tool.poetry.group.dev.dependencies]`, `[tool.poetry.group.ml.dependencies]`). Or
accept that `requirements.txt` always needs manual merge.

**`src/domain.py` — rare but catastrophic conflict**
ML engineer might add `confidence_score: Optional[float]` to Transaction.
Dashboard engineer might add `display_name: str` for UI purposes.
**Any simultaneous modification to `Transaction` fields will conflict and break everything.**

Mitigation: Only one engineer owns `domain.py`. All domain changes go through a pull
request review gate.

**`conftest.py` — low probability but painful**
If any engineer adds fixtures that need to be shared. Currently 2 lines; easy to conflict
if both add module-level fixtures.

Mitigation: Move shared fixtures to `tests/conftest.py` sections divided by concern.

---

## SECTION 14 — Maintainability Review

### Parts that will age well (5 years)

**`src/domain.py`** — A `@dataclass` with 9 fields. Dataclasses are stable Python.
No framework dependencies. Will work on Python 3.20 the same as today.

**`src/ingestion/csv_parser.py`** — Pure Python stdlib (`csv`, `datetime`). No external
dependencies. The column alias mapping and date format list will expand but the architecture
is sound. This file will outlast every ML framework in the project.

**`src/ingestion/transaction_store.py`** — SQLite via stdlib `sqlite3`. SQLite will be
supported forever. The schema is simple. Migrations are manual but SQLite's stability means
schema changes are rare. The parameterized queries protect against injection for the long term.

**`src/forecasting/forecaster.py`** — NumPy + pandas EWMA. Both are extremely stable
libraries with long-term maintenance commitments. No deep ML framework dependency.

---

### Parts that will become painful (5 years)

**`langchain==0.1.20`** — Already 2 major versions behind. LangChain has breaking changes
in every minor version. In 5 years, 0.1.20 will be completely obsolete and incompatible
with any new Python version or dependency. **This will be the most painful legacy dependency.**

**`src/api/app.py` with `_get_components()`** — If not fixed, every new feature added by
every engineer goes through this function. It will accumulate more and more components
and become unmaintainable. The longer it lives, the more painful it is to refactor.

**`src/categorization/categorizer.py` — training data coupling** — The trained model
embeds knowledge of 44 specific merchant names. In 5 years, the training data will be
stale. New merchant names (from new companies, rebranded businesses) will all be "Other".
The model will require periodic retraining with updated data, and the training pipeline
(`scripts/train_categorizer.py`) will need to be maintained.

**`src/api/vector_store.py` (NumPy implementation)** — At scale (10k+ transactions),
the full in-memory matrix becomes impractical. In 5 years this will need to be replaced
with a proper vector database. The interface is clean (easy to swap), but the migration
effort grows with the size of stored data.

**`pytest.ini` + `conftest.py` path hack** — The `sys.path.insert` pattern is a workaround
for not having a proper Python package structure. In 5 years, the correct fix is to add a
`pyproject.toml` with `[tool.pytest.ini_options] pythonpath = ["src"]` and make `src/` a
proper package. The hack works but is fragile in CI environments.


---

## SECTION 15 — Production Code Review

Comments a senior staff engineer would leave on a PR for this codebase.
No compliments. Actionable only.

---

**`src/api/app.py` — `_get_components()` function**
> This function is called on every HTTP request and instantiates `SentenceTransformer`, which
> loads a ~200MB PyTorch model from disk every time. This is a ~4 second overhead per request.
> Use FastAPI `lifespan` for startup initialization. This is blocking blocker for production.

---

**`src/api/app.py` — `ingest()` endpoint, line ~51**
> `tmp_path = Path(f"data/raw/{filename}")` — `filename` is user-supplied. A filename of
> `../../src/config.py` resolves to the config source file and would overwrite it.
> Use `safe_filename = Path(file.filename).name` to strip path components before constructing `tmp_path`.

---

**`src/api/app.py` — `chat()` endpoint, lines 110-115**
> `from dotenv import load_dotenv; load_dotenv()` inside a request handler.
> (1) This import is at request time — move it to module level.
> (2) `load_dotenv()` with no arguments defaults to `override=True`, which can override
> system environment variables. `config.py` uses `override=False`. Inconsistency.
> (3) `from anomaly.anomaly_detector import AnomalyDetector` and `from forecasting.forecaster import Forecaster`
> are already imported at module level — these local imports are dead code.

---

**`src/api/app.py` — ingest pipeline, lines ~75-88**
> After inserting N new transactions, the code calls `vector_store.index(txn)` for ALL
> transactions in the database (via `store.get_all()`), not just the newly inserted ones.
> With 1000 existing transactions and 10 new ones, this calls `model.encode()` 1010 times
> instead of 10. This is O(n²) in total ingest operations.

---

**`src/anomaly/anomaly_detector.py` — `fit_and_score()`, line ~64**
> `with store._get_connection() as conn:` — directly accessing a private method of
> `TransactionStore`. Add `TransactionStore.update_anomaly_scores(scores: dict[int, tuple[bool, float]])` 
> as a public method. This breaks the encapsulation contract and couples `AnomalyDetector`
> to `TransactionStore`'s internal implementation.

---

**`src/anomaly/anomaly_detector.py` — score normalization, line ~57**
> `normalized_scores = np.clip(-raw_scores, 0, 1).tolist()` produces scores in [0, 0.07]
> range, not [0, 1], because `IsolationForest.decision_function` returns values in [-0.2, 0.2].
> Use min-max normalization:
> ```python
> inverted = -raw_scores
> min_s, max_s = inverted.min(), inverted.max()
> normalized_scores = ((inverted - min_s) / (max_s - min_s + 1e-10)).tolist()
> ```

---

**`src/api/vector_store.py` — `_load()`, lines ~47-54**
> `except Exception as e:` — exception `e` is captured but never logged.
> A corrupted vector store silently resets to empty. Add:
> `logger.error(f"Failed to load vector store from {self._persist_dir}: {e}")`

---

**`src/api/vector_store.py` — `_find_index()`, lines ~56-60**
> O(n) linear scan on every `index()` and `delete()` call. Add a dict:
> `self._id_to_index: dict[int, int] = {}` maintained in sync with `_metadata`.
> `_find_index` becomes `return self._id_to_index.get(transaction_id, -1)`.
> Required before this file handles >1000 transactions.

---

**`src/api/vector_store.py` — `_save()`, line ~42**
> No atomicity between `np.save()` and `json.dump()`. If the process is killed between
> the two writes, the files are inconsistent. Use a write-then-rename pattern:
> save to `.npy.tmp` and `.json.tmp`, then `os.replace()` (atomic on POSIX).
> On Windows, `os.replace()` is also atomic for files on the same drive.

---

**`src/ingestion/pdf_parser.py` — `PDFParser.__init__()`, line ~14**
> `self._field_mapper = CSVParser()` to access private methods `._canonical_field_name()`,
> `._parse_date()`, `._parse_amount()`. This is calling private methods of another class.
> Extract a `FieldMapper` utility with public methods. `CSVParser` and `PDFParser` both
> use `FieldMapper`. Remove the `CSVParser` dependency from `PDFParser`.

---

**`src/ingestion/pdf_parser.py` — `parse()`, line ~42**
> `total_rows_attempted` is incremented but never returned. Requirement 3.7 says the
> summary must contain "total number of rows attempted." `ParseSummary` has no `attempted`
> field. Either add `attempted: int = 0` to `ParseSummary`, or document that this
> requirement is not yet met.

---

**`src/categorization/categorizer.py` — `predict()`, line ~96**
> `except Exception:` — bare except with no logging. If `sklearn` raises an unexpected
> internal error (e.g., version incompatibility), the error is silently converted to
> `category="Other"`. This makes debugging impossible. Add at minimum:
> `logger.debug(f"Categorizer prediction failed for merchant '{transaction.merchant}': {e}")`

---

**`src/agent/agent.py` — `FinancialAgent.__init__()`, line ~57**
> `api_key = os.getenv("LLM_API_KEY")` — if `LLM_API_KEY` is not set, `api_key=None`.
> `ChatGroq(api_key=None, ...)` doesn't raise at construction. The error surfaces during
> the first `chat()` call. Fail fast: `if not api_key: raise ValueError("LLM_API_KEY not set")`.

---

**`src/agent/agent.py` — untyped parameters throughout**
> `def __init__(self, store, vector_store, forecaster, anomaly_detector):`
> `def chat(self, message, session_id):`
> All parameters must have type annotations. Without them, static analysis cannot catch
> incorrect usage.

---

**`src/config.py` — `Settings._load()`, lines ~39-45**
> `self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")` — `os.getenv()` returns `Optional[str]`.
> The class annotation says `SQLITE_DB_PATH: str`. After `_validate_required()` confirms
> the variable exists, the value is guaranteed non-None, but the type system doesn't know
> this. Use `self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH") or ""` or cast explicitly
> to satisfy type checkers.

---

**`tests/unit/test_categorizer.py` — `test_predict_batch_handles_individual_errors_without_aborting`**
> This test wraps its assertions inside `except RuntimeError: pass`. If `predict_batch`
> does NOT handle the error (the very behavior being tested), the `RuntimeError` propagates
> to the `except` block, the assertions never run, and the test passes silently — a false positive.
> Remove the `try/except`. Mock at the pipeline level instead of at the `predict()` level.

---

**`src/agent/tools.py` — all four tools**
> All four tools use `except Exception as e:` and return error strings to the LLM.
> The LLM then tells the user "there was an error" without specifics.
> Improve: distinguish between `ValueError` (bad input, user-actionable) and other exceptions
> (internal errors, should be logged at ERROR level and not exposed to users).


---

## SECTION 16 — Final Repository Knowledge

Things visible from the code itself that haven't been stated in any previous document:

**FK-01: `ParseSummary.attempted` is missing — this is a spec violation visible in code**
`pdf_parser.py` line 34: `total_rows_attempted = 0` is computed but never put into
`ParseSummary`. Requirement 3.7 explicitly says the summary must contain "total number
of rows attempted." `ParseSummary` dataclass has no `attempted` field. The count is
computed correctly but silently discarded. This is the only spec acceptance criterion
with code that computes the right value but fails to return it.

**FK-02: `_parse_amount()` strips `$` and `,` but not `£`, `€`, `₹`**
`csv_parser.py`: `cleaned = raw_value.strip().replace("$", "").replace(",", "")`.
Only USD formatting is handled. A bank CSV from the UK (`£87.43`) or India (`₹87.43`)
would fail to parse — `float("£87.43")` raises `ValueError`, the row is skipped with
a warning. For a project with IST timezone metadata (commit timestamps), this is likely
to encounter Indian bank statements with `₹` symbols.

**FK-03: `SyntheticGenerator.write_csv()` always writes to `synthetic_transactions.csv`**
`output_path = output_dir / "synthetic_transactions.csv"` — hardcoded filename.
You can't call `write_csv()` twice with the same `output_dir` to get two files with
different names. The second call silently overwrites the first. If generating test
fixtures for multiple scenarios, this is a limitation.

**FK-04: `date_span_days` in `SyntheticGenerator.generate()` is computed once, not validated for zero**
`date_span_days = (end_date - start_date).days`. If `start_date == end_date`, `date_span_days=0`.
`rng.randint(0, 0)` always returns 0. All transactions get the same date.
`rng.uniform(min_amt, max_amt)` still works. `rng.choice(...)` still works. No error,
but a degenerate dataset is produced silently. Forecaster would raise because all transactions
are on the same day (1 distinct day < 14 required).

**FK-05: `Forecaster._get_daily_totals()` returns a DataFrame with pandas Timestamps, not `datetime.date`**
`last_date = df["ds"].iloc[-1]`. The `df["ds"]` column contains Python `datetime.date` objects
(from `daily` dict keys), but pandas may convert them to `pd.Timestamp` depending on
pandas version and operation. The check `if hasattr(last_date, "date"): last_date = last_date.date()`
handles this. But `ForecastPoint.date` is typed as `datetime.date`. If `last_date` is ever
a naive datetime, `last_date + timedelta(days=i + 1)` still works. If it's a `pd.Timestamp`,
the `hasattr("date")` check converts it correctly. This is a type fragility that could
break on pandas version changes.

**FK-06: `VectorStore.search()` imports `date` inside the method**
```python
from datetime import date
try:
    txn_date = date.fromisoformat(meta["date"])
```
This is an import inside a function body — executed on every call to `search()`. Python
caches module imports, so this doesn't re-import the module each time, but it's non-idiomatic.
`date` should be imported at the top of `vector_store.py` with the other imports.

**FK-07: `calculate_total` tool's `"all"` check is case-sensitive for `cleaned.lower()`**
```python
if not cleaned or cleaned.lower() == "all":
```
`cleaned.lower()` is applied, so "ALL", "All", "aLL" all work. But the initial strip:
`cleaned = cleaned.strip().strip('"').strip("'").strip()` strips only leading/trailing
characters. If the LLM passes `query="all transactions"`, `cleaned.lower() == "all"` fails
because it has extra text. The tool then queries category `"all transactions"` which returns
empty (no category named that). The fallback to `store.get_all()` only triggers for
exactly `"all"` (after strip).

**FK-08: `Categorizer.train()` does NOT validate that labels are from `CANONICAL_CATEGORIES`**
```python
labels = [txn.category for txn in labeled_transactions]
```
The training data can have ANY category strings — the LabelEncoder will happily encode
them. If training data has `"Food"` instead of `"Dining"`, the model learns `"Food"` as
a valid category. `predict()` would then return `"Food"` with high confidence, which is
NOT in `CANONICAL_CATEGORIES`. The `CANONICAL_CATEGORIES` check is only in the assertion
in `test_categorizer.py`. The production code has no validation that trained categories
match the canonical set.

**FK-09: `VectorStore.index()` with `transaction.id=None` creates a permanent orphan entry**
If `index()` is called with a transaction where `id=None` (happens if called before SQLite
insert), `_find_index(None)` returns -1 (not found). A new entry is appended to `_metadata`
with `transaction_id: null`. This entry can never be found by `_find_index(None)` again
because `_find_index` returns the FIRST match — but subsequent calls with `transaction.id=None`
will match the first null entry and update it, not create a new one. So at most one
null-id orphan can exist. But this null-id entry will appear in search results, reconstructed
as `Transaction(id=None, ...)`.

**FK-10: `TransactionStore.delete()` does not verify the row was actually deleted**
```python
def delete(self, transaction_id: int) -> None:
    with self._get_connection() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (transaction_id,))
```
If `transaction_id` doesn't exist, no error is raised and `rowcount=0`. The delete is
a silent no-op. Unlike `VectorStore.delete()` which is explicitly documented as a no-op
for missing IDs, `TransactionStore.delete()` has no documentation about this behavior.
This is consistent with the VectorStore behavior but undocumented.

**FK-11: The `ParseSummary.warnings` list grows unboundedly**
In `csv_parser.py`, every skipped row appends to `summary.warnings`. For a 10,000-row
CSV with many bad rows, `summary.warnings` could contain thousands of strings.
In `app.py`: `return IngestResponse(..., warnings=summary.warnings[:10])` — only the
first 10 are returned. But the full list is still in memory. For a very large CSV with
many bad rows, the warnings list is a memory concern.

**FK-12: `SyntheticGenerator` uses `rng.uniform(min_amt, max_amt)` which is inclusive on both ends**
`random.Random.uniform(a, b)` returns `a <= N <= b`. The spec says amounts are "within"
the range — this is correct. But `round(..., 2)` can theoretically produce values
slightly outside bounds due to floating-point rounding. E.g., `round(random.uniform(5.0, 300.0), 2)`
could theoretically produce `300.0` since `300.0` rounds to `300.00` — which is exactly
at the boundary. The test `assert min_amt <= txn.amount <= max_amt` uses `<=` so boundary
values pass. Not a bug, but worth knowing that the spec's range is inclusive, not exclusive.

---

*End of SOURCE_CODE_INSPECTION.md*
