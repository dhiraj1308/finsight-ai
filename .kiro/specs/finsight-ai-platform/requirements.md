# Requirements Document

## Introduction

FinSight AI is a personal finance intelligence platform that ingests bank and credit card
statements, parses them into structured transaction data, and applies machine learning to
auto-categorize transactions, detect anomalies, and forecast future spending. Users can
interact with their financial data through a natural-language chat interface backed by a
tool-using LLM agent grounded in a vector database of transactions. The platform exposes
all capabilities through a FastAPI backend and a Streamlit dashboard frontend.

The system is designed for solo use by a single developer/user, running on CPU-only
hardware, built incrementally over 30 days using Python, pandas, scikit-learn,
sentence-transformers, ChromaDB, LangChain, Prophet, FastAPI, Streamlit, and SQLite
(development database).

---

## Glossary

- **Platform**: The FinSight AI application as a whole.
- **Ingestion_Module**: The component responsible for reading raw PDF and CSV statement files and producing structured transaction records.
- **Statement**: A bank or credit card statement file in PDF or CSV format.
- **Transaction**: A single financial event with the fields: `date`, `merchant`, `amount`, and `category`.
- **Transaction_Store**: The SQLite database that persists all parsed transactions.
- **Synthetic_Generator**: The component that generates realistic fake transaction data for development and testing purposes.
- **CSV_Parser**: The sub-component of the Ingestion_Module that handles CSV statement files.
- **PDF_Parser**: The sub-component of the Ingestion_Module that handles PDF statement files.
- **Categorizer**: The ML-based component that assigns a spending category to each transaction.
- **Category**: A human-readable label for a type of spending (e.g., "Groceries", "Utilities", "Entertainment").
- **Anomaly_Detector**: The unsupervised ML component that flags statistically unusual transactions.
- **Anomaly**: A transaction whose amount or pattern deviates significantly from the user's historical norms.
- **Forecaster**: The time-series forecasting component that predicts future spending per category.
- **Forecast**: A projected spending amount for a future time period within a given category.
- **Vector_Store**: The ChromaDB vector database that stores transaction embeddings for semantic retrieval.
- **Embedding_Model**: The sentence-transformers model used to convert transaction text into vector embeddings.
- **Agent**: The LangChain-based LLM agent that answers natural-language financial questions using tools.
- **RAG**: Retrieval-Augmented Generation — the pattern of grounding LLM responses in retrieved transaction context.
- **Tool**: A callable function exposed to the Agent (e.g., retrieve transactions, run forecast, calculate total).
- **API**: The FastAPI backend that exposes all platform capabilities as HTTP endpoints.
- **Dashboard**: The Streamlit frontend that displays charts and hosts the chat interface.
- **Pretty_Printer**: The component that serializes structured Transaction objects back into a canonical CSV format.

---

## Requirements

### Requirement 1: Synthetic Transaction Data Generation

**User Story:** As a developer, I want to generate realistic synthetic transaction data, so that I can develop and test the platform without needing real bank statements.

#### Acceptance Criteria

1. THE Synthetic_Generator SHALL produce transaction records containing `date`, `merchant`, `amount`, and `category` fields.
2. WHEN the Synthetic_Generator is invoked with a record count N (where 1 ≤ N ≤ 100,000) and a date range, THE Synthetic_Generator SHALL produce exactly N transaction records whose `date` values fall within the inclusive start and end dates of that range.
3. THE Synthetic_Generator SHALL produce merchant names and amounts that are realistic for each category using the following per-category amount ranges: Groceries $5–$300, Utilities $20–$300, Entertainment $5–$150, Dining $5–$200, Transport $2–$100, Healthcare $10–$500, Shopping $10–$400, Subscriptions $5–$50.
4. THE Synthetic_Generator SHALL write output to a CSV file in the `data/synthetic/` directory.
5. WHEN a seed value is provided to the Synthetic_Generator, THE Synthetic_Generator SHALL produce records with identical field values in identical row order on every invocation using that seed.
6. THE Synthetic_Generator SHALL cover at least 8 spending categories including Groceries, Utilities, Entertainment, Dining, Transport, Healthcare, Shopping, and Subscriptions.
7. WHEN the Synthetic_Generator is invoked with an invalid record count (N < 1 or N > 100,000) or a date range where start date is after end date, THE Synthetic_Generator SHALL raise a descriptive error identifying the invalid parameter and produce no output file.
8. THE Synthetic_Generator SHALL use a default record count of 3,000 transactions when no explicit count is provided; this default is chosen to keep local iteration fast on CPU-only hardware (faster Isolation Forest fits, faster embedding generation, faster Prophet fitting) while the generator remains capable of scaling to 100,000 when needed. All example invocations, default parameters, and test fixtures SHALL reference counts in the 2,000–5,000 range rather than larger values.

---

### Requirement 2: CSV Statement Parsing

**User Story:** As a user, I want to upload a CSV bank or credit card statement, so that the platform can extract my transactions without manual data entry.

#### Acceptance Criteria

1. WHEN a valid CSV statement file (UTF-8 encoded, ≤ 100 MB, containing at least 1 data row) is provided, THE CSV_Parser SHALL parse each data row into a Transaction record with `date`, `merchant`, and `amount` fields; the `category` field is optional in the input and defaults to empty string if absent.
2. WHEN a CSV file contains a header row with column names differing from the canonical schema, THE CSV_Parser SHALL map the following known column name variants (case-insensitive) to canonical fields: "Description" / "Narration" / "Details" → `merchant`; "Debit" / "Withdrawal" / "Charge" → `amount`; "Transaction Date" / "Posted Date" / "Trans. Date" → `date`.
3. IF a CSV row is missing a required field (`date`, `merchant`, or `amount`), THEN THE CSV_Parser SHALL skip that row and record a parse warning containing the 1-based row number and the name of the missing field.
4. IF the `date` field in a CSV row cannot be parsed as a valid calendar date, THEN THE CSV_Parser SHALL skip that row and record a parse warning containing the 1-based row number and the raw value that failed to parse.
5. IF the `amount` field in a CSV row cannot be parsed as a numeric value, THEN THE CSV_Parser SHALL skip that row and record a parse warning containing the 1-based row number and the raw value that failed to parse.
6. IF the CSV file is empty or cannot be decoded as UTF-8, THEN THE CSV_Parser SHALL return zero parsed records and one file-level error message describing the failure; it SHALL NOT raise an unhandled exception.
7. WHEN parsing is complete, THE CSV_Parser SHALL return a summary containing the count of successfully parsed rows and the count of skipped rows.
8. WHEN a list of Transaction records is serialized by THE Pretty_Printer, THE Pretty_Printer SHALL write a valid CSV file whose header row matches the canonical schema (`date`, `merchant`, `amount`, `category`) with fields in that exact order.
9. FOR ALL non-empty lists of valid Transaction records, parsing a canonical CSV file, serializing the resulting records with THE Pretty_Printer, then parsing the output again SHALL produce a list of Transaction records where each record has field values identical (character-for-character) to the corresponding record from the first parse.

---

### Requirement 3: PDF Statement Parsing

**User Story:** As a user, I want to upload a PDF bank statement, so that the platform can extract my transactions from the document without manual data entry.

#### Acceptance Criteria

1. WHEN a valid PDF statement file (≤ 10 MB, ≤ 100 pages) is provided, THE PDF_Parser SHALL extract transaction rows and parse each into a Transaction record with `date` (a parseable calendar date), `merchant` (a non-empty string), and `amount` (a numeric value with positive indicating a debit charge) fields.
2. THE PDF_Parser SHALL use pdfplumber to extract text content from each page of the PDF.
3. IF a PDF file is password-protected, THEN THE PDF_Parser SHALL return the error message "File is password-protected and cannot be read." and zero parsed records, without raising an unhandled exception.
4. IF a PDF file exceeds 10 MB or 100 pages, or is not a valid PDF file, THEN THE PDF_Parser SHALL return a descriptive error message identifying the rejection reason and zero parsed records.
5. IF a PDF file contains no recognizable transaction table (no page yields at least one row containing a parseable date and a numeric amount), THEN THE PDF_Parser SHALL return the error message "No recognizable transaction table found." and zero parsed records.
6. IF a row extracted from a PDF page cannot be mapped to a valid Transaction (missing or unparseable `date`, `merchant`, or `amount` field), THEN THE PDF_Parser SHALL skip that row and record a parse warning containing the page number, row index on that page, and the name(s) of the failing field(s).
7. WHEN parsing is complete, THE PDF_Parser SHALL return a summary containing the total number of rows attempted, the count of successfully parsed records, and the count of skipped rows.

---

### Requirement 4: Transaction Storage

**User Story:** As a user, I want my parsed transactions to be persisted, so that I can analyze them across multiple sessions without re-uploading statements.

#### Acceptance Criteria

1. THE Transaction_Store SHALL persist Transaction records in a SQLite database with columns: `id`, `date`, `merchant`, `amount`, `category`, `is_anomaly` (default `false` at insert time), `anomaly_score`, and `source_file`.
2. WHEN a list of Transaction records is written to the Transaction_Store, THE Transaction_Store SHALL assign a unique `id` to each record and SHALL return the count of records successfully inserted and the count skipped as duplicates.
3. WHEN a Transaction record with an identical (`date`, `merchant`, `amount`, `source_file`) combination already exists in the Transaction_Store, THE Transaction_Store SHALL skip the duplicate and not insert a second record.
4. WHEN transactions are queried by date range, THE Transaction_Store SHALL return all records whose `date` falls within the inclusive start and end dates; IF `start_date` is after `end_date`, THE Transaction_Store SHALL raise a descriptive error and return no records.
5. WHEN transactions are queried by category, THE Transaction_Store SHALL return all records matching the specified category string exactly (case-insensitive); WHEN a query returns no matching records, THE Transaction_Store SHALL return an empty list.
6. WHEN all transactions are queried without filters, THE Transaction_Store SHALL return all records ordered by `date` descending; WHEN no records exist, THE Transaction_Store SHALL return an empty list.

---

### Requirement 5: Transaction Categorization

**User Story:** As a user, I want my transactions to be automatically categorized, so that I can understand my spending by type without labeling each transaction manually.

#### Acceptance Criteria

1. WHEN a Transaction record with a `merchant` and `amount` is provided, THE Categorizer SHALL assign one Category label from the following canonical set: Groceries, Utilities, Entertainment, Dining, Transport, Healthcare, Shopping, Subscriptions, Other, Uncategorized.
2. THE Categorizer SHALL use a machine learning classifier trained on labeled transaction data rather than hardcoded keyword rules.
3. THE Categorizer SHALL use TF-IDF or sentence-transformer embeddings of the `merchant` field as input features to the classifier.
4. WHEN the Categorizer is trained on a labeled dataset, THE Categorizer SHALL achieve a weighted F1 score of at least 0.80 on a held-out validation set of at least 200 transactions.
5. WHEN a transaction's classifier confidence score is below 0.60, THE Categorizer SHALL assign the category "Other" and set a `needs_review` flag on the Transaction record to `true`.
6. WHEN the Categorizer's batch prediction method is invoked with a list of Transaction records, THE Categorizer SHALL return the same list with `category` fields populated for all records; IF any individual record raises an error during prediction, THEN that record's `category` SHALL be set to "Other" with `needs_review` set to `true`, and batch processing SHALL continue without aborting.

---

### Requirement 6: Anomaly Detection

**User Story:** As a user, I want unusual transactions to be flagged automatically, so that I can quickly identify potential fraud or unexpected charges.

#### Acceptance Criteria

1. WHEN the Anomaly_Detector is run against the Transaction_Store, THE Anomaly_Detector SHALL evaluate each transaction and set its `is_anomaly` flag to `true` or `false` based on the Isolation Forest decision function.
2. THE Anomaly_Detector SHALL use Isolation Forest from scikit-learn and SHALL NOT require labeled anomaly data.
3. THE Anomaly_Detector SHALL use the `amount` field (numeric) and the label-encoded `category` field as input features when computing anomaly scores.
4. WHEN a transaction is flagged as an anomaly, THE Anomaly_Detector SHALL record a normalized anomaly score in the range [0.0, 1.0] (higher = more anomalous) alongside the boolean `is_anomaly` flag in the Transaction_Store.
5. WHEN the Anomaly_Detector is re-run after new transactions are added, THE Anomaly_Detector SHALL re-evaluate all transactions and update `is_anomaly` flags and anomaly scores accordingly.
6. THE Anomaly_Detector SHALL expose a method that returns all transactions currently flagged as anomalies, ordered by anomaly score descending.
7. WHEN the Transaction_Store contains fewer than 10 transactions, THE Anomaly_Detector SHALL return a descriptive error indicating insufficient data for anomaly detection and SHALL NOT modify any `is_anomaly` flags.

---

### Requirement 7: Spending Forecasting

**User Story:** As a user, I want to see a forecast of my future spending per category, so that I can plan my budget for upcoming weeks or months.

#### Acceptance Criteria

1. WHEN a category and a forecast horizon in days (1–365 inclusive) are provided, THE Forecaster SHALL return a Forecast with projected daily spending amounts (floored at 0.0) for each day in the horizon.
2. THE Forecaster SHALL use the Prophet library to fit a time-series model on historical daily transaction totals per category, where "historical data" means distinct calendar days on which at least one transaction exists.
3. WHEN fewer than 14 distinct calendar days of historical data exist for a category, THE Forecaster SHALL return a descriptive error indicating insufficient data for that category and SHALL NOT attempt to fit a model.
4. THE Forecaster SHALL include a lower bound (floored at 0.0) and upper bound representing a 95% confidence interval alongside the point estimate in each Forecast.
5. WHEN the Forecaster is queried for a category with no transactions, THE Forecaster SHALL return a descriptive error indicating no data exists for that category.
6. THE Forecaster SHALL expose a method that returns forecasts for all categories simultaneously; for categories with insufficient data, the method SHALL include a per-category error entry rather than aborting the entire call.
7. WHEN a forecast horizon outside the range 1–365 days is provided, THE Forecaster SHALL return a descriptive error identifying the invalid horizon and produce no forecast.

---

### Requirement 8: Vector Store and Embeddings

**User Story:** As a developer, I want transactions to be indexed in a vector database, so that the Agent can retrieve semantically relevant transactions to ground its answers.

#### Acceptance Criteria

1. WHEN a Transaction record is inserted into the Transaction_Store, THE Vector_Store SHALL also index an embedding of that transaction's text representation formatted as space-separated fields in this exact order: `merchant`, `category`, `amount`, `date`.
2. THE Embedding_Model SHALL use a sentence-transformers model that runs on CPU without a GPU.
3. WHEN a natural-language query string is provided with a K value (1 ≤ K ≤ 100), THE Vector_Store SHALL return the top-K most semantically similar transactions; IF fewer than K embeddings exist, THE Vector_Store SHALL return all available embeddings.
4. THE Vector_Store SHALL use ChromaDB as the underlying vector database with a persistent local collection that survives process restarts.
5. WHEN a transaction is deleted from the Transaction_Store, THE Vector_Store SHALL also remove the corresponding embedding; IF no embedding exists for that transaction `id`, the deletion SHALL be a no-op that returns successfully.
6. IF a transaction embedding already exists in the Vector_Store for a given transaction `id`, THEN THE Vector_Store SHALL update the existing embedding rather than inserting a duplicate.
7. IF Vector_Store indexing fails during a Transaction_Store insert, THE Transaction_Store insert SHALL be rolled back and an error describing the indexing failure SHALL be returned, leaving both stores in a consistent state.

---

### Requirement 9: LLM Agent with Financial Tools

**User Story:** As a user, I want to ask natural-language questions about my finances and receive accurate, grounded answers, so that I can get insights without writing queries or reading raw data.

#### Acceptance Criteria

1. WHEN a natural-language financial question is submitted to the Agent, THE Agent SHALL produce an answer that references data returned by at least one tool invocation and SHALL NOT assert facts not present in the tool-returned data.
2. THE Agent SHALL use LangChain's agent framework with at least the following tools: `retrieve_transactions` (RAG over Vector_Store), `calculate_total` (sum transactions by filter), `run_forecast` (invoke the Forecaster), and `get_anomalies` (retrieve flagged transactions).
3. WHEN the Agent invokes a tool, THE Agent SHALL pass only typed parameters matching that tool's defined schema and SHALL NOT pass free-text as a parameter value disguised as a structured parameter.
4. WHEN all invoked tools return zero or empty results for a question, THE Agent SHALL respond stating it could not find relevant data and SHALL NOT fabricate transaction details or amounts.
5. THE Agent SHALL maintain a conversational memory of at least the last 5 user+agent exchange pairs within a session, resetting memory when a new `session_id` is received.
6. WHEN a question is outside the domain of personal finance (transactions, spending, budgets, forecasts, anomalies), THE Agent SHALL respond stating it is scoped to financial data analysis and SHALL NOT attempt to answer the out-of-scope question.
7. WHEN a tool raises an error during execution, THE Agent SHALL inform the user that the tool encountered an error and SHALL NOT fabricate data to substitute for the failed tool result.

---

### Requirement 10: FastAPI Backend

**User Story:** As a developer, I want all platform capabilities exposed as REST API endpoints, so that the frontend and external clients can interact with the system programmatically.

#### Acceptance Criteria

1. THE API SHALL expose a `POST /ingest` endpoint that accepts a statement file (PDF or CSV), triggers ingestion, parsing, categorization, and Vector_Store indexing, and returns the count of successfully ingested transactions on success.
2. THE API SHALL expose a `GET /transactions` endpoint that returns transactions filtered by optional query parameters `start_date` and `end_date` (ISO 8601 format, e.g. `2024-01-15`) and `category`; WHEN no transactions match the filters, THE API SHALL return HTTP 200 with an empty list.
3. THE API SHALL expose a `GET /anomalies` endpoint that returns all transactions flagged as anomalies, ordered by anomaly score descending.
4. THE API SHALL expose a `GET /forecast/{category}` endpoint that accepts a `days` query parameter (integer, 1–365 inclusive, default 30) and returns a Forecast for that category.
5. THE API SHALL expose a `POST /chat` endpoint that accepts a `message` string (≤ 2000 characters) and `session_id` string (≤ 128 characters) and returns the Agent's response.
6. WHEN a request to any endpoint contains invalid parameters, THE API SHALL return an HTTP 422 response with a structured error message describing the validation failure.
7. WHEN an internal error occurs during request processing, THE API SHALL return an HTTP 500 response with an error message indicating the nature of the failure and SHALL log the full stack trace.
8. THE API SHALL include OpenAPI documentation auto-generated by FastAPI, accessible at `/docs`.
9. IF the uploaded file in `POST /ingest` is not a PDF or CSV file, THEN THE API SHALL return HTTP 422 and perform no ingestion.
10. IF the `category` path parameter in `GET /forecast/{category}` has insufficient historical data, THEN THE API SHALL return HTTP 422 with an error message indicating insufficient data for that category.

---

### Requirement 11: Streamlit Dashboard Frontend

**User Story:** As a user, I want a visual dashboard to explore my finances, so that I can understand spending patterns and chat with my data without using the API directly.

#### Acceptance Criteria

1. WHEN the Dashboard page loads, THE Dashboard SHALL fetch spending-by-category data from the API and display it as a bar or pie chart; IF no transaction data exists, THE Dashboard SHALL display an empty-state message rather than a broken or blank chart area.
2. THE Dashboard SHALL display a monthly spending trend line chart showing total spending per month over the available history; IF no transaction data exists, THE Dashboard SHALL display an empty-state message.
3. THE Dashboard SHALL display a list of anomalous transactions with their merchant, amount, date, and anomaly score; IF no anomalies exist, THE Dashboard SHALL display an empty-state message.
4. THE Dashboard SHALL display a forecast chart per category showing projected spending with confidence intervals for the next 30 days; IF a category has insufficient data for forecasting, THE Dashboard SHALL display a per-category message indicating insufficient data.
5. THE Dashboard SHALL include a file upload widget that allows the user to upload a PDF or CSV statement and submit it to the `POST /ingest` API endpoint.
6. WHEN the ingestion call completes, THE Dashboard SHALL display a success message with the count of ingested transactions, or an error message describing the failure, before accepting the next upload.
7. THE Dashboard SHALL include a chat interface with a text input field and a conversation history area that displays prior user messages and Agent responses within the current browser session.
8. WHEN the user submits a message in the chat interface, THE Dashboard SHALL send the message to the `POST /chat` API endpoint and display the Agent's response in the conversation history area.
9. IF the API is unreachable when the Dashboard attempts any API call, THE Dashboard SHALL display an error banner indicating the backend is unavailable rather than showing an empty or broken UI.

---

### Requirement 12: Configuration and Environment Management

**User Story:** As a developer, I want all configuration values (database paths, model names, API keys) managed through environment variables, so that the platform can be configured without modifying source code.

#### Acceptance Criteria

1. THE Platform SHALL read the following required configuration values from environment variables or a `.env` file: `SQLITE_DB_PATH`, `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL_NAME`, `LLM_API_KEY`; and the following optional values with defaults: `PROPHET_YEARLY_SEASONALITY` (default `true`), `PROPHET_WEEKLY_SEASONALITY` (default `true`), `LOG_LEVEL` (default `INFO`).
2. WHEN a required environment variable (`SQLITE_DB_PATH`, `CHROMA_PERSIST_DIR`, `EMBEDDING_MODEL_NAME`, or `LLM_API_KEY`) is missing at startup, THE Platform SHALL log an error message identifying the missing variable name, exit with a non-zero exit code, and make no API endpoints available; WHEN an optional environment variable is absent, THE Platform SHALL use its documented default value without raising an error.
3. THE Platform SHALL use python-dotenv to load `.env` files; system environment variables SHALL take precedence over values defined in the `.env` file when both are present.
4. THE Platform SHALL provide a `.env.example` file listing all required and optional environment variables with placeholder values and inline comments describing each variable's purpose and valid value range.
