# Requirements Document

## Introduction

FinSight AI allows users to upload credit card and bank statement PDFs for automated transaction
ingestion. Some financial institutions issue statements that are encrypted (password-protected)
at the PDF level. Currently, the `PDFParser` catches the `PDFPasswordIncorrect` exception and
returns a generic file error, which the backend surfaces as an HTTP 422. The frontend treats
this identically to any other parse failure — the user receives no actionable guidance.

This feature introduces end-to-end handling for password-protected PDFs:

1. The backend detects an encrypted PDF and returns a dedicated, machine-readable signal
   instead of a generic error.
2. The frontend recognises that signal and prompts the user for a decryption password
   without requiring a re-upload.
3. The frontend submits the original file together with the password; the backend decrypts
   and processes the PDF normally.
4. Wrong passwords are handled gracefully — the user is re-prompted with a clear message.
5. The password is never written to disk, logs, or any persistent store.

---

## Glossary

- **Upload_Page**: The Streamlit view rendered by `src/frontend/views/upload.py` that
  accepts file uploads from the user.
- **API_Client**: The `APIClient` class in `src/frontend/services/api.py` that issues HTTP
  requests to the FastAPI backend on behalf of the frontend.
- **Ingest_Endpoint**: The FastAPI `POST /ingest` route defined in `src/api/app.py`.
- **PDF_Parser**: The `PDFParser` class in `src/ingestion/pdf_parser.py` that opens and
  reads PDF files using `pdfplumber`.
- **Ingest_Response**: The Pydantic model `IngestResponse` in `src/api/models.py` returned
  by the `Ingest_Endpoint` on success.
- **Password_Required_Response**: A new HTTP 4xx response body that the `Ingest_Endpoint`
  returns when it detects an encrypted PDF and no password has been supplied.
- **Decryption_Password**: A string supplied by the user, held only in memory for the
  duration of a single parse attempt, used to unlock an encrypted PDF.
- **Session_State**: Streamlit's `st.session_state` dictionary, scoped to one browser
  session; used to track whether a password-prompt is active and to hold the uploaded
  file buffer between Streamlit re-runs.

---

## Requirements

### Requirement 1: Detect Encrypted PDFs and Return a Distinct Signal

**User Story:** As a backend developer, I want the `Ingest_Endpoint` to return a dedicated,
machine-readable response when it receives an encrypted PDF without a password, so that the
frontend can distinguish this case from an unrecoverable parse failure and prompt the user
accordingly.

#### Acceptance Criteria

1. WHEN the `Ingest_Endpoint` receives a PDF file that is encrypted and no
   `Decryption_Password` is supplied, THEN THE `Ingest_Endpoint` SHALL return HTTP status
   code 422 with a JSON body containing a field `error_code` equal to the string
   `"PASSWORD_REQUIRED"` and a human-readable `detail` field.

2. WHEN the `Ingest_Endpoint` receives a PDF file that is encrypted and an incorrect
   `Decryption_Password` is supplied, THEN THE `Ingest_Endpoint` SHALL return HTTP status
   code 422 with a JSON body containing a field `error_code` equal to the string
   `"PASSWORD_INCORRECT"` and a human-readable `detail` field.

3. WHEN the `Ingest_Endpoint` receives a PDF file that is not encrypted, THE
   `Ingest_Endpoint` SHALL proceed with normal parsing and SHALL NOT include an `error_code`
   field in the response body.

4. WHEN the `Ingest_Endpoint` receives a CSV file, THE `Ingest_Endpoint` SHALL proceed with
   normal parsing regardless of any `password` parameter and SHALL NOT include an
   `error_code` field in the response body.

5. THE `Ingest_Endpoint` SHALL accept an optional `password` form field alongside the
   existing `file` field in the `multipart/form-data` request body.

---

### Requirement 2: PDF Parser Supports Decryption Passwords

**User Story:** As a backend developer, I want the `PDF_Parser` to accept an optional
`Decryption_Password` and attempt to open encrypted PDFs with it, so that statement data can
be extracted from encrypted files when the correct password is provided.

#### Acceptance Criteria

1. WHEN `PDF_Parser.parse()` is called with a `Decryption_Password` and the file is
   encrypted, THE `PDF_Parser` SHALL attempt to open the PDF using that password via
   `pdfplumber`.

2. WHEN `PDF_Parser.parse()` opens an encrypted PDF with the correct `Decryption_Password`,
   THE `PDF_Parser` SHALL extract transactions and return them as if the file were
   unencrypted.

3. WHEN `PDF_Parser.parse()` is called without a `Decryption_Password` and the file is
   encrypted, THE `PDF_Parser` SHALL return an empty transaction list and a `ParseSummary`
   whose `file_errors` list contains exactly one entry with `error_code` equal to
   `"PASSWORD_REQUIRED"`.

4. WHEN `PDF_Parser.parse()` is called with an incorrect `Decryption_Password`, THE
   `PDF_Parser` SHALL return an empty transaction list and a `ParseSummary` whose
   `file_errors` list contains exactly one entry with `error_code` equal to
   `"PASSWORD_INCORRECT"`.

5. THE `PDF_Parser` SHALL NOT write the `Decryption_Password` to any file, log entry,
   exception message, or `ParseSummary` field.

6. THE `PDF_Parser` SHALL NOT retain the `Decryption_Password` in any instance variable or
   class-level attribute after `parse()` returns.

---

### Requirement 3: API Response Models Include the Error Code Field

**User Story:** As a backend developer, I want the Pydantic response models to carry the
`error_code` field so that the frontend receives a strongly-typed, documented contract it
can parse reliably.

#### Acceptance Criteria

1. THE `Ingest_Endpoint` SHALL return a response body that includes a nullable `error_code`
   string field; the field SHALL be absent (or `null`) for successful ingestions.

2. WHEN the `Ingest_Endpoint` returns `error_code` equal to `"PASSWORD_REQUIRED"` or
   `"PASSWORD_INCORRECT"`, THE `Ingest_Endpoint` SHALL set the HTTP status code to 422.

3. THE `Password_Required_Response` model SHALL include at minimum the fields `error_code`
   (string) and `detail` (string).

---

### Requirement 4: Frontend Detects the Password-Required Signal

**User Story:** As a user, I want the upload page to recognise when my PDF is
password-protected, so that I am prompted to enter my password rather than seeing an
unhelpful generic error.

#### Acceptance Criteria

1. WHEN the `API_Client` receives a response whose `error_code` is `"PASSWORD_REQUIRED"`,
   THE `API_Client` SHALL raise a dedicated `PasswordRequiredError` exception (not a generic
   `RuntimeError`) carrying the original filename.

2. WHEN the `API_Client` receives a response whose `error_code` is `"PASSWORD_INCORRECT"`,
   THE `API_Client` SHALL raise a dedicated `PasswordIncorrectError` exception (not a
   generic `RuntimeError`) carrying the original filename.

3. WHEN the `Upload_Page` catches a `PasswordRequiredError`, THE `Upload_Page` SHALL
   set a flag in `Session_State` indicating that a password prompt is required and SHALL
   display a `st.text_input` field labelled "PDF Password" with `type="password"` and a
   "Unlock & Upload" submit button.

4. WHEN the `Upload_Page` catches a `PasswordIncorrectError`, THE `Upload_Page` SHALL
   retain the password prompt visible and SHALL display an inline error message reading
   "Incorrect password — please try again."

5. WHILE the password prompt is visible, THE `Upload_Page` SHALL NOT show the generic
   upload error that would otherwise appear for a 422 response.

---

### Requirement 5: Frontend Re-Submits the File with the Supplied Password

**User Story:** As a user, I want to enter my PDF password once and have the upload proceed
automatically, so that I do not need to select the file again.

#### Acceptance Criteria

1. WHEN the user clicks "Unlock & Upload" and the password input is non-empty, THE
   `Upload_Page` SHALL re-submit the previously uploaded file from `Session_State`
   together with the `Decryption_Password` to the `Ingest_Endpoint` in a single request.

2. WHEN the user clicks "Unlock & Upload" and the password input is empty, THE
   `Upload_Page` SHALL display a validation message "Password cannot be empty" and SHALL NOT
   issue a request to the `Ingest_Endpoint`.

3. WHEN the `Ingest_Endpoint` accepts the re-submitted file with the correct
   `Decryption_Password` and returns a successful `Ingest_Response`, THE `Upload_Page`
   SHALL display the standard success result view and SHALL clear the password prompt and
   the `Session_State` entry for the buffered file.

4. WHEN the password-prompt flow is active, THE `Upload_Page` SHALL keep the file
   selection widget disabled so that the user cannot accidentally select a different file
   while the prompt is open.

5. THE `API_Client` SHALL accept an optional `password` parameter in `upload_statement()`
   and, when provided, SHALL include it as a `password` form field in the
   `multipart/form-data` POST request to the `Ingest_Endpoint`.

---

### Requirement 6: Password Security and Non-Persistence

**User Story:** As a security-conscious user, I want my PDF password to remain in memory
only and never be stored anywhere, so that my financial data stays protected even if the
system is compromised.

#### Acceptance Criteria

1. THE `Ingest_Endpoint` SHALL NOT write the `Decryption_Password` to any file on disk,
   including the `data/raw/` staging directory path, log files, or any database record.

2. THE `Ingest_Endpoint` SHALL NOT include the `Decryption_Password` in any log statement
   at any log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL).

3. THE `Ingest_Endpoint` SHALL NOT persist the decrypted PDF byte stream to disk; the
   decryption SHALL occur entirely in memory during the parse operation.

4. THE `Upload_Page` SHALL store the file buffer in `Session_State` using the Streamlit
   in-memory buffer only; THE `Upload_Page` SHALL NOT write the file buffer to disk during
   the password-prompt phase.

5. WHEN the `Upload_Page` completes the upload (success or unrecoverable error), THE
   `Upload_Page` SHALL remove the file buffer and any password value from `Session_State`.

6. THE `Upload_Page` SHALL render the password `st.text_input` with `type="password"` so
   that the entered characters are masked in the browser.

---

### Requirement 7: Graceful Handling of Repeated Wrong Passwords

**User Story:** As a user, I want to be able to attempt multiple passwords without restarting
the upload flow, so that I can correct typos or try alternative passwords efficiently.

#### Acceptance Criteria

1. WHEN the user submits an incorrect `Decryption_Password` and the `Ingest_Endpoint`
   returns `error_code` `"PASSWORD_INCORRECT"`, THE `Upload_Page` SHALL keep the password
   prompt visible, clear the password input field, and display the message "Incorrect
   password — please try again."

2. THE `Upload_Page` SHALL allow the user to attempt a new password up to 5 consecutive
   times within a single upload session before replacing the prompt with an unrecoverable
   error message that reads "Too many failed attempts. Please re-upload the file."

3. WHEN the unrecoverable error state is reached, THE `Upload_Page` SHALL clear the
   `Session_State` entry for the buffered file and restore the file-selection widget to its
   default enabled state.

4. WHEN the user navigates away from the `Upload_Page` and returns, THE `Upload_Page` SHALL
   reset the attempt counter and any password prompt state.

---

### Requirement 8: Non-PDF Files Are Unaffected

**User Story:** As a user uploading a CSV file, I want the upload flow to work exactly as it
did before this feature, so that the new password-handling code introduces no regressions.

#### Acceptance Criteria

1. WHEN the user uploads a CSV file, THE `Ingest_Endpoint` SHALL process the file using the
   existing `CSVParser` path and SHALL return a standard `Ingest_Response` with no
   `error_code` field.

2. WHEN the user uploads a CSV file, THE `Upload_Page` SHALL display the standard success
   or error result without triggering the password-prompt path.

3. WHEN the user uploads an unencrypted PDF file, THE `Upload_Page` SHALL display the
   standard success or error result without triggering the password-prompt path.

---

### Requirement 9: Password Field Is Transmitted Securely

**User Story:** As a security-conscious developer, I want the password to travel over the
network only under HTTPS and never appear in query strings, so that it cannot be intercepted
or logged by a proxy.

#### Acceptance Criteria

1. THE `API_Client` SHALL transmit the `Decryption_Password` exclusively as a
   `multipart/form-data` body field named `password`; THE `API_Client` SHALL NOT include the
   password as a URL path segment or a query parameter.

2. IF the `FINSIGHT_API_URL` environment variable specifies an HTTPS endpoint, THEN THE
   `API_Client` SHALL send requests over HTTPS when submitting a `Decryption_Password`.
