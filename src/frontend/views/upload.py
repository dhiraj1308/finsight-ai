"""Upload page — ingest a CSV or PDF bank statement into FinSight AI."""
from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.services.api import APIClient, PasswordIncorrectError, PasswordRequiredError
from frontend.utils import page_header

_ACCEPTED_TYPES = ["csv", "pdf"]
_MAX_PASSWORD_ATTEMPTS = 5


def _clear_password_state() -> None:
    """Remove all password-prompt keys from session state."""
    for key in ("pdf_bytes", "pdf_name", "pw_prompt", "pw_attempts"):
        st.session_state.pop(key, None)


def _show_result(result: dict[str, Any]) -> None:
    """Render the ingest response as metric cards with optional warnings.

    Displays every field returned by the backend.  Fields not present in the
    response are silently omitted so the page remains forward-compatible if
    the backend is extended.

    Parameters
    ----------
    result:
        Parsed ``IngestResponse`` payload from ``APIClient.upload_statement``.
    """
    st.success("✅ Statement uploaded successfully.")

    ingested: int = result.get("ingested", 0)
    skipped: int = result.get("skipped", 0)
    categories: int | None = result.get("categories_created")
    anomalies: int | None = result.get("anomalies_detected")

    # Build only the metric columns that have data
    metrics: list[tuple[str, int]] = [
        ("Transactions Ingested", ingested),
        ("Transactions Skipped", skipped),
    ]
    if categories is not None:
        metrics.append(("Categories Created", categories))
    if anomalies is not None:
        metrics.append(("Anomalies Detected", anomalies))

    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    warnings: list[str] = result.get("warnings", [])
    if warnings:
        with st.expander(f"⚠️ Warnings ({len(warnings)})"):
            for warning in warnings:
                st.warning(warning)


def render(client: APIClient) -> None:
    """Render the Upload Statement page."""
    page_header(
        "📤 Upload Statement",
        subtitle="Upload bank statements for analysis",
    )

    pw_prompt_active = st.session_state.get("pw_prompt", False)

    # ------------------------------------------------------------------
    # BRANCH A: Normal upload (no active password prompt)
    # ------------------------------------------------------------------
    if not pw_prompt_active:
        uploaded_file = st.file_uploader(
            "Select a statement file",
            type=_ACCEPTED_TYPES,
            help="Accepted formats: CSV, PDF",
        )

        if uploaded_file is not None:
            st.caption(f"Selected: **{uploaded_file.name}**")

        if st.button(
            "Upload",
            disabled=uploaded_file is None,
            use_container_width=True,
        ):
            raw_bytes = bytes(uploaded_file.getbuffer())
            try:
                with st.spinner(f"Uploading {uploaded_file.name}…"):
                    result = client.upload_statement(
                        file_bytes=raw_bytes,
                        filename=uploaded_file.name,
                    )
                _show_result(result)
            except PasswordRequiredError:
                st.session_state["pdf_bytes"] = raw_bytes
                st.session_state["pdf_name"] = uploaded_file.name
                st.session_state["pw_prompt"] = True
                st.session_state["pw_attempts"] = 0
                st.rerun()
            except RuntimeError as exc:
                st.error(f"Upload failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    # ------------------------------------------------------------------
    # BRANCH B: Password prompt active
    # ------------------------------------------------------------------
    else:
        pdf_name = st.session_state.get("pdf_name", "your PDF")
        st.info(f"🔒 **{pdf_name}** is password-protected. Enter the password to continue.")

        # Keep file uploader visible but disabled
        st.file_uploader(
            "Select a statement file",
            type=_ACCEPTED_TYPES,
            disabled=True,
            help="Accepted formats: CSV, PDF",
        )

        password = st.text_input("PDF Password", type="password", key="pw_input")

        if st.button("Unlock & Upload", use_container_width=True):
            if not password:
                st.warning("Password cannot be empty")
            else:
                try:
                    with st.spinner("Unlocking and uploading…"):
                        result = client.upload_statement(
                            file_bytes=st.session_state["pdf_bytes"],
                            filename=st.session_state["pdf_name"],
                            password=password,
                        )
                    _show_result(result)
                    _clear_password_state()
                except PasswordIncorrectError:
                    st.session_state["pw_attempts"] = (
                        st.session_state.get("pw_attempts", 0) + 1
                    )
                    if st.session_state["pw_attempts"] >= _MAX_PASSWORD_ATTEMPTS:
                        st.error(
                            "Too many failed attempts. Please re-upload the file."
                        )
                        _clear_password_state()
                        st.rerun()
                    else:
                        attempts_left = (
                            _MAX_PASSWORD_ATTEMPTS - st.session_state["pw_attempts"]
                        )
                        st.error(
                            f"Incorrect password — please try again."
                            f" ({attempts_left} attempt(s) remaining)"
                        )
                except RuntimeError as exc:
                    st.error(f"Upload failed: {exc}")
                    _clear_password_state()
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)
                    _clear_password_state()
