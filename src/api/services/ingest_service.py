"""Ingest service — post-parse business pipeline for statement ingestion.

This module contains the shared orchestration that runs after a CSV or PDF
has already been parsed into Transaction objects.  It is deliberately free of
FastAPI/HTTP concerns so it can be tested directly without a TestClient.

Responsibilities:
  1. Assign source_file to every parsed transaction.
  2. Categorize transactions (if the categorizer is trained).
  3. Insert transactions into the SQLite store (deduplication is handled
     by TransactionStore's unique index).
  4. Incrementally index only newly-inserted transactions into the vector
     store (skip-if-already-indexed logic).
  5. Run anomaly detection when the total transaction count meets the
     minimum threshold.
  6. Build and return an IngestResponse.

Each step gracefully handles its own failure modes:
  - Vector indexing failures are logged as warnings; ingest still succeeds.
  - Anomaly detection failures are logged as warnings; anomalies_detected
    is returned as None rather than propagating an error.
"""
from __future__ import annotations

import logging
from typing import List

from api.models import IngestResponse
from domain import Transaction
from ingestion.csv_parser import ParseSummary

logger = logging.getLogger(__name__)

# Minimum total stored transactions required to run anomaly detection.
# Must match AnomalyDetector.MIN_TRANSACTIONS (currently 10).
_ANOMALY_MIN_TRANSACTIONS = 10


class IngestService:
    """Orchestrates the post-parse ingestion pipeline.

    Instances are stateless with respect to individual ingest operations —
    all mutable state lives in the components passed at construction time.
    """

    def __init__(self, store, vector_store, categorizer, anomaly_detector):
        self._store = store
        self._vector_store = vector_store
        self._categorizer = categorizer
        self._anomaly_detector = anomaly_detector

    def process(
        self,
        transactions: List[Transaction],
        source_filename: str,
        parse_warnings: List[str],
    ) -> IngestResponse:
        """Run the shared post-parse pipeline and return an IngestResponse.

        Parameters
        ----------
        transactions:
            The list of Transaction objects produced by a CSV or PDF parser.
            May be empty (e.g., when parsing produced no rows).
        source_filename:
            The original uploaded filename.  Assigned to every transaction's
            ``source_file`` field and used as the deduplication key component.
        parse_warnings:
            Per-row parse warnings from ParseSummary.warnings.
            Capped to 10 in the response.

        Returns
        -------
        IngestResponse
            Contains inserted/skipped counts, capped warnings, and optional
            anomaly and needs-review counts.
        """
        # ── 1. Source-file assignment ──────────────────────────────────────
        for txn in transactions:
            txn.source_file = source_filename

        # ── 2. Categorization ──────────────────────────────────────────────
        if self._categorizer._is_trained:
            transactions = self._categorizer.predict_batch(transactions)
            needs_review_count: int | None = sum(
                1 for t in transactions if t.needs_review
            )
        else:
            needs_review_count = None

        # ── 3. Database insertion ──────────────────────────────────────────
        inserted, skipped = self._store.insert(transactions)

        # ── 4. Incremental vector indexing ─────────────────────────────────
        # Only index transactions that are not already in the vector store.
        # Fetching all stored IDs before the loop prevents re-embedding the
        # entire database on every upload.
        already_indexed: frozenset[int] = self._vector_store.indexed_ids
        all_txns = self._store.get_all()
        for txn in all_txns:
            if txn.id is not None and txn.id not in already_indexed:
                try:
                    self._vector_store.index(txn)
                except Exception as exc:
                    logger.warning(f"Vector indexing failed for {txn.id}: {exc}")

        # ── 5. Anomaly detection ───────────────────────────────────────────
        try:
            if len(all_txns) >= _ANOMALY_MIN_TRANSACTIONS:
                anomaly_count: int | None = self._anomaly_detector.fit_and_score(
                    self._store
                )
            else:
                anomaly_count = None
        except Exception as exc:
            anomaly_count = None
            logger.warning(f"Anomaly detection failed: {exc}")

        # ── 6. Response ────────────────────────────────────────────────────
        return IngestResponse(
            ingested=inserted,
            skipped=skipped,
            warnings=parse_warnings[:10],
            anomalies_detected=anomaly_count,
            needs_review_count=needs_review_count,
        )
