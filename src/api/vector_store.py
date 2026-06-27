from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from domain import Transaction

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_FILE = "vector_store.npy"
METADATA_FILE = "vector_store_metadata.json"


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_path = self._persist_dir / COLLECTION_FILE
        self._metadata_path = self._persist_dir / METADATA_FILE
        self._model = SentenceTransformer(embedding_model_name)
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: list[dict] = []
        self._load()

    def _transaction_text(self, transaction: Transaction) -> str:
        return f"{transaction.merchant} {transaction.category} {transaction.amount} {transaction.date}"

    def _save(self) -> None:
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(str(self._collection_path), self._embeddings)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f)

    def _load(self) -> None:
        if self._collection_path.exists() and self._metadata_path.exists():
            try:
                self._embeddings = np.load(str(self._collection_path))
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
            except Exception as e:
                self._embeddings = None
                self._metadata = []

    def _find_index(self, transaction_id: int) -> int:
        for i, meta in enumerate(self._metadata):
            if meta.get("transaction_id") == transaction_id:
                return i
        return -1

    def index(self, transaction: Transaction) -> None:
        text = self._transaction_text(transaction)
        embedding = self._model.encode([text])[0].astype(np.float32)
        existing_index = self._find_index(transaction.id)
        metadata_entry = {
            "transaction_id": transaction.id,
            "date": str(transaction.date),
            "merchant": transaction.merchant,
            "amount": transaction.amount,
            "category": transaction.category,
        }
        if existing_index >= 0:
            self._embeddings[existing_index] = embedding
            self._metadata[existing_index] = metadata_entry
        else:
            if self._embeddings is None or len(self._embeddings) == 0:
                self._embeddings = embedding.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, embedding])
            self._metadata.append(metadata_entry)
        self._save()

    def search(self, query: str, k: int) -> list[Transaction]:
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        query_embedding = self._model.encode([query])[0].astype(np.float32)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = self._embeddings / norms
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-10
        normalized_query = query_embedding / query_norm
        similarities = normalized @ normalized_query
        actual_k = min(k, len(self._metadata))
        top_indices = np.argsort(similarities)[::-1][:actual_k]
        results = []
        for idx in top_indices:
            meta = self._metadata[idx]
            from datetime import date
            try:
                txn_date = date.fromisoformat(meta["date"])
            except (ValueError, KeyError):
                txn_date = date.today()
            results.append(Transaction(
                id=meta.get("transaction_id"),
                date=txn_date,
                merchant=meta.get("merchant", ""),
                amount=meta.get("amount", 0.0),
                category=meta.get("category", ""),
            ))
        return results

    def delete(self, transaction_id: int) -> None:
        index = self._find_index(transaction_id)
        if index < 0:
            return
        self._metadata.pop(index)
        if self._embeddings is not None and len(self._embeddings) > 0:
            self._embeddings = np.delete(self._embeddings, index, axis=0)
            if len(self._embeddings) == 0:
                self._embeddings = None
        self._save()

    @property
    def count(self) -> int:
        return len(self._metadata)
