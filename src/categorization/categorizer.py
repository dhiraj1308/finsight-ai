from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from domain import Transaction

CANONICAL_CATEGORIES = frozenset(
    {
        "Groceries",
        "Utilities",
        "Entertainment",
        "Dining",
        "Transport",
        "Healthcare",
        "Shopping",
        "Subscriptions",
        "Other",
        "Uncategorized",
    }
)

CONFIDENCE_THRESHOLD = 0.60


class Categorizer:
    """ML-based transaction categorizer using TF-IDF + LogisticRegression."""

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._is_trained: bool = False

    def train(self, labeled_transactions: list[Transaction]) -> float:
        """
        Train classifier on labeled transactions.
        Returns weighted F1 score on held-out validation set.
        """
        if not labeled_transactions:
            raise ValueError("Cannot train on empty transaction list.")

        texts = [txn.merchant for txn in labeled_transactions]
        labels = [txn.category for txn in labeled_transactions]

        unique_categories = set(labels)
        if len(unique_categories) < 2:
            raise ValueError(
                f"Need at least 2 distinct categories to train, "
                f"got: {unique_categories}"
            )

        encoded_labels = self._label_encoder.fit_transform(labels)

        # Use stratified split only when dataset is large enough.
        # Stratified split requires test set >= n_classes samples.
        # With test_size=0.2: need total >= n_classes / 0.2 + 1.
        n_classes = len(unique_categories)
        min_samples_for_stratify = int(n_classes / 0.2) + 1
        use_stratify = len(texts) >= min_samples_for_stratify

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels if use_stratify else None,
        )

        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        max_features=10_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=5.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        self._pipeline.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self._pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="weighted")
        return float(f1)

    def predict(self, transaction: Transaction) -> Transaction:
        """
        Predict category for a single transaction.
        Sets needs_review=True if confidence < CONFIDENCE_THRESHOLD.
        """
        if not self._is_trained or self._pipeline is None:
            raise RuntimeError(
                "Categorizer must be trained or loaded before calling predict()."
            )

        try:
            proba = self._pipeline.predict_proba([transaction.merchant])[0]
            confidence = float(np.max(proba))
            predicted_index = int(np.argmax(proba))
            predicted_category = self._label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            if confidence < CONFIDENCE_THRESHOLD:
                transaction.category = "Other"
                transaction.needs_review = True
            else:
                transaction.category = predicted_category
                transaction.needs_review = False

        except Exception:
            transaction.category = "Other"
            transaction.needs_review = True

        return transaction

    def predict_batch(self, transactions: list[Transaction]) -> list[Transaction]:
        """
        Predict categories for a list of transactions.
        Individual prediction errors set category='Other', needs_review=True
        without aborting the batch.
        """
        return [self.predict(txn) for txn in transactions]

    def save(self, path: Path) -> None:
        """Save trained pipeline and label encoder to disk."""
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained Categorizer.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "label_encoder": self._label_encoder,
                "is_trained": True,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load trained pipeline and label encoder from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at: {path}")
        saved = joblib.load(path)
        self._pipeline = saved["pipeline"]
        self._label_encoder = saved["label_encoder"]
        if self._pipeline is not None and self._label_encoder is not None:
            self._is_trained = True
        else:
            raise RuntimeError(
                "Loaded model file is missing pipeline or label encoder."
            )