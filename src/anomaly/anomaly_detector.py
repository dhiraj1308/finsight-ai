from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from domain import Transaction
from ingestion.transaction_store import TransactionStore

MIN_TRANSACTIONS = 10


class AnomalyDetector:
    """
    Detects anomalous transactions using Isolation Forest.
    Uses amount and label-encoded category as features.
    No labeled anomaly data required — fully unsupervised.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self._contamination = contamination
        self._random_state = random_state

    def fit_and_score(self, store: TransactionStore) -> int:
        """
        Fits Isolation Forest on all transactions in the store.
        Updates is_anomaly and anomaly_score for every record.
        Returns count of flagged anomalies.
        Raises ValueError if store has fewer than MIN_TRANSACTIONS records.
        """
        transactions = store.get_all()

        if len(transactions) < MIN_TRANSACTIONS:
            raise ValueError(
                f"Anomaly detection requires at least {MIN_TRANSACTIONS} "
                f"transactions, but store contains {len(transactions)}."
            )

        # Build feature matrix: [amount, label_encoded_category]
        label_encoder = LabelEncoder()
        categories = [txn.category for txn in transactions]
        encoded_categories = label_encoder.fit_transform(categories)

        amounts = np.array([txn.amount for txn in transactions])
        feature_matrix = np.column_stack([amounts, encoded_categories])

        # Fit Isolation Forest
        model = IsolationForest(
            contamination=self._contamination,
            random_state=self._random_state,
        )
        model.fit(feature_matrix)

        # decision_function returns negative scores for anomalies.
        # We invert and clip to [0.0, 1.0]: higher = more anomalous.
        raw_scores = model.decision_function(feature_matrix)
        normalized_scores = np.clip(-raw_scores, 0, 1).tolist()

        # Isolation Forest predict: -1 = anomaly, 1 = normal
        predictions = model.predict(feature_matrix)

        # Update every transaction in the store
        anomaly_count = 0
        with store._get_connection() as conn:
            for txn, score, prediction in zip(
                transactions, normalized_scores, predictions
            ):
                is_anomaly = prediction == -1
                if is_anomaly:
                    anomaly_count += 1
                conn.execute(
                    """
                    UPDATE transactions
                    SET is_anomaly = ?, anomaly_score = ?
                    WHERE id = ?
                    """,
                    (int(is_anomaly), float(score), txn.id),
                )

        return anomaly_count

    def get_anomalies(self, store: TransactionStore) -> list[Transaction]:
        """
        Returns all transactions flagged as anomalies,
        ordered by anomaly_score descending (highest score first).
        """
        all_txns = store.get_all()
        anomalies = [txn for txn in all_txns if txn.is_anomaly]
        return sorted(anomalies, key=lambda t: t.anomaly_score or 0.0, reverse=True)