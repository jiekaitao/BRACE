"""Movement similarity search across sessions via VectorAI.

Stores motion segment feature vectors and retrieves similar past movements
for form comparison, progress tracking, and injury risk pattern matching.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np


class MovementSearchEngine:
    """Search engine for finding similar past motion segments.

    Usage::

        engine = MovementSearchEngine(store)
        engine.store_segment(features, {"activity_label": "squat", ...})
        similar = engine.find_similar(features, top_k=5)
    """

    def __init__(self, store):
        """
        Args:
            store: A VectorAIStore instance.
        """
        self._store = store

    def store_segment(
        self,
        features: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """Store a motion segment's feature vector with metadata.

        Args:
            features: (42,) SRP feature vector for the motion segment.
            metadata: Dict with keys like activity_label, session_id,
                      person_id, risk_score. A timestamp is added automatically.
        """
        meta = dict(metadata)
        if "timestamp" not in meta:
            meta["timestamp"] = time.time()

        self._store.store_motion_segment(
            features=features,
            activity_label=meta.get("activity_label", "unknown"),
            session_id=meta.get("session_id", ""),
            person_id=meta.get("person_id", ""),
            risk_score=meta.get("risk_score", 0.0),
            timestamp=meta.get("timestamp"),
        )

    def find_similar(
        self,
        features: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Find similar past motion segments by vector similarity.

        Args:
            features: (42,) SRP feature vector to query against.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters (e.g. {"person_id": "S1"}).

        Returns:
            List of {"score": float, "metadata": dict} sorted by descending
            similarity.
        """
        return self._store.find_similar_movements(
            query_features=features,
            top_k=top_k,
            filters=filters,
        )
