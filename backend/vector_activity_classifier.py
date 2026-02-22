"""VectorAI-based fast-path activity classifier.

Uses semantic similarity search against stored activity templates
to classify movement segments without calling the Gemini API.
Gracefully degrades if VectorAI is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VectorActivityClassifier:
    """Classify activities by nearest-neighbor search in VectorAI."""

    def __init__(self, store: Any | None = None, threshold: float = 0.75):
        self._store = store
        self._threshold = threshold

    @property
    def available(self) -> bool:
        """True if the VectorAI store exists and is reachable."""
        if self._store is None:
            return False
        try:
            return self._store.health_check()
        except Exception:
            return False

    def classify(self, features: np.ndarray, top_k: int = 1) -> str:
        """Classify a feature vector by nearest-neighbor search.

        Args:
            features: SRP feature vector (28D or 42D).
            top_k: Number of nearest neighbors to consider.

        Returns:
            Activity label string, or "unknown" if no confident match.
        """
        if self._store is None:
            return "unknown"

        try:
            results = self._store.find_similar_movements(
                features, top_k=top_k,
            )
        except Exception:
            return "unknown"

        if not results:
            return "unknown"

        best = results[0]
        score = best.get("score", 0.0)
        if score < self._threshold:
            return "unknown"

        return best.get("metadata", {}).get("activity_label", "unknown")

    def seed_templates(self, templates: dict[str, np.ndarray]) -> None:
        """Store activity template vectors for future classification.

        Args:
            templates: Mapping of activity_label -> feature vector.
        """
        if self._store is None:
            return

        for label, features in templates.items():
            try:
                self._store.store_movement_segment(
                    features,
                    metadata={"activity_label": label, "is_template": True},
                )
            except Exception as e:
                print(f"[vector-clf] seed failed for {label}: {e}", flush=True)
