"""Activity classification via vector similarity search (replaces Gemini for known activities).

Uses VectorAIStore to search activity_templates collection for the nearest
labeled reference movement. Falls back to None (caller should use Gemini) when
no match is found above the confidence threshold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VectorActivityClassifier:
    """Classify physical activities by matching motion features to stored templates.

    Usage::

        classifier = VectorActivityClassifier(store, threshold=0.80)
        label = classifier.classify(motion_features)
        if label is None:
            label = gemini_classifier.classify_activity(...)  # fallback
    """

    def __init__(self, store, threshold: float = 0.80):
        """
        Args:
            store: A VectorAIStore instance.
            threshold: Minimum cosine similarity for a confident classification.
        """
        self._store = store
        self._threshold = threshold

    def classify(self, motion_features: np.ndarray) -> str | None:
        """Classify a motion segment by vector similarity to stored templates.

        Args:
            motion_features: (42,) SRP feature vector for the motion segment.

        Returns:
            Activity label string if a match is found above threshold, else None.
        """
        label, confidence = self._store.classify_activity(
            motion_features, threshold=self._threshold
        )
        return label

    def seed_templates(self, labeled_segments: list[dict]) -> None:
        """Bulk-insert labeled activity templates into VectorAI.

        Args:
            labeled_segments: List of dicts with keys:
                - "features": np.ndarray (42,) SRP feature vector
                - "activity_name": str label (e.g. "squat", "lunge")
                - "source": str (optional, default "labeled_data")
        """
        for seg in labeled_segments:
            self._store.store_activity_template(
                features=seg["features"],
                activity_name=seg["activity_name"],
                source=seg.get("source", "labeled_data"),
            )
