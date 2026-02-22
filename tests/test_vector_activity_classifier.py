"""Tests for VectorAI-based fast-path activity classifier."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from vector_activity_classifier import VectorActivityClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_store(search_results=None):
    """Create a mock VectorAI store."""
    store = MagicMock()
    store.find_similar_movements.return_value = search_results or []
    store.health_check.return_value = True
    return store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVectorActivityClassifier:
    def test_classify_known(self):
        """Returns activity label from closest matching template."""
        store = _make_mock_store(search_results=[
            {"score": 0.95, "metadata": {"activity_label": "basketball shooting"}},
            {"score": 0.80, "metadata": {"activity_label": "basketball dribbling"}},
        ])
        clf = VectorActivityClassifier(store, threshold=0.8)
        features = np.random.randn(28).astype(np.float32)
        label = clf.classify(features)
        assert label == "basketball shooting"

    def test_classify_unknown_below_threshold(self):
        """Returns 'unknown' when best match is below threshold."""
        store = _make_mock_store(search_results=[
            {"score": 0.3, "metadata": {"activity_label": "squat"}},
        ])
        clf = VectorActivityClassifier(store, threshold=0.8)
        features = np.random.randn(28).astype(np.float32)
        label = clf.classify(features)
        assert label == "unknown"

    def test_classify_empty_results(self):
        """Returns 'unknown' when no results found."""
        store = _make_mock_store(search_results=[])
        clf = VectorActivityClassifier(store, threshold=0.8)
        features = np.random.randn(28).astype(np.float32)
        label = clf.classify(features)
        assert label == "unknown"

    def test_classify_store_none(self):
        """Returns 'unknown' when store is None (graceful degradation)."""
        clf = VectorActivityClassifier(None, threshold=0.8)
        features = np.random.randn(28).astype(np.float32)
        label = clf.classify(features)
        assert label == "unknown"

    def test_seed_templates(self):
        """seed_templates stores feature vectors with labels."""
        store = _make_mock_store()
        clf = VectorActivityClassifier(store, threshold=0.8)
        templates = {
            "basketball shooting": np.random.randn(28).astype(np.float32),
            "basketball dribbling": np.random.randn(28).astype(np.float32),
        }
        clf.seed_templates(templates)
        assert store.store_movement_segment.call_count == 2

    def test_seed_templates_no_store(self):
        """seed_templates is no-op when store is None."""
        clf = VectorActivityClassifier(None)
        templates = {"squat": np.random.randn(28).astype(np.float32)}
        clf.seed_templates(templates)  # should not raise

    def test_classify_exception_graceful(self):
        """API exception returns 'unknown'."""
        store = _make_mock_store()
        store.find_similar_movements.side_effect = RuntimeError("connection lost")
        clf = VectorActivityClassifier(store, threshold=0.8)
        features = np.random.randn(28).astype(np.float32)
        label = clf.classify(features)
        assert label == "unknown"

    def test_available_property(self):
        """available is True when store exists and is healthy."""
        store = _make_mock_store()
        clf = VectorActivityClassifier(store)
        assert clf.available is True

        clf_none = VectorActivityClassifier(None)
        assert clf_none.available is False
