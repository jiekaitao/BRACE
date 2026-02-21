"""TDD tests for VectorActivityClassifier — activity classification via vector search."""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure fake cortex module
_fake_cortex = types.ModuleType("cortex")


class _FakeCortexClient:
    def __init__(self, host="localhost", port=5555):
        pass
    def create_collection(self, name, dimension, description=""):
        pass
    def insert(self, collection, vectors, metadata=None):
        pass
    def search(self, collection, query, top_k=5, filters=None):
        return []
    def health(self):
        return {"status": "ok"}


_fake_cortex.CortexClient = _FakeCortexClient
sys.modules.setdefault("cortex", _fake_cortex)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from vectorai_store import VectorAIStore  # noqa: E402
from vector_activity_classifier import VectorActivityClassifier  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_store() -> tuple[VectorAIStore, MagicMock]:
    client = MagicMock()
    client.health.return_value = {"status": "ok"}
    client.search.return_value = []
    store = VectorAIStore.__new__(VectorAIStore)
    store._client = client
    store._available = True
    store._host = "localhost"
    store._port = 5555
    return store, client


def _make_features(dim: int = 42, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    feat = rng.randn(dim).astype(np.float32)
    return feat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVectorActivityClassifier:
    def test_classify_known_activity(self):
        """When a matching template exists with high confidence, return the activity label."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)

        client.search.return_value = [
            {"id": "tpl_1", "score": 0.92, "metadata": {"activity_name": "squat", "source": "labeled_data"}},
        ]

        features = _make_features(42, seed=10)
        result = classifier.classify(features)
        assert result == "squat"

    def test_classify_unknown_returns_none(self):
        """When no template matches above threshold, return None."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)

        client.search.return_value = [
            {"id": "tpl_1", "score": 0.55, "metadata": {"activity_name": "squat"}},
        ]

        features = _make_features(42, seed=20)
        result = classifier.classify(features)
        assert result is None

    def test_classify_empty_db_returns_none(self):
        """When DB is empty, return None."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)
        client.search.return_value = []

        features = _make_features(42, seed=30)
        result = classifier.classify(features)
        assert result is None

    def test_seed_templates_from_labeled_data(self):
        """seed_templates() should bulk-insert labeled activity vectors."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)

        labeled_segments = [
            {"features": _make_features(42, seed=i), "activity_name": f"activity_{i}"}
            for i in range(10)
        ]

        classifier.seed_templates(labeled_segments)
        assert client.insert.call_count == 10

    def test_latency_under_10ms(self):
        """Vector classification should be very fast (mock client = near-zero latency)."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)

        client.search.return_value = [
            {"id": "tpl_1", "score": 0.92, "metadata": {"activity_name": "squat"}},
        ]

        features = _make_features(42, seed=40)

        t0 = time.monotonic()
        for _ in range(100):
            classifier.classify(features)
        elapsed = (time.monotonic() - t0) / 100

        # Each call should be well under 10ms with a mock client
        assert elapsed < 0.010, f"Average latency {elapsed*1000:.2f}ms exceeds 10ms"

    def test_fallback_to_gemini_when_no_match(self):
        """When VectorAI returns no match, classifier returns None (caller falls back to Gemini)."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store)
        client.search.return_value = []

        features = _make_features(42, seed=50)
        result = classifier.classify(features)
        assert result is None  # Caller responsible for Gemini fallback

    def test_classify_with_custom_threshold(self):
        """Classifier respects a custom threshold."""
        store, client = _mock_store()
        classifier = VectorActivityClassifier(store, threshold=0.90)

        client.search.return_value = [
            {"id": "tpl_1", "score": 0.88, "metadata": {"activity_name": "lunge"}},
        ]

        features = _make_features(42, seed=60)
        result = classifier.classify(features)
        assert result is None  # 0.88 < 0.90

    def test_classify_when_store_unavailable(self):
        """When store is unavailable, classifier returns None without crashing."""
        store = VectorAIStore.__new__(VectorAIStore)
        store._client = None
        store._available = False
        store._host = "localhost"
        store._port = 5555

        classifier = VectorActivityClassifier(store)
        features = _make_features(42, seed=70)
        result = classifier.classify(features)
        assert result is None
