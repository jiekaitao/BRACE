"""TDD tests for MovementSearchEngine — movement similarity search across sessions."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

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
from vector_movement_search import MovementSearchEngine  # noqa: E402


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
    return rng.randn(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMovementSearchEngine:
    def test_store_movement_segment(self):
        """store_segment() should insert vector + metadata into motion_segments."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        features = _make_features(42, seed=1)
        engine.store_segment(
            features,
            metadata={
                "activity_label": "squat",
                "session_id": "sess_001",
                "person_id": "S1",
                "risk_score": 0.2,
            },
        )

        client.insert.assert_called_once()
        call_args = client.insert.call_args
        assert call_args.args[0] == "motion_segments"

    def test_find_similar_movements_same_activity(self):
        """find_similar() returns movements of the same activity type."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        client.search.return_value = [
            {
                "id": "seg_1",
                "score": 0.95,
                "metadata": {
                    "activity_label": "squat",
                    "session_id": "sess_001",
                    "person_id": "S1",
                    "risk_score": 0.2,
                },
            },
            {
                "id": "seg_2",
                "score": 0.90,
                "metadata": {
                    "activity_label": "squat",
                    "session_id": "sess_001",
                    "person_id": "S1",
                    "risk_score": 0.3,
                },
            },
        ]

        features = _make_features(42, seed=2)
        results = engine.find_similar(features, top_k=5)
        assert len(results) == 2
        assert all(r["metadata"]["activity_label"] == "squat" for r in results)

    def test_find_similar_movements_cross_session(self):
        """find_similar() returns movements from different sessions."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        client.search.return_value = [
            {
                "id": "seg_1",
                "score": 0.92,
                "metadata": {"activity_label": "lunge", "session_id": "sess_001", "person_id": "S1"},
            },
            {
                "id": "seg_2",
                "score": 0.87,
                "metadata": {"activity_label": "lunge", "session_id": "sess_002", "person_id": "S1"},
            },
        ]

        features = _make_features(42, seed=3)
        results = engine.find_similar(features, top_k=5)
        sessions = {r["metadata"]["session_id"] for r in results}
        assert len(sessions) == 2

    def test_filter_by_person_id(self):
        """find_similar() passes person_id filter to the store."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        features = _make_features(42, seed=4)
        engine.find_similar(features, top_k=5, filters={"person_id": "S1"})

        call_args = client.search.call_args
        filters_passed = call_args.kwargs.get("filters")
        assert filters_passed == {"person_id": "S1"}

    def test_filter_by_activity_label(self):
        """find_similar() passes activity_label filter to the store."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        features = _make_features(42, seed=5)
        engine.find_similar(features, top_k=5, filters={"activity_label": "squat"})

        call_args = client.search.call_args
        filters_passed = call_args.kwargs.get("filters")
        assert filters_passed == {"activity_label": "squat"}

    def test_empty_results_for_novel_movement(self):
        """A completely novel movement returns empty results."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)
        client.search.return_value = []

        features = _make_features(42, seed=999)
        results = engine.find_similar(features, top_k=5)
        assert results == []

    def test_store_segment_includes_timestamp(self):
        """store_segment() should include a timestamp in metadata."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        features = _make_features(42, seed=6)
        engine.store_segment(
            features,
            metadata={
                "activity_label": "deadlift",
                "session_id": "sess_003",
                "person_id": "S2",
            },
        )

        call_args = client.insert.call_args
        meta = call_args.kwargs.get("metadata") or call_args.args[2]
        assert "timestamp" in meta[0]

    def test_find_similar_sorted_by_score(self):
        """Results should be sorted by descending similarity score."""
        store, client = _mock_store()
        engine = MovementSearchEngine(store)

        client.search.return_value = [
            {"id": "s1", "score": 0.95, "metadata": {"activity_label": "squat"}},
            {"id": "s2", "score": 0.80, "metadata": {"activity_label": "squat"}},
            {"id": "s3", "score": 0.70, "metadata": {"activity_label": "lunge"}},
        ]

        features = _make_features(42, seed=7)
        results = engine.find_similar(features, top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_store_unavailable_returns_empty(self):
        """When store is unavailable, find_similar returns empty list."""
        store = VectorAIStore.__new__(VectorAIStore)
        store._client = None
        store._available = False
        store._host = "localhost"
        store._port = 5555

        engine = MovementSearchEngine(store)
        features = _make_features(42, seed=8)
        results = engine.find_similar(features, top_k=5)
        assert results == []
