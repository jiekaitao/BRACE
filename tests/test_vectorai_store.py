"""Unit tests for VectorAIStore — mocked CortexClient (no real DB needed)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Provide a fake 'cortex' module so import succeeds without pip install
# ---------------------------------------------------------------------------
_fake_cortex = types.ModuleType("cortex")


class _FakeCortexClient:
    """Minimal stub matching the CortexClient interface."""

    def __init__(self, host="localhost", port=5555):
        self.host = host
        self.port = port

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

# Now import the module under test
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "backend"))
from vectorai_store import VectorAIStore  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Return a MagicMock pretending to be CortexClient."""
    client = MagicMock()
    client.health.return_value = {"status": "ok"}
    client.search.return_value = []
    return client


@pytest.fixture
def store(mock_client):
    """VectorAIStore with an injected mock client (skips real connection)."""
    s = VectorAIStore.__new__(VectorAIStore)
    s._client = mock_client
    s._available = True
    s._host = "localhost"
    s._port = 5555
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectAndCreateCollections:
    def test_creates_three_collections(self, mock_client):
        """On init, store should create person_embeddings, motion_segments, activity_templates."""
        with patch("vectorai_store._GRPC_AVAILABLE", False), \
             patch("vectorai_store.CortexClient", return_value=mock_client):
            store = VectorAIStore(host="localhost", port=5555)

        calls = mock_client.create_collection.call_args_list
        names = {c.args[0] if c.args else c.kwargs.get("name") for c in calls}
        assert "person_embeddings" in names
        assert "motion_segments" in names
        assert "activity_templates" in names

    def test_health_check_ok(self, store, mock_client):
        assert store.health_check() is True
        mock_client.health.assert_called_once()

    def test_health_check_failure(self, store, mock_client):
        mock_client.health.side_effect = Exception("connection refused")
        assert store.health_check() is False


class TestPersonEmbeddings:
    def test_insert_person_embedding(self, store, mock_client):
        emb = np.random.randn(768).astype(np.float32)
        store.store_person_embedding(emb, person_id="athlete_1", session_id="sess_001")

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args.args[0] == "person_embeddings" or call_args.kwargs.get("collection") == "person_embeddings"

    def test_search_person_by_embedding(self, store, mock_client):
        query = np.random.randn(768).astype(np.float32)
        mock_client.search.return_value = [
            {"id": "vec_1", "score": 0.92, "metadata": {"person_id": "athlete_1", "session_id": "sess_001"}},
        ]

        result = store.find_person(query, threshold=0.85)
        assert result is not None
        assert result["person_id"] == "athlete_1"
        assert result["score"] >= 0.85

    def test_search_person_no_match(self, store, mock_client):
        query = np.random.randn(768).astype(np.float32)
        mock_client.search.return_value = [
            {"id": "vec_1", "score": 0.50, "metadata": {"person_id": "athlete_2"}},
        ]

        result = store.find_person(query, threshold=0.85)
        assert result is None

    def test_search_person_empty_db(self, store, mock_client):
        query = np.random.randn(768).astype(np.float32)
        mock_client.search.return_value = []

        result = store.find_person(query, threshold=0.85)
        assert result is None


class TestMotionSegments:
    def test_insert_motion_segment(self, store, mock_client):
        features = np.random.randn(42).astype(np.float32)
        store.store_motion_segment(
            features,
            activity_label="squat",
            session_id="sess_001",
            person_id="athlete_1",
            risk_score=0.3,
        )

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args.args[0] == "motion_segments" or call_args.kwargs.get("collection") == "motion_segments"

    def test_search_similar_movements(self, store, mock_client):
        query = np.random.randn(42).astype(np.float32)
        mock_client.search.return_value = [
            {
                "id": "seg_1",
                "score": 0.95,
                "metadata": {
                    "activity_label": "squat",
                    "session_id": "sess_001",
                    "person_id": "athlete_1",
                    "risk_score": 0.2,
                },
            },
            {
                "id": "seg_2",
                "score": 0.88,
                "metadata": {
                    "activity_label": "squat",
                    "session_id": "sess_002",
                    "person_id": "athlete_1",
                    "risk_score": 0.4,
                },
            },
        ]

        results = store.find_similar_movements(query, top_k=5)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    def test_search_similar_movements_with_filters(self, store, mock_client):
        query = np.random.randn(42).astype(np.float32)
        mock_client.search.return_value = []

        store.find_similar_movements(query, top_k=3, filters={"person_id": "athlete_1"})
        call_args = mock_client.search.call_args
        # Verify filters were passed through
        assert call_args.kwargs.get("filters") == {"person_id": "athlete_1"} or \
            (len(call_args.args) > 3 and call_args.args[3] == {"person_id": "athlete_1"})


class TestActivityTemplates:
    def test_insert_activity_template(self, store, mock_client):
        features = np.random.randn(42).astype(np.float32)
        store.store_activity_template(features, activity_name="squat", source="labeled_data")

        mock_client.insert.assert_called_once()
        call_args = mock_client.insert.call_args
        assert call_args.args[0] == "activity_templates" or call_args.kwargs.get("collection") == "activity_templates"

    def test_classify_activity_by_vector(self, store, mock_client):
        features = np.random.randn(42).astype(np.float32)
        mock_client.search.return_value = [
            {"id": "tpl_1", "score": 0.92, "metadata": {"activity_name": "squat", "source": "labeled_data"}},
        ]

        label, confidence = store.classify_activity(features, threshold=0.80)
        assert label == "squat"
        assert confidence >= 0.80

    def test_classify_activity_no_match(self, store, mock_client):
        features = np.random.randn(42).astype(np.float32)
        mock_client.search.return_value = [
            {"id": "tpl_1", "score": 0.60, "metadata": {"activity_name": "squat"}},
        ]

        label, confidence = store.classify_activity(features, threshold=0.80)
        assert label is None
        assert confidence < 0.80

    def test_classify_activity_empty_db(self, store, mock_client):
        features = np.random.randn(42).astype(np.float32)
        mock_client.search.return_value = []

        label, confidence = store.classify_activity(features, threshold=0.80)
        assert label is None


class TestGracefulFallback:
    def test_fallback_when_unavailable(self):
        """When VectorAI connection fails, all operations return None gracefully."""
        with patch("vectorai_store.CortexClient", side_effect=Exception("connection refused")):
            store = VectorAIStore(host="bad-host", port=9999)

        assert store._available is False

        # All operations should return None / empty without crashing
        emb = np.random.randn(768).astype(np.float32)
        store.store_person_embedding(emb, person_id="test", session_id="s1")
        assert store.find_person(emb) is None

        feat = np.random.randn(42).astype(np.float32)
        store.store_motion_segment(feat, activity_label="squat", session_id="s1", person_id="p1")
        assert store.find_similar_movements(feat) == []

        store.store_activity_template(feat, activity_name="squat")
        label, conf = store.classify_activity(feat)
        assert label is None

        assert store.health_check() is False

    def test_store_degrades_on_insert_error(self, store, mock_client):
        """If insert raises, store should not crash."""
        mock_client.insert.side_effect = Exception("write timeout")
        emb = np.random.randn(768).astype(np.float32)
        # Should not raise
        store.store_person_embedding(emb, person_id="test", session_id="s1")

    def test_store_degrades_on_search_error(self, store, mock_client):
        """If search raises, return None/empty."""
        mock_client.search.side_effect = Exception("read timeout")
        emb = np.random.randn(768).astype(np.float32)
        assert store.find_person(emb) is None
        assert store.find_similar_movements(np.random.randn(42).astype(np.float32)) == []
        label, _ = store.classify_activity(np.random.randn(42).astype(np.float32))
        assert label is None
