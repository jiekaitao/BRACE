"""Integration tests for VectorAI cross-session person re-ID.

These tests use a mocked VectorAIStore to verify the integration between
IdentityResolver and VectorAI without requiring a running VectorAI container.
For real-DB tests, start the VectorAI container and set VECTORAI_HOST.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure fake cortex module is available
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(dim: int = 768, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def _mock_store() -> tuple[VectorAIStore, MagicMock]:
    """Create a VectorAIStore with a mock client."""
    client = MagicMock()
    client.health.return_value = {"status": "ok"}
    client.search.return_value = []

    store = VectorAIStore.__new__(VectorAIStore)
    store._client = client
    store._available = True
    store._host = "localhost"
    store._port = 5555
    return store, client


# ---------------------------------------------------------------------------
# Tests — Cross-session Person Re-ID
# ---------------------------------------------------------------------------


class TestPersonReID:
    def test_person_stored_after_identification(self):
        """After identifying a person, their embedding is stored in VectorAI."""
        store, client = _mock_store()
        emb = _make_embedding(768, seed=42)

        store.store_person_embedding(emb, person_id="S1", session_id="sess_001")

        client.insert.assert_called_once()
        call_args = client.insert.call_args
        assert call_args.args[0] == "person_embeddings"
        meta = call_args.kwargs.get("metadata") or call_args.args[2]
        assert meta[0]["person_id"] == "S1"
        assert meta[0]["session_id"] == "sess_001"

    def test_person_recognized_across_sessions(self):
        """A person stored in session 1 can be found in session 2."""
        store, client = _mock_store()
        emb_session1 = _make_embedding(768, seed=42)
        emb_session2 = emb_session1 + np.random.randn(768).astype(np.float32) * 0.05
        emb_session2 /= np.linalg.norm(emb_session2)

        # Simulate: session 1 stored, session 2 queries
        client.search.return_value = [
            {
                "id": "vec_42",
                "score": 0.93,
                "metadata": {"person_id": "S1", "session_id": "sess_001"},
            }
        ]

        result = store.find_person(emb_session2, threshold=0.85)
        assert result is not None
        assert result["person_id"] == "S1"
        assert result["session_id"] == "sess_001"
        assert result["score"] >= 0.85

    def test_unknown_person_not_matched(self):
        """A completely different person should not match any stored embedding."""
        store, client = _mock_store()
        # New person with very different embedding
        new_emb = _make_embedding(768, seed=999)

        # DB returns low-similarity matches
        client.search.return_value = [
            {
                "id": "vec_1",
                "score": 0.35,
                "metadata": {"person_id": "S1", "session_id": "sess_001"},
            }
        ]

        result = store.find_person(new_emb, threshold=0.85)
        assert result is None

    def test_confidence_threshold_filtering(self):
        """Only matches above the confidence threshold are returned."""
        store, client = _mock_store()
        emb = _make_embedding(768, seed=42)

        # Multiple results with varying scores
        client.search.return_value = [
            {"id": "v1", "score": 0.82, "metadata": {"person_id": "S2", "session_id": "s2"}},
            {"id": "v2", "score": 0.75, "metadata": {"person_id": "S3", "session_id": "s3"}},
        ]

        # threshold=0.85 should reject all
        result = store.find_person(emb, threshold=0.85)
        assert result is None

        # threshold=0.80 should accept the best
        result = store.find_person(emb, threshold=0.80)
        assert result is not None
        assert result["person_id"] == "S2"

    def test_multiple_sessions_stored(self):
        """Multiple embeddings from different sessions can be stored for the same person."""
        store, client = _mock_store()

        for i in range(5):
            emb = _make_embedding(768, seed=42 + i)
            store.store_person_embedding(emb, person_id="S1", session_id=f"sess_{i:03d}")

        assert client.insert.call_count == 5

    def test_find_person_returns_best_match(self):
        """When multiple people match, the highest-scoring one is returned."""
        store, client = _mock_store()
        emb = _make_embedding(768, seed=42)

        client.search.return_value = [
            {"id": "v1", "score": 0.95, "metadata": {"person_id": "S1", "session_id": "s1"}},
            {"id": "v2", "score": 0.88, "metadata": {"person_id": "S2", "session_id": "s2"}},
        ]

        result = store.find_person(emb, threshold=0.85)
        assert result["person_id"] == "S1"
        assert result["score"] == 0.95
