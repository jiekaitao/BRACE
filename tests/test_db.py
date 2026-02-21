"""Tests for MongoDB integration: db.py client helpers, schema indexes, and CRUD operations.

All tests use mocking -- no running MongoDB instance required.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Test: MONGODB_URI env var parsing
# ---------------------------------------------------------------------------

class TestConnectionConfig:
    """Tests for connection string configuration."""

    def test_default_uri(self):
        """get_client() uses default URI when MONGODB_URI is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MONGODB_URI", None)
            # Re-import to pick up env
            import importlib
            import db as db_mod
            importlib.reload(db_mod)
            assert db_mod.MONGODB_URI == "mongodb://localhost:27017/brace"

    def test_custom_uri_from_env(self):
        """get_client() uses MONGODB_URI from environment."""
        custom = "mongodb://mongo:27017/testdb"
        with patch.dict(os.environ, {"MONGODB_URI": custom}):
            import importlib
            import db as db_mod
            importlib.reload(db_mod)
            assert db_mod.MONGODB_URI == custom


# ---------------------------------------------------------------------------
# Test: Async client helpers (motor)
# ---------------------------------------------------------------------------

class TestAsyncHelpers:
    """Tests for motor-based async helpers."""

    def test_get_client_returns_motor_client(self):
        """get_client() returns a motor AsyncIOMotorClient instance."""
        import db as db_mod
        with patch("db.motor.motor_asyncio.AsyncIOMotorClient") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value = mock_instance
            client = db_mod.get_client()
            assert client is mock_instance
            MockClient.assert_called_once_with(db_mod.MONGODB_URI)

    def test_get_db_returns_database(self):
        """get_db() returns the database from the client."""
        import db as db_mod
        mock_client = MagicMock()
        mock_db = MagicMock()
        mock_client.get_default_database.return_value = mock_db
        with patch.object(db_mod, "get_client", return_value=mock_client):
            result = db_mod.get_db()
            assert result is mock_db

    def test_get_collection_returns_collection(self):
        """get_collection() returns a named collection from the database."""
        import db as db_mod
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        with patch.object(db_mod, "get_db", return_value=mock_db):
            result = db_mod.get_collection("users")
            mock_db.__getitem__.assert_called_once_with("users")
            assert result is mock_collection


# ---------------------------------------------------------------------------
# Test: Sync client helpers (pymongo)
# ---------------------------------------------------------------------------

class TestSyncHelpers:
    """Tests for pymongo-based sync helpers."""

    def test_get_sync_db_returns_database(self):
        """get_sync_db() returns a pymongo Database."""
        import db as db_mod
        with patch("db.pymongo.MongoClient") as MockClient:
            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_client.get_default_database.return_value = mock_db
            MockClient.return_value = mock_client
            result = db_mod.get_sync_db()
            assert result is mock_db
            MockClient.assert_called_once_with(db_mod.MONGODB_URI)

    def test_get_sync_collection_returns_collection(self):
        """get_sync_collection() returns a named pymongo collection."""
        import db as db_mod
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        with patch.object(db_mod, "get_sync_db", return_value=mock_db):
            result = db_mod.get_sync_collection("chat_sessions")
            mock_db.__getitem__.assert_called_once_with("chat_sessions")
            assert result is mock_collection


# ---------------------------------------------------------------------------
# Test: ensure_indexes()
# ---------------------------------------------------------------------------

class TestEnsureIndexes:
    """Tests for index creation on all collections."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_users_unique_index(self):
        """ensure_indexes() creates a unique index on users.username."""
        import db as db_mod
        mock_db = MagicMock()
        mock_users = AsyncMock()
        mock_chat = AsyncMock()
        mock_workouts = AsyncMock()
        mock_db.__getitem__ = lambda self, name: {
            "users": mock_users,
            "chat_sessions": mock_chat,
            "workout_summaries": mock_workouts,
        }[name]
        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()
            # Check users got a unique index on username
            mock_users.create_index.assert_called_once()
            call_args = mock_users.create_index.call_args
            assert "username" in str(call_args)
            assert call_args[1].get("unique") is True

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_chat_sessions_index(self):
        """ensure_indexes() creates an index on chat_sessions.user_id."""
        import db as db_mod
        mock_db = MagicMock()
        mock_users = AsyncMock()
        mock_chat = AsyncMock()
        mock_workouts = AsyncMock()
        mock_db.__getitem__ = lambda self, name: {
            "users": mock_users,
            "chat_sessions": mock_chat,
            "workout_summaries": mock_workouts,
        }[name]
        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()
            mock_chat.create_index.assert_called_once()
            call_args = mock_chat.create_index.call_args
            assert "user_id" in str(call_args)

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_workout_compound_index(self):
        """ensure_indexes() creates a compound index on workout_summaries (user_id, created_at desc)."""
        import db as db_mod
        mock_db = MagicMock()
        mock_users = AsyncMock()
        mock_chat = AsyncMock()
        mock_workouts = AsyncMock()
        mock_db.__getitem__ = lambda self, name: {
            "users": mock_users,
            "chat_sessions": mock_chat,
            "workout_summaries": mock_workouts,
        }[name]
        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()
            mock_workouts.create_index.assert_called_once()
            call_args = mock_workouts.create_index.call_args
            # Should contain user_id ascending and created_at descending
            index_spec = call_args[0][0]
            assert index_spec[0] == ("user_id", 1)
            assert index_spec[1] == ("created_at", -1)


# ---------------------------------------------------------------------------
# Test: Schema document helpers
# ---------------------------------------------------------------------------

class TestSchemaHelpers:
    """Tests for document creation helpers."""

    def test_make_user_document(self):
        """make_user_doc() creates a well-formed user document."""
        import db as db_mod
        doc = db_mod.make_user_doc("alice")
        assert doc["username"] == "alice"
        assert isinstance(doc["created_at"], datetime)
        assert doc["injury_profile"] is None
        assert doc["risk_modifiers"] is None

    def test_make_chat_session_document(self):
        """make_chat_session_doc() creates a well-formed chat session document."""
        import db as db_mod
        doc = db_mod.make_chat_session_doc("user123")
        assert doc["user_id"] == "user123"
        assert doc["messages"] == []
        assert doc["extracted_profile"] is None
        assert isinstance(doc["created_at"], datetime)
        assert isinstance(doc["updated_at"], datetime)

    def test_make_workout_summary_document(self):
        """make_workout_summary_doc() creates a well-formed workout summary document."""
        import db as db_mod
        doc = db_mod.make_workout_summary_doc(
            user_id="user123",
            duration_sec=120.5,
            clusters={"0": {"label": "squat"}},
            injury_risks=[{"type": "acl", "severity": "moderate"}],
            fatigue_score=0.75,
            video_name="test.mp4",
        )
        assert doc["user_id"] == "user123"
        assert doc["video_name"] == "test.mp4"
        assert doc["duration_sec"] == 120.5
        assert doc["clusters"] == {"0": {"label": "squat"}}
        assert len(doc["injury_risks"]) == 1
        assert doc["fatigue_score"] == 0.75
        assert isinstance(doc["created_at"], datetime)

    def test_make_workout_summary_defaults(self):
        """make_workout_summary_doc() uses sensible defaults."""
        import db as db_mod
        doc = db_mod.make_workout_summary_doc(
            user_id="u1",
            duration_sec=60.0,
            clusters={},
            injury_risks=[],
        )
        assert doc["video_name"] is None
        assert doc["fatigue_score"] is None
