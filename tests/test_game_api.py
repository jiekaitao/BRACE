"""Tests for basketball game API endpoints and document helpers.

All tests use mocking — no running MongoDB or GPU pipeline required.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ---------------------------------------------------------------------------
# Test: Game document helpers
# ---------------------------------------------------------------------------

class TestGameDocHelpers:
    """Tests for game-related document creation helpers."""

    def test_make_game_doc(self):
        """make_game_doc() creates a well-formed game document."""
        import db as db_mod
        doc = db_mod.make_game_doc("session-123", "/app/uploads/test.mp4")
        assert doc["session_id"] == "session-123"
        assert doc["video_path"] == "/app/uploads/test.mp4"
        assert doc["status"] == "queued"
        assert doc["progress"] == 0.0
        assert doc["player_count"] == 0
        assert doc["team_colors"] == []
        assert doc["error"] is None
        assert isinstance(doc["created_at"], datetime)
        assert isinstance(doc["updated_at"], datetime)

    def test_make_game_player_doc(self):
        """make_game_player_doc() creates a well-formed game player document."""
        import db as db_mod
        doc = db_mod.make_game_player_doc(
            game_id="game-1",
            subject_id=5,
            jersey_number=23,
            team_id=0,
            jersey_color="red",
        )
        assert doc["game_id"] == "game-1"
        assert doc["subject_id"] == 5
        assert doc["jersey_number"] == 23
        assert doc["team_id"] == 0
        assert doc["jersey_color"] == "red"
        assert doc["injury_events"] == []
        assert doc["analysis_summary"] is None
        assert doc["risk_status"] is None
        assert doc["risk_history"] == []
        assert doc["workload"] is None
        assert doc["pull_recommended"] is False
        assert doc["pull_reasons"] == []

    def test_make_game_player_doc_defaults(self):
        """make_game_player_doc() handles default values."""
        import db as db_mod
        doc = db_mod.make_game_player_doc(game_id="g1", subject_id=1)
        assert doc["jersey_number"] is None
        assert doc["team_id"] is None
        assert doc["jersey_color"] is None

    def test_make_player_frame_doc(self):
        """make_player_frame_doc() creates a well-formed frame snapshot."""
        import db as db_mod
        quality = {"form_score": 85.0, "anomaly_score": 0.1}
        biomech = {"fppa_left": 12.5, "fppa_right": 11.0}
        doc = db_mod.make_player_frame_doc(
            game_id="game-1",
            subject_id=3,
            frame_idx=300,
            quality=quality,
            biomechanics=biomech,
        )
        assert doc["game_id"] == "game-1"
        assert doc["subject_id"] == 3
        assert doc["frame_idx"] == 300
        assert doc["quality"] == quality
        assert doc["biomechanics"] == biomech
        assert isinstance(doc["created_at"], datetime)

    def test_make_player_frame_doc_defaults(self):
        """make_player_frame_doc() handles missing quality/biomechanics."""
        import db as db_mod
        doc = db_mod.make_player_frame_doc("g1", 1, 0)
        assert doc["quality"] is None
        assert doc["biomechanics"] is None


# ---------------------------------------------------------------------------
# Test: Game indexes
# ---------------------------------------------------------------------------

class TestGameIndexes:
    """Tests for game-related MongoDB indexes in ensure_indexes()."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_games_indexes(self):
        """ensure_indexes() creates indexes on the games collection."""
        import db as db_mod

        # Create async mock collections
        mock_collections = {}
        for name in ["users", "chat_sessions", "workout_summaries",
                      "games", "game_players", "player_frames"]:
            mock_collections[name] = AsyncMock()

        mock_db = MagicMock()
        mock_db.__getitem__ = lambda self, name: mock_collections[name]

        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()

            # games collection should have 2 index calls (status, created_at)
            assert mock_collections["games"].create_index.call_count == 2

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_game_players_indexes(self):
        """ensure_indexes() creates compound unique index on game_players."""
        import db as db_mod

        mock_collections = {}
        for name in ["users", "chat_sessions", "workout_summaries",
                      "games", "game_players", "player_frames"]:
            mock_collections[name] = AsyncMock()

        mock_db = MagicMock()
        mock_db.__getitem__ = lambda self, name: mock_collections[name]

        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()

            # game_players should have 2 index calls
            assert mock_collections["game_players"].create_index.call_count == 2
            # First call should be the unique compound index
            first_call = mock_collections["game_players"].create_index.call_args_list[0]
            assert first_call[1].get("unique") is True

    @pytest.mark.asyncio
    async def test_ensure_indexes_creates_player_frames_indexes(self):
        """ensure_indexes() creates indexes on player_frames with TTL."""
        import db as db_mod

        mock_collections = {}
        for name in ["users", "chat_sessions", "workout_summaries",
                      "games", "game_players", "player_frames"]:
            mock_collections[name] = AsyncMock()

        mock_db = MagicMock()
        mock_db.__getitem__ = lambda self, name: mock_collections[name]

        with patch.object(db_mod, "get_db", return_value=mock_db):
            await db_mod.ensure_indexes()

            # player_frames should have 2 index calls (compound + TTL)
            assert mock_collections["player_frames"].create_index.call_count == 2
            # Second call should have TTL
            ttl_call = mock_collections["player_frames"].create_index.call_args_list[1]
            assert ttl_call[1].get("expireAfterSeconds") == 30 * 24 * 3600


# ---------------------------------------------------------------------------
# Test: Basketball processor imports
# ---------------------------------------------------------------------------

class TestBasketballProcessorImports:
    """Tests that basketball_processor module structure is correct."""

    def test_import_basketball_processor(self):
        """basketball_processor module can be imported."""
        from backend.basketball_processor import process_basketball_game
        assert callable(process_basketball_game)

    def test_basketball_classify_prompt_has_basketball_terms(self):
        """The basketball-specific prompt includes basketball activity labels."""
        from backend.basketball_processor import _BASKETBALL_CLASSIFY_PROMPT
        assert "dunk" in _BASKETBALL_CLASSIFY_PROMPT
        assert "crossover" in _BASKETBALL_CLASSIFY_PROMPT
        assert "layup" in _BASKETBALL_CLASSIFY_PROMPT
        assert "defensive slide" in _BASKETBALL_CLASSIFY_PROMPT

    def test_snapshot_interval_constant(self):
        """Snapshot interval should be 300 frames."""
        from backend.basketball_processor import _SNAPSHOT_INTERVAL
        assert _SNAPSHOT_INTERVAL == 300

    def test_jersey_detect_threshold_constant(self):
        """Jersey detection should wait 90 frames (~3s at 30fps)."""
        from backend.basketball_processor import _JERSEY_DETECT_FRAME_THRESHOLD
        assert _JERSEY_DETECT_FRAME_THRESHOLD == 90


# ---------------------------------------------------------------------------
# Test: Gemini classifier prompt parameter
# ---------------------------------------------------------------------------

class TestGeminiClassifierPromptParam:
    """Tests for the optional prompt parameter in classify_activity()."""

    def test_classify_activity_accepts_prompt_kwarg(self):
        """classify_activity() should accept an optional prompt parameter."""
        import inspect
        from backend.gemini_classifier import GeminiActivityClassifier

        sig = inspect.signature(GeminiActivityClassifier.classify_activity)
        params = list(sig.parameters.keys())
        assert "prompt" in params

    def test_classify_activity_prompt_default_is_none(self):
        """The prompt parameter should default to None."""
        import inspect
        from backend.gemini_classifier import GeminiActivityClassifier

        sig = inspect.signature(GeminiActivityClassifier.classify_activity)
        prompt_param = sig.parameters["prompt"]
        assert prompt_param.default is None
