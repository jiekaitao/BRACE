"""Tests for game-related database document builders and indexes."""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from db import (
    make_game_doc,
    make_game_player_doc,
    make_player_frame_doc,
    make_guideline_doc,
    make_workout_summary_doc,
)


# ---------------------------------------------------------------------------
# make_game_doc tests
# ---------------------------------------------------------------------------

class TestMakeGameDoc:
    def test_required_fields(self):
        doc = make_game_doc(
            session_id="abc123",
            video_name="game.mp4",
        )
        assert doc["session_id"] == "abc123"
        assert doc["video_name"] == "game.mp4"
        assert doc["status"] == "pending"
        assert isinstance(doc["created_at"], datetime)
        assert doc["player_count"] == 0
        assert doc["total_frames"] == 0

    def test_optional_fields(self):
        doc = make_game_doc(
            session_id="abc",
            video_name="game.mp4",
            sport="basketball",
            user_id="user1",
        )
        assert doc["sport"] == "basketball"
        assert doc["user_id"] == "user1"

    def test_status_default_pending(self):
        doc = make_game_doc(session_id="x", video_name="v.mp4")
        assert doc["status"] == "pending"


# ---------------------------------------------------------------------------
# make_game_player_doc tests
# ---------------------------------------------------------------------------

class TestMakeGamePlayerDoc:
    def test_required_fields(self):
        doc = make_game_player_doc(
            game_id="game1",
            subject_id=1,
            label="S1",
        )
        assert doc["game_id"] == "game1"
        assert doc["subject_id"] == 1
        assert doc["label"] == "S1"
        assert doc["jersey_number"] is None
        assert doc["jersey_color"] is None
        assert doc["risk_status"] == "GREEN"

    def test_with_jersey(self):
        doc = make_game_player_doc(
            game_id="game1",
            subject_id=1,
            label="S1",
            jersey_number=23,
            jersey_color="red",
        )
        assert doc["jersey_number"] == 23
        assert doc["jersey_color"] == "red"

    def test_with_risk_status(self):
        doc = make_game_player_doc(
            game_id="g", subject_id=1, label="S1", risk_status="RED"
        )
        assert doc["risk_status"] == "RED"


# ---------------------------------------------------------------------------
# make_player_frame_doc tests
# ---------------------------------------------------------------------------

class TestMakePlayerFrameDoc:
    def test_required_fields(self):
        doc = make_player_frame_doc(
            game_id="g1",
            subject_id=1,
            frame_index=42,
            video_time=1.4,
        )
        assert doc["game_id"] == "g1"
        assert doc["subject_id"] == 1
        assert doc["frame_index"] == 42
        assert doc["video_time"] == 1.4
        assert doc["quality"] is None
        assert doc["activity_label"] is None

    def test_with_quality(self):
        quality = {"form_score": 85.0, "injury_risks": []}
        doc = make_player_frame_doc(
            game_id="g1", subject_id=1, frame_index=0, video_time=0.0,
            quality=quality,
        )
        assert doc["quality"] == quality

    def test_with_activity(self):
        doc = make_player_frame_doc(
            game_id="g1", subject_id=1, frame_index=0, video_time=0.0,
            activity_label="shooting",
        )
        assert doc["activity_label"] == "shooting"


# ---------------------------------------------------------------------------
# make_guideline_doc tests
# ---------------------------------------------------------------------------

class TestMakeGuidelineDoc:
    def test_required_fields(self):
        doc = make_guideline_doc(
            user_id="u1",
            activity="basketball shooting",
            guidelines=["Keep elbow aligned", "Follow through"],
        )
        assert doc["user_id"] == "u1"
        assert doc["activity"] == "basketball shooting"
        assert len(doc["guidelines"]) == 2
        assert isinstance(doc["created_at"], datetime)

    def test_with_injury_context(self):
        doc = make_guideline_doc(
            user_id="u1",
            activity="running",
            guidelines=["Watch knee valgus"],
            injury_context="ACL recovery",
        )
        assert doc["injury_context"] == "ACL recovery"


# ---------------------------------------------------------------------------
# make_workout_summary_doc expanded fields tests
# ---------------------------------------------------------------------------

class TestMakeWorkoutSummaryDocExpanded:
    def test_existing_fields_preserved(self):
        doc = make_workout_summary_doc(
            user_id="u1",
            duration_sec=120.0,
            clusters={},
            injury_risks=[],
        )
        assert doc["user_id"] == "u1"
        assert doc["duration_sec"] == 120.0

    def test_game_id_field(self):
        doc = make_workout_summary_doc(
            user_id="u1",
            duration_sec=60.0,
            clusters={},
            injury_risks=[],
            game_id="game123",
        )
        assert doc["game_id"] == "game123"

    def test_activity_label_field(self):
        doc = make_workout_summary_doc(
            user_id="u1",
            duration_sec=60.0,
            clusters={},
            injury_risks=[],
            activity_label="basketball",
        )
        assert doc["activity_label"] == "basketball"
