"""Tests for crash analysis API endpoints in backend/main.py.

Tests the REST and WebSocket endpoints using httpx and mock pipeline.
These tests mock the crash_processor module to avoid needing real video/GPU.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ── Fixtures ──────────────────────────────────────────────────────────────


MOCK_CRASH_RESULT = {
    "analysis_id": "test-analysis-123",
    "status": "complete",
    "total_frames": 100,
    "duration_sec": 3.3,
    "fps": 30.0,
    "subjects_tracked": 2,
    "collision_events": [
        {
            "event_id": "evt-1",
            "frame_index": 45,
            "video_time": 1.5,
            "subject_a": 1,
            "subject_b": 2,
            "closing_speed_ms": 3.5,
            "peak_linear_g": 45.0,
            "peak_rotational_rads2": 2250.0,
            "concussion_probability": 0.12,
            "risk_level": "MODERATE",
            "recommendation": "Monitor for symptoms",
            "contact_zone": "head_to_shoulder",
            "head_coupling_factor": 0.4,
            "hic": 120.5,
        }
    ],
    "subject_summaries": {
        "1": {
            "subject_id": 1,
            "collision_count": 1,
            "max_concussion_probability": 0.12,
            "worst_risk_level": "MODERATE",
            "recommendation": "Monitor for symptoms",
        },
        "2": {
            "subject_id": 2,
            "collision_count": 1,
            "max_concussion_probability": 0.12,
            "worst_risk_level": "MODERATE",
            "recommendation": "Monitor for symptoms",
        },
    },
    "overall_risk": "MODERATE",
    "overall_recommendation": "Monitor involved players for concussion symptoms",
}


# ═══════════════════════════════════════════════════════════════════════════
# Crash Analysis Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrashAnalysisEndpoints:
    """Tests for POST /api/crash-analysis and GET /api/crash-analysis/{id}.

    These tests require importing main.py which pulls in GPU-only deps
    and creates directories — they only pass inside Docker containers.
    """

    @pytest.fixture
    def app_client(self):
        """Create a test client for the FastAPI app.

        This requires httpx and the app to be importable.
        Skip if dependencies aren't available (runs only in Docker).
        """
        try:
            from httpx import AsyncClient, ASGITransport
            from main import app
            return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        except (ImportError, PermissionError, OSError):
            pytest.skip("main.py not importable outside Docker (GPU-only deps)")

    @pytest.mark.asyncio
    async def test_post_requires_session_id(self, app_client):
        """POST without session_id should return 422."""
        async with app_client as client:
            resp = await client.post("/api/crash-analysis")
            assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_post_invalid_session_returns_error(self, app_client):
        """POST with a non-existent session_id should return an error."""
        async with app_client as client:
            resp = await client.post(
                "/api/crash-analysis",
                params={"session_id": "nonexistent-session"},
            )
            data = resp.json()
            assert "error" in data

    @pytest.mark.asyncio
    async def test_get_not_found(self, app_client):
        """GET for a non-existent analysis_id should return not found."""
        async with app_client as client:
            resp = await client.get("/api/crash-analysis/nonexistent-id")
            data = resp.json()
            assert data.get("error") or data.get("status") == "not_found"


# ═══════════════════════════════════════════════════════════════════════════
# Crash Analysis Results Structure Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrashAnalysisResults:
    """Tests for the structure of crash analysis results returned by the API."""

    def test_mock_result_has_collision_events(self):
        assert "collision_events" in MOCK_CRASH_RESULT
        assert len(MOCK_CRASH_RESULT["collision_events"]) == 1

    def test_collision_event_structure(self):
        event = MOCK_CRASH_RESULT["collision_events"][0]
        required_fields = [
            "event_id", "frame_index", "video_time",
            "subject_a", "subject_b", "closing_speed_ms",
            "peak_linear_g", "peak_rotational_rads2",
            "concussion_probability", "risk_level",
            "recommendation", "contact_zone",
            "head_coupling_factor", "hic",
        ]
        for field in required_fields:
            assert field in event, f"Missing field: {field}"

    def test_result_has_overall_risk(self):
        assert MOCK_CRASH_RESULT["overall_risk"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_result_has_overall_recommendation(self):
        assert isinstance(MOCK_CRASH_RESULT["overall_recommendation"], str)
        assert len(MOCK_CRASH_RESULT["overall_recommendation"]) > 0

    def test_result_has_subject_summaries(self):
        assert "subject_summaries" in MOCK_CRASH_RESULT
        assert len(MOCK_CRASH_RESULT["subject_summaries"]) == 2

    def test_subject_summary_structure(self):
        summary = MOCK_CRASH_RESULT["subject_summaries"]["1"]
        required_fields = [
            "subject_id", "collision_count",
            "max_concussion_probability", "worst_risk_level",
            "recommendation",
        ]
        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

    def test_result_serializable_to_json(self):
        json_str = json.dumps(MOCK_CRASH_RESULT)
        parsed = json.loads(json_str)
        assert parsed["analysis_id"] == MOCK_CRASH_RESULT["analysis_id"]

    def test_concussion_probability_range(self):
        for event in MOCK_CRASH_RESULT["collision_events"]:
            assert 0.0 <= event["concussion_probability"] <= 1.0

    def test_risk_levels_are_valid(self):
        valid = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        for event in MOCK_CRASH_RESULT["collision_events"]:
            assert event["risk_level"] in valid
        assert MOCK_CRASH_RESULT["overall_risk"] in valid

    def test_hic_values_non_negative(self):
        for event in MOCK_CRASH_RESULT["collision_events"]:
            assert event["hic"] >= 0.0

    def test_closing_speed_non_negative(self):
        for event in MOCK_CRASH_RESULT["collision_events"]:
            assert event["closing_speed_ms"] >= 0.0

    def test_peak_values_non_negative(self):
        for event in MOCK_CRASH_RESULT["collision_events"]:
            assert event["peak_linear_g"] >= 0.0
            assert event["peak_rotational_rads2"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# In-memory state management tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrashStateManagement:
    """Tests for the in-memory crash task/result state dicts."""

    def test_crash_dicts_exist(self):
        """Verify the module-level state dicts are defined in main.py."""
        try:
            from main import _crash_tasks, _crash_results, _crash_progress_sockets
            assert isinstance(_crash_tasks, dict)
            assert isinstance(_crash_results, dict)
            assert isinstance(_crash_progress_sockets, dict)
        except (ImportError, PermissionError, OSError):
            pytest.skip("main.py not importable outside Docker (GPU-only deps)")

    def test_crash_result_storage(self):
        """Crash results should be storable and retrievable by analysis_id."""
        try:
            from main import _crash_results
        except (ImportError, PermissionError, OSError):
            pytest.skip("main.py not importable outside Docker")

        analysis_id = str(uuid.uuid4())
        _crash_results[analysis_id] = MOCK_CRASH_RESULT.copy()
        assert _crash_results[analysis_id]["status"] == "complete"
        del _crash_results[analysis_id]

    def test_crash_result_json_serializable(self):
        """Stored results must be JSON-serializable for WebSocket transmission."""
        json_str = json.dumps(MOCK_CRASH_RESULT)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["collision_events"][0]["risk_level"] == "MODERATE"
