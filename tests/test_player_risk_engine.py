"""Tests for backend/player_risk_engine.py — player risk tracking during games."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
from player_risk_engine import (
    RiskStatus,
    InjuryEvent,
    PlayerWorkload,
    PlayerRiskState,
    PlayerRiskEngine,
)


# ── 1. RiskStatus enum (3 tests) ────────────────────────────────────────────


class TestRiskStatus:
    """Tests for the RiskStatus IntEnum."""

    def test_enum_members_exist(self):
        """GREEN, YELLOW, RED members exist."""
        assert hasattr(RiskStatus, "GREEN")
        assert hasattr(RiskStatus, "YELLOW")
        assert hasattr(RiskStatus, "RED")

    def test_ordering(self):
        """GREEN < YELLOW < RED."""
        assert RiskStatus.GREEN.value < RiskStatus.YELLOW.value
        assert RiskStatus.YELLOW.value < RiskStatus.RED.value

    def test_string_representation(self):
        """String names are readable."""
        assert RiskStatus.GREEN.name == "GREEN"
        assert RiskStatus.YELLOW.name == "YELLOW"
        assert RiskStatus.RED.name == "RED"


# ── 2. InjuryEvent dataclass (4 tests) ──────────────────────────────────────


class TestInjuryEvent:
    """Tests for the InjuryEvent dataclass."""

    def test_create_with_required_fields(self):
        """Can create an InjuryEvent with required fields."""
        ev = InjuryEvent(
            risk_name="knee_valgus",
            severity="medium",
            joint="left_knee",
            timestamp=10.5,
            frame_index=315,
        )
        assert ev.risk_name == "knee_valgus"
        assert ev.severity == "medium"
        assert ev.joint == "left_knee"
        assert ev.timestamp == 10.5
        assert ev.frame_index == 315

    def test_severity_values(self):
        """Severity accepts 'medium' and 'high'."""
        med = InjuryEvent("r", "medium", "j", 0.0, 0)
        high = InjuryEvent("r", "high", "j", 0.0, 0)
        assert med.severity == "medium"
        assert high.severity == "high"

    def test_consolidation_key_same(self):
        """Two events with same risk_name and joint share a consolidation key."""
        e1 = InjuryEvent("knee_valgus", "medium", "left_knee", 1.0, 30)
        e2 = InjuryEvent("knee_valgus", "medium", "left_knee", 3.0, 90)
        assert e1.key == e2.key

    def test_consolidation_key_different_joints(self):
        """Events with different joints have different consolidation keys."""
        e1 = InjuryEvent("knee_valgus", "medium", "left_knee", 1.0, 30)
        e2 = InjuryEvent("knee_valgus", "medium", "right_knee", 1.0, 30)
        assert e1.key != e2.key


# ── 3. PlayerWorkload dataclass (4 tests) ───────────────────────────────────


class TestPlayerWorkload:
    """Tests for the PlayerWorkload dataclass."""

    def test_create_with_fields(self):
        """Can create with all numeric fields."""
        w = PlayerWorkload(
            total_frames=900,
            active_seconds=25.0,
            high_intensity_seconds=10.0,
            rest_seconds=5.0,
        )
        assert w.total_frames == 900
        assert w.active_seconds == 25.0
        assert w.high_intensity_seconds == 10.0
        assert w.rest_seconds == 5.0

    def test_intensity_ratio(self):
        """intensity_ratio = high_intensity_seconds / active_seconds."""
        w = PlayerWorkload(active_seconds=20.0, high_intensity_seconds=10.0)
        assert w.intensity_ratio == pytest.approx(0.5)

    def test_intensity_ratio_zero_active(self):
        """intensity_ratio is 0 when no active time."""
        w = PlayerWorkload(active_seconds=0.0, high_intensity_seconds=0.0)
        assert w.intensity_ratio == 0.0

    def test_fatigue_estimate(self):
        """fatigue_estimate increases with intensity and duration."""
        fresh = PlayerWorkload(active_seconds=10.0, high_intensity_seconds=0.0)
        tired = PlayerWorkload(active_seconds=600.0, high_intensity_seconds=500.0)
        assert tired.fatigue_estimate > fresh.fatigue_estimate

    def test_default_values(self):
        """Default values are all zero."""
        w = PlayerWorkload()
        assert w.total_frames == 0
        assert w.active_seconds == 0.0
        assert w.high_intensity_seconds == 0.0
        assert w.rest_seconds == 0.0


# ── 4. PlayerRiskState dataclass (3 tests) ──────────────────────────────────


class TestPlayerRiskState:
    """Tests for the PlayerRiskState dataclass."""

    def test_fields_present(self):
        """Contains status, events, workload, pull_recommended, pull_reason."""
        s = PlayerRiskState()
        assert hasattr(s, "status")
        assert hasattr(s, "events")
        assert hasattr(s, "workload")
        assert hasattr(s, "pull_recommended")
        assert hasattr(s, "pull_reason")

    def test_default_status_green(self):
        """Default status is GREEN."""
        s = PlayerRiskState()
        assert s.status == RiskStatus.GREEN

    def test_default_pull_false(self):
        """Default pull_recommended is False."""
        s = PlayerRiskState()
        assert s.pull_recommended is False


# ── 5. PlayerRiskEngine class (17 tests) ────────────────────────────────────


class TestPlayerRiskEngineConstructor:
    """Constructor tests for PlayerRiskEngine."""

    def test_default_thresholds(self):
        """Default thresholds match specification."""
        e = PlayerRiskEngine()
        assert e._yellow_count == 3
        assert e._red_count == 6
        assert e._consolidation_window == 5.0
        assert e._fatigue_yellow == pytest.approx(0.6)
        assert e._fatigue_red == pytest.approx(0.8)

    def test_custom_thresholds(self):
        """Custom thresholds are accepted."""
        e = PlayerRiskEngine(
            yellow_event_count=2,
            red_event_count=4,
            consolidation_window_sec=10.0,
            fatigue_yellow_threshold=0.5,
            fatigue_red_threshold=0.7,
        )
        assert e._yellow_count == 2
        assert e._red_count == 4
        assert e._consolidation_window == 10.0
        assert e._fatigue_yellow == pytest.approx(0.5)
        assert e._fatigue_red == pytest.approx(0.7)


# --- Helpers for building quality dicts ---

def _make_quality(risks=None):
    """Build a quality dict with optional injury_risks list."""
    q = {}
    if risks is not None:
        q["injury_risks"] = risks
    return q


def _risk(name="knee_valgus", severity="medium", joint="left_knee", desc=""):
    """Shorthand for an injury risk dict as MovementQualityTracker would emit."""
    return {
        "risk_name": name,
        "severity": severity,
        "joint": joint,
        "description": desc,
    }


class TestProcessFrame:
    """Tests for PlayerRiskEngine.process_frame()."""

    def test_returns_player_risk_state(self):
        """process_frame returns a PlayerRiskState."""
        e = PlayerRiskEngine()
        result = e.process_frame(quality=None, frame_index=0)
        assert isinstance(result, PlayerRiskState)

    def test_no_risks_is_green(self):
        """Single frame with no risks → GREEN."""
        e = PlayerRiskEngine()
        state = e.process_frame(quality=_make_quality(), frame_index=0)
        assert state.status == RiskStatus.GREEN
        assert len(state.events) == 0

    def test_medium_risk_creates_event(self):
        """Frame with medium risk → creates InjuryEvent."""
        e = PlayerRiskEngine()
        q = _make_quality([_risk("knee_valgus", "medium", "left_knee")])
        state = e.process_frame(quality=q, frame_index=0, video_time=0.0)
        assert len(state.events) == 1
        assert state.events[0].severity == "medium"

    def test_accumulate_to_yellow(self):
        """3+ medium risk events (different timestamps beyond window) → YELLOW."""
        e = PlayerRiskEngine()
        for i in range(4):
            # Space events beyond consolidation window (>5s apart)
            t = i * 10.0
            q = _make_quality([_risk("knee_valgus", "medium", "left_knee")])
            state = e.process_frame(quality=q, frame_index=i * 300, video_time=t)
        assert state.status == RiskStatus.YELLOW

    def test_accumulate_to_red(self):
        """6+ risk events → RED."""
        e = PlayerRiskEngine()
        for i in range(7):
            t = i * 10.0
            q = _make_quality([_risk("hip_drop", "medium", "pelvis")])
            state = e.process_frame(quality=q, frame_index=i * 300, video_time=t)
        assert state.status == RiskStatus.RED


class TestConsolidation:
    """Tests for _consolidate_injury_risks()."""

    def test_same_key_within_window_consolidated(self):
        """Two events same risk+joint within 5s → 1 consolidated event."""
        e = PlayerRiskEngine()
        events = [
            InjuryEvent("knee_valgus", "medium", "left_knee", 1.0, 30),
            InjuryEvent("knee_valgus", "medium", "left_knee", 3.0, 90),
        ]
        result = e._consolidate_injury_risks(events)
        assert len(result) == 1

    def test_same_key_beyond_window_not_consolidated(self):
        """Two events same risk+joint >5s apart → 2 events."""
        e = PlayerRiskEngine()
        events = [
            InjuryEvent("knee_valgus", "medium", "left_knee", 1.0, 30),
            InjuryEvent("knee_valgus", "medium", "left_knee", 10.0, 300),
        ]
        result = e._consolidate_injury_risks(events)
        assert len(result) == 2

    def test_different_risk_names_no_consolidation(self):
        """Events with different risk_names → no consolidation."""
        e = PlayerRiskEngine()
        events = [
            InjuryEvent("knee_valgus", "medium", "left_knee", 1.0, 30),
            InjuryEvent("hip_drop", "medium", "left_knee", 2.0, 60),
        ]
        result = e._consolidate_injury_risks(events)
        assert len(result) == 2


class TestDetermineStatus:
    """Tests for _determine_status()."""

    def test_few_events_green(self):
        """0-2 events → GREEN."""
        e = PlayerRiskEngine()
        events = [InjuryEvent("r", "medium", "j", 0, 0)] * 2
        assert e._determine_status(events) == RiskStatus.GREEN

    def test_medium_count_yellow(self):
        """3-5 events → YELLOW."""
        e = PlayerRiskEngine()
        events = [InjuryEvent("r", "medium", "j", i * 10, i * 300) for i in range(4)]
        assert e._determine_status(events) == RiskStatus.YELLOW

    def test_high_count_red(self):
        """6+ events → RED."""
        e = PlayerRiskEngine()
        events = [InjuryEvent("r", "medium", "j", i * 10, i * 300) for i in range(6)]
        assert e._determine_status(events) == RiskStatus.RED

    def test_high_severity_at_least_yellow(self):
        """Any single 'high' severity event → at least YELLOW."""
        e = PlayerRiskEngine()
        events = [InjuryEvent("knee_valgus", "high", "left_knee", 0, 0)]
        status = e._determine_status(events)
        assert status >= RiskStatus.YELLOW


class TestPullRecommendation:
    """Tests for _check_pull_recommendation()."""

    def test_red_status_pull(self):
        """RED status → pull_recommended=True with reason about risk count."""
        e = PlayerRiskEngine()
        # Drive to RED via events
        for i in range(7):
            t = i * 10.0
            q = _make_quality([_risk("hip_drop", "medium", "pelvis")])
            e.process_frame(quality=q, frame_index=i * 300, video_time=t)
        assert e.state.pull_recommended is True
        assert "RED" in e.state.pull_reason or "risk" in e.state.pull_reason.lower()

    def test_fatigue_plus_yellow_pull(self):
        """High fatigue + YELLOW → pull_recommended=True with fatigue mention."""
        e = PlayerRiskEngine(fatigue_yellow_threshold=0.3)
        # Generate enough events for YELLOW (3 distinct)
        for i in range(3):
            t = i * 10.0
            q = _make_quality([_risk("hip_drop", "medium", "pelvis")])
            e.process_frame(quality=q, frame_index=i * 300, video_time=t)
        # Manually push workload to trigger fatigue
        e._state.workload.active_seconds = 600.0
        e._state.workload.high_intensity_seconds = 500.0
        # Re-evaluate pull
        e._state.pull_recommended, e._state.pull_reason = e._check_pull_recommendation()
        assert e.state.pull_recommended is True
        assert "fatigue" in e.state.pull_reason.lower()


class TestGetPlayerSummary:
    """Tests for get_player_summary()."""

    def test_summary_keys(self):
        """Summary dict has required keys."""
        e = PlayerRiskEngine()
        e.process_frame(quality=None, frame_index=0)
        s = e.get_player_summary()
        expected_keys = {"status", "event_count", "events", "workload", "pull_recommended", "pull_reason"}
        assert expected_keys.issubset(set(s.keys()))

    def test_summary_reflects_state(self):
        """Summary accurately reflects accumulated state."""
        e = PlayerRiskEngine()
        # Add two events spaced beyond consolidation window
        for i in range(2):
            t = i * 10.0
            q = _make_quality([_risk("knee_valgus", "medium", "left_knee")])
            e.process_frame(quality=q, frame_index=i * 300, video_time=t)
        s = e.get_player_summary()
        assert s["status"] == "GREEN"
        assert s["event_count"] == 2
        assert len(s["events"]) == 2
        assert s["events"][0]["risk_name"] == "knee_valgus"
        assert s["pull_recommended"] is False
        assert s["workload"]["total_frames"] == 2
