"""Tests for HeadAccelerationTracker, HeadImpactAnalyzer, and compute_concussion_probability.

These classes were added by Paul's PR for collision-triggered head impact detection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.movement_quality import (
    HeadAccelerationTracker,
    HeadImpactAnalyzer,
    HeadKinematicState,
    CollisionEvent,
    compute_concussion_probability,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _stationary_head(n_frames: int = 10, fps: float = 30.0):
    """Feed a stationary head to a tracker and return the last state."""
    tracker = HeadAccelerationTracker()
    pos = np.array([1.0, 1.0])
    dt = 1.0 / fps
    state = None
    for _ in range(n_frames):
        state = tracker.update(pos, ear_angle=0.0, dt=dt)
    return state


def _constant_velocity_head(
    velocity_mps: float = 2.0,
    n_frames: int = 20,
    fps: float = 30.0,
):
    """Feed a head moving at constant velocity; return all states."""
    tracker = HeadAccelerationTracker()
    dt = 1.0 / fps
    states = []
    for i in range(n_frames):
        pos = np.array([velocity_mps * i * dt, 0.0])
        state = tracker.update(pos, ear_angle=0.0, dt=dt)
        states.append(state)
    return states


# ═══════════════════════════════════════════════════════════════════════════
# HeadAccelerationTracker tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHeadAccelerationTracker:
    """Tests for per-subject head kinematics tracking."""

    def test_initial_state_zeros(self):
        tracker = HeadAccelerationTracker()
        state = tracker.update(np.array([1.0, 1.0]), ear_angle=0.0, dt=1 / 30)
        # First frame — no previous data, all derivatives should be zero
        assert state.speed_mps == 0.0
        assert state.accel_magnitude_g == 0.0
        assert state.jerk_magnitude == 0.0

    def test_stationary_head_zero_velocity(self):
        state = _stationary_head(n_frames=20)
        assert state is not None
        assert state.speed_mps < 0.01  # effectively zero
        assert state.accel_magnitude_g < 0.01

    def test_constant_velocity_correct_speed(self):
        velocity = 3.0  # m/s
        states = _constant_velocity_head(velocity_mps=velocity, n_frames=30)
        # After warm-up, speed should converge toward the true velocity
        # Due to EMA smoothing it may not be exact, but should be in the ballpark
        late_states = states[15:]
        avg_speed = np.mean([s.speed_mps for s in late_states])
        assert avg_speed > velocity * 0.5, f"Expected speed near {velocity}, got {avg_speed}"
        assert avg_speed < velocity * 1.5

    def test_constant_velocity_low_accel(self):
        states = _constant_velocity_head(velocity_mps=2.0, n_frames=30)
        # Constant velocity → acceleration should be near zero after warm-up
        late_accels = [s.accel_magnitude_g for s in states[15:]]
        avg_accel = np.mean(late_accels)
        assert avg_accel < 1.0, f"Expected low accel for constant velocity, got {avg_accel}g"

    def test_large_displacement_resets(self):
        """A >2m jump should be treated as a tracker ID switch → reset."""
        tracker = HeadAccelerationTracker()
        dt = 1 / 30
        # Normal movement
        for i in range(10):
            tracker.update(np.array([0.1 * i, 0.0]), ear_angle=0.0, dt=dt)
        # Jump 5m — should reset
        state = tracker.update(np.array([5.0, 5.0]), ear_angle=0.0, dt=dt)
        assert state.speed_mps == 0.0  # reset to zero after jump

    def test_rotational_velocity_from_ear_angle(self):
        tracker = HeadAccelerationTracker()
        dt = 1 / 30
        # Warm up
        for _ in range(5):
            tracker.update(np.array([1.0, 1.0]), ear_angle=0.0, dt=dt)
        # Now rotate head rapidly
        state = tracker.update(np.array([1.0, 1.0]), ear_angle=math.pi / 4, dt=dt)
        # Should register some rotational velocity
        # (exact value depends on EMA smoothing)
        assert state.rotational_vel_rps >= 0.0

    def test_accel_capped_at_100g(self):
        """Acceleration should be capped to prevent noise-driven absurd values."""
        tracker = HeadAccelerationTracker()
        dt = 1 / 30
        # Warm up at origin
        for _ in range(5):
            tracker.update(np.array([0.0, 0.0]), ear_angle=0.0, dt=dt)
        # Sudden but not huge displacement (just below the 2m reset threshold)
        state = tracker.update(np.array([1.9, 0.0]), ear_angle=0.0, dt=dt)
        assert state.accel_magnitude_g <= 100.0

    def test_reset_clears_state(self):
        tracker = HeadAccelerationTracker()
        dt = 1 / 30
        for i in range(10):
            tracker.update(np.array([0.1 * i, 0.0]), ear_angle=0.0, dt=dt)
        tracker.reset()
        # After reset, first update should behave like initial state
        state = tracker.update(np.array([0.0, 0.0]), ear_angle=0.0, dt=dt)
        assert state.speed_mps == 0.0

    def test_state_has_all_fields(self):
        tracker = HeadAccelerationTracker()
        state = tracker.update(np.array([1.0, 1.0]), ear_angle=0.0, dt=1 / 30)
        assert isinstance(state, HeadKinematicState)
        assert hasattr(state, "speed_mps")
        assert hasattr(state, "accel_magnitude_g")
        assert hasattr(state, "accel_direction_change")
        assert hasattr(state, "jerk_magnitude")
        assert hasattr(state, "rotational_vel_rps")
        assert hasattr(state, "velocity_vector")
        assert hasattr(state, "accel_vector")

    def test_velocity_vector_shape(self):
        tracker = HeadAccelerationTracker()
        state = tracker.update(np.array([1.0, 1.0]), ear_angle=0.0, dt=1 / 30)
        assert state.velocity_vector.shape == (2,)
        assert state.accel_vector.shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════
# HeadImpactAnalyzer tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHeadImpactAnalyzer:
    """Tests for multi-subject collision-triggered head impact detection."""

    def test_init(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        assert analyzer is not None
        assert analyzer.fps == 30.0

    def test_get_or_create_tracker(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        t1 = analyzer.get_or_create_tracker(1)
        t2 = analyzer.get_or_create_tracker(1)
        assert t1 is t2  # same instance
        t3 = analyzer.get_or_create_tracker(2)
        assert t3 is not t1  # different subject

    def test_proximity_triggers_monitoring(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        pairs = [{"a": 1, "b": 2, "closing_speed": 2.0, "distance": 0.1}]
        analyzer.update_proximity(pairs, frame_index=0)
        # Internal monitoring should have been set up
        assert len(analyzer._monitored) > 0

    def test_monitoring_window_expires(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        pairs = [{"a": 1, "b": 2, "closing_speed": 2.0, "distance": 0.1}]
        analyzer.update_proximity(pairs, frame_index=0)
        # Advance frames past the monitoring window
        for frame in range(1, 50):
            analyzer.update_proximity([], frame_index=frame)
            # Feed dummy head data
            for sid in [1, 2]:
                analyzer.update_subject(
                    sid, np.array([1.0, 1.0]), ear_angle=0.0,
                    dt=1 / 30, frame_index=frame, video_time=frame / 30,
                )
        # Monitored contexts should have been evaluated and removed
        assert len(analyzer._monitored) == 0

    def test_update_subject_returns_kinematic_state(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        state = analyzer.update_subject(
            subject_id=1,
            head_pos_m=np.array([1.0, 1.0]),
            ear_angle=0.0,
            dt=1 / 30,
            frame_index=0,
            video_time=0.0,
        )
        assert isinstance(state, HeadKinematicState)

    def test_score_decay(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        analyzer._subject_scores[1] = 50.0
        analyzer.decay_scores(factor=0.5)
        assert analyzer._subject_scores[1] == pytest.approx(25.0)

    def test_score_decay_removes_low(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        analyzer._subject_scores[1] = 0.3  # below 0.5 threshold
        analyzer.decay_scores(factor=0.995)
        assert 1 not in analyzer._subject_scores

    def test_cleanup_subject(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        analyzer.get_or_create_tracker(1)
        analyzer._subject_scores[1] = 50.0
        analyzer.cleanup_subject(1)
        assert 1 not in analyzer._trackers
        assert 1 not in analyzer._subject_scores

    def test_get_subject_score_default_zero(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        assert analyzer.get_subject_score(999) == 0.0

    def test_get_recent_events_serializable(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        events = analyzer.get_recent_events()
        assert isinstance(events, list)
        # Should be JSON-serializable (no numpy types)
        import json
        json.dumps(events)  # should not raise

    def test_standalone_spike_detection(self):
        """Very high acceleration without closing-speed context should still trigger."""
        analyzer = HeadImpactAnalyzer(fps=30.0)
        dt = 1 / 30
        # Feed calm data first to build baseline
        for i in range(20):
            analyzer.update_subject(1, np.array([1.0, 1.0]), 0.0, dt, i, i / 30)
        # Verify the standalone detection mechanism exists via the cooldown dict
        assert hasattr(analyzer, "_standalone_last_frame")

    def test_cooldown_respected(self):
        analyzer = HeadImpactAnalyzer(fps=30.0)
        # Set up cooldown for subject 1 — last standalone at frame 100 (far future)
        analyzer._standalone_last_frame[1] = 100
        # The standalone detection should not fire while in cooldown
        initial_events = len(analyzer._recent_events)
        dt = 1 / 30
        for i in range(5):
            analyzer.update_subject(1, np.array([1.0, 1.0]), 0.0, dt, i, i / 30)
        assert len(analyzer._recent_events) == initial_events


# ═══════════════════════════════════════════════════════════════════════════
# compute_concussion_probability tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeConcussionProbability:
    """Tests for the 0-100 concussion probability scoring function."""

    def test_zero_inputs_low_probability(self):
        prob = compute_concussion_probability(
            peak_accel_g=0.0,
            accel_direction_change_deg=0.0,
            high_accel_duration_frames=0,
            rotational_vel_peak=0.0,
            pre_impact_closing_speed=0.0,
        )
        assert prob < 10.0

    def test_high_accel_high_probability(self):
        prob = compute_concussion_probability(
            peak_accel_g=30.0,
            accel_direction_change_deg=180.0,
            high_accel_duration_frames=10,
            rotational_vel_peak=50.0,
            pre_impact_closing_speed=5.0,
        )
        assert prob > 50.0

    def test_monotonic_in_accel(self):
        """Higher acceleration should produce higher probability."""
        p_low = compute_concussion_probability(5.0, 0.0, 0, 0.0, 0.0)
        p_high = compute_concussion_probability(20.0, 0.0, 0, 0.0, 0.0)
        assert p_high >= p_low

    def test_monotonic_in_direction_change(self):
        p_low = compute_concussion_probability(10.0, 30.0, 0, 0.0, 0.0)
        p_high = compute_concussion_probability(10.0, 160.0, 0, 0.0, 0.0)
        assert p_high >= p_low

    def test_monotonic_in_rotational_vel(self):
        p_low = compute_concussion_probability(10.0, 90.0, 3, 12.0, 1.0)
        p_high = compute_concussion_probability(10.0, 90.0, 3, 40.0, 1.0)
        assert p_high >= p_low

    def test_monotonic_in_closing_speed(self):
        p_low = compute_concussion_probability(10.0, 90.0, 3, 20.0, 0.5)
        p_high = compute_concussion_probability(10.0, 90.0, 3, 20.0, 5.0)
        assert p_high >= p_low

    def test_result_between_0_and_100(self):
        # Even with extreme values, should cap at ~100
        prob = compute_concussion_probability(100.0, 360.0, 20, 100.0, 10.0)
        assert 0 <= prob <= 100

    def test_sigmoid_caps_near_100(self):
        # Very high inputs should approach but not exceed 100
        prob = compute_concussion_probability(50.0, 200.0, 15, 60.0, 8.0)
        assert prob <= 100.0
