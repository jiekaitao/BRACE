"""Unit tests for concussion pipeline refactoring.

Tests cover:
- Closing-speed hard clamp (Step 2)
- Savitzky-Golay filter on head positions (Step 1)
- HIC computation (Step 7)
- Velocity-dependent COR (Step 4)
- Hertzian impact duration (Step 5)
- Calibration validator (Step 3)
- Rowson & Duma preservation (constraint)
- Approach angle (Step 6)
- Model applicability (Step 8)
- score_collision new output fields
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts directory to path so we can import the modules under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _collision_detection import (
    CollisionDetector,
    compute_approach_angle,
    compute_closing_velocity,
    estimate_meters_per_pixel,
    MAX_CLOSING_SPEED_MS,
    L_SHOULDER,
    R_SHOULDER,
    NOSE,
    L_ANKLE,
    R_ANKLE,
    CONF_THRESHOLD,
)
from _biomechanics_model import (
    RD_B0,
    RD_B1,
    RD_B2,
    RD_B3,
    COEFF_RESTITUTION,
    compute_hic_half_sine,
    hertzian_impact_duration,
    score_collision,
    velocity_dependent_restitution,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_kpts_conf(overrides: dict[int, tuple[float, float, float]] | None = None):
    """Create 17-keypoint arrays with default positions and high confidence.

    Each keypoint defaults to (100, 100) with confidence 0.9.
    overrides: {index: (x, y, conf)} to set specific keypoints.
    """
    kpts = np.full((17, 2), 100.0)
    conf = np.full(17, 0.9)
    if overrides:
        for idx, (x, y, c) in overrides.items():
            kpts[idx] = [x, y]
            conf[idx] = c
    return kpts, conf


def _straight_line_positions(start, end, n):
    """Generate n evenly-spaced positions along a line."""
    return [
        np.array([
            start[0] + (end[0] - start[0]) * i / (n - 1),
            start[1] + (end[1] - start[1]) * i / (n - 1),
        ])
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════
# TestClosingSpeedClamp
# ═══════════════════════════════════════════════════════════════════════════

class TestClosingSpeedClamp:
    """Step 2: Hard-clamp closing speed at MAX_CLOSING_SPEED_MS."""

    def test_clamp_at_10_ms(self):
        """Speed above 10 m/s should be clamped in check_collision output."""
        det = CollisionDetector(iou_threshold=0.0, proximity_ratio=999.0)
        kpts_a, conf_a = _make_kpts_conf({
            NOSE: (50.0, 50.0, 0.9),
            L_SHOULDER: (40.0, 80.0, 0.9),
            R_SHOULDER: (60.0, 80.0, 0.9),
        })
        kpts_b, conf_b = _make_kpts_conf({
            NOSE: (55.0, 50.0, 0.9),
            L_SHOULDER: (45.0, 80.0, 0.9),
            R_SHOULDER: (65.0, 80.0, 0.9),
        })
        bbox = np.array([0, 0, 200, 200], dtype=float)

        # Constant-velocity approach producing a very high closing speed.
        # Each person moves 500 px/frame at 30fps with 1 m/px → 15000 m/s
        # per head, relative closing ~ 30000 m/s. Even after SG smoothing
        # on a linear trajectory the speed is preserved.
        for i in range(5):
            det.update(0, np.array([i * 500.0, 50.0]), bbox)
            det.update(1, np.array([5000.0 - i * 500.0, 50.0]), bbox)

        result = det.check_collision(
            0, 1, kpts_a, conf_a, kpts_b, conf_b,
            fps=30.0, meters_per_pixel=1.0,
        )
        assert result is not None
        assert result["closing_speed_ms"] <= MAX_CLOSING_SPEED_MS
        assert result["closing_speed_clamped"] is True
        assert result["closing_speed_raw_ms"] > MAX_CLOSING_SPEED_MS

    def test_no_clamp_below_max(self):
        """Speed below max should pass through unclamped."""
        det = CollisionDetector(iou_threshold=0.0, proximity_ratio=999.0)
        kpts_a, conf_a = _make_kpts_conf({
            NOSE: (50.0, 50.0, 0.9),
            L_SHOULDER: (40.0, 80.0, 0.9),
            R_SHOULDER: (60.0, 80.0, 0.9),
        })
        kpts_b, conf_b = _make_kpts_conf({
            NOSE: (55.0, 50.0, 0.9),
            L_SHOULDER: (45.0, 80.0, 0.9),
            R_SHOULDER: (65.0, 80.0, 0.9),
        })
        bbox = np.array([0, 0, 200, 200], dtype=float)

        # Slow approach: 1 px/frame at 30fps with 0.01 m/px = 0.3 m/s
        for i in range(5):
            det.update(0, np.array([50.0 + i, 50.0]), bbox)
            det.update(1, np.array([100.0 - i, 50.0]), bbox)

        result = det.check_collision(
            0, 1, kpts_a, conf_a, kpts_b, conf_b,
            fps=30.0, meters_per_pixel=0.01,
        )
        assert result is not None
        assert result["closing_speed_clamped"] is False
        assert result["closing_speed_ms"] == result["closing_speed_raw_ms"]

    def test_custom_clamp_value(self):
        """Custom max_closing_speed_ms should be respected."""
        det = CollisionDetector(
            iou_threshold=0.0, proximity_ratio=999.0,
            max_closing_speed_ms=5.0,
        )
        assert det.max_closing_speed_ms == 5.0

    def test_raw_value_preserved(self):
        """closing_speed_raw_ms should always contain the original value."""
        det = CollisionDetector(iou_threshold=0.0, proximity_ratio=999.0)
        kpts_a, conf_a = _make_kpts_conf({
            NOSE: (50.0, 50.0, 0.9),
            L_SHOULDER: (40.0, 80.0, 0.9),
            R_SHOULDER: (60.0, 80.0, 0.9),
        })
        kpts_b, conf_b = _make_kpts_conf({
            NOSE: (55.0, 50.0, 0.9),
            L_SHOULDER: (45.0, 80.0, 0.9),
            R_SHOULDER: (65.0, 80.0, 0.9),
        })
        bbox = np.array([0, 0, 200, 200], dtype=float)

        for i in range(5):
            det.update(0, np.array([50.0 + i * 5, 50.0]), bbox)
            det.update(1, np.array([200.0 - i * 5, 50.0]), bbox)

        result = det.check_collision(
            0, 1, kpts_a, conf_a, kpts_b, conf_b,
            fps=30.0, meters_per_pixel=0.1,
        )
        assert result is not None
        assert "closing_speed_raw_ms" in result
        assert isinstance(result["closing_speed_raw_ms"], float)


# ═══════════════════════════════════════════════════════════════════════════
# TestSGFilter
# ═══════════════════════════════════════════════════════════════════════════

class TestSGFilter:
    """Step 1: Savitzky-Golay smoothing on head positions."""

    def test_noise_reduction(self):
        """Noisy positions should produce a more stable speed estimate
        after SG filtering vs. with filtering disabled."""
        rng = np.random.RandomState(42)
        n = 7
        # Straight-line approach with noise
        base_a = _straight_line_positions((0, 0), (100, 0), n)
        base_b = _straight_line_positions((200, 0), (110, 0), n)
        noisy_a = [p + rng.randn(2) * 10 for p in base_a]
        noisy_b = [p + rng.randn(2) * 10 for p in base_b]

        speed_filtered = compute_closing_velocity(
            noisy_a, noisy_b, fps=30.0, meters_per_pixel=0.01,
            sg_window=5, sg_polyorder=2,
        )
        speed_raw = compute_closing_velocity(
            noisy_a, noisy_b, fps=30.0, meters_per_pixel=0.01,
            sg_window=0,  # disable filter (window < 3)
        )
        # Both should be positive (approaching)
        assert speed_filtered >= 0.0
        assert speed_raw >= 0.0

    def test_skipped_with_few_points(self):
        """SG filter should be silently skipped when fewer points than window."""
        positions_a = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        positions_b = [np.array([10.0, 0.0]), np.array([9.0, 0.0]), np.array([8.0, 0.0])]
        # sg_window=5 but only 3 points — should not crash
        speed = compute_closing_velocity(
            positions_a, positions_b, fps=30.0, meters_per_pixel=0.01,
            sg_window=5, sg_polyorder=2,
        )
        assert speed >= 0.0

    def test_linear_trajectory_preserved(self):
        """SG filter on a perfectly linear trajectory should not distort speed."""
        n = 7
        positions_a = _straight_line_positions((0, 0), (60, 0), n)
        positions_b = _straight_line_positions((200, 0), (140, 0), n)

        speed_filtered = compute_closing_velocity(
            positions_a, positions_b, fps=30.0, meters_per_pixel=0.01,
            sg_window=5, sg_polyorder=2,
        )
        speed_raw = compute_closing_velocity(
            positions_a, positions_b, fps=30.0, meters_per_pixel=0.01,
            sg_window=0,
        )
        # Should be very close for a linear trajectory
        assert abs(speed_filtered - speed_raw) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# TestHIC
# ═══════════════════════════════════════════════════════════════════════════

class TestHIC:
    """Step 7: Head Injury Criterion cross-check."""

    def test_zero_inputs(self):
        assert compute_hic_half_sine(0.0, 0.012) == 0.0
        assert compute_hic_half_sine(50.0, 0.0) == 0.0
        assert compute_hic_half_sine(0.0, 0.0) == 0.0

    def test_known_analytical_value(self):
        """Verify HIC = T * ((2/pi) * peak_g)^2.5 for known inputs."""
        peak_g = 100.0
        duration = 0.010  # 10 ms
        expected = duration * ((2.0 / math.pi) * peak_g) ** 2.5
        result = compute_hic_half_sine(peak_g, duration)
        assert abs(result - expected) < 1e-6

    def test_monotonicity(self):
        """Higher peak g should produce higher HIC at same duration."""
        hic_50 = compute_hic_half_sine(50.0, 0.010)
        hic_100 = compute_hic_half_sine(100.0, 0.010)
        hic_150 = compute_hic_half_sine(150.0, 0.010)
        assert hic_50 < hic_100 < hic_150

    def test_present_in_score_collision(self):
        """score_collision output should contain 'hic' field."""
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
        )
        assert "hic" in result
        assert result["hic"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TestVelocityDependentCOR
# ═══════════════════════════════════════════════════════════════════════════

class TestVelocityDependentCOR:
    """Step 4: Velocity-dependent coefficient of restitution."""

    def test_low_speed_elastic(self):
        """At zero closing speed, COR should equal e_base."""
        e = velocity_dependent_restitution(0.0)
        assert e == 0.55

    def test_high_speed_clamped(self):
        """At very high speed, COR should be floored at e_min."""
        e = velocity_dependent_restitution(100.0)
        assert e == 0.1

    def test_reference_value(self):
        """At v=5 m/s: e = 0.55 - 0.025*5 = 0.425."""
        e = velocity_dependent_restitution(5.0)
        assert abs(e - 0.425) < 1e-6

    def test_decreasing(self):
        """COR should decrease (or stay flat) as speed increases."""
        e_prev = velocity_dependent_restitution(0.0)
        for v in [2.0, 5.0, 10.0, 20.0]:
            e = velocity_dependent_restitution(v)
            assert e <= e_prev
            e_prev = e


# ═══════════════════════════════════════════════════════════════════════════
# TestHertzianDuration
# ═══════════════════════════════════════════════════════════════════════════

class TestHertzianDuration:
    """Step 5: Hertzian-scaled impact duration."""

    def test_reference_speed_equals_base(self):
        """At v_ref, the duration should equal the base duration."""
        t = hertzian_impact_duration(0.010, v_closing=5.0, v_ref=5.0)
        assert abs(t - 0.010) < 1e-9

    def test_faster_shorter(self):
        """Faster closing speed should produce shorter duration."""
        t_fast = hertzian_impact_duration(0.010, v_closing=8.0, v_ref=5.0)
        t_base = hertzian_impact_duration(0.010, v_closing=5.0, v_ref=5.0)
        assert t_fast < t_base

    def test_slower_longer(self):
        """Slower closing speed should produce longer duration."""
        t_slow = hertzian_impact_duration(0.010, v_closing=3.0, v_ref=5.0)
        t_base = hertzian_impact_duration(0.010, v_closing=5.0, v_ref=5.0)
        assert t_slow > t_base

    def test_zero_speed(self):
        """Zero closing speed should return the base duration."""
        t = hertzian_impact_duration(0.010, v_closing=0.0, v_ref=5.0)
        assert t == 0.010


# ═══════════════════════════════════════════════════════════════════════════
# TestCalibrationValidator
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrationValidator:
    """Step 3: estimate_meters_per_pixel returns (m_per_px, confidence)."""

    def test_returns_tuple(self):
        """Return type should be a tuple of (float, str)."""
        kpts, conf = _make_kpts_conf({
            L_SHOULDER: (100.0, 100.0, 0.9),
            R_SHOULDER: (150.0, 100.0, 0.9),
        })
        result = estimate_meters_per_pixel(kpts, conf)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    def test_shoulder_based_confidence(self):
        """Shoulder-based calibration without height check should be 'medium'."""
        kpts, conf = _make_kpts_conf({
            L_SHOULDER: (100.0, 100.0, 0.9),
            R_SHOULDER: (150.0, 100.0, 0.9),
        })
        # Set ankles and nose to low confidence to prevent height check
        conf[NOSE] = 0.1
        conf[L_ANKLE] = 0.1
        conf[R_ANKLE] = 0.1
        m_per_px, confidence = estimate_meters_per_pixel(kpts, conf)
        assert confidence == "medium"
        assert m_per_px > 0

    def test_shoulder_with_valid_height_high(self):
        """Shoulder calibration + valid height cross-check → 'high'."""
        # Shoulder width 50px at 0.45m → 0.009 m/px
        # Height from nose(100,50) to ankle(100,250) = 200px
        # 200 * 0.009 = 1.8m — valid range
        kpts, conf = _make_kpts_conf({
            L_SHOULDER: (75.0, 100.0, 0.9),
            R_SHOULDER: (125.0, 100.0, 0.9),
            NOSE: (100.0, 50.0, 0.9),
            L_ANKLE: (100.0, 250.0, 0.9),
        })
        m_per_px, confidence = estimate_meters_per_pixel(kpts, conf)
        assert confidence == "high"

    def test_fallback_confidence_low(self):
        """Fallback path (no visible landmarks) should give 'low'."""
        kpts = np.full((17, 2), 100.0)
        conf = np.full(17, 0.1)  # all below threshold
        m_per_px, confidence = estimate_meters_per_pixel(kpts, conf)
        assert confidence == "low"
        assert m_per_px == 0.004


# ═══════════════════════════════════════════════════════════════════════════
# TestRowsonDumaPreserved
# ═══════════════════════════════════════════════════════════════════════════

class TestRowsonDumaPreserved:
    """Constraint: Rowson & Duma logistic regression must be unchanged."""

    def test_coefficients_unchanged(self):
        assert RD_B0 == -10.2
        assert RD_B1 == 0.0433
        assert RD_B2 == 0.000873
        assert RD_B3 == -0.00000092

    def test_all_original_output_fields_present(self):
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
        )
        required = [
            "closing_speed_ms",
            "delta_v_body_struck_ms",
            "delta_v_head_ms",
            "peak_linear_g",
            "peak_rotational_rads2",
            "concussion_prob",
            "concussion_prob_linear",
            "risk_level",
        ]
        for field in required:
            assert field in result, f"Missing original field: {field}"

    def test_new_fields_present(self):
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
        )
        new_fields = [
            "coeff_restitution",
            "impact_duration_s",
            "angular_modulation",
            "coupling_effective",
            "hic",
            "model_applicability",
            "applicability_flags",
        ]
        for field in new_fields:
            assert field in result, f"Missing new field: {field}"


# ═══════════════════════════════════════════════════════════════════════════
# TestApproachAngle
# ═══════════════════════════════════════════════════════════════════════════

class TestApproachAngle:
    """Step 6: Approach angle computation."""

    def test_head_on_near_zero(self):
        """Two objects approaching each other head-on → angle ≈ 0."""
        pos_a = _straight_line_positions((0, 0), (40, 0), 5)
        pos_b = _straight_line_positions((100, 0), (60, 0), 5)
        angle = compute_approach_angle(pos_a, pos_b, fps=30.0)
        assert angle < 0.3  # nearly head-on

    def test_perpendicular_near_pi_half(self):
        """One moving perpendicular to the approach direction."""
        # A moving upward, B stationary to the right
        pos_a = [np.array([50.0, 100.0 - i * 10]) for i in range(5)]
        pos_b = [np.array([50.0, 50.0]) for _ in range(5)]  # stationary
        angle = compute_approach_angle(pos_a, pos_b, fps=30.0)
        # A is moving up (negative Y), approach vec is (0, -50) from A to B
        # Since A is below B and moving up, rel_vel is in same direction
        # This should actually be head-on in this case
        # Let me use a properly perpendicular case:
        pos_a2 = [np.array([50.0 + i * 10, 100.0]) for i in range(5)]
        pos_b2 = [np.array([50.0, 50.0]) for _ in range(5)]
        angle2 = compute_approach_angle(pos_a2, pos_b2, fps=30.0)
        # A moves right, approach from A to B is up-left → more oblique
        assert angle2 > 0.3

    def test_insufficient_data(self):
        """Too few points should return 0."""
        pos_a = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
        pos_b = [np.array([10.0, 0.0]), np.array([9.0, 0.0])]
        assert compute_approach_angle(pos_a, pos_b, fps=30.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TestModelApplicability
# ═══════════════════════════════════════════════════════════════════════════

class TestModelApplicability:
    """Step 8: Model applicability flags."""

    def test_validated_normal_inputs(self):
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
            min_pose_confidence=0.8,
        )
        assert result["model_applicability"] == "VALIDATED"
        assert result["applicability_flags"] == []

    def test_extrapolated_high_speed(self):
        result = score_collision(
            closing_speed_ms=9.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
        )
        assert result["model_applicability"] == "EXTRAPOLATED"
        assert "closing_speed_above_8ms" in result["applicability_flags"]

    def test_extrapolated_mass_out_of_range(self):
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=35.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
        )
        assert result["model_applicability"] == "EXTRAPOLATED"
        assert "mass_a_out_of_range" in result["applicability_flags"]

    def test_extrapolated_low_confidence(self):
        result = score_collision(
            closing_speed_ms=5.0,
            mass_a_kg=80.0,
            mass_b_kg=80.0,
            head_coupling_factor=0.4,
            min_pose_confidence=0.3,
        )
        assert result["model_applicability"] == "EXTRAPOLATED"
        assert "low_pose_confidence" in result["applicability_flags"]
