"""Tests for backend/risk_profile.py — personalized risk threshold scaling."""

import pytest

from backend.risk_profile import (
    RiskModifiers,
    DEFAULT_THRESHOLDS,
    apply_modifiers,
)
from backend.movement_quality import evaluate_injury_risks


# --- RiskModifiers dataclass ---


class TestRiskModifiersCreation:
    def test_default_all_ones(self):
        m = RiskModifiers()
        assert m.fppa_scale == 1.0
        assert m.hip_drop_scale == 1.0
        assert m.trunk_lean_scale == 1.0
        assert m.asymmetry_scale == 1.0
        assert m.angular_velocity_scale == 1.0
        assert m.monitor_joints == []

    def test_custom_values(self):
        m = RiskModifiers(fppa_scale=0.65, hip_drop_scale=0.7, monitor_joints=["left_knee"])
        assert m.fppa_scale == 0.65
        assert m.hip_drop_scale == 0.7
        assert m.trunk_lean_scale == 1.0
        assert m.monitor_joints == ["left_knee"]

    def test_to_dict(self):
        m = RiskModifiers(fppa_scale=0.5, monitor_joints=["right_knee"])
        d = m.to_dict()
        assert d["fppa_scale"] == 0.5
        assert d["monitor_joints"] == ["right_knee"]
        assert isinstance(d, dict)

    def test_from_dict_full(self):
        d = {
            "fppa_scale": 0.65,
            "hip_drop_scale": 0.7,
            "trunk_lean_scale": 0.8,
            "asymmetry_scale": 0.9,
            "angular_velocity_scale": 0.5,
            "monitor_joints": ["left_knee", "right_knee"],
        }
        m = RiskModifiers.from_dict(d)
        assert m.fppa_scale == 0.65
        assert m.hip_drop_scale == 0.7
        assert m.trunk_lean_scale == 0.8
        assert m.asymmetry_scale == 0.9
        assert m.angular_velocity_scale == 0.5
        assert m.monitor_joints == ["left_knee", "right_knee"]

    def test_from_dict_partial(self):
        d = {"fppa_scale": 0.5}
        m = RiskModifiers.from_dict(d)
        assert m.fppa_scale == 0.5
        assert m.hip_drop_scale == 1.0  # default
        assert m.monitor_joints == []

    def test_from_dict_none(self):
        m = RiskModifiers.from_dict(None)
        assert m.fppa_scale == 1.0
        assert m.hip_drop_scale == 1.0

    def test_from_dict_empty(self):
        m = RiskModifiers.from_dict({})
        assert m.fppa_scale == 1.0

    def test_roundtrip(self):
        original = RiskModifiers(fppa_scale=0.65, monitor_joints=["left_knee"])
        restored = RiskModifiers.from_dict(original.to_dict())
        assert restored.fppa_scale == original.fppa_scale
        assert restored.monitor_joints == original.monitor_joints


# --- apply_modifiers ---


class TestApplyModifiers:
    def test_none_returns_defaults(self):
        result = apply_modifiers(None)
        assert result == {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}

    def test_default_modifiers_match_defaults(self):
        result = apply_modifiers(RiskModifiers())
        for metric in DEFAULT_THRESHOLDS:
            for level in DEFAULT_THRESHOLDS[metric]:
                assert result[metric][level] == pytest.approx(DEFAULT_THRESHOLDS[metric][level])

    def test_fppa_scale(self):
        m = RiskModifiers(fppa_scale=0.65)
        result = apply_modifiers(m)
        assert result["fppa"]["medium"] == pytest.approx(15.0 * 0.65)
        assert result["fppa"]["high"] == pytest.approx(25.0 * 0.65)
        # Other thresholds unchanged
        assert result["hip_drop"]["medium"] == pytest.approx(8.0)
        assert result["hip_drop"]["high"] == pytest.approx(12.0)

    def test_hip_drop_scale(self):
        m = RiskModifiers(hip_drop_scale=0.7)
        result = apply_modifiers(m)
        assert result["hip_drop"]["medium"] == pytest.approx(8.0 * 0.7)
        assert result["hip_drop"]["high"] == pytest.approx(12.0 * 0.7)

    def test_trunk_lean_scale(self):
        m = RiskModifiers(trunk_lean_scale=0.7)
        result = apply_modifiers(m)
        assert result["trunk_lean"]["medium"] == pytest.approx(15.0 * 0.7)
        assert result["trunk_lean"]["high"] == pytest.approx(25.0 * 0.7)

    def test_asymmetry_scale(self):
        m = RiskModifiers(asymmetry_scale=0.8)
        result = apply_modifiers(m)
        assert result["asymmetry"]["medium"] == pytest.approx(15.0 * 0.8)
        assert result["asymmetry"]["high"] == pytest.approx(25.0 * 0.8)

    def test_angular_velocity_scale(self):
        m = RiskModifiers(angular_velocity_scale=0.8)
        result = apply_modifiers(m)
        assert result["angular_velocity"]["medium"] == pytest.approx(500.0 * 0.8)

    def test_multiple_scales(self):
        m = RiskModifiers(fppa_scale=0.5, hip_drop_scale=0.6, trunk_lean_scale=0.7)
        result = apply_modifiers(m)
        assert result["fppa"]["medium"] == pytest.approx(15.0 * 0.5)
        assert result["hip_drop"]["medium"] == pytest.approx(8.0 * 0.6)
        assert result["trunk_lean"]["medium"] == pytest.approx(15.0 * 0.7)
        # Unset scales remain at 1.0
        assert result["asymmetry"]["medium"] == pytest.approx(15.0)

    def test_stacking_via_manual_multiply(self):
        """Stacking means the user computes fppa_scale = 0.65 * 0.8 = 0.52."""
        m = RiskModifiers(fppa_scale=0.65 * 0.8)
        result = apply_modifiers(m)
        assert result["fppa"]["medium"] == pytest.approx(15.0 * 0.52)
        assert result["fppa"]["high"] == pytest.approx(25.0 * 0.52)


# --- Integration with evaluate_injury_risks ---


class TestEvaluateInjuryRisksWithModifiers:
    def test_fppa_below_default_no_risk(self):
        """FPPA=14 with default thresholds -> no risk."""
        bio = {"fppa_left": 14.0, "fppa_right": 0.0}
        risks = evaluate_injury_risks(bio)
        fppa_risks = [r for r in risks if r["risk"] == "knee_valgus"]
        assert len(fppa_risks) == 0

    def test_fppa_below_default_with_modifier_triggers_risk(self):
        """FPPA=14 with fppa_scale=0.65 -> medium risk (14 > 9.75)."""
        bio = {"fppa_left": 14.0, "fppa_right": 0.0}
        m = RiskModifiers(fppa_scale=0.65)
        risks = evaluate_injury_risks(bio, modifiers=m)
        fppa_risks = [r for r in risks if r["risk"] == "knee_valgus"]
        assert len(fppa_risks) == 1
        assert fppa_risks[0]["severity"] == "medium"
        assert fppa_risks[0]["joint"] == "left_knee"
        assert fppa_risks[0]["threshold"] == pytest.approx(15.0 * 0.65)

    def test_hip_drop_below_default_no_risk(self):
        """Hip drop=7 with default thresholds -> no risk."""
        bio = {"hip_drop": 7.0}
        risks = evaluate_injury_risks(bio)
        hd_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hd_risks) == 0

    def test_hip_drop_below_default_with_modifier_triggers_risk(self):
        """Hip drop=7 with hip_drop_scale=0.7 -> medium risk (7 > 5.6)."""
        bio = {"hip_drop": 7.0}
        m = RiskModifiers(hip_drop_scale=0.7)
        risks = evaluate_injury_risks(bio, modifiers=m)
        hd_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hd_risks) == 1
        assert hd_risks[0]["severity"] == "medium"
        assert hd_risks[0]["threshold"] == pytest.approx(8.0 * 0.7)

    def test_trunk_lean_with_modifier(self):
        """Trunk lean=12 with trunk_lean_scale=0.7 -> medium risk (12 > 10.5)."""
        bio = {"trunk_lean": 12.0}
        m = RiskModifiers(trunk_lean_scale=0.7)
        risks = evaluate_injury_risks(bio, modifiers=m)
        tl_risks = [r for r in risks if r["risk"] == "trunk_lean"]
        assert len(tl_risks) == 1
        assert tl_risks[0]["severity"] == "medium"

    def test_asymmetry_with_modifier(self):
        """Asymmetry=13 with asymmetry_scale=0.8 -> medium risk (13 > 12)."""
        bio = {"asymmetry": 13.0}
        m = RiskModifiers(asymmetry_scale=0.8)
        risks = evaluate_injury_risks(bio, modifiers=m)
        asym_risks = [r for r in risks if r["risk"] == "asymmetry"]
        assert len(asym_risks) == 1
        assert asym_risks[0]["severity"] == "medium"

    def test_angular_velocity_with_modifier(self):
        """Velocity=450 with angular_velocity_scale=0.8 -> risk (450 > 400)."""
        bio = {}
        av = {"left_knee": 450.0}
        m = RiskModifiers(angular_velocity_scale=0.8)
        risks = evaluate_injury_risks(bio, angular_velocities=av, modifiers=m)
        av_risks = [r for r in risks if r["risk"] == "angular_velocity_spike"]
        assert len(av_risks) == 1
        assert av_risks[0]["threshold"] == pytest.approx(500.0 * 0.8)

    def test_modifiers_ignored_when_profile_provided(self):
        """When a movement profile is provided, modifiers are NOT used."""
        bio = {"fppa_left": 14.0, "fppa_right": 0.0}
        m = RiskModifiers(fppa_scale=0.65)

        # Create a minimal fake profile with thresholds
        class FakeThreshold:
            def __init__(self):
                self.enabled = True
                self.metric = "fppa"
                self.joint = "left_knee"
                self.risk_name = "knee_valgus"
                self.medium = 15.0
                self.high = 25.0

        class FakeProfile:
            def __init__(self):
                self.thresholds = [FakeThreshold()]

        profile = FakeProfile()
        # With profile, the profile's thresholds are used, not modifiers
        risks = evaluate_injury_risks(bio, profile=profile, modifiers=m)
        # 14 < 15 (profile threshold), so no risk from profile path
        fppa_risks = [r for r in risks if r["risk"] == "knee_valgus"]
        assert len(fppa_risks) == 0

    def test_high_severity_with_modifier(self):
        """FPPA=20 with fppa_scale=0.65 -> high risk (20 > 16.25)."""
        bio = {"fppa_left": 20.0, "fppa_right": 0.0}
        m = RiskModifiers(fppa_scale=0.65)
        risks = evaluate_injury_risks(bio, modifiers=m)
        fppa_risks = [r for r in risks if r["risk"] == "knee_valgus"]
        assert len(fppa_risks) == 1
        assert fppa_risks[0]["severity"] == "high"
        assert fppa_risks[0]["threshold"] == pytest.approx(25.0 * 0.65)

    def test_default_modifiers_same_as_no_modifiers(self):
        """RiskModifiers() should produce identical results to no modifiers."""
        bio = {
            "fppa_left": 20.0, "fppa_right": 10.0,
            "hip_drop": 10.0, "trunk_lean": 18.0, "asymmetry": 20.0,
        }
        av = {"left_knee": 550.0, "right_knee": 200.0}
        risks_none = evaluate_injury_risks(bio, angular_velocities=av)
        risks_default = evaluate_injury_risks(bio, angular_velocities=av, modifiers=RiskModifiers())
        # Same number and content of risks
        assert len(risks_none) == len(risks_default)
        for r1, r2 in zip(risks_none, risks_default):
            assert r1["joint"] == r2["joint"]
            assert r1["risk"] == r2["risk"]
            assert r1["severity"] == r2["severity"]
            assert r1["value"] == pytest.approx(r2["value"])
            assert r1["threshold"] == pytest.approx(r2["threshold"])
