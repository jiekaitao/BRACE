"""Tests for backend/voice_alerts.py — voice alert generation with cooldown and dedup."""

import pytest

from backend.voice_alerts import VoiceAlertGenerator, RISK_DESCRIPTIONS, JOINT_SPOKEN


def _high_risk(risk="acl_valgus", joint="left_knee"):
    return {"risk": risk, "joint": joint, "severity": "high", "value": 20.0, "threshold": 15.0}


def _medium_risk(risk="hip_drop", joint="pelvis"):
    return {"risk": risk, "joint": joint, "severity": "medium", "value": 8.0, "threshold": 7.0}


def _low_risk(risk="trunk_lean", joint="trunk"):
    return {"risk": risk, "joint": joint, "severity": "low", "value": 3.0, "threshold": 10.0}


class TestVoiceAlertGenerator:
    """Unit tests for VoiceAlertGenerator."""

    def test_no_risks_returns_none(self):
        gen = VoiceAlertGenerator()
        assert gen.generate_alert_text([], current_time=0.0) is None

    def test_high_severity_returns_alert(self):
        gen = VoiceAlertGenerator()
        text = gen.generate_alert_text([_high_risk()], current_time=0.0)
        assert text is not None
        assert "Warning" in text
        assert "knee valgus" in text

    def test_low_severity_returns_none(self):
        gen = VoiceAlertGenerator()
        assert gen.generate_alert_text([_low_risk()], current_time=0.0) is None

    def test_cooldown_returns_none_within_window(self):
        gen = VoiceAlertGenerator(cooldown_sec=8.0)
        # First alert at t=0
        text1 = gen.generate_alert_text([_high_risk()], current_time=0.0)
        assert text1 is not None
        # Second alert at t=5 (within cooldown) -> None
        text2 = gen.generate_alert_text(
            [_high_risk(risk="hip_drop", joint="pelvis")], current_time=5.0
        )
        assert text2 is None

    def test_high_severity_takes_precedence_over_medium(self):
        gen = VoiceAlertGenerator()
        risks = [_medium_risk(), _high_risk()]
        text = gen.generate_alert_text(risks, current_time=0.0)
        assert text is not None
        # Should pick the high risk (acl_valgus), not the medium (hip_drop)
        assert "knee valgus" in text
        assert "Warning" in text

    def test_sustained_medium_alerts_after_threshold(self):
        gen = VoiceAlertGenerator(cooldown_sec=0.0, sustained_threshold_sec=3.0)
        risk = _medium_risk()
        # First frame: starts tracking, no alert
        assert gen.generate_alert_text([risk], current_time=0.0) is None
        # At 2s: still within threshold
        assert gen.generate_alert_text([risk], current_time=2.0) is None
        # At 3.5s: past threshold, should alert
        text = gen.generate_alert_text([risk], current_time=3.5)
        assert text is not None
        assert "hips" in text

    def test_alert_text_includes_risk_and_joint(self):
        gen = VoiceAlertGenerator()
        text = gen.generate_alert_text([_high_risk()], current_time=0.0)
        assert "knee valgus" in text
        assert "left knee" in text

    def test_alert_text_includes_guideline_context(self):
        gen = VoiceAlertGenerator()
        guideline = {"name": "squat", "display_name": "Squat"}
        text = gen.generate_alert_text(
            [_high_risk()], active_guideline=guideline, current_time=0.0
        )
        assert text is not None
        assert text.startswith("During Squat:")

    def test_multiple_risks_picks_highest_severity(self):
        gen = VoiceAlertGenerator()
        risks = [
            _medium_risk(risk="hip_drop", joint="pelvis"),
            _high_risk(risk="acl_valgus", joint="right_knee"),
            _medium_risk(risk="trunk_lean", joint="trunk"),
        ]
        text = gen.generate_alert_text(risks, current_time=0.0)
        assert "knee valgus" in text
        assert "right knee" in text

    def test_cooldown_resets_after_timeout(self):
        gen = VoiceAlertGenerator(cooldown_sec=8.0)
        # First alert
        text1 = gen.generate_alert_text([_high_risk()], current_time=0.0)
        assert text1 is not None
        # Within cooldown
        assert gen.generate_alert_text(
            [_high_risk(risk="hip_drop", joint="pelvis")], current_time=5.0
        ) is None
        # After cooldown expires
        text2 = gen.generate_alert_text(
            [_high_risk(risk="hip_drop", joint="pelvis")], current_time=9.0
        )
        assert text2 is not None
        assert "hips" in text2

    def test_dedup_same_alert_returns_none(self):
        gen = VoiceAlertGenerator(cooldown_sec=0.0)
        # Same risk twice in a row (after cooldown)
        text1 = gen.generate_alert_text([_high_risk()], current_time=0.0)
        assert text1 is not None
        text2 = gen.generate_alert_text([_high_risk()], current_time=1.0)
        assert text2 is None  # Same text -> dedup
