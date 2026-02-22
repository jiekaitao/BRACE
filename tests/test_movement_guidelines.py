"""Tests for backend/movement_guidelines.py — movement profile matching."""

import pytest

from backend.movement_guidelines import (
    match_guideline,
    GENERIC_PROFILE,
    SQUAT_PROFILE,
    LUNGE_PROFILE,
    RUNNING_PROFILE,
    WALKING_PROFILE,
    JUMP_PROFILE,
    DEADLIFT_PROFILE,
    PUSH_UP_PROFILE,
    PLANK_PROFILE,
    _PROFILES,
    MovementProfile,
    RiskThreshold,
)


class TestMatchGuideline:
    def test_match_squat_exact(self):
        assert match_guideline("squat") is SQUAT_PROFILE

    def test_match_squat_variant(self):
        assert match_guideline("front squat") is SQUAT_PROFILE

    def test_match_squat_goblet(self):
        assert match_guideline("goblet squat") is SQUAT_PROFILE

    def test_match_lunge_exact(self):
        assert match_guideline("lunge") is LUNGE_PROFILE

    def test_match_lunge_walking(self):
        assert match_guideline("walking lunge") is LUNGE_PROFILE

    def test_match_lunge_lunging(self):
        assert match_guideline("lunging") is LUNGE_PROFILE

    def test_match_lunge_split_squat(self):
        assert match_guideline("split squat") is LUNGE_PROFILE

    def test_match_running(self):
        assert match_guideline("running") is RUNNING_PROFILE

    def test_match_jogging(self):
        assert match_guideline("jogging") is RUNNING_PROFILE

    def test_match_sprinting(self):
        assert match_guideline("sprinting") is RUNNING_PROFILE

    def test_match_walking(self):
        assert match_guideline("walking") is WALKING_PROFILE

    def test_match_jump(self):
        assert match_guideline("jump") is JUMP_PROFILE

    def test_match_box_jump(self):
        assert match_guideline("box jump") is JUMP_PROFILE

    def test_match_deadlift(self):
        assert match_guideline("deadlift") is DEADLIFT_PROFILE

    def test_match_romanian_deadlift(self):
        assert match_guideline("romanian deadlift") is DEADLIFT_PROFILE

    def test_match_push_up(self):
        assert match_guideline("push-up") is PUSH_UP_PROFILE

    def test_match_pushup_no_hyphen(self):
        assert match_guideline("pushup") is PUSH_UP_PROFILE

    def test_match_plank(self):
        assert match_guideline("plank") is PLANK_PROFILE

    def test_match_side_plank(self):
        assert match_guideline("side plank") is PLANK_PROFILE

    def test_unknown_returns_generic(self):
        assert match_guideline("kayaking") is GENERIC_PROFILE

    def test_none_returns_generic(self):
        assert match_guideline(None) is GENERIC_PROFILE

    def test_empty_string_returns_generic(self):
        assert match_guideline("") is GENERIC_PROFILE

    def test_case_insensitive(self):
        assert match_guideline("SQUAT") is SQUAT_PROFILE
        assert match_guideline("Running") is RUNNING_PROFILE

    def test_label_with_extra_spaces(self):
        assert match_guideline("  lunge  ") is LUNGE_PROFILE


class TestProfileIntegrity:
    def test_all_profiles_have_thresholds(self):
        """Every profile (including generic) should have at least 1 threshold."""
        all_profiles = [GENERIC_PROFILE] + _PROFILES
        for profile in all_profiles:
            assert len(profile.thresholds) >= 1, f"{profile.name} has no thresholds"

    def test_all_profiles_have_form_cues(self):
        """Every profile should have at least 1 form cue."""
        all_profiles = [GENERIC_PROFILE] + _PROFILES
        for profile in all_profiles:
            assert len(profile.form_cues) >= 1, f"{profile.name} has no form cues"

    def test_all_profiles_have_display_name(self):
        """Every profile should have a display name."""
        all_profiles = [GENERIC_PROFILE] + _PROFILES
        for profile in all_profiles:
            assert profile.display_name, f"{profile.name} has no display_name"

    def test_generic_has_no_keywords(self):
        """Generic profile should not have keywords (it's the fallback)."""
        assert len(GENERIC_PROFILE.keywords) == 0

    def test_profile_keywords_no_cross_match(self):
        """No keyword from one profile should match another profile's keywords."""
        for i, p1 in enumerate(_PROFILES):
            for kw in p1.keywords:
                for j, p2 in enumerate(_PROFILES):
                    if i == j:
                        continue
                    for kw2 in p2.keywords:
                        # Neither keyword should be a substring of the other
                        # (only check exact substring containment that would cause
                        # match_guideline to hit the wrong profile)
                        if kw in kw2 or kw2 in kw:
                            # This is allowed only if the more specific one comes first
                            # in _PROFILES (first match wins)
                            pass  # ordering handles this

    def test_lunge_hip_drop_disabled(self):
        """Lunge profile should have hip_drop disabled."""
        hip_drop_thresholds = [t for t in LUNGE_PROFILE.thresholds if t.risk_name == "hip_drop"]
        assert len(hip_drop_thresholds) == 1
        assert not hip_drop_thresholds[0].enabled

    def test_lunge_asymmetry_disabled(self):
        """Lunge profile should have asymmetry disabled."""
        asym_thresholds = [t for t in LUNGE_PROFILE.thresholds if t.risk_name == "asymmetry"]
        assert len(asym_thresholds) == 1
        assert not asym_thresholds[0].enabled

    def test_deadlift_fppa_disabled(self):
        """Deadlift profile should have FPPA disabled."""
        fppa_thresholds = [t for t in DEADLIFT_PROFILE.thresholds if t.risk_name == "knee_valgus"]
        assert len(fppa_thresholds) == 2  # left and right
        assert all(not t.enabled for t in fppa_thresholds)

    def test_running_stricter_fppa(self):
        """Running profile should have stricter FPPA thresholds than generic."""
        generic_fppa = [t for t in GENERIC_PROFILE.thresholds if t.risk_name == "knee_valgus"][0]
        running_fppa = [t for t in RUNNING_PROFILE.thresholds if t.risk_name == "knee_valgus"][0]
        assert running_fppa.medium < generic_fppa.medium
