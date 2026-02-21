"""Tests for basketball movement profiles in backend/movement_guidelines.py."""

import pytest

from backend.movement_guidelines import (
    match_guideline,
    GENERIC_PROFILE,
    JUMP_PROFILE,
    RUNNING_PROFILE,
    BASKETBALL_LANDING_PROFILE,
    BASKETBALL_CUTTING_PROFILE,
    BASKETBALL_SHOOTING_PROFILE,
    BASKETBALL_DRIBBLING_PROFILE,
    BASKETBALL_DEFENSE_PROFILE,
    _PROFILES,
)


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

class TestBasketballLandingMatch:
    def test_dunk(self):
        assert match_guideline("dunk") is BASKETBALL_LANDING_PROFILE

    def test_dunking(self):
        assert match_guideline("dunking") is BASKETBALL_LANDING_PROFILE

    def test_block(self):
        assert match_guideline("block") is BASKETBALL_LANDING_PROFILE

    def test_rebound(self):
        assert match_guideline("rebound") is BASKETBALL_LANDING_PROFILE

    def test_slam_dunk(self):
        assert match_guideline("slam dunk") is BASKETBALL_LANDING_PROFILE

    def test_alley_oop(self):
        assert match_guideline("alley-oop") is BASKETBALL_LANDING_PROFILE


class TestBasketballCuttingMatch:
    def test_crossover(self):
        assert match_guideline("crossover") is BASKETBALL_CUTTING_PROFILE

    def test_euro_step(self):
        assert match_guideline("euro step") is BASKETBALL_CUTTING_PROFILE

    def test_cutting(self):
        assert match_guideline("cutting") is BASKETBALL_CUTTING_PROFILE

    def test_direction_change(self):
        assert match_guideline("direction change") is BASKETBALL_CUTTING_PROFILE


class TestBasketballShootingMatch:
    def test_shooting(self):
        assert match_guideline("shooting") is BASKETBALL_SHOOTING_PROFILE

    def test_jump_shot(self):
        assert match_guideline("jump shot") is BASKETBALL_SHOOTING_PROFILE

    def test_free_throw(self):
        assert match_guideline("free throw") is BASKETBALL_SHOOTING_PROFILE

    def test_layup(self):
        assert match_guideline("layup") is BASKETBALL_SHOOTING_PROFILE

    def test_three_pointer(self):
        assert match_guideline("three pointer") is BASKETBALL_SHOOTING_PROFILE


class TestBasketballDribblingMatch:
    def test_dribbling(self):
        assert match_guideline("dribbling") is BASKETBALL_DRIBBLING_PROFILE

    def test_ball_handling(self):
        assert match_guideline("ball handling") is BASKETBALL_DRIBBLING_PROFILE


class TestBasketballDefenseMatch:
    def test_defensive_slide(self):
        assert match_guideline("defensive slide") is BASKETBALL_DEFENSE_PROFILE

    def test_guarding(self):
        assert match_guideline("guarding") is BASKETBALL_DEFENSE_PROFILE

    def test_closeout(self):
        assert match_guideline("closeout") is BASKETBALL_DEFENSE_PROFILE

    def test_defensive_shuffle(self):
        assert match_guideline("defensive shuffle") is BASKETBALL_DEFENSE_PROFILE


class TestBasketballCaseInsensitive:
    def test_upper_case(self):
        assert match_guideline("DUNK") is BASKETBALL_LANDING_PROFILE
        assert match_guideline("CROSSOVER") is BASKETBALL_CUTTING_PROFILE

    def test_mixed_case(self):
        assert match_guideline("Jump Shot") is BASKETBALL_SHOOTING_PROFILE


# ---------------------------------------------------------------------------
# Non-basketball labels still match generic/existing profiles
# ---------------------------------------------------------------------------

class TestBasketballDoesNotOverrideOthers:
    def test_generic_jump_still_matches_jump_profile(self):
        """'jump' alone should match the generic JUMP_PROFILE, not basketball."""
        assert match_guideline("jump") is JUMP_PROFILE

    def test_running_still_matches(self):
        assert match_guideline("running") is RUNNING_PROFILE

    def test_unknown_still_generic(self):
        assert match_guideline("swimming") is GENERIC_PROFILE


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------

class TestBasketballThresholdIntegrity:
    @pytest.mark.parametrize("profile", [
        BASKETBALL_LANDING_PROFILE,
        BASKETBALL_CUTTING_PROFILE,
        BASKETBALL_SHOOTING_PROFILE,
        BASKETBALL_DRIBBLING_PROFILE,
        BASKETBALL_DEFENSE_PROFILE,
    ])
    def test_has_thresholds(self, profile):
        assert len(profile.thresholds) >= 5

    @pytest.mark.parametrize("profile", [
        BASKETBALL_LANDING_PROFILE,
        BASKETBALL_CUTTING_PROFILE,
        BASKETBALL_SHOOTING_PROFILE,
        BASKETBALL_DRIBBLING_PROFILE,
        BASKETBALL_DEFENSE_PROFILE,
    ])
    def test_has_form_cues(self, profile):
        assert len(profile.form_cues) >= 3

    @pytest.mark.parametrize("profile", [
        BASKETBALL_LANDING_PROFILE,
        BASKETBALL_CUTTING_PROFILE,
        BASKETBALL_SHOOTING_PROFILE,
        BASKETBALL_DRIBBLING_PROFILE,
        BASKETBALL_DEFENSE_PROFILE,
    ])
    def test_has_keywords(self, profile):
        assert len(profile.keywords) >= 2

    def test_landing_strictest_fppa(self):
        """Landing profile should have the tightest FPPA thresholds."""
        landing_fppa = [t for t in BASKETBALL_LANDING_PROFILE.thresholds
                        if t.risk_name == "acl_valgus" and t.enabled][0]
        assert landing_fppa.medium == 8.0
        assert landing_fppa.high == 15.0

    def test_cutting_tight_fppa(self):
        """Cutting profile should have tight FPPA for plant-leg ACL risk."""
        cutting_fppa = [t for t in BASKETBALL_CUTTING_PROFILE.thresholds
                        if t.risk_name == "acl_valgus" and t.enabled][0]
        assert cutting_fppa.medium == 10.0
        assert cutting_fppa.high == 16.0

    def test_defense_low_angular_velocity(self):
        """Defense profile should have lower angular velocity thresholds."""
        defense_av = [t for t in BASKETBALL_DEFENSE_PROFILE.thresholds
                      if t.risk_name == "angular_velocity_spike"][0]
        landing_av = [t for t in BASKETBALL_LANDING_PROFILE.thresholds
                      if t.risk_name == "angular_velocity_spike"][0]
        assert defense_av.medium < landing_av.medium

    def test_shooting_asymmetry_disabled(self):
        """Shooting profile should have asymmetry disabled (dominant hand)."""
        asym = [t for t in BASKETBALL_SHOOTING_PROFILE.thresholds
                if t.risk_name == "asymmetry"]
        assert len(asym) == 1
        assert not asym[0].enabled

    def test_cutting_asymmetry_disabled(self):
        """Cutting profile should have asymmetry disabled (plant leg dominant)."""
        asym = [t for t in BASKETBALL_CUTTING_PROFILE.thresholds
                if t.risk_name == "asymmetry"]
        assert len(asym) == 1
        assert not asym[0].enabled

    def test_all_basketball_profiles_in_profiles_list(self):
        """All 5 basketball profiles should be in the _PROFILES list."""
        basketball_profiles = [
            BASKETBALL_LANDING_PROFILE,
            BASKETBALL_CUTTING_PROFILE,
            BASKETBALL_SHOOTING_PROFILE,
            BASKETBALL_DRIBBLING_PROFILE,
            BASKETBALL_DEFENSE_PROFILE,
        ]
        for p in basketball_profiles:
            assert p in _PROFILES, f"{p.name} not in _PROFILES"
