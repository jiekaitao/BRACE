"""Tests for basketball movement profiles in backend/movement_guidelines.py."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from movement_guidelines import (
    match_guideline,
    MovementProfile,
    GENERIC_PROFILE,
    SQUAT_PROFILE,
    RUNNING_PROFILE,
    LANDING_PROFILE,
    CUTTING_PROFILE,
    SHOOTING_PROFILE,
    DRIBBLING_PROFILE,
    DEFENSE_PROFILE,
)


# ---------------------------------------------------------------------------
# 1. Profile existence tests
# ---------------------------------------------------------------------------

class TestBasketballProfileExistence:
    def test_landing_profile_exists(self):
        assert isinstance(LANDING_PROFILE, MovementProfile)

    def test_cutting_profile_exists(self):
        assert isinstance(CUTTING_PROFILE, MovementProfile)

    def test_shooting_profile_exists(self):
        assert isinstance(SHOOTING_PROFILE, MovementProfile)

    def test_dribbling_profile_exists(self):
        assert isinstance(DRIBBLING_PROFILE, MovementProfile)

    def test_defense_profile_exists(self):
        assert isinstance(DEFENSE_PROFILE, MovementProfile)


# ---------------------------------------------------------------------------
# 2. Keyword matching tests via match_guideline()
# ---------------------------------------------------------------------------

class TestBasketballKeywordMatching:
    def test_basketball_landing(self):
        assert match_guideline("basketball landing") is LANDING_PROFILE

    def test_cutting_movement(self):
        assert match_guideline("cutting movement") is CUTTING_PROFILE

    def test_shooting_form(self):
        assert match_guideline("shooting form") is SHOOTING_PROFILE

    def test_dribbling(self):
        assert match_guideline("dribbling") is DRIBBLING_PROFILE

    def test_defensive_stance(self):
        assert match_guideline("defensive stance") is DEFENSE_PROFILE

    def test_defense(self):
        assert match_guideline("defense") is DEFENSE_PROFILE

    def test_case_insensitive_landing(self):
        assert match_guideline("BASKETBALL LANDING") is LANDING_PROFILE

    def test_case_insensitive_shooting(self):
        assert match_guideline("Basketball Shooting") is SHOOTING_PROFILE

    def test_crossover_matches_cutting(self):
        assert match_guideline("crossover") is CUTTING_PROFILE

    def test_dribble_matches_dribbling(self):
        assert match_guideline("dribble") is DRIBBLING_PROFILE

    def test_ball_handling_matches_dribbling(self):
        assert match_guideline("ball handling") is DRIBBLING_PROFILE

    def test_free_throw_matches_shooting(self):
        assert match_guideline("free throw") is SHOOTING_PROFILE

    def test_layup_matches_shooting(self):
        assert match_guideline("layup") is SHOOTING_PROFILE

    def test_defensive_slide_matches_defense(self):
        assert match_guideline("defensive slide") is DEFENSE_PROFILE

    def test_guarding_matches_defense(self):
        assert match_guideline("guarding") is DEFENSE_PROFILE


# ---------------------------------------------------------------------------
# 3. Threshold integrity tests
# ---------------------------------------------------------------------------

class TestBasketballThresholdIntegrity:
    @pytest.fixture(params=[
        LANDING_PROFILE,
        CUTTING_PROFILE,
        SHOOTING_PROFILE,
        DRIBBLING_PROFILE,
        DEFENSE_PROFILE,
    ], ids=lambda p: p.name)
    def profile(self, request):
        return request.param

    def test_at_least_5_thresholds(self, profile):
        assert len(profile.thresholds) >= 5, (
            f"{profile.name} has only {len(profile.thresholds)} thresholds"
        )

    def test_medium_less_than_high(self, profile):
        for t in profile.thresholds:
            assert t.medium <= t.high, (
                f"{profile.name}: {t.risk_name} medium={t.medium} >= high={t.high}"
            )

    def test_all_thresholds_positive(self, profile):
        for t in profile.thresholds:
            assert t.medium > 0, (
                f"{profile.name}: {t.risk_name} medium={t.medium} not positive"
            )
            assert t.high > 0, (
                f"{profile.name}: {t.risk_name} high={t.high} not positive"
            )

    def test_has_form_cues(self, profile):
        assert len(profile.form_cues) >= 1, (
            f"{profile.name} has no form cues"
        )


# ---------------------------------------------------------------------------
# 4. Existing profiles unaffected
# ---------------------------------------------------------------------------

class TestExistingProfilesUnaffected:
    def test_squat_still_matches(self):
        assert match_guideline("squat") is SQUAT_PROFILE

    def test_running_still_matches(self):
        assert match_guideline("running") is RUNNING_PROFILE

    def test_bare_landing_still_matches_jump(self):
        """Bare 'landing' (without 'basketball') should still match JUMP_PROFILE."""
        from movement_guidelines import JUMP_PROFILE
        assert match_guideline("landing") is JUMP_PROFILE

    def test_unknown_still_generic(self):
        assert match_guideline("kayaking") is GENERIC_PROFILE

    def test_none_still_generic(self):
        assert match_guideline(None) is GENERIC_PROFILE


# ---------------------------------------------------------------------------
# 5. Basketball-specific threshold values
# ---------------------------------------------------------------------------

class TestBasketballSpecificThresholds:
    # --- LANDING ---
    def test_landing_tight_fppa(self):
        fppa = [t for t in LANDING_PROFILE.thresholds if t.risk_name == "acl_valgus"]
        assert len(fppa) >= 1
        assert fppa[0].medium == pytest.approx(10.0)
        assert fppa[0].high == pytest.approx(18.0)

    def test_landing_angular_velocity_enabled(self):
        av = [t for t in LANDING_PROFILE.thresholds
              if t.risk_name == "angular_velocity_spike"]
        assert len(av) >= 1
        assert all(t.enabled for t in av)

    def test_landing_angular_velocity_threshold(self):
        av = [t for t in LANDING_PROFILE.thresholds
              if t.risk_name == "angular_velocity_spike"]
        assert av[0].medium == pytest.approx(600.0)

    # --- CUTTING ---
    def test_cutting_tight_fppa(self):
        fppa = [t for t in CUTTING_PROFILE.thresholds if t.risk_name == "acl_valgus"]
        assert len(fppa) >= 1
        assert fppa[0].medium == pytest.approx(10.0)
        assert fppa[0].high == pytest.approx(18.0)

    def test_cutting_hip_drop_sensitive(self):
        hd = [t for t in CUTTING_PROFILE.thresholds if t.risk_name == "hip_drop"]
        assert len(hd) == 1
        assert hd[0].medium == pytest.approx(6.0)
        assert hd[0].high == pytest.approx(10.0)
        assert hd[0].enabled

    # --- SHOOTING ---
    def test_shooting_trunk_lean_important(self):
        tl = [t for t in SHOOTING_PROFILE.thresholds if t.risk_name == "trunk_lean"]
        assert len(tl) == 1
        assert tl[0].medium == pytest.approx(12.0)
        assert tl[0].enabled

    def test_shooting_hip_drop_disabled(self):
        hd = [t for t in SHOOTING_PROFILE.thresholds if t.risk_name == "hip_drop"]
        assert len(hd) == 1
        assert not hd[0].enabled

    def test_shooting_asymmetry_disabled(self):
        asym = [t for t in SHOOTING_PROFILE.thresholds if t.risk_name == "asymmetry"]
        assert len(asym) == 1
        assert not asym[0].enabled

    # --- DRIBBLING ---
    def test_dribbling_relaxed_fppa(self):
        fppa = [t for t in DRIBBLING_PROFILE.thresholds if t.risk_name == "acl_valgus"]
        assert len(fppa) >= 1
        assert fppa[0].medium == pytest.approx(15.0)
        assert fppa[0].high == pytest.approx(25.0)

    def test_dribbling_asymmetry_disabled(self):
        asym = [t for t in DRIBBLING_PROFILE.thresholds if t.risk_name == "asymmetry"]
        assert len(asym) == 1
        assert not asym[0].enabled

    # --- DEFENSE ---
    def test_defense_hip_drop_monitoring(self):
        hd = [t for t in DEFENSE_PROFILE.thresholds if t.risk_name == "hip_drop"]
        assert len(hd) == 1
        assert hd[0].enabled
        assert hd[0].medium == pytest.approx(6.0)
        assert hd[0].high == pytest.approx(10.0)

    def test_defense_asymmetry_disabled(self):
        asym = [t for t in DEFENSE_PROFILE.thresholds if t.risk_name == "asymmetry"]
        assert len(asym) == 1
        assert not asym[0].enabled
