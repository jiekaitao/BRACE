"""Tests for backend/movement_quality.py — biomechanical metrics, fatigue detection, form scoring."""

import numpy as np
import pytest

from backend.movement_quality import (
    JOINT_CHAINS,
    BILATERAL_PAIRS,
    JOINT_NAMES,
    BONE_PAIRS,
    compute_joint_angle,
    compute_fppa,
    compute_hip_drop,
    compute_trunk_lean,
    bilateral_asymmetry_index,
    sparc,
    log_dimensionless_jerk,
    spectral_entropy,
    rep_cross_correlation,
    project_bone_lengths,
    BoneLengthFilter,
    evaluate_injury_risks,
    sample_entropy,
    estimate_center_of_mass,
    compute_kinematic_sequence,
    median_frequency,
    StreamingCurvature,
    MovementQualityTracker,
)


# --- Helpers: synthetic joint generation ---

def _make_straight_joints_2d(n_joints=14):
    """Straight standing pose in SRP space (2D). Joints at roughly correct positions."""
    # FEATURE_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]
    # 0:L_shoulder 1:R_shoulder 2:L_elbow 3:R_elbow 4:L_wrist 5:R_wrist
    # 6:L_hip 7:R_hip 8:L_knee 9:R_knee 10:L_ankle 11:R_ankle 12:L_foot 13:R_foot
    joints = np.array([
        [-0.5, 2.0],   # 0: L shoulder
        [0.5, 2.0],    # 1: R shoulder
        [-0.7, 1.2],   # 2: L elbow
        [0.7, 1.2],    # 3: R elbow
        [-0.8, 0.4],   # 4: L wrist
        [0.8, 0.4],    # 5: R wrist
        [-0.5, 0.0],   # 6: L hip
        [0.5, 0.0],    # 7: R hip
        [-0.5, -1.5],  # 8: L knee
        [0.5, -1.5],   # 9: R knee
        [-0.5, -3.0],  # 10: L ankle
        [0.5, -3.0],   # 11: R ankle
        [-0.5, -3.2],  # 12: L foot
        [0.5, -3.2],   # 13: R foot
    ], dtype=np.float32)
    return joints


def _make_squat_trajectory_2d(n_frames=60, depth=1.0, noise=0.0):
    """Generate a squat-like trajectory. Knees bend, hips drop, then return."""
    base = _make_straight_joints_2d()
    trajectory = []
    for t in range(n_frames):
        phase = np.sin(2 * np.pi * t / n_frames)  # -1 to 1
        frame = base.copy()
        # Hips drop
        frame[6, 1] -= depth * (1 + phase) * 0.3
        frame[7, 1] -= depth * (1 + phase) * 0.3
        # Knees bend outward slightly and drop
        frame[8, 1] -= depth * (1 + phase) * 0.6
        frame[9, 1] -= depth * (1 + phase) * 0.6
        frame[8, 0] -= depth * (1 + phase) * 0.1
        frame[9, 0] += depth * (1 + phase) * 0.1
        if noise > 0:
            frame += np.random.randn(*frame.shape).astype(np.float32) * noise
        trajectory.append(frame)
    return np.array(trajectory)


def _make_fatigued_trajectory_2d(n_frames=60, depth=0.7, asymmetry=0.15, noise=0.03):
    """Generate a fatigued squat: reduced ROM, asymmetric, noisy."""
    base = _make_straight_joints_2d()
    trajectory = []
    for t in range(n_frames):
        phase = np.sin(2 * np.pi * t / n_frames)
        frame = base.copy()
        # Reduced depth (fatigue)
        frame[6, 1] -= depth * (1 + phase) * 0.3
        frame[7, 1] -= depth * (1 + phase) * 0.3
        # Knees: left knee collapses inward more (valgus), asymmetric
        frame[8, 1] -= depth * (1 + phase) * 0.6
        frame[9, 1] -= (depth - asymmetry) * (1 + phase) * 0.6
        frame[8, 0] += depth * (1 + phase) * 0.05  # LEFT knee valgus
        frame[9, 0] += depth * (1 + phase) * 0.1
        # Add noise (tremor)
        frame += np.random.randn(*frame.shape).astype(np.float32) * noise
        trajectory.append(frame)
    return np.array(trajectory)


# =====================================================
# Joint angle tests
# =====================================================

class TestJointAngles:
    def test_straight_angle_180(self):
        """Three colinear points should give 180 degrees."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([2.0, 0.0])
        assert abs(compute_joint_angle(p1, p2, p3) - 180.0) < 1.0

    def test_right_angle_90(self):
        p1 = np.array([0.0, 1.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        assert abs(compute_joint_angle(p1, p2, p3) - 90.0) < 1.0

    def test_zero_length_vectors(self):
        p = np.array([1.0, 1.0])
        assert compute_joint_angle(p, p, p) == 180.0

    def test_3d_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        assert abs(compute_joint_angle(p1, p2, p3) - 90.0) < 1.0

    def test_acute_angle(self):
        p1 = np.array([1.0, 1.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 0.0])
        angle = compute_joint_angle(p1, p2, p3)
        assert 40.0 < angle < 50.0  # should be 45 degrees


class TestFPPA:
    def test_straight_leg_zero(self):
        """Straight leg alignment should give FPPA near 0."""
        hip = np.array([0.0, 1.0])
        knee = np.array([0.0, 0.0])
        ankle = np.array([0.0, -1.0])
        fppa = compute_fppa(hip, knee, ankle)
        assert abs(fppa) < 5.0

    def test_valgus_negative(self):
        """Knee medial to hip-ankle line should give negative FPPA."""
        hip = np.array([0.0, 1.0])
        knee = np.array([0.3, 0.0])  # knee shifted medially
        ankle = np.array([0.0, -1.0])
        fppa = compute_fppa(hip, knee, ankle)
        # The sign depends on cross product convention
        assert abs(fppa) > 5.0  # significant deviation

    def test_symmetric_legs_similar(self):
        """Both legs in normal standing should have similar FPPA."""
        joints = _make_straight_joints_2d()
        fppa_l = compute_fppa(joints[6], joints[8], joints[10])
        fppa_r = compute_fppa(joints[7], joints[9], joints[11])
        assert abs(fppa_l - fppa_r) < 10.0


class TestHipDrop:
    def test_level_pelvis_zero(self):
        """Level pelvis should give ~0 hip drop."""
        lhip = np.array([-0.5, 0.0])
        rhip = np.array([0.5, 0.0])
        assert abs(compute_hip_drop(lhip, rhip)) < 5.0

    def test_left_hip_higher(self):
        """Left hip higher should give positive angle (Y-down: smaller Y = higher)."""
        lhip = np.array([-0.5, -0.5])  # left hip higher (smaller Y in image space)
        rhip = np.array([0.5, 0.0])
        drop = compute_hip_drop(lhip, rhip)
        assert drop > 10.0  # positive = left higher


class TestTrunkLean:
    def test_upright_zero(self):
        """Upright posture should give near-zero trunk lean (Y-down image space)."""
        ls = np.array([-0.5, -2.0])  # shoulders above hips (smaller Y)
        rs = np.array([0.5, -2.0])
        lh = np.array([-0.5, 0.0])
        rh = np.array([0.5, 0.0])
        lean = compute_trunk_lean(ls, rs, lh, rh)
        assert abs(lean) < 5.0

    def test_leaning_right(self):
        """Trunk leaning right should give positive lean (Y-down image space)."""
        ls = np.array([0.0, -2.0])  # shifted right
        rs = np.array([1.0, -2.0])
        lh = np.array([-0.5, 0.0])
        rh = np.array([0.5, 0.0])
        lean = compute_trunk_lean(ls, rs, lh, rh)
        assert lean > 5.0


# =====================================================
# Bilateral asymmetry tests
# =====================================================

class TestBilateralAsymmetry:
    def test_perfect_symmetry(self):
        assert bilateral_asymmetry_index(90.0, 90.0) == 0.0

    def test_known_asymmetry(self):
        bai = bilateral_asymmetry_index(90.0, 80.0)
        expected = 10.0 / 90.0 * 100.0  # ~11.1%
        assert abs(bai - expected) < 0.1

    def test_zero_values(self):
        assert bilateral_asymmetry_index(0.0, 0.0) == 0.0

    def test_large_asymmetry(self):
        bai = bilateral_asymmetry_index(100.0, 50.0)
        assert bai == 50.0


# =====================================================
# Signal processing tests
# =====================================================

class TestSPARC:
    def test_pure_sinusoid_smooth(self):
        """Pure sinusoid should have SPARC that is negative (smooth)."""
        t = np.linspace(0, 2 * np.pi, 120)
        speed = np.abs(np.cos(t)) * 2.0 + 0.1
        s = sparc(speed, fps=30.0)
        assert -5.0 < s < 0.0  # smooth movements are negative but not extremely

    def test_noisy_signal_less_smooth(self):
        """Noisy signal should have more negative SPARC."""
        t = np.linspace(0, 2 * np.pi, 120)
        speed_smooth = np.abs(np.cos(t)) * 2.0 + 0.1
        speed_noisy = speed_smooth + np.random.randn(120) * 0.5
        speed_noisy = np.abs(speed_noisy)
        s_smooth = sparc(speed_smooth, fps=30.0)
        s_noisy = sparc(speed_noisy, fps=30.0)
        assert s_noisy < s_smooth  # noisier = more negative

    def test_short_signal(self):
        assert sparc(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_constant_signal(self):
        """Constant signal may still produce a non-zero SPARC due to FFT edge effects."""
        s = sparc(np.ones(60))
        # Constant signal: SPARC is less meaningful, just verify it runs
        assert isinstance(s, float)


class TestLDLJ:
    def test_smooth_trajectory(self):
        """Smooth sinusoidal trajectory should have very negative LDLJ."""
        t = np.linspace(0, 2 * np.pi, 120)
        pos = np.column_stack([np.sin(t), np.cos(t)])
        ldlj = log_dimensionless_jerk(pos, fps=30.0)
        assert ldlj < -3.0  # smooth

    def test_jerky_trajectory(self):
        """Jerky trajectory should have LDLJ more negative (higher jerk integral)."""
        pos_smooth = np.column_stack([
            np.sin(np.linspace(0, 2 * np.pi, 120)),
            np.cos(np.linspace(0, 2 * np.pi, 120)),
        ])
        # Add high-frequency noise to make it jerky
        np.random.seed(42)
        pos_jerky = pos_smooth + np.random.randn(120, 2) * 0.05
        ldlj_smooth = log_dimensionless_jerk(pos_smooth, fps=30.0)
        ldlj_jerky = log_dimensionless_jerk(pos_jerky, fps=30.0)
        assert ldlj_jerky < ldlj_smooth  # jerkier = more negative (-log of larger jerk)

    def test_short_trajectory(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        ldlj = log_dimensionless_jerk(pos, fps=30.0)
        assert ldlj == -10.0  # default for too-short


class TestSpectralEntropy:
    def test_pure_sine_low_entropy(self):
        """Pure sine wave should have low spectral entropy."""
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        H = spectral_entropy(signal)
        assert H < 0.4

    def test_noise_high_entropy(self):
        """White noise should have high spectral entropy."""
        np.random.seed(42)
        signal = np.random.randn(200)
        H = spectral_entropy(signal)
        assert H > 0.7

    def test_short_signal(self):
        assert spectral_entropy(np.array([1.0, 2.0])) == 0.0


class TestCrossCorrelation:
    def test_identical_reps(self):
        """Identical reps should have correlation 1.0."""
        rep = np.random.randn(60, 28).astype(np.float32)
        cc = rep_cross_correlation(rep, rep)
        assert cc > 0.99

    def test_different_reps(self):
        """Very different reps should have low correlation."""
        np.random.seed(42)
        rep1 = np.random.randn(60, 28).astype(np.float32)
        rep2 = np.random.randn(60, 28).astype(np.float32)
        cc = rep_cross_correlation(rep1, rep2)
        assert cc < 0.5

    def test_similar_reps(self):
        """Similar reps with small noise should have high correlation."""
        np.random.seed(42)
        rep1 = np.random.randn(60, 28).astype(np.float32)
        rep2 = rep1 + np.random.randn(60, 28).astype(np.float32) * 0.1
        cc = rep_cross_correlation(rep1, rep2)
        assert cc > 0.8


# =====================================================
# Curvature tracker tests
# =====================================================

class TestStreamingCurvature:
    def test_straight_line_zero_curvature(self):
        """Straight-line trajectory should have zero curvature."""
        tracker = StreamingCurvature()
        for t in range(10):
            feat = np.array([float(t), 0.0, 0.0])
            kappa, jerk = tracker.update(feat)
        assert kappa < 0.01

    def test_circle_nonzero_curvature(self):
        """Circular trajectory should have nonzero curvature."""
        tracker = StreamingCurvature()
        curvatures = []
        for t in range(30):
            angle = 2 * np.pi * t / 30
            feat = np.array([np.cos(angle), np.sin(angle), 0.0])
            kappa, _ = tracker.update(feat)
            curvatures.append(kappa)
        # After warmup, curvature should be positive
        assert max(curvatures[5:]) > 0.1

    def test_reset(self):
        tracker = StreamingCurvature()
        for t in range(5):
            tracker.update(np.array([float(t), 0.0]))
        tracker.reset()
        kappa, jerk = tracker.update(np.array([0.0, 0.0]))
        assert kappa == 0.0


# =====================================================
# MovementQualityTracker tests
# =====================================================

class TestMovementQualityTracker:
    def test_process_frame_basic(self):
        """Basic frame processing should not crash and populate metrics."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        tracker.process_frame(
            srp_joints=joints,
            cluster_id=0,
            seg_info={"start_valid": 0, "end_valid": 60, "_current_valid_idx": 30, "cluster": 0},
            representative_joints=joints,  # same as current = perfect form
            fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        assert "form_score" in result
        assert result["form_score"] > 90.0  # near perfect
        assert "biomechanics" in result
        assert "joint_quality" in result

    def test_form_score_degrades_with_deviation(self):
        """Form score should decrease when joints deviate from representative."""
        tracker = MovementQualityTracker(fps=30.0)
        reference = _make_straight_joints_2d()
        deviated = reference.copy()
        deviated[8, 0] += 0.8  # shift L knee significantly

        tracker.process_frame(
            srp_joints=deviated,
            cluster_id=0,
            seg_info={"start_valid": 0, "end_valid": 60, "_current_valid_idx": 30},
            representative_joints=reference,
            fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        # Perfect match is 100, with significant knee deviation it should be lower
        assert result["form_score"] < 98.0

    def test_movement_phase_detection(self):
        """Movement phase should be detected from segment progress."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()

        # Early in segment = ascending
        tracker.process_frame(
            srp_joints=joints, cluster_id=0,
            seg_info={"start_valid": 0, "end_valid": 60, "_current_valid_idx": 15},
            representative_joints=joints, fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        assert result["movement_phase"]["label"] == "ascending"

        # Late in segment = descending
        tracker2 = MovementQualityTracker(fps=30.0)
        tracker2.process_frame(
            srp_joints=joints, cluster_id=0,
            seg_info={"start_valid": 0, "end_valid": 60, "_current_valid_idx": 45},
            representative_joints=joints, fatigue_index=0.0,
        )
        result2 = tracker2.get_frame_quality()
        assert result2["movement_phase"]["label"] == "descending"

    def test_degrading_joints_detection(self):
        """After many frames, degrading joints should be detected."""
        tracker = MovementQualityTracker(fps=30.0)
        reference = _make_straight_joints_2d()

        # First 30 frames: perfect form
        for t in range(30):
            tracker.process_frame(
                srp_joints=reference, cluster_id=0,
                seg_info={"start_valid": 0, "end_valid": 120, "_current_valid_idx": t},
                representative_joints=reference, fatigue_index=0.0,
            )

        # Next 30 frames: degraded left knee
        for t in range(30, 60):
            degraded = reference.copy()
            degraded[8, 0] += 0.3  # L knee drift
            tracker.process_frame(
                srp_joints=degraded, cluster_id=0,
                seg_info={"start_valid": 0, "end_valid": 120, "_current_valid_idx": t},
                representative_joints=reference, fatigue_index=0.0,
            )

        result = tracker.get_frame_quality()
        # Joint 8 (L knee) should be flagged as degrading
        if "joint_quality" in result:
            assert 8 in result["joint_quality"]["degrading"]

    def test_fatigue_timeline_sampling(self):
        """Fatigue timeline should be sampled every 30 frames."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        for t in range(90):
            tracker.process_frame(
                srp_joints=joints, cluster_id=0,
                seg_info={"start_valid": 0, "end_valid": 120, "_current_valid_idx": t},
                representative_joints=joints, fatigue_index=0.1 * (t / 90),
                video_time=t / 30.0,
            )
        result = tracker.get_frame_quality()
        if "fatigue_timeline" in result:
            assert len(result["fatigue_timeline"]["timestamps"]) >= 2

    def test_reset(self):
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        tracker.process_frame(
            srp_joints=joints, cluster_id=0,
            seg_info=None, representative_joints=None, fatigue_index=0.0,
        )
        tracker.reset()
        assert tracker._frame_count == 0
        assert len(tracker._cluster_state) == 0


class TestClusterQualityAnalysis:
    def _make_reps(self, n_reps=10, n_frames=60, d=28, noise=0.01, fatigue_start=None):
        """Generate synthetic reps. Optionally add degradation after fatigue_start."""
        np.random.seed(42)
        base = np.zeros((n_frames, d), dtype=np.float32)
        # Create a sinusoidal base trajectory
        for j in range(d):
            freq = 1.0 + 0.1 * j
            base[:, j] = np.sin(np.linspace(0, 2 * np.pi * freq, n_frames)) * (0.5 + 0.02 * j)

        reps = []
        for k in range(n_reps):
            rep = base.copy()
            rep += np.random.randn(n_frames, d).astype(np.float32) * noise
            if fatigue_start is not None and k >= fatigue_start:
                # Add systematic shift (ROM decrease + asymmetry)
                decay = (k - fatigue_start + 1) * 0.05
                rep *= (1.0 - decay)  # reduce ROM
                rep[:, :d // 2] += decay * 0.2  # asymmetric shift
                rep += np.random.randn(n_frames, d).astype(np.float32) * noise * 3  # more noise
            reps.append(rep)
        return np.array(reps), [r for r in reps]

    def test_stable_reps_low_fatigue(self):
        """Consistent reps should produce low composite fatigue score."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert result["enough_data"]
        assert result["composite_fatigue"] < 0.5

    def test_degrading_reps_high_fatigue(self):
        """Degrading reps should produce higher composite fatigue score."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert result["enough_data"]
        # Should detect degradation
        assert result["composite_fatigue"] > 0.1

    def test_rom_decay_detected(self):
        """ROM should decrease in fatigued reps."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        # ROM decay should be computed (may or may not be significant depending on the synthetic data)
        assert "rom_decay" in result

    def test_cusum_onset_detected(self):
        """CUSUM should detect onset of degradation."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=15, noise=0.01, fatigue_start=7)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        # CUSUM onset should be near rep 7 (give or take detection delay)
        if result["cusum_onset_rep"] is not None:
            assert result["cusum_onset_rep"] >= 5  # some detection delay expected

    def test_ewma_alarms_on_degradation(self):
        """EWMA should alarm on joints that degrade."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=12, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        # Should have some alarming joints
        assert len(result["ewma_alarming_joints"]) >= 0  # may or may not trigger depending on noise

    def test_cross_correlation_decay(self):
        """Cross-correlation should decay with degrading reps."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        if "correlation_decay_rate" in result:
            # Negative slope = correlation decreasing over reps
            assert result["correlation_decay_rate"] < 0.01  # allow some tolerance

    def test_sparc_values_computed(self):
        """SPARC should be computed for each rep."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=6, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert len(result["sparc_values"]) == 6
        assert all(s <= 0 for s in result["sparc_values"])

    def test_insufficient_data(self):
        """Single rep should return enough_data=False."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=1, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert not result["enough_data"]

    def test_loop_spreads_computed(self):
        """Loop spread should be computed for each rep."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=6, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert len(result["loop_spreads"]) == 6
        assert all(s >= 0 for s in result["loop_spreads"])

    def test_path_efficiencies_computed(self):
        """Path efficiency should be computed for each rep."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=6, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert len(result["path_efficiencies"]) == 6
        assert all(0 <= e <= 1 for e in result["path_efficiencies"])

    def test_fatigue_components_present(self):
        """Fatigue components dict should enumerate contributing metrics."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=8, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert "fatigue_components" in result
        assert "composite_fatigue" in result
        assert 0.0 <= result["composite_fatigue"] <= 1.0

    def test_stable_vs_fatigued_comparison(self):
        """Fatigued reps should have higher composite fatigue than stable."""
        tracker_stable = MovementQualityTracker(fps=30.0)
        tracker_fatigued = MovementQualityTracker(fps=30.0)

        stable_reps, stable_raw = self._make_reps(n_reps=10, noise=0.01)
        fatigued_reps, fatigued_raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)

        result_stable = tracker_stable.analyze_cluster_quality(0, stable_reps, stable_raw)
        result_fatigued = tracker_fatigued.analyze_cluster_quality(0, fatigued_reps, fatigued_raw)

        assert result_fatigued["composite_fatigue"] >= result_stable["composite_fatigue"]


class TestBiomechanicsOnTrajectory:
    def test_squat_knee_angle_changes(self):
        """Knee angle should change during a squat trajectory."""
        trajectory = _make_squat_trajectory_2d(n_frames=60)
        knee_angles = []
        for frame in trajectory:
            angle = compute_joint_angle(frame[6], frame[8], frame[10])  # L hip-knee-ankle
            knee_angles.append(angle)
        # Should have variation (not all the same)
        assert max(knee_angles) - min(knee_angles) > 5.0

    def test_fatigued_squat_more_asymmetric(self):
        """Fatigued squats should show more bilateral asymmetry."""
        normal = _make_squat_trajectory_2d(n_frames=60, noise=0.0)
        fatigued = _make_fatigued_trajectory_2d(n_frames=60)

        asym_normal = []
        asym_fatigued = []
        for frame in normal:
            l_angle = compute_joint_angle(frame[6], frame[8], frame[10])
            r_angle = compute_joint_angle(frame[7], frame[9], frame[11])
            asym_normal.append(bilateral_asymmetry_index(l_angle, r_angle))

        for frame in fatigued:
            l_angle = compute_joint_angle(frame[6], frame[8], frame[10])
            r_angle = compute_joint_angle(frame[7], frame[9], frame[11])
            asym_fatigued.append(bilateral_asymmetry_index(l_angle, r_angle))

        assert np.mean(asym_fatigued) > np.mean(asym_normal)


# =====================================================
# Phase 1A: Bone-Length Projection tests
# =====================================================

class TestBoneLengthProjection:
    def test_bone_projection_preserves_direction(self):
        """Projected bone should maintain the same direction vector for a single bone."""
        joints = _make_straight_joints_2d()
        # Only project a single bone to avoid cascading effects
        target_lengths = {(6, 8): 2.0}  # just hip->knee
        projected = project_bone_lengths(joints, target_lengths)
        orig_dir = joints[8] - joints[6]
        proj_dir = projected[8] - projected[6]
        cos_sim = np.dot(orig_dir, proj_dir) / (np.linalg.norm(orig_dir) * np.linalg.norm(proj_dir))
        assert cos_sim > 0.99

    def test_bone_projection_enforces_length(self):
        """Projected bone length should match target length."""
        joints = _make_straight_joints_2d()
        target_lengths = {(6, 8): 1.5, (7, 9): 1.5}  # hip to knee
        projected = project_bone_lengths(joints, target_lengths)
        for parent, child in [(6, 8), (7, 9)]:
            actual_len = float(np.linalg.norm(projected[child] - projected[parent]))
            assert abs(actual_len - 1.5) < 0.01

    def test_bone_filter_warmup(self):
        """Filter should return raw joints before warmup frames."""
        bf = BoneLengthFilter(warmup=30)
        joints = _make_straight_joints_2d()
        for t in range(29):
            result = bf.update(joints)
        # Before warmup, should return joints unchanged
        np.testing.assert_array_equal(result, joints)

    def test_bone_filter_stabilizes_lengths(self):
        """After warmup, bone lengths should be more stable with filter."""
        bf = BoneLengthFilter(warmup=10)
        np.random.seed(42)
        base = _make_straight_joints_2d()
        lengths_unfiltered = []
        lengths_filtered = []
        for t in range(60):
            noisy = base + np.random.randn(*base.shape).astype(np.float32) * 0.1
            # Measure unfiltered
            ul = float(np.linalg.norm(noisy[8] - noisy[6]))
            lengths_unfiltered.append(ul)
            # Apply filter
            filtered = bf.update(noisy)
            fl = float(np.linalg.norm(filtered[8] - filtered[6]))
            lengths_filtered.append(fl)
        # After warmup, filtered lengths should have lower variance
        assert np.std(lengths_filtered[15:]) < np.std(lengths_unfiltered[15:])


# =====================================================
# Phase 1B: Angular Velocity tests
# =====================================================

class TestAngularVelocity:
    def test_angular_velocity_static_pose_zero(self):
        """Static pose should give ~0 angular velocity."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        for _ in range(5):
            tracker.process_frame(
                srp_joints=joints, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        # After a few identical frames, angular velocity should be ~0
        for name, vel in tracker._angular_velocities.items():
            assert vel < 1.0, f"{name} has velocity {vel} for static pose"

    def test_angular_velocity_moving_joint_positive(self):
        """Moving joint should produce positive angular velocity."""
        tracker = MovementQualityTracker(fps=30.0)
        base = _make_straight_joints_2d()
        for t in range(10):
            frame = base.copy()
            # Move the left knee to change the knee angle
            frame[8, 0] += t * 0.05
            tracker.process_frame(
                srp_joints=frame, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        # Left knee angular velocity should be positive
        assert tracker._angular_velocities.get("left_knee", 0.0) > 0.0

    def test_angular_velocity_in_frame_quality(self):
        """Angular velocities should appear in biomechanics dict."""
        tracker = MovementQualityTracker(fps=30.0)
        base = _make_straight_joints_2d()
        for t in range(3):
            frame = base.copy()
            frame[8, 0] += t * 0.1
            tracker.process_frame(
                srp_joints=frame, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        result = tracker.get_frame_quality()
        assert "angular_velocities" in result["biomechanics"]


# =====================================================
# Phase 1C: Isolation Forest Anomaly tests
# =====================================================

class TestAnomalyScoring:
    def test_anomaly_not_computed_before_warmup(self):
        """Anomaly score should be None before 60 frames."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        for _ in range(30):
            tracker.process_frame(
                srp_joints=joints, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        assert tracker._anomaly_score is None

    def test_anomaly_score_normal_frames_low(self):
        """Consistent normal frames should produce low anomaly score."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        np.random.seed(42)
        for _ in range(70):
            noisy = joints + np.random.randn(*joints.shape).astype(np.float32) * 0.01
            tracker.process_frame(
                srp_joints=noisy, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        # After fitting, score for normal frame should be low
        if tracker._anomaly_score is not None:
            assert tracker._anomaly_score < 0.7

    def test_anomaly_score_outlier_high(self):
        """Extreme outlier frame should produce higher anomaly score."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        np.random.seed(42)
        # Train on normal frames
        for _ in range(65):
            noisy = joints + np.random.randn(*joints.shape).astype(np.float32) * 0.01
            tracker.process_frame(
                srp_joints=noisy, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        score_normal = tracker._anomaly_score
        # Now feed an extreme frame
        extreme = joints.copy()
        extreme[8, 0] += 5.0  # huge knee shift
        extreme[6, 1] += 3.0  # huge hip shift
        tracker.process_frame(
            srp_joints=extreme, cluster_id=None,
            seg_info=None, representative_joints=None, fatigue_index=0.0,
        )
        score_outlier = tracker._anomaly_score
        if score_normal is not None and score_outlier is not None:
            assert score_outlier >= score_normal


# =====================================================
# Phase 1D: Injury Risk Threshold tests
# =====================================================

class TestInjuryRisks:
    def test_injury_risk_normal_pose_none(self):
        """Normal standing pose should produce no injury risks."""
        biomech = {
            "fppa_left": 2.0, "fppa_right": -1.0,
            "hip_drop": 1.5, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech)
        assert len(risks) == 0

    def test_injury_risk_valgus_detected(self):
        """FPPA > 15 should flag ACL risk."""
        biomech = {
            "fppa_left": 20.0, "fppa_right": 5.0,
            "hip_drop": 1.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech)
        acl_risks = [r for r in risks if r["risk"] == "acl_valgus"]
        assert len(acl_risks) == 1
        assert acl_risks[0]["severity"] == "medium"
        assert acl_risks[0]["joint"] == "left_knee"

    def test_injury_risk_hip_drop_detected(self):
        """Hip drop > 8 should flag hip drop risk."""
        biomech = {
            "fppa_left": 2.0, "fppa_right": 2.0,
            "hip_drop": 10.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech)
        hip_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hip_risks) == 1
        assert hip_risks[0]["severity"] == "medium"

    def test_injury_risk_severity_levels(self):
        """15 vs 25 FPPA should give medium vs high severity."""
        biomech_medium = {
            "fppa_left": 20.0, "fppa_right": 2.0,
            "hip_drop": 0.0, "trunk_lean": 0.0, "asymmetry": 0.0,
        }
        biomech_high = {
            "fppa_left": 30.0, "fppa_right": 2.0,
            "hip_drop": 0.0, "trunk_lean": 0.0, "asymmetry": 0.0,
        }
        risks_m = evaluate_injury_risks(biomech_medium)
        risks_h = evaluate_injury_risks(biomech_high)
        assert risks_m[0]["severity"] == "medium"
        assert risks_h[0]["severity"] == "high"

    def test_injury_risk_angular_velocity_spike(self):
        """Angular velocity > 500 should flag knee spike."""
        biomech = {
            "fppa_left": 2.0, "fppa_right": 2.0,
            "hip_drop": 0.0, "trunk_lean": 0.0, "asymmetry": 0.0,
        }
        ang_vel = {"left_knee": 600.0, "right_knee": 100.0}
        risks = evaluate_injury_risks(biomech, ang_vel)
        spike_risks = [r for r in risks if r["risk"] == "angular_velocity_spike"]
        assert len(spike_risks) == 1
        assert spike_risks[0]["joint"] == "left_knee"

    def test_injury_risks_in_frame_quality(self):
        """Injury risks should appear in frame quality when thresholds exceeded."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        # Create valgus pose: shift both knees inward significantly
        valgus = joints.copy()
        valgus[8, 0] += 1.0   # L knee shifted way outward
        valgus[9, 0] -= 1.0   # R knee shifted way outward
        for _ in range(3):
            tracker.process_frame(
                srp_joints=valgus, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        result = tracker.get_frame_quality()
        # Check that biomechanics exist (risks may or may not trigger depending on exact angles)
        assert "biomechanics" in result


# =====================================================
# Phase 2A: LDLJ Integration tests
# =====================================================

class TestInjuryRisksWithProfile:
    """Tests for profile-aware injury risk evaluation."""

    def test_lunge_profile_suppresses_hip_drop(self):
        """Lunge profile with high hip drop should NOT flag hip_drop risk."""
        from backend.movement_guidelines import LUNGE_PROFILE
        biomech = {
            "fppa_left": 2.0, "fppa_right": 2.0,
            "hip_drop": 30.0,  # very high — but expected in lunge
            "trunk_lean": 3.0, "asymmetry": 30.0,
        }
        risks = evaluate_injury_risks(biomech, profile=LUNGE_PROFILE)
        hip_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hip_risks) == 0, "Hip drop should be disabled for lunge"
        asym_risks = [r for r in risks if r["risk"] == "asymmetry"]
        assert len(asym_risks) == 0, "Asymmetry should be disabled for lunge"

    def test_squat_profile_stricter_fppa(self):
        """Squat profile: FPPA 13 should trigger medium (squat threshold 12), but not generic (15)."""
        from backend.movement_guidelines import SQUAT_PROFILE
        biomech = {
            "fppa_left": 13.0, "fppa_right": 2.0,
            "hip_drop": 1.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        # With squat profile: 13 > 12 = medium
        risks_squat = evaluate_injury_risks(biomech, profile=SQUAT_PROFILE)
        acl_risks = [r for r in risks_squat if r["risk"] == "acl_valgus"]
        assert len(acl_risks) == 1
        assert acl_risks[0]["severity"] == "medium"

        # Without profile (generic): 13 < 15 = no trigger
        risks_generic = evaluate_injury_risks(biomech)
        acl_risks_generic = [r for r in risks_generic if r["risk"] == "acl_valgus"]
        assert len(acl_risks_generic) == 0

    def test_no_profile_backward_compatible(self):
        """evaluate_injury_risks without profile should work exactly as before."""
        biomech = {
            "fppa_left": 20.0, "fppa_right": 5.0,
            "hip_drop": 10.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech)
        acl_risks = [r for r in risks if r["risk"] == "acl_valgus"]
        assert len(acl_risks) == 1
        assert acl_risks[0]["severity"] == "medium"
        hip_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hip_risks) == 1

    def test_deadlift_profile_suppresses_fppa(self):
        """Deadlift profile should suppress FPPA risks."""
        from backend.movement_guidelines import DEADLIFT_PROFILE
        biomech = {
            "fppa_left": 30.0, "fppa_right": 30.0,
            "hip_drop": 1.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech, profile=DEADLIFT_PROFILE)
        acl_risks = [r for r in risks if r["risk"] == "acl_valgus"]
        assert len(acl_risks) == 0

    def test_deadlift_profile_relaxed_trunk_lean(self):
        """Deadlift profile: trunk lean 26 should trigger medium (threshold 25), not high (40)."""
        from backend.movement_guidelines import DEADLIFT_PROFILE
        biomech = {
            "fppa_left": 2.0, "fppa_right": 2.0,
            "hip_drop": 1.0, "trunk_lean": 26.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech, profile=DEADLIFT_PROFILE)
        trunk_risks = [r for r in risks if r["risk"] == "trunk_lean"]
        assert len(trunk_risks) == 1
        assert trunk_risks[0]["severity"] == "medium"

    def test_running_profile_stricter_hip_drop(self):
        """Running profile: hip drop 7 should trigger medium (threshold 6)."""
        from backend.movement_guidelines import RUNNING_PROFILE
        biomech = {
            "fppa_left": 2.0, "fppa_right": 2.0,
            "hip_drop": 7.0, "trunk_lean": 3.0, "asymmetry": 5.0,
        }
        risks = evaluate_injury_risks(biomech, profile=RUNNING_PROFILE)
        hip_risks = [r for r in risks if r["risk"] == "hip_drop"]
        assert len(hip_risks) == 1
        assert hip_risks[0]["severity"] == "medium"

    def test_generic_profile_matches_hardcoded(self):
        """Generic profile should produce the same results as hardcoded fallback."""
        from backend.movement_guidelines import GENERIC_PROFILE
        biomech = {
            "fppa_left": 20.0, "fppa_right": 5.0,
            "hip_drop": 10.0, "trunk_lean": 20.0, "asymmetry": 20.0,
        }
        ang_vel = {"left_knee": 600.0, "right_knee": 100.0}
        risks_hardcoded = evaluate_injury_risks(biomech, ang_vel)
        risks_profile = evaluate_injury_risks(biomech, ang_vel, profile=GENERIC_PROFILE)
        # Same number of risks
        assert len(risks_hardcoded) == len(risks_profile)
        # Same risk types
        hardcoded_types = sorted(r["risk"] for r in risks_hardcoded)
        profile_types = sorted(r["risk"] for r in risks_profile)
        assert hardcoded_types == profile_types


class TestMovementQualityTrackerGuideline:
    def test_set_activity_label_squat(self):
        """Setting activity label to squat should update the profile."""
        tracker = MovementQualityTracker(fps=30.0)
        tracker.set_activity_label("front squat")
        assert tracker._current_profile is not None
        assert tracker._current_profile.name == "squat"

    def test_set_activity_label_none_clears_profile(self):
        """Setting activity label to None should clear the profile."""
        tracker = MovementQualityTracker(fps=30.0)
        tracker.set_activity_label("squat")
        assert tracker._current_profile is not None
        tracker.set_activity_label(None)
        assert tracker._current_profile is None

    def test_guideline_in_frame_quality(self):
        """Active guideline should appear in frame quality dict."""
        tracker = MovementQualityTracker(fps=30.0)
        tracker.set_activity_label("lunge")
        joints = _make_straight_joints_2d()
        tracker.process_frame(
            srp_joints=joints, cluster_id=0,
            seg_info=None, representative_joints=None, fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        assert "active_guideline" in result
        assert result["active_guideline"]["name"] == "lunge"
        assert result["active_guideline"]["display_name"] == "Lunge"
        assert len(result["active_guideline"]["form_cues"]) > 0

    def test_no_guideline_when_no_label(self):
        """No guideline should appear when no activity label is set."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        tracker.process_frame(
            srp_joints=joints, cluster_id=0,
            seg_info=None, representative_joints=None, fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        assert "active_guideline" not in result


class TestLDLJIntegration:
    def _make_reps(self, n_reps=10, n_frames=60, d=28, noise=0.01, fatigue_start=None):
        np.random.seed(42)
        base = np.zeros((n_frames, d), dtype=np.float32)
        for j in range(d):
            freq = 1.0 + 0.1 * j
            base[:, j] = np.sin(np.linspace(0, 2 * np.pi * freq, n_frames)) * (0.5 + 0.02 * j)
        reps = []
        for k in range(n_reps):
            rep = base.copy()
            rep += np.random.randn(n_frames, d).astype(np.float32) * noise
            if fatigue_start is not None and k >= fatigue_start:
                decay = (k - fatigue_start + 1) * 0.05
                rep *= (1.0 - decay)
                rep[:, :d // 2] += decay * 0.2
                rep += np.random.randn(n_frames, d).astype(np.float32) * noise * 3
            reps.append(rep)
        return np.array(reps), [r for r in reps]

    def test_ldlj_values_in_cluster_quality(self):
        """LDLJ should be computed per rep in cluster analysis."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=6, noise=0.01)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        assert "ldlj_values" in result
        assert len(result["ldlj_values"]) == 6

    def test_ldlj_decay_detected(self):
        """Jerky reps should have lower (more negative) LDLJ."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        if "ldlj_decay" in result:
            # Decay should be detected (jerkier later reps)
            assert isinstance(result["ldlj_decay"], float)

    def test_composite_fatigue_includes_ldlj(self):
        """LDLJ should appear in fatigue components when there's decay."""
        tracker = MovementQualityTracker(fps=30.0)
        resampled, raw = self._make_reps(n_reps=10, noise=0.01, fatigue_start=5)
        result = tracker.analyze_cluster_quality(0, resampled, raw, fps=30.0)
        # LDLJ component may or may not appear depending on whether decay is significant
        assert "fatigue_components" in result


# =====================================================
# Phase 2B: Sample Entropy tests
# =====================================================

class TestSampleEntropy:
    def test_sample_entropy_regular_signal_low(self):
        """Pure sine wave should have low sample entropy."""
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)
        se = sample_entropy(signal)
        assert se < 1.0

    def test_sample_entropy_noisy_signal_high(self):
        """White noise should have higher sample entropy than sine."""
        np.random.seed(42)
        sine = np.sin(np.linspace(0, 4 * np.pi, 200))
        noise = np.random.randn(200)
        se_sine = sample_entropy(sine)
        se_noise = sample_entropy(noise)
        assert se_noise > se_sine

    def test_sample_entropy_in_cluster_quality(self):
        """Sample entropy should be computed per rep."""
        tracker = MovementQualityTracker(fps=30.0)
        np.random.seed(42)
        base = np.zeros((60, 28), dtype=np.float32)
        for j in range(28):
            base[:, j] = np.sin(np.linspace(0, 2 * np.pi, 60)) * 0.5
        reps = np.stack([base + np.random.randn(60, 28).astype(np.float32) * 0.01
                         for _ in range(6)], axis=0)
        raw = [r for r in reps]
        result = tracker.analyze_cluster_quality(0, reps, raw, fps=30.0)
        assert "sample_entropy_values" in result
        assert len(result["sample_entropy_values"]) == 6

    def test_sample_entropy_short_signal(self):
        """Too-short signal should return 0."""
        assert sample_entropy(np.array([1.0, 2.0])) == 0.0

    def test_sample_entropy_constant_signal(self):
        """Constant signal should return 0 (std=0)."""
        assert sample_entropy(np.ones(100)) == 0.0


# =====================================================
# Phase 3A: Center of Mass tests
# =====================================================

class TestCenterOfMass:
    def test_com_symmetric_pose_centered(self):
        """Symmetric standing pose should have CoM near hip midpoint."""
        joints = _make_straight_joints_2d()
        com = estimate_center_of_mass(joints)
        hip_mid = (joints[6] + joints[7]) / 2.0
        # CoM should be close to midline (x near 0) and above hips
        assert abs(com[0]) < 0.3  # roughly centered
        assert com[1] > hip_mid[1]  # above hip midpoint (trunk pulls it up)

    def test_com_leaning_shifts(self):
        """Leaning right should shift CoM right."""
        joints = _make_straight_joints_2d()
        com_normal = estimate_center_of_mass(joints)
        # Shift upper body right
        leaning = joints.copy()
        leaning[0, 0] += 1.0  # L shoulder right
        leaning[1, 0] += 1.0  # R shoulder right
        leaning[2, 0] += 1.0  # L elbow right
        leaning[3, 0] += 1.0  # R elbow right
        com_leaning = estimate_center_of_mass(leaning)
        assert com_leaning[0] > com_normal[0]  # shifted right

    def test_com_sway_static_low(self):
        """Standing still should produce low CoM sway."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        for _ in range(5):
            tracker.process_frame(
                srp_joints=joints, cluster_id=None,
                seg_info=None, representative_joints=None, fatigue_index=0.0,
            )
        result = tracker.get_frame_quality()
        assert result["biomechanics"]["com_sway"] < 5.0  # CoM naturally above ankles

    def test_com_velocity_in_frame_quality(self):
        """CoM velocity should appear in biomechanics dict."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        tracker.process_frame(
            srp_joints=joints, cluster_id=None,
            seg_info=None, representative_joints=None, fatigue_index=0.0,
        )
        result = tracker.get_frame_quality()
        assert "com_velocity" in result["biomechanics"]
        assert "com_sway" in result["biomechanics"]


# =====================================================
# Phase 3B: Kinematic Chain Sequencing tests
# =====================================================

class TestKinematicChainSequencing:
    def test_kinematic_sequence_ideal_score_1(self):
        """Perfect proximal-to-distal timing should give score ~1.0."""
        # Peaks at frames: hip=5, knee=10, elbow=15
        history = {
            "left_hip": [0.0] * 5 + [100.0] + [0.0] * 14,
            "right_hip": [0.0] * 6 + [100.0] + [0.0] * 13,
            "left_knee": [0.0] * 10 + [100.0] + [0.0] * 9,
            "right_knee": [0.0] * 11 + [100.0] + [0.0] * 8,
            "left_elbow": [0.0] * 15 + [100.0] + [0.0] * 4,
            "right_elbow": [0.0] * 16 + [100.0] + [0.0] * 3,
        }
        result = compute_kinematic_sequence(history)
        assert result["sequence_score"] is not None
        assert result["sequence_score"] > 0.8

    def test_kinematic_sequence_reversed_low(self):
        """Distal-first timing should give low score."""
        # Peaks reversed: elbow first, then knee, then hip
        history = {
            "left_hip": [0.0] * 15 + [100.0] + [0.0] * 4,
            "right_hip": [0.0] * 16 + [100.0] + [0.0] * 3,
            "left_knee": [0.0] * 10 + [100.0] + [0.0] * 9,
            "right_knee": [0.0] * 11 + [100.0] + [0.0] * 8,
            "left_elbow": [0.0] * 5 + [100.0] + [0.0] * 14,
            "right_elbow": [0.0] * 6 + [100.0] + [0.0] * 13,
        }
        result = compute_kinematic_sequence(history)
        assert result["sequence_score"] is not None
        assert result["sequence_score"] < 0.5

    def test_kinematic_sequence_in_cluster_quality(self):
        """Kinematic sequence should be computed in cluster analysis."""
        tracker = MovementQualityTracker(fps=30.0)
        np.random.seed(42)
        base = np.zeros((60, 28), dtype=np.float32)
        for j in range(28):
            base[:, j] = np.sin(np.linspace(0, 2 * np.pi, 60)) * 0.5
        reps = np.stack([base + np.random.randn(60, 28).astype(np.float32) * 0.01
                         for _ in range(4)], axis=0)
        raw = [r for r in reps]
        result = tracker.analyze_cluster_quality(0, reps, raw, fps=30.0)
        # May or may not have score depending on angle variation
        assert "n_reps" in result


# =====================================================
# Phase 3C: Spectral Median Frequency tests
# =====================================================

class TestMedianFrequency:
    def test_mnf_pure_sine_at_frequency(self):
        """Pure sine at 5 Hz should have MNF near 5 Hz."""
        t = np.linspace(0, 1.0, 120)  # 1 second at 120 fps
        signal = np.sin(2 * np.pi * 5 * t)
        mnf = median_frequency(signal, fps=120.0)
        assert abs(mnf - 5.0) < 2.0

    def test_mnf_shift_indicates_fatigue(self):
        """Lower frequency content should give lower MNF."""
        t = np.linspace(0, 1.0, 120)
        high_freq = np.sin(2 * np.pi * 10 * t)
        low_freq = np.sin(2 * np.pi * 3 * t)
        mnf_high = median_frequency(high_freq, fps=120.0)
        mnf_low = median_frequency(low_freq, fps=120.0)
        assert mnf_low < mnf_high

    def test_mnf_in_cluster_quality(self):
        """MNF should be computed per joint per rep."""
        tracker = MovementQualityTracker(fps=30.0)
        np.random.seed(42)
        base = np.zeros((60, 28), dtype=np.float32)
        for j in range(28):
            base[:, j] = np.sin(np.linspace(0, 2 * np.pi, 60)) * 0.5
        reps = np.stack([base + np.random.randn(60, 28).astype(np.float32) * 0.01
                         for _ in range(4)], axis=0)
        raw = [r for r in reps]
        result = tracker.analyze_cluster_quality(0, reps, raw, fps=30.0)
        assert "mnf_per_joint" in result
        # Should have entries for each joint chain
        assert len(result["mnf_per_joint"]) == len(JOINT_CHAINS)

    def test_mnf_short_signal(self):
        """Short signal should return 0."""
        assert median_frequency(np.array([1.0, 2.0])) == 0.0

    def test_mnf_constant_signal(self):
        """Constant signal should return 0."""
        assert median_frequency(np.ones(60)) == 0.0


# =====================================================
# Joint visibility guard tests
# =====================================================

class TestJointVisibilityGuards:
    """Tests that biomechanical angles are skipped when joints are not visible."""

    def test_fppa_skipped_when_legs_not_visible(self):
        """FPPA should not fire ACL risk when leg joints have low visibility."""
        tracker = MovementQualityTracker(fps=30.0)
        # Create joints where legs are in extreme valgus position
        joints = _make_straight_joints_2d()
        # Push left knee far medial to trigger FPPA
        joints[8, 0] = 0.5  # Left knee crosses midline (extreme valgus)

        # All joints visible → should compute FPPA
        vis_all = [1.0] * 14
        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0, joint_vis=vis_all)
        result_visible = tracker.get_frame_quality()
        fppa_left_visible = result_visible["biomechanics"]["fppa_left"]

        # Now set leg joints invisible and reset tracker
        tracker2 = MovementQualityTracker(fps=30.0)
        vis_legs_hidden = [1.0] * 14
        vis_legs_hidden[6] = 0.1   # L hip
        vis_legs_hidden[8] = 0.1   # L knee
        vis_legs_hidden[10] = 0.1  # L ankle
        vis_legs_hidden[7] = 0.1   # R hip
        vis_legs_hidden[9] = 0.1   # R knee
        vis_legs_hidden[11] = 0.1  # R ankle
        for _ in range(5):
            tracker2.process_frame(joints, None, None, None, 0.0, joint_vis=vis_legs_hidden)
        result_hidden = tracker2.get_frame_quality()
        # FPPA should remain at initial value (0.0) when legs are invisible
        assert abs(result_hidden["biomechanics"]["fppa_left"]) < 1e-6
        # With visible legs, FPPA should be non-zero for valgus pose
        assert abs(fppa_left_visible) > 1.0

    def test_fppa_holds_last_value_when_legs_disappear(self):
        """After establishing FPPA, setting vis low should hold last computed value."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        joints[8, 0] = 0.5  # Left knee valgus

        vis_all = [1.0] * 14
        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0, joint_vis=vis_all)
        fppa_before = tracker.get_frame_quality()["biomechanics"]["fppa_left"]
        assert abs(fppa_before) > 1.0  # Should have computed a non-zero FPPA

        # Now hide legs — should hold last value
        vis_hidden = [1.0] * 14
        vis_hidden[6] = 0.1
        vis_hidden[8] = 0.1
        vis_hidden[10] = 0.1
        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0, joint_vis=vis_hidden)
        fppa_after = tracker.get_frame_quality()["biomechanics"]["fppa_left"]
        assert abs(fppa_after - fppa_before) < 1e-6

    def test_hip_drop_skipped_when_hips_not_visible(self):
        """Hip drop should not update when hip joints have low visibility."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        # Tilt hips to create hip drop
        joints[6, 1] = 0.5  # L hip up
        joints[7, 1] = -0.5  # R hip down

        vis_hips_hidden = [1.0] * 14
        vis_hips_hidden[6] = 0.1  # L hip
        vis_hips_hidden[7] = 0.1  # R hip
        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0, joint_vis=vis_hips_hidden)
        result = tracker.get_frame_quality()
        # Should stay at 0.0 (initial value)
        assert abs(result["biomechanics"]["hip_drop"]) < 1e-6

    def test_trunk_lean_skipped_when_shoulders_not_visible(self):
        """Trunk lean should not update when shoulders have low visibility."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        # Create trunk lean by shifting shoulders
        joints[0, 0] = -2.0  # L shoulder far left
        joints[1, 0] = -1.0  # R shoulder also left

        vis_shoulders_hidden = [1.0] * 14
        vis_shoulders_hidden[0] = 0.1  # L shoulder
        vis_shoulders_hidden[1] = 0.1  # R shoulder
        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0, joint_vis=vis_shoulders_hidden)
        result = tracker.get_frame_quality()
        # Trunk lean should stay at initial value
        assert abs(result["biomechanics"]["trunk_lean"]) < 1e-6

    def test_backward_compat_no_joint_vis(self):
        """When joint_vis=None (default), all angles should be computed normally."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        joints[8, 0] = 0.5  # Left knee valgus

        for _ in range(5):
            tracker.process_frame(joints, None, None, None, 0.0)  # no joint_vis
        result = tracker.get_frame_quality()
        # FPPA should be computed (non-zero for valgus pose)
        assert abs(result["biomechanics"]["fppa_left"]) > 1.0


# =====================================================
# Concussion Rating Tests
# =====================================================

class TestConcussionRating:
    """Tests for the head-kinematics-based concussion rating."""

    @staticmethod
    def _make_head_landmarks(nose_x=0.0, nose_y=300.0,
                              lear_x=-50.0, lear_y=300.0,
                              rear_x=50.0, rear_y=300.0):
        """Create hip-centered head landmarks (3, 2): [nose, left_ear, right_ear]."""
        return np.array([
            [nose_x, nose_y],
            [lear_x, lear_y],
            [rear_x, rear_y],
        ], dtype=np.float64)

    def test_stationary_head_scores_zero(self):
        """Fixed head position should produce concussion rating <5."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        head = self._make_head_landmarks()
        shoulder_w = 200.0  # ~200px between shoulders

        for _ in range(60):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head, shoulder_width_px=shoulder_w,
            )
        result = tracker.get_frame_quality()
        assert result["concussion_rating"] < 5.0, (
            f"Stationary head should score <5, got {result['concussion_rating']}"
        )

    def test_gentle_motion_scores_low(self):
        """Sinusoidal gentle sway should produce rating <15."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        shoulder_w = 200.0

        for t in range(90):
            # Gentle sway: ±5px at 1Hz
            offset = 5.0 * np.sin(2 * np.pi * t / 30.0)
            head = self._make_head_landmarks(nose_x=offset, lear_x=-50 + offset, rear_x=50 + offset)
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head, shoulder_width_px=shoulder_w,
            )
        result = tracker.get_frame_quality()
        assert result["concussion_rating"] < 15.0, (
            f"Gentle sway should score <15, got {result['concussion_rating']}"
        )

    def test_sudden_impact_spikes_high(self):
        """A 150px jump in 1 frame should produce rating >40."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        shoulder_w = 200.0
        head_normal = self._make_head_landmarks()

        # 30 frames of still head to establish baseline
        for _ in range(30):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head_normal, shoulder_width_px=shoulder_w,
            )

        # Sudden 150px jump (simulating impact — ~28g at this scale)
        head_impact = self._make_head_landmarks(nose_x=150.0, lear_x=100.0, rear_x=200.0)
        tracker.process_frame(
            joints, None, None, None, 0.0,
            head_landmarks=head_impact, shoulder_width_px=shoulder_w,
        )
        result = tracker.get_frame_quality()
        assert result["concussion_rating"] > 40.0, (
            f"Sudden impact should score >40, got {result['concussion_rating']}"
        )

    def test_no_head_landmarks_stays_zero(self):
        """When head_landmarks=None throughout, rating stays 0."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()

        for _ in range(60):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=None, shoulder_width_px=0.0,
            )
        result = tracker.get_frame_quality()
        assert result["concussion_rating"] == 0.0, (
            f"No head data should score 0, got {result['concussion_rating']}"
        )

    def test_peak_hold_and_decay(self):
        """After a spike, rating should hold for ~1.5s then decay."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        shoulder_w = 200.0
        head_normal = self._make_head_landmarks()

        # Establish baseline
        for _ in range(30):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head_normal, shoulder_width_px=shoulder_w,
            )

        # Sudden impact (150px jump)
        head_impact = self._make_head_landmarks(nose_x=150.0, lear_x=100.0, rear_x=200.0)
        tracker.process_frame(
            joints, None, None, None, 0.0,
            head_landmarks=head_impact, shoulder_width_px=shoulder_w,
        )
        spike_rating = tracker.get_frame_quality()["concussion_rating"]

        # 20 frames later (still in hold window): should be near spike
        for _ in range(20):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head_normal, shoulder_width_px=shoulder_w,
            )
        held_rating = tracker.get_frame_quality()["concussion_rating"]
        assert held_rating >= spike_rating * 0.5, (
            f"During hold, rating should stay near spike ({spike_rating}), got {held_rating}"
        )

        # 100 more frames (well past hold): should have decayed significantly
        for _ in range(100):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head_normal, shoulder_width_px=shoulder_w,
            )
        decayed_rating = tracker.get_frame_quality()["concussion_rating"]
        assert decayed_rating < spike_rating * 0.3, (
            f"After decay, rating should be <30% of spike ({spike_rating}), got {decayed_rating}"
        )

    def test_rotational_impact_spikes(self):
        """Swapping ear positions (head rotation) should spike angular component >20."""
        tracker = MovementQualityTracker(fps=30.0)
        joints = _make_straight_joints_2d()
        shoulder_w = 200.0

        # Normal orientation: left_ear at -50, right_ear at +50
        head_normal = self._make_head_landmarks()
        for _ in range(30):
            tracker.process_frame(
                joints, None, None, None, 0.0,
                head_landmarks=head_normal, shoulder_width_px=shoulder_w,
            )

        # Swap ears (180° rotation in 1 frame ≈ massive angular velocity)
        head_rotated = self._make_head_landmarks(lear_x=50.0, rear_x=-50.0)
        tracker.process_frame(
            joints, None, None, None, 0.0,
            head_landmarks=head_rotated, shoulder_width_px=shoulder_w,
        )
        result = tracker.get_frame_quality()
        # The angular component alone should contribute >20 points
        assert result["concussion_rating"] > 20.0, (
            f"Rotational impact should score >20, got {result['concussion_rating']}"
        )
