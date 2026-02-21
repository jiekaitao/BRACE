"""Tests for baseline builder."""

import numpy as np
import pytest

from brace.core.baseline import build_baseline
from brace.data.joint_map import NUM_KINECT_JOINTS, FEATURE_DIM


def _make_normal_gait_sequence(n_frames=120, period=30, noise=0.005, seed=42):
    """Generate a synthetic 'normal' gait sequence with periodic ankle motion."""
    rng = np.random.RandomState(seed)
    seq = np.zeros((n_frames, NUM_KINECT_JOINTS, 3), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32)

    # Set anchor joints
    seq[:, 12, 0] = -0.1   # HipLeft
    seq[:, 16, 0] = 0.1    # HipRight
    seq[:, 4, :] = [-0.15, -0.3, 0]
    seq[:, 8, :] = [0.15, -0.3, 0]

    # Periodic ankle motion
    seq[:, 14, 1] = 0.1 * np.sin(2 * np.pi * t / period)
    seq[:, 18, 1] = 0.1 * np.sin(2 * np.pi * t / period + np.pi)

    # Knee motion (half amplitude)
    seq[:, 13, 1] = 0.05 * np.sin(2 * np.pi * t / period)
    seq[:, 17, 1] = 0.05 * np.sin(2 * np.pi * t / period + np.pi)

    # Add small noise
    seq += rng.randn(*seq.shape).astype(np.float32) * noise

    return seq


def test_build_baseline_produces_valid_structure():
    """Baseline should contain all expected keys with correct shapes."""
    seqs = [_make_normal_gait_sequence(seed=i) for i in range(3)]
    baseline = build_baseline(seqs)

    assert "mean_trajectory" in baseline
    assert "std_trajectory" in baseline
    assert "feat_mean" in baseline
    assert "feat_std" in baseline
    assert "distance_calibration" in baseline
    assert "n_cycles" in baseline

    assert baseline["mean_trajectory"].shape[1] == FEATURE_DIM
    assert baseline["std_trajectory"].shape[1] == FEATURE_DIM
    assert baseline["feat_mean"].shape[0] == FEATURE_DIM
    assert baseline["n_cycles"] > 0


def test_baseline_distance_calibration():
    """Distance calibration should produce p50 < p90 < p99."""
    seqs = [_make_normal_gait_sequence(seed=i) for i in range(5)]
    baseline = build_baseline(seqs)
    cal = baseline["distance_calibration"]

    assert cal["p50"] <= cal["p90"] <= cal["p99"]
    assert cal["p50"] >= 0


def test_baseline_cycle_count():
    """Should extract multiple cycles from multiple sequences."""
    seqs = [_make_normal_gait_sequence(n_frames=180, seed=i) for i in range(3)]
    baseline = build_baseline(seqs)

    # Each 180-frame sequence with period 30 should give ~5 cycles
    assert baseline["n_cycles"] >= 5


def test_baseline_std_positive():
    """Std trajectory should be all positive (no zeros due to robust_std)."""
    seqs = [_make_normal_gait_sequence(seed=i) for i in range(3)]
    baseline = build_baseline(seqs)

    assert np.all(baseline["std_trajectory"] > 0)
    assert np.all(baseline["feat_std"] > 0)
