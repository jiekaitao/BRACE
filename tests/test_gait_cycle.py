"""Tests for gait cycle detection on synthetic periodic data."""

import numpy as np
import pytest

from brace.core.gait_cycle import (
    detect_heel_strikes,
    segment_gait_cycles,
    resample_cycle,
    extract_resampled_cycles,
)
from brace.data.joint_map import ANKLE_LEFT, NUM_KINECT_JOINTS


def _make_periodic_skeleton(n_frames=180, period=30, amplitude=0.1):
    """Create skeleton sequence with sinusoidal ankle oscillation.

    Simulates 6 gait cycles of 30 frames each (1 second each at 30fps).
    """
    seq = np.zeros((n_frames, NUM_KINECT_JOINTS, 3), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32)

    # Ankle oscillates vertically (axis=1)
    ankle_y = amplitude * np.sin(2 * np.pi * t / period)
    seq[:, ANKLE_LEFT, 1] = ankle_y

    # Give hips some width so SRP normalization doesn't degenerate
    seq[:, 12, 0] = -0.1  # HipLeft
    seq[:, 16, 0] = 0.1   # HipRight
    seq[:, 4, :] = [-0.15, -0.3, 0]   # ShoulderLeft
    seq[:, 8, :] = [0.15, -0.3, 0]    # ShoulderRight

    return seq


def test_detect_heel_strikes_count():
    """Should detect approximately the right number of heel strikes."""
    seq = _make_periodic_skeleton(n_frames=180, period=30)
    strikes = detect_heel_strikes(seq, ankle_joint=ANKLE_LEFT, axis=1, fs=30.0, min_cycle_frames=15)

    # 180 frames / 30 period = 6 cycles → ~5-6 valleys
    assert 4 <= len(strikes) <= 7, f"Expected 4-7 strikes, got {len(strikes)}"


def test_detect_heel_strikes_spacing():
    """Strikes should be roughly evenly spaced at the period interval."""
    seq = _make_periodic_skeleton(n_frames=300, period=30)
    strikes = detect_heel_strikes(seq, ankle_joint=ANKLE_LEFT, axis=1, fs=30.0, min_cycle_frames=15)

    if len(strikes) >= 2:
        diffs = np.diff(strikes)
        mean_diff = np.mean(diffs)
        # Should be close to period=30
        assert 20 < mean_diff < 40, f"Mean strike spacing {mean_diff} not near period 30"


def test_segment_gait_cycles():
    """Segmentation should produce cycles of roughly the right length."""
    seq = _make_periodic_skeleton(n_frames=180, period=30)
    cycles = segment_gait_cycles(seq, fs=30.0, min_cycle_frames=15)

    assert len(cycles) >= 2, f"Expected at least 2 cycles, got {len(cycles)}"
    for cycle in cycles:
        assert 15 <= cycle.shape[0] <= 60, f"Cycle length {cycle.shape[0]} out of range"


def test_resample_cycle():
    """Resampling should produce exactly the target length."""
    cycle = np.random.randn(45, NUM_KINECT_JOINTS, 3).astype(np.float32)
    resampled = resample_cycle(cycle, target_length=60)

    assert resampled.shape == (60, NUM_KINECT_JOINTS, 3)


def test_resample_preserves_endpoints():
    """First and last frames should be approximately preserved after resampling."""
    rng = np.random.RandomState(42)
    cycle = rng.randn(40, NUM_KINECT_JOINTS, 3).astype(np.float32)
    resampled = resample_cycle(cycle, target_length=80)

    np.testing.assert_allclose(resampled[0], cycle[0], atol=1e-5)
    np.testing.assert_allclose(resampled[-1], cycle[-1], atol=1e-5)


def test_extract_resampled_cycles():
    """Full pipeline should produce cycles of the target length."""
    seq = _make_periodic_skeleton(n_frames=180, period=30)
    cycles = extract_resampled_cycles(seq, target_length=60, fs=30.0)

    assert len(cycles) >= 1
    for cycle in cycles:
        assert cycle.shape == (60, NUM_KINECT_JOINTS, 3)
