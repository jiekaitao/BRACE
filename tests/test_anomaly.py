"""Tests for anomaly detection: normal should score low, pathological should score high."""

import numpy as np
import pytest

from brace.core.baseline import build_baseline
from brace.core.anomaly import score_sequence, score_sequence_aggregate
from brace.data.joint_map import NUM_KINECT_JOINTS


def _make_gait_sequence(n_frames=120, period=30, noise=0.005, ankle_amp=0.1, seed=42):
    """Generate a gait sequence with configurable parameters."""
    rng = np.random.RandomState(seed)
    seq = np.zeros((n_frames, NUM_KINECT_JOINTS, 3), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32)

    seq[:, 12, 0] = -0.1
    seq[:, 16, 0] = 0.1
    seq[:, 4, :] = [-0.15, -0.3, 0]
    seq[:, 8, :] = [0.15, -0.3, 0]

    seq[:, 14, 1] = ankle_amp * np.sin(2 * np.pi * t / period)
    seq[:, 18, 1] = ankle_amp * np.sin(2 * np.pi * t / period + np.pi)
    seq[:, 13, 1] = (ankle_amp / 2) * np.sin(2 * np.pi * t / period)
    seq[:, 17, 1] = (ankle_amp / 2) * np.sin(2 * np.pi * t / period + np.pi)

    seq += rng.randn(*seq.shape).astype(np.float32) * noise
    return seq


def _make_pathological_sequence(n_frames=120, period=30, seed=100):
    """Generate a sequence with large asymmetric deviations (simulating limping)."""
    rng = np.random.RandomState(seed)
    seq = _make_gait_sequence(n_frames, period, noise=0.005, seed=seed)
    t = np.arange(n_frames, dtype=np.float32)

    # Add large asymmetric deviation to left side (simulating antalgic gait)
    seq[:, 14, 1] += 0.15 * np.sin(2 * np.pi * t / period * 0.5)  # Different frequency
    seq[:, 13, 0] += 0.1  # Lateral knee shift
    seq[:, 14, 0] += 0.08  # Lateral ankle shift

    return seq


@pytest.fixture
def baseline():
    """Build a baseline from synthetic normal gait."""
    seqs = [_make_gait_sequence(n_frames=180, seed=i) for i in range(5)]
    return build_baseline(seqs)


def test_normal_scores_low(baseline):
    """Normal gait (similar to training data) should have low anomaly score."""
    normal_seq = _make_gait_sequence(n_frames=120, seed=99)
    result = score_sequence_aggregate(normal_seq, baseline)

    # Normal gait against its own baseline should be relatively low
    assert result["mean_anomaly_score"] < 5.0, f"Normal score too high: {result['mean_anomaly_score']}"
    assert result["n_cycles"] >= 1


def test_pathological_scores_higher_than_normal(baseline):
    """Pathological gait should score higher than normal gait."""
    normal_seq = _make_gait_sequence(n_frames=120, seed=99)
    patho_seq = _make_pathological_sequence(n_frames=120, seed=200)

    normal_result = score_sequence_aggregate(normal_seq, baseline)
    patho_result = score_sequence_aggregate(patho_seq, baseline)

    assert patho_result["mean_anomaly_score"] > normal_result["mean_anomaly_score"], \
        f"Pathological ({patho_result['mean_anomaly_score']:.3f}) not higher than normal ({normal_result['mean_anomaly_score']:.3f})"


def test_score_has_joint_scores(baseline):
    """Score result should include per-joint deviation information."""
    seq = _make_gait_sequence(n_frames=120, seed=42)
    results = score_sequence(seq, baseline)

    assert len(results) >= 1
    result = results[0]
    assert "joint_scores" in result
    assert "worst_joints" in result
    assert "phase_scores" in result
    assert len(result["worst_joints"]) == 3


def test_score_sequence_aggregate(baseline):
    """Aggregate scoring should produce valid statistics."""
    seq = _make_gait_sequence(n_frames=180, seed=42)
    result = score_sequence_aggregate(seq, baseline)

    assert "mean_anomaly_score" in result
    assert "median_anomaly_score" in result
    assert "max_anomaly_score" in result
    assert result["n_cycles"] >= 1
    assert len(result["per_cycle_scores"]) == result["n_cycles"]
    assert result["mean_anomaly_score"] >= 0
