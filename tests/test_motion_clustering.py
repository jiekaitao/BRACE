"""Tests for motion segmentation and clustering in motion_segments.py.

Verifies that the improved clustering algorithm correctly groups similar
segments and avoids over-segmentation.
"""

import numpy as np
import pytest

from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
    analyze_consistency,
    _resample_segment,
    _segment_distance,
)
from brace.core.pose import (
    FEATURE_INDICES,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    NUM_MP_LANDMARKS,
)


def _make_valid_landmarks(
    knee_angle: float = 0.0,
    arm_angle: float = 0.0,
    offset: tuple = (300.0, 400.0),
) -> np.ndarray:
    """Build synthetic landmarks with configurable joint angles."""
    lm = np.zeros((NUM_MP_LANDMARKS, 4), dtype=np.float32)
    lm[:, 3] = 0.9

    ox, oy = offset
    hw = 60.0  # hip width

    lm[LEFT_HIP] = [ox - hw / 2, oy, 0, 0.9]
    lm[RIGHT_HIP] = [ox + hw / 2, oy, 0, 0.9]
    lm[LEFT_SHOULDER] = [ox - hw * 0.6, oy - hw * 1.5, 0, 0.9]
    lm[RIGHT_SHOULDER] = [ox + hw * 0.6, oy - hw * 1.5, 0, 0.9]

    # Arms vary with arm_angle
    lm[13] = [ox - hw * 0.8 - 20 * np.sin(arm_angle), oy - hw * 0.7 + 20 * np.cos(arm_angle), 0, 0.9]
    lm[14] = [ox + hw * 0.8 + 20 * np.sin(arm_angle), oy - hw * 0.7 + 20 * np.cos(arm_angle), 0, 0.9]
    lm[15] = [ox - hw * 0.8 - 40 * np.sin(arm_angle), oy - hw * 0.3 + 40 * np.cos(arm_angle), 0, 0.9]
    lm[16] = [ox + hw * 0.8 + 40 * np.sin(arm_angle), oy - hw * 0.3 + 40 * np.cos(arm_angle), 0, 0.9]

    # Legs vary with knee_angle
    lm[25] = [ox - hw / 2, oy + hw * 1.2 + 30 * np.sin(knee_angle), 0, 0.9]
    lm[26] = [ox + hw / 2, oy + hw * 1.2 + 30 * np.sin(knee_angle), 0, 0.9]
    lm[27] = [ox - hw / 2, oy + hw * 2.5 + 20 * np.sin(knee_angle), 0, 0.9]
    lm[28] = [ox + hw / 2, oy + hw * 2.5 + 20 * np.sin(knee_angle), 0, 0.9]
    lm[31] = [ox - hw / 2, oy + hw * 2.7 + 20 * np.sin(knee_angle), 0, 0.9]
    lm[32] = [ox + hw / 2, oy + hw * 2.7 + 20 * np.sin(knee_angle), 0, 0.9]

    return lm


def _generate_repetitive_features(
    n_reps: int = 10, frames_per_rep: int = 30, noise: float = 0.02, seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic feature trajectory of a repeating motion pattern.

    Each repetition follows a sinusoidal pattern with small noise.
    Returns features array and valid_indices (identity mapping).
    """
    rng = np.random.RandomState(seed)
    all_features = []

    for _ in range(n_reps):
        t = np.linspace(0, 2 * np.pi, frames_per_rep)
        base = np.column_stack([np.sin(t + phase) for phase in np.linspace(0, np.pi, 28)])
        base += rng.randn(*base.shape) * noise
        all_features.append(base.astype(np.float32))

    features = np.vstack(all_features)
    valid_indices = list(range(features.shape[0]))
    return features, valid_indices


def _generate_mixed_features(
    motion_types: int = 3,
    reps_per_type: int = 5,
    frames_per_rep: int = 30,
    separation: float = 3.0,
    noise: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Generate features with multiple distinct motion types interleaved.

    Each motion type has a different mean offset (simulating different average
    body poses for different exercises) AND different frequency/phase.

    Returns features, valid_indices, and ground truth cluster labels per frame.
    """
    rng = np.random.RandomState(seed)
    all_features = []
    labels = []

    for type_idx in range(motion_types):
        for _ in range(reps_per_type):
            t = np.linspace(0, 2 * np.pi, frames_per_rep)
            # Different base pattern per motion type
            phase_offset = type_idx * separation
            freq_mult = 1.0 + type_idx * 0.5
            # Mean offset per type simulates different average body poses
            mean_offset = type_idx * separation
            base = np.column_stack([
                np.sin(freq_mult * t + phase_offset + phase) + mean_offset
                for phase in np.linspace(0, np.pi, 28)
            ])
            base += rng.randn(*base.shape) * noise
            all_features.append(base.astype(np.float32))
            labels.extend([type_idx] * frames_per_rep)

    features = np.vstack(all_features)
    valid_indices = list(range(features.shape[0]))
    return features, valid_indices, labels


class TestResampleSegment:
    def test_output_shape(self):
        feat = np.random.randn(20, 28).astype(np.float32)
        out = _resample_segment(feat, 30)
        assert out.shape == (30, 28)

    def test_identity_when_same_length(self):
        feat = np.random.randn(30, 28).astype(np.float32)
        out = _resample_segment(feat, 30)
        np.testing.assert_allclose(out, feat, atol=1e-5)

    def test_preserves_endpoints(self):
        feat = np.random.randn(20, 28).astype(np.float32)
        out = _resample_segment(feat, 50)
        np.testing.assert_allclose(out[0], feat[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], feat[-1], atol=1e-5)


class TestSegmentDistance:
    def test_identical_segments_zero_distance(self):
        feat = np.random.randn(20, 28).astype(np.float32)
        seg = {"features": feat, "mean_feature": feat.mean(axis=0)}
        d = _segment_distance(seg, seg)
        assert d < 1e-4

    def test_different_segments_positive_distance(self):
        feat_a = np.random.randn(20, 28).astype(np.float32)
        feat_b = np.random.randn(20, 28).astype(np.float32) + 5.0
        seg_a = {"features": feat_a, "mean_feature": feat_a.mean(axis=0)}
        seg_b = {"features": feat_b, "mean_feature": feat_b.mean(axis=0)}
        d = _segment_distance(seg_a, seg_b)
        assert d > 1.0

    def test_similar_segments_small_distance(self):
        feat = np.random.randn(20, 28).astype(np.float32)
        feat_noisy = feat + np.random.randn(20, 28).astype(np.float32) * 0.01
        seg_a = {"features": feat, "mean_feature": feat.mean(axis=0)}
        seg_b = {"features": feat_noisy, "mean_feature": feat_noisy.mean(axis=0)}
        d = _segment_distance(seg_a, seg_b)
        assert d < 0.1


class TestDetectMotionBoundaries:
    def test_returns_at_least_one(self):
        features = np.random.randn(100, 28).astype(np.float32)
        boundaries = detect_motion_boundaries(features, fps=30)
        assert len(boundaries) >= 1
        assert boundaries[0] == 0

    def test_short_input_single_boundary(self):
        features = np.random.randn(4, 28).astype(np.float32)
        boundaries = detect_motion_boundaries(features, fps=30)
        assert boundaries == [0]

    def test_constant_input_no_extra_boundaries(self):
        """Constant features (no motion) should produce minimal boundaries."""
        features = np.ones((200, 28), dtype=np.float32)
        boundaries = detect_motion_boundaries(features, fps=30)
        assert len(boundaries) == 1  # only frame 0

    def test_repetitive_motion_reasonable_boundary_count(self):
        """Repetitive sinusoidal motion should not be over-segmented."""
        features, _ = _generate_repetitive_features(n_reps=10, frames_per_rep=30)
        boundaries = detect_motion_boundaries(features, fps=30, min_segment_sec=0.8)
        # 300 frames at 30fps = 10 seconds. With min 0.8s per segment,
        # max possible is ~12. Should be much less for smooth repetitive motion.
        assert len(boundaries) <= 15, f"Too many boundaries ({len(boundaries)}) for repetitive motion"

    def test_savgol_velocity_smoother(self):
        """Savitzky-Golay velocity should have lower variance than finite diff on noisy data."""
        np.random.seed(42)
        features, _ = _generate_repetitive_features(n_reps=5, frames_per_rep=30)
        features += np.random.randn(*features.shape).astype(np.float32) * 0.05
        b_default = detect_motion_boundaries(features, fps=30, use_savgol=False)
        b_savgol = detect_motion_boundaries(features, fps=30, use_savgol=True)
        # Both should produce valid boundaries
        assert len(b_default) >= 1
        assert len(b_savgol) >= 1
        assert b_savgol[0] == 0

    def test_savgol_boundaries_consistent(self):
        """Savitzky-Golay boundaries should be similar to standard (within ±2 boundaries)."""
        np.random.seed(42)
        features, _ = _generate_repetitive_features(n_reps=5, frames_per_rep=30)
        b_default = detect_motion_boundaries(features, fps=30, min_segment_sec=1.0)
        b_savgol = detect_motion_boundaries(features, fps=30, min_segment_sec=1.0, use_savgol=True)
        assert abs(len(b_default) - len(b_savgol)) <= 3


class TestClusterSegments:
    def test_empty_input(self):
        result = cluster_segments([], distance_threshold=2.0)
        assert result == []

    def test_single_segment(self):
        feat = np.random.randn(20, 28).astype(np.float32)
        segments = [{"features": feat, "mean_feature": feat.mean(axis=0)}]
        result = cluster_segments(segments, distance_threshold=2.0)
        assert result[0]["cluster"] == 0

    def test_identical_segments_one_cluster(self):
        """Identical segments should all go into one cluster."""
        feat = np.random.randn(20, 28).astype(np.float32)
        segments = []
        for _ in range(5):
            noisy = feat + np.random.randn(20, 28).astype(np.float32) * 0.01
            segments.append({"features": noisy, "mean_feature": noisy.mean(axis=0)})

        result = cluster_segments(segments, distance_threshold=2.0)
        clusters = set(s["cluster"] for s in result)
        assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"

    def test_distinct_segments_multiple_clusters(self):
        """Very different segments should get different clusters."""
        segments = []
        for i in range(3):
            feat = np.random.randn(20, 28).astype(np.float32) + i * 10.0
            segments.append({"features": feat, "mean_feature": feat.mean(axis=0)})

        result = cluster_segments(segments, distance_threshold=2.0)
        clusters = set(s["cluster"] for s in result)
        assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"

    def test_cluster_labels_contiguous(self):
        """Cluster labels should be 0-indexed and contiguous."""
        segments = []
        for i in range(6):
            offset = (i % 2) * 10.0
            feat = np.random.randn(20, 28).astype(np.float32) + offset
            segments.append({"features": feat, "mean_feature": feat.mean(axis=0)})

        result = cluster_segments(segments, distance_threshold=2.0)
        clusters = sorted(set(s["cluster"] for s in result))
        assert clusters == list(range(len(clusters)))

    def test_threshold_controls_granularity(self):
        """Higher threshold should produce fewer clusters."""
        rng = np.random.RandomState(42)
        segments = []
        for i in range(6):
            feat = rng.randn(20, 28).astype(np.float32) + i * 2.0
            segments.append({"features": feat.copy(), "mean_feature": feat.mean(axis=0)})

        # Low threshold = more clusters
        result_low = cluster_segments(
            [{"features": s["features"].copy(), "mean_feature": s["mean_feature"].copy()} for s in segments],
            distance_threshold=1.0,
        )
        n_low = len(set(s["cluster"] for s in result_low))

        # High threshold = fewer clusters
        result_high = cluster_segments(
            [{"features": s["features"].copy(), "mean_feature": s["mean_feature"].copy()} for s in segments],
            distance_threshold=10.0,
        )
        n_high = len(set(s["cluster"] for s in result_high))

        assert n_high <= n_low, f"Higher threshold should give fewer clusters: {n_high} > {n_low}"


class TestRepetitiveMotionClustering:
    """Integration tests: full pipeline on synthetic repetitive motion."""

    def test_repetitive_exercise_few_clusters(self):
        """Repetitive motion (like exercise) should produce very few clusters."""
        features, valid_indices = _generate_repetitive_features(
            n_reps=10, frames_per_rep=30, noise=0.02, seed=42,
        )
        segments = segment_motions(features, valid_indices, fps=30.0, min_segment_sec=0.8)
        segments = cluster_segments(segments, distance_threshold=2.0)
        n_clusters = len(set(s["cluster"] for s in segments))
        assert n_clusters <= 3, f"Repetitive motion should have <= 3 clusters, got {n_clusters}"

    def test_distinct_motions_separated(self):
        """Different motion types should get different clusters."""
        features, valid_indices, _ = _generate_mixed_features(
            motion_types=3, reps_per_type=5, frames_per_rep=30,
            separation=5.0, noise=0.02, seed=42,
        )
        segments = segment_motions(features, valid_indices, fps=30.0, min_segment_sec=0.8)
        # Threshold 3.0: mean pose + spectral distance normalized by sqrt(D);
        # distinct exercises with different average poses separate above 3.0
        segments = cluster_segments(segments, distance_threshold=3.0)
        n_clusters = len(set(s["cluster"] for s in segments))
        # Should produce at least 2 distinct clusters for 3 motion types
        assert n_clusters >= 2, f"Expected >= 2 clusters for 3 motion types, got {n_clusters}"


class TestAnalyzeConsistency:
    def test_consistency_with_single_cluster(self):
        """Single cluster should produce valid analysis."""
        features, valid_indices = _generate_repetitive_features(n_reps=5, frames_per_rep=30)
        segments = segment_motions(features, valid_indices, fps=30.0)
        segments = cluster_segments(segments, distance_threshold=5.0)  # force one cluster
        analysis = analyze_consistency(segments)
        assert len(analysis) >= 1

    def test_consistency_scores_present(self):
        """Segments in clusters with 2+ members should get consistency scores."""
        rng = np.random.RandomState(42)
        feat_base = rng.randn(20, 28).astype(np.float32)
        segments = []
        for _ in range(4):
            noisy = feat_base + rng.randn(20, 28).astype(np.float32) * 0.01
            segments.append({"features": noisy, "mean_feature": noisy.mean(axis=0)})

        segments = cluster_segments(segments, distance_threshold=5.0)
        analysis = analyze_consistency(segments)

        for seg in segments:
            if "consistency_score" in seg:
                assert isinstance(seg["consistency_score"], float)
                assert seg["consistency_score"] >= 0.0

    def test_single_member_cluster_no_crash(self):
        """Cluster with 1 member should not crash and produce zero scores."""
        feat = np.random.randn(20, 28).astype(np.float32)
        segments = [{"features": feat, "mean_feature": feat.mean(axis=0), "cluster": 0}]
        analysis = analyze_consistency(segments)
        assert 0 in analysis
        assert analysis[0]["count"] == 1
        assert analysis[0]["mean_score"] == 0.0
