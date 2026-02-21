"""Tests for normalize_frame_3d_real() and 3D feature extraction."""

import numpy as np
import pytest

from brace.core.motion_segments import (
    normalize_frame_3d_real,
    normalize_frame,
    feature_vector,
)
from brace.core.pose import (
    FEATURE_INDICES,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    NUM_MP_LANDMARKS,
)


def _make_3d_landmarks(
    pelvis_offset: tuple = (200.0, 300.0, 2.0),
    hip_width: float = 30.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Build synthetic 3D landmarks with real depth for testing.

    Creates a standing humanoid at the given pelvis position with the
    given hip width. All feature joints get plausible 3D positions.
    The default pelvis_offset includes non-zero Z to ensure real depth detection.
    """
    lm = np.zeros((NUM_MP_LANDMARKS, 4), dtype=np.float32)

    px, py, pz = pelvis_offset
    hw = hip_width * scale

    # Hips
    lm[LEFT_HIP] = [px - hw / 2, py, pz, 0.9]
    lm[RIGHT_HIP] = [px + hw / 2, py, pz, 0.9]

    # Shoulders (above hips, slightly wider)
    sh_width = hw * 1.2
    sh_y = py - hw * 2.0
    lm[LEFT_SHOULDER] = [px - sh_width / 2, sh_y, pz + hw * 0.1, 0.9]
    lm[RIGHT_SHOULDER] = [px + sh_width / 2, sh_y, pz - hw * 0.1, 0.9]

    # Elbows
    lm[13] = [px - sh_width / 2 - hw * 0.5, sh_y + hw * 0.8, pz + hw * 0.2, 0.85]
    lm[14] = [px + sh_width / 2 + hw * 0.5, sh_y + hw * 0.8, pz - hw * 0.2, 0.85]

    # Wrists
    lm[15] = [px - sh_width / 2 - hw * 0.8, sh_y + hw * 1.5, pz + hw * 0.3, 0.8]
    lm[16] = [px + sh_width / 2 + hw * 0.8, sh_y + hw * 1.5, pz - hw * 0.3, 0.8]

    # Knees
    lm[25] = [px - hw / 2, py + hw * 1.5, pz - hw * 0.05, 0.9]
    lm[26] = [px + hw / 2, py + hw * 1.5, pz + hw * 0.05, 0.9]

    # Ankles
    lm[27] = [px - hw / 2, py + hw * 3.0, pz - hw * 0.02, 0.9]
    lm[28] = [px + hw / 2, py + hw * 3.0, pz + hw * 0.02, 0.9]

    # Feet
    lm[31] = [px - hw / 2, py + hw * 3.1, pz - hw * 0.04, 0.8]
    lm[32] = [px + hw / 2, py + hw * 3.1, pz + hw * 0.04, 0.8]

    return lm


class TestNormalizeFrame3dReal:
    def test_output_shape(self):
        lm = _make_3d_landmarks()
        norm = normalize_frame_3d_real(lm)
        assert norm is not None
        assert norm.shape == (33, 3)

    def test_pelvis_at_origin(self):
        """Pelvis midpoint should be at (0, 0, 0) in normalized frame."""
        lm = _make_3d_landmarks(pelvis_offset=(200.0, 300.0, 5.0))
        norm = normalize_frame_3d_real(lm)
        assert norm is not None

        pelvis = (norm[LEFT_HIP] + norm[RIGHT_HIP]) / 2.0
        assert abs(pelvis[0]) < 1e-4
        assert abs(pelvis[1]) < 1e-4
        assert abs(pelvis[2]) < 1e-4

    def test_hip_width_is_one(self):
        """Distance between hips should be 1.0 in normalized coordinates."""
        lm = _make_3d_landmarks(hip_width=50.0)
        norm = normalize_frame_3d_real(lm)
        assert norm is not None

        hip_dist = float(np.linalg.norm(norm[LEFT_HIP] - norm[RIGHT_HIP]))
        assert hip_dist == pytest.approx(1.0, abs=1e-3)

    def test_translation_invariance(self):
        """Shifting all landmarks should not change normalized feature joints."""
        lm1 = _make_3d_landmarks(pelvis_offset=(100.0, 200.0, 1.0))
        lm2 = _make_3d_landmarks(pelvis_offset=(500.0, 300.0, 10.0))

        norm1 = normalize_frame_3d_real(lm1)
        norm2 = normalize_frame_3d_real(lm2)
        assert norm1 is not None and norm2 is not None

        # Compare only feature joints (unmapped zero-landmarks are not meaningful)
        np.testing.assert_allclose(
            norm1[FEATURE_INDICES], norm2[FEATURE_INDICES], atol=1e-3
        )

    def test_scale_invariance(self):
        """Uniformly scaling all landmarks should not change normalized feature joints."""
        lm1 = _make_3d_landmarks(hip_width=30.0, scale=1.0)
        lm2 = _make_3d_landmarks(hip_width=30.0, scale=2.5)

        norm1 = normalize_frame_3d_real(lm1)
        norm2 = normalize_frame_3d_real(lm2)
        assert norm1 is not None and norm2 is not None

        # Compare only feature joints
        np.testing.assert_allclose(
            norm1[FEATURE_INDICES], norm2[FEATURE_INDICES], atol=1e-3
        )

    def test_z_column_nonzero(self):
        """Z column should contain real depth-based values, not all zeros."""
        lm = _make_3d_landmarks()
        # Confirm input has real Z
        assert abs(float(lm[LEFT_SHOULDER, 2])) > 0.01

        norm = normalize_frame_3d_real(lm)
        assert norm is not None

        # At least some joints should have non-zero Z in normalized frame
        z_vals = norm[FEATURE_INDICES, 2]
        assert np.any(np.abs(z_vals) > 1e-4), "Normalized Z should have non-zero values"

    def test_returns_none_for_low_visibility(self):
        """Should return None if anchor joints have low visibility."""
        lm = _make_3d_landmarks()
        lm[LEFT_HIP, 3] = 0.1  # below 0.3 threshold
        assert normalize_frame_3d_real(lm) is None

    def test_returns_none_for_degenerate_hips(self):
        """Should return None if hips overlap (zero hip width)."""
        lm = _make_3d_landmarks()
        lm[LEFT_HIP, :3] = lm[RIGHT_HIP, :3]
        assert normalize_frame_3d_real(lm) is None

    def test_2d_normalize_still_works(self):
        """Existing 2D normalize_frame should still work independently."""
        lm = _make_3d_landmarks()
        norm2d = normalize_frame(lm)
        assert norm2d is not None
        assert norm2d.shape == (33, 2)


class TestFeatureVector3D:
    def test_feature_vector_42d_from_3d(self):
        """feature_vector on (33, 3) normalized data should produce 42D vector."""
        lm = _make_3d_landmarks()
        norm = normalize_frame_3d_real(lm)
        assert norm is not None

        feat = feature_vector(norm)
        assert feat.shape == (14 * 3,)  # 42D

    def test_feature_vector_28d_from_2d(self):
        """feature_vector on (33, 2) normalized data should produce 28D vector."""
        lm = _make_3d_landmarks()
        norm = normalize_frame(lm)
        assert norm is not None

        feat = feature_vector(norm)
        assert feat.shape == (14 * 2,)  # 28D

    def test_feature_vector_no_nan(self):
        """Feature vector from valid 3D landmarks should have no NaN."""
        lm = _make_3d_landmarks()
        norm = normalize_frame_3d_real(lm)
        assert norm is not None

        feat = feature_vector(norm)
        assert not np.any(np.isnan(feat))
        assert not np.any(np.isinf(feat))

    def test_feature_vector_invariant_to_translation(self):
        """Feature vectors from translated poses should be identical."""
        lm1 = _make_3d_landmarks(pelvis_offset=(0.0, 0.0, 0.0))
        lm2 = _make_3d_landmarks(pelvis_offset=(100.0, 200.0, 3.0))

        norm1 = normalize_frame_3d_real(lm1)
        norm2 = normalize_frame_3d_real(lm2)
        assert norm1 is not None and norm2 is not None

        feat1 = feature_vector(norm1)
        feat2 = feature_vector(norm2)
        np.testing.assert_allclose(feat1, feat2, atol=1e-3)


class TestStreamingAnalyzer3DDetection:
    def test_has_real_depth_true_for_3d(self):
        """StreamingAnalyzer._has_real_depth should return True for 3D landmarks."""
        from backend.streaming_analyzer import StreamingAnalyzer

        lm = _make_3d_landmarks()
        assert StreamingAnalyzer._has_real_depth(lm) is True

    def test_has_real_depth_false_for_2d(self):
        """StreamingAnalyzer._has_real_depth should return False when Z is zero."""
        from backend.streaming_analyzer import StreamingAnalyzer

        lm = _make_3d_landmarks()
        lm[:, 2] = 0.0  # zero out all Z values
        assert StreamingAnalyzer._has_real_depth(lm) is False

    def test_process_frame_with_3d_landmarks(self):
        """Process frame should succeed with 3D landmarks."""
        from backend.streaming_analyzer import StreamingAnalyzer

        a = StreamingAnalyzer()
        lm = _make_3d_landmarks()
        result = a.process_frame(lm)
        assert result is not None
        assert "frame_index" in result

    def test_dimension_locked_per_session(self):
        """Once dimension is locked (2D or 3D), it should stay locked."""
        from backend.streaming_analyzer import StreamingAnalyzer

        a = StreamingAnalyzer()
        lm_3d = _make_3d_landmarks()

        # First frame with 3D data locks to 3D
        a.process_frame(lm_3d)
        assert a._norm_dims == 3

    def test_dimension_locked_2d(self):
        """2D landmarks should lock to 2D."""
        from backend.streaming_analyzer import StreamingAnalyzer

        a = StreamingAnalyzer()
        lm = _make_3d_landmarks()
        lm[:, 2] = 0.0  # make 2D

        a.process_frame(lm)
        assert a._norm_dims == 2
