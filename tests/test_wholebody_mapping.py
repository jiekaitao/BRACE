"""Tests for wholebody133_to_mediapipe33 mapping."""

import numpy as np
import pytest

from brace.core.pose import (
    wholebody133_to_mediapipe33,
    COCO_WHOLEBODY_TO_MP,
    NUM_MP_LANDMARKS,
    FEATURE_INDICES,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
)


def _make_wholebody_keypoints(z_value: float = 1.5) -> np.ndarray:
    """Create synthetic 133-keypoint wholebody data with real Z depth."""
    kpts = np.zeros((133, 4), dtype=np.float32)
    rng = np.random.RandomState(42)

    for i in range(133):
        kpts[i, 0] = 100.0 + rng.randn() * 50  # x pixel
        kpts[i, 1] = 200.0 + rng.randn() * 50  # y pixel
        kpts[i, 2] = z_value + rng.randn() * 0.3  # z depth
        kpts[i, 3] = 0.8 + rng.rand() * 0.2  # confidence

    return kpts


class TestWholebodyMapping:
    def test_output_shape(self):
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)
        assert mp.shape == (33, 4)

    def test_z_values_preserved(self):
        """Real depth values from RTMW3D should be preserved, not zeroed."""
        kpts = _make_wholebody_keypoints(z_value=2.0)
        mp = wholebody133_to_mediapipe33(kpts)

        # Check that mapped body joints have non-zero Z
        body_mp_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for mp_idx in body_mp_indices:
            assert mp[mp_idx, 2] != 0.0, f"MP joint {mp_idx} Z should be non-zero"

    def test_body_keypoints_mapped_correctly(self):
        """Body keypoints (COCO 5-16) should map to correct MediaPipe indices."""
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)

        # COCO body shoulder (5) -> MP left_shoulder (11)
        assert mp[11, 0] == pytest.approx(kpts[5, 0], abs=1e-4)
        assert mp[11, 1] == pytest.approx(kpts[5, 1], abs=1e-4)
        assert mp[11, 2] == pytest.approx(kpts[5, 2], abs=1e-4)

        # COCO body right_shoulder (6) -> MP right_shoulder (12)
        assert mp[12, 0] == pytest.approx(kpts[6, 0], abs=1e-4)

        # COCO hips (11, 12) -> MP hips (23, 24)
        assert mp[23, 0] == pytest.approx(kpts[11, 0], abs=1e-4)
        assert mp[24, 0] == pytest.approx(kpts[12, 0], abs=1e-4)

    def test_foot_keypoints_mapped(self):
        """Foot keypoints (wholebody 17-22) should map to MP foot landmarks."""
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)

        # left_heel (wb 19) -> MP left_heel (29)
        assert mp[29, 0] == pytest.approx(kpts[19, 0], abs=1e-4)
        # right_heel (wb 22) -> MP right_heel (30)
        assert mp[30, 0] == pytest.approx(kpts[22, 0], abs=1e-4)

    def test_nose_mapped(self):
        """Nose (wb 0) -> MP nose (0)."""
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)
        assert mp[0, 0] == pytest.approx(kpts[0, 0], abs=1e-4)

    def test_higher_confidence_wins(self):
        """When multiple wholebody kpts map to same MP index, higher conf wins."""
        kpts = np.zeros((133, 4), dtype=np.float32)

        # Both wb 17 (big toe) and wb 18 (small toe) map to MP 31
        kpts[17] = [100, 200, 1.5, 0.9]  # higher conf
        kpts[18] = [110, 210, 1.6, 0.7]  # lower conf

        mp = wholebody133_to_mediapipe33(kpts)
        assert mp[31, 0] == pytest.approx(100.0, abs=1e-4)  # from wb 17

    def test_unmapped_joints_are_zero(self):
        """MediaPipe joints with no mapping should remain zero."""
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)

        # MP indices for finger details (17-22) are not directly mapped from
        # COCO body. Check that pinky/index/thumb remain zero (no mapping).
        for mp_idx in [17, 18, 19, 20, 21, 22]:
            assert mp[mp_idx, 0] == 0.0

    def test_all_feature_joints_mapped(self):
        """All 14 SRP feature joints should have non-zero values after mapping."""
        kpts = _make_wholebody_keypoints()
        mp = wholebody133_to_mediapipe33(kpts)

        for fi in FEATURE_INDICES:
            assert mp[fi, 3] > 0.0, f"Feature joint {fi} should have visibility > 0"

    def test_anchor_joints_have_depth(self):
        """SRP anchor joints (hips, shoulders) should have real Z depth."""
        kpts = _make_wholebody_keypoints(z_value=3.0)
        mp = wholebody133_to_mediapipe33(kpts)

        for j in [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]:
            assert abs(mp[j, 2]) > 0.1, f"Anchor joint {j} should have real depth"

    def test_zero_input_produces_zero_output(self):
        """All-zero input should produce all-zero output."""
        kpts = np.zeros((133, 4), dtype=np.float32)
        mp = wholebody133_to_mediapipe33(kpts)
        assert np.all(mp == 0)

    def test_mapping_constant_covers_all_feature_joints(self):
        """COCO_WHOLEBODY_TO_MP should cover all 14 feature joint MP indices."""
        mapped_mp = set(COCO_WHOLEBODY_TO_MP.values())
        for fi in FEATURE_INDICES:
            assert fi in mapped_mp, f"Feature index {fi} not in COCO_WHOLEBODY_TO_MP values"
