"""Tests for SRP normalization: translation, scale, and rotation invariance in 3D."""

import numpy as np
import pytest

from brace.core.srp import normalize_to_body_frame_3d, procrustes_align_3d
from brace.data.joint_map import HIP_LEFT, HIP_RIGHT, SHOULDER_LEFT, SHOULDER_RIGHT, NUM_KINECT_JOINTS


def _make_skeleton(hip_left, hip_right, shoulder_left, shoulder_right):
    """Create a 25-joint skeleton with specified anchor joints, others random but consistent."""
    rng = np.random.RandomState(42)
    joints = rng.randn(NUM_KINECT_JOINTS, 3).astype(np.float32) * 0.1
    joints[HIP_LEFT] = hip_left
    joints[HIP_RIGHT] = hip_right
    joints[SHOULDER_LEFT] = shoulder_left
    joints[SHOULDER_RIGHT] = shoulder_right
    return joints


def test_translation_invariance():
    """Translating the skeleton should not change the normalized output."""
    skel = _make_skeleton([-0.1, 0, 0.5], [0.1, 0, 0.5], [-0.15, -0.3, 0.5], [0.15, -0.3, 0.5])
    shifted = skel + np.array([5.0, -3.0, 10.0])

    norm1, _ = normalize_to_body_frame_3d(skel)
    norm2, _ = normalize_to_body_frame_3d(shifted)

    np.testing.assert_allclose(norm1, norm2, atol=1e-4)


def test_scale_invariance():
    """Scaling the skeleton uniformly should not change the normalized output."""
    skel = _make_skeleton([-0.1, 0, 0.5], [0.1, 0, 0.5], [-0.15, -0.3, 0.5], [0.15, -0.3, 0.5])
    scaled = skel * 3.7

    norm1, _ = normalize_to_body_frame_3d(skel)
    norm2, _ = normalize_to_body_frame_3d(scaled)

    np.testing.assert_allclose(norm1, norm2, atol=1e-4)


def test_combined_translation_and_scale():
    """Both translation and scale together should be invariant."""
    skel = _make_skeleton([-0.1, 0, 0.5], [0.1, 0, 0.5], [-0.15, -0.3, 0.5], [0.15, -0.3, 0.5])
    transformed = skel * 2.5 + np.array([100, -50, 30])

    norm1, _ = normalize_to_body_frame_3d(skel)
    norm2, _ = normalize_to_body_frame_3d(transformed)

    np.testing.assert_allclose(norm1, norm2, atol=1e-4)


def test_pelvis_at_origin():
    """After normalization, the pelvis (midpoint of hips) should be at origin."""
    skel = _make_skeleton([-0.2, 0.1, 0.5], [0.2, 0.1, 0.5], [-0.2, -0.3, 0.5], [0.2, -0.3, 0.5])
    norm, _ = normalize_to_body_frame_3d(skel)

    pelvis_norm = (norm[HIP_LEFT] + norm[HIP_RIGHT]) / 2.0
    np.testing.assert_allclose(pelvis_norm, [0, 0, 0], atol=1e-4)


def test_hip_width_is_one():
    """After normalization, hip width should be ~1.0 (hip-width units)."""
    skel = _make_skeleton([-0.15, 0, 0.5], [0.15, 0, 0.5], [-0.2, -0.4, 0.5], [0.2, -0.4, 0.5])
    norm, _ = normalize_to_body_frame_3d(skel)

    hip_dist = np.linalg.norm(norm[HIP_LEFT] - norm[HIP_RIGHT])
    np.testing.assert_allclose(hip_dist, 1.0, atol=1e-3)


def test_batch_normalization():
    """Normalizing a batch should give same results as individual frames."""
    rng = np.random.RandomState(123)
    batch = rng.randn(5, NUM_KINECT_JOINTS, 3).astype(np.float32)
    # Make hips and shoulders reasonable
    for i in range(5):
        batch[i, HIP_LEFT] = [-0.1 + rng.randn() * 0.01, rng.randn() * 0.01, 0.5]
        batch[i, HIP_RIGHT] = [0.1 + rng.randn() * 0.01, rng.randn() * 0.01, 0.5]
        batch[i, SHOULDER_LEFT] = [-0.15, -0.3, 0.5]
        batch[i, SHOULDER_RIGHT] = [0.15, -0.3, 0.5]

    norm_batch, _ = normalize_to_body_frame_3d(batch)

    for i in range(5):
        norm_single, _ = normalize_to_body_frame_3d(batch[i])
        np.testing.assert_allclose(norm_batch[i], norm_single, atol=1e-4)


def test_procrustes_align_3d_identity():
    """Aligning identical point sets should return the same points."""
    rng = np.random.RandomState(42)
    pts = rng.randn(8, 3).astype(np.float32)

    aligned, R, s, t = procrustes_align_3d(pts, pts)
    np.testing.assert_allclose(aligned, pts, atol=1e-4)
    assert abs(s - 1.0) < 0.01


def test_procrustes_align_3d_rotation():
    """Procrustes should recover alignment after a known rotation."""
    rng = np.random.RandomState(42)
    pts = rng.randn(10, 3).astype(np.float32)

    # Rotate 45 degrees around z-axis
    theta = np.pi / 4
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = pts @ R_true.T

    aligned, R, s, t = procrustes_align_3d(pts, rotated)
    np.testing.assert_allclose(aligned, pts, atol=1e-3)


def test_procrustes_align_3d_scale_and_translation():
    """Procrustes should handle combined scale + translation."""
    rng = np.random.RandomState(42)
    pts = rng.randn(8, 3).astype(np.float32)
    transformed = pts * 2.5 + np.array([10, -5, 3])

    aligned, R, s, t = procrustes_align_3d(pts, transformed)
    np.testing.assert_allclose(aligned, pts, atol=1e-3)
