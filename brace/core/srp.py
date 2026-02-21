"""SRP normalization — Scale, Rotation, Position invariance for 3D skeletons.

Ported from EXPERIMENT_PT_coach/pt_coach/common.py, extended to full 3D
(Kinect v2 gives real depth, unlike MediaPipe's estimated z).
"""

from __future__ import annotations

import numpy as np

from ..data.joint_map import (
    HIP_LEFT,
    HIP_RIGHT,
    SHOULDER_LEFT,
    SHOULDER_RIGHT,
)


def _unit3(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize a 3D vector to unit length."""
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def normalize_to_body_frame_3d(
    joints: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert 3D skeleton into a body-centric coordinate frame.

    Body frame construction:
        - Origin: pelvis = midpoint(hip_left, hip_right)
        - Scale: hip_width (all coordinates in hip-width units)
        - X-axis: hip_left → hip_right direction
        - Y-axis: Gram-Schmidt orthogonalization of shoulder_center - pelvis
        - Z-axis: cross product (X × Y), forming right-handed frame

    Args:
        joints: (25, 3) or (N, 25, 3) joint positions.

    Returns:
        norm_joints: same shape as input, in body-frame coordinates.
        frame_info: dict with pelvis, axes, scale.
    """
    single = joints.ndim == 2
    if single:
        joints = joints[None, :, :]

    pts = joints.astype(np.float64)
    n_frames = pts.shape[0]
    out = np.zeros_like(pts)

    # We build one frame per sequence (from frame 0), not per-frame
    # This preserves temporal consistency within a sequence.
    lhip = pts[:, HIP_LEFT, :]    # (N, 3)
    rhip = pts[:, HIP_RIGHT, :]
    lsh = pts[:, SHOULDER_LEFT, :]
    rsh = pts[:, SHOULDER_RIGHT, :]

    pelvis = (lhip + rhip) * 0.5  # (N, 3)
    hip_vec = lhip - rhip          # (N, 3)
    hip_width = np.linalg.norm(hip_vec, axis=1, keepdims=True)  # (N, 1)
    hip_width = np.maximum(hip_width, 1e-4)

    # Per-frame body axes
    x_axis = hip_vec / hip_width  # (N, 3)

    shoulder_center = (lsh + rsh) * 0.5
    up_guess = shoulder_center - pelvis  # (N, 3)

    # Gram-Schmidt: remove x-component from up_guess
    dot_xu = np.sum(up_guess * x_axis, axis=1, keepdims=True)  # (N, 1)
    up_proj = up_guess - dot_xu * x_axis

    up_norm = np.linalg.norm(up_proj, axis=1, keepdims=True)
    # Fallback for degenerate cases
    for i in range(n_frames):
        if up_norm[i, 0] < 1e-6:
            up_proj[i] = np.array([0.0, 1.0, 0.0])
            up_norm[i, 0] = 1.0

    y_axis = up_proj / up_norm  # (N, 3)
    z_axis = np.cross(x_axis, y_axis)  # (N, 3)

    # Transform all joints into body frame
    for i in range(n_frames):
        rel = pts[i] - pelvis[i]  # (25, 3)
        hw = hip_width[i, 0]
        # Project onto body axes and scale by hip width
        out[i, :, 0] = (rel @ x_axis[i]) / hw
        out[i, :, 1] = (rel @ y_axis[i]) / hw
        out[i, :, 2] = (rel @ z_axis[i]) / hw

    norm = out.astype(np.float32)
    info = {
        "pelvis": pelvis.astype(np.float32),
        "x_axis": x_axis.astype(np.float32),
        "y_axis": y_axis.astype(np.float32),
        "z_axis": z_axis.astype(np.float32),
        "scale": hip_width.astype(np.float32),
    }

    if single:
        norm = norm[0]
        info = {k: v[0] for k, v in info.items()}

    return norm, info


def procrustes_align_3d(
    target_pts: np.ndarray,
    source_pts: np.ndarray,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Procrustes alignment in 3D: find optimal rotation + uniform scale.

    Finds rotation R and scale s that minimizes:
        ||target - (s * source @ R.T + t)||^2

    Args:
        target_pts: (K, 3) target points.
        source_pts: (K, 3) source points to align onto target.
        allow_reflection: if False, constrain to proper rotation (det R = +1).

    Returns:
        aligned: (K, 3) source points after alignment.
        rotation: (3, 3) rotation matrix.
        scale: scalar scale factor.
        translation: (3,) translation vector.
    """
    t = target_pts.astype(np.float64)
    s = source_pts.astype(np.float64)

    t_mean = t.mean(axis=0)
    s_mean = s.mean(axis=0)
    t_c = t - t_mean
    s_c = s - s_mean

    t_norm = np.linalg.norm(t_c)
    s_norm = np.linalg.norm(s_c)
    if t_norm < 1e-8 or s_norm < 1e-8:
        return source_pts.copy(), np.eye(3, dtype=np.float32), 1.0, np.zeros(3, dtype=np.float32)

    scale = t_norm / s_norm

    # Optimal rotation via SVD of cross-covariance
    M = t_c.T @ s_c  # (3, 3)
    U, S, Vt = np.linalg.svd(M)

    if not allow_reflection:
        d = np.linalg.det(U @ Vt)
        D = np.diag([1.0, 1.0, d])
        R = U @ D @ Vt
    else:
        R = U @ Vt

    aligned = scale * (s_c @ R.T) + t_mean

    return (
        aligned.astype(np.float32),
        R.astype(np.float32),
        float(scale),
        (t_mean - scale * s_mean @ R.T).astype(np.float32),
    )
