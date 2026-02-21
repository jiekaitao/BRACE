"""Body proportion extraction for identity verification.

Uses limb length ratios (normalized by hip width) as a scale-invariant
biometric signal. Complementary to appearance embeddings for re-ID.
"""

from __future__ import annotations

import numpy as np

# MediaPipe landmark indices for limb pairs
# Each tuple: (joint_a, joint_b) — the Euclidean distance forms one limb length
_LIMB_PAIRS = [
    (11, 13),  # left upper arm
    (13, 15),  # left forearm
    (12, 14),  # right upper arm
    (14, 16),  # right forearm
    (23, 25),  # left thigh
    (25, 27),  # left shin
    (24, 26),  # right thigh
    (26, 28),  # right shin
    (11, 23),  # left torso (shoulder to hip)
    (12, 24),  # right torso (shoulder to hip)
]

# Hips used for normalization
_LEFT_HIP = 23
_RIGHT_HIP = 24

_MIN_VISIBILITY = 0.3


def compute_limb_lengths(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """Compute 10 limb lengths normalized by hip width.

    Args:
        landmarks_xyzv: (33, 4) array [x, y, z, visibility] in any coordinate system.

    Returns:
        (10,) float32 array of hip-width-normalized limb lengths, or None if
        hips are not visible or hip width is degenerate.
    """
    if landmarks_xyzv is None or landmarks_xyzv.shape[0] < 33:
        return None

    # Check hip visibility
    if (landmarks_xyzv[_LEFT_HIP, 3] < _MIN_VISIBILITY or
            landmarks_xyzv[_RIGHT_HIP, 3] < _MIN_VISIBILITY):
        return None

    hip_l = landmarks_xyzv[_LEFT_HIP, :2]
    hip_r = landmarks_xyzv[_RIGHT_HIP, :2]
    hip_width = np.linalg.norm(hip_r - hip_l)

    if hip_width < 1e-6:
        return None

    lengths = np.zeros(len(_LIMB_PAIRS), dtype=np.float32)
    for i, (a, b) in enumerate(_LIMB_PAIRS):
        if (landmarks_xyzv[a, 3] < _MIN_VISIBILITY or
                landmarks_xyzv[b, 3] < _MIN_VISIBILITY):
            lengths[i] = 0.0
        else:
            dist = np.linalg.norm(landmarks_xyzv[a, :2] - landmarks_xyzv[b, :2])
            lengths[i] = dist / hip_width

    return lengths


class ProportionAccumulator:
    """Accumulates limb length samples and computes a stable proportion vector."""

    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self._samples: list[np.ndarray] = []

    def add(self, limb_lengths: np.ndarray) -> None:
        """Add a (10,) limb length vector."""
        self._samples.append(limb_lengths.copy())

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def get_proportion_vector(self) -> np.ndarray | None:
        """Return median proportion vector, or None if not enough samples."""
        if len(self._samples) < self.min_samples:
            return None
        return np.median(np.stack(self._samples), axis=0).astype(np.float32)

    def similarity(self, other: "ProportionAccumulator") -> float | None:
        """Cosine similarity between this and another accumulator's proportion vectors."""
        v1 = self.get_proportion_vector()
        v2 = other.get_proportion_vector()
        if v1 is None or v2 is None:
            return None
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
