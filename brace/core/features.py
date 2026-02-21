"""Feature extraction and z-score scaling for skeleton data."""

from __future__ import annotations

import numpy as np

from ..data.joint_map import FEATURE_LANDMARKS, FEATURE_DIM


def feature_vector(norm_joints: np.ndarray, feature_indices: list[int] | None = None) -> np.ndarray:
    """Flatten selected landmarks into a single feature vector.

    Args:
        norm_joints: (25, 3) normalized joint positions.
        feature_indices: which joint indices to use. Defaults to FEATURE_LANDMARKS.

    Returns:
        1D float32 array of length len(feature_indices) * 3.
    """
    if feature_indices is None:
        feature_indices = FEATURE_LANDMARKS
    return norm_joints[feature_indices, :].reshape(-1).astype(np.float32)


def extract_features_sequence(norm_sequence: np.ndarray, feature_indices: list[int] | None = None) -> np.ndarray:
    """Extract feature vectors for an entire sequence.

    Args:
        norm_sequence: (N, 25, 3) normalized skeleton sequence.
        feature_indices: which joints to use.

    Returns:
        (N, D) feature matrix where D = len(feature_indices) * 3.
    """
    if feature_indices is None:
        feature_indices = FEATURE_LANDMARKS
    n = norm_sequence.shape[0]
    d = len(feature_indices) * 3
    features = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        features[i] = feature_vector(norm_sequence[i], feature_indices)
    return features


def robust_std(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-feature standard deviation, floored at eps to avoid division by zero."""
    std = np.std(a, axis=0)
    return np.where(std < eps, 1.0, std)


def z_score_scale(features: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features.

    Args:
        features: (N, D) feature matrix.
        mean: precomputed mean. If None, computed from features.
        std: precomputed std. If None, computed from features.

    Returns:
        scaled: (N, D) z-score normalized features.
        mean: (D,) mean used.
        std: (D,) std used.
    """
    if mean is None:
        mean = np.mean(features, axis=0).astype(np.float32)
    if std is None:
        std = robust_std(features).astype(np.float32)
    scaled = ((features - mean[None, :]) / std[None, :]).astype(np.float32)
    return scaled, mean, std


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix (N, N)."""
    xx = np.sum(x * x, axis=1, keepdims=True)
    d2 = xx + xx.T - 2.0 * (x @ x.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)
