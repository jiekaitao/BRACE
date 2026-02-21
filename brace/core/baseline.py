"""Build a motion baseline from normal gait data for a single subject."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .srp import normalize_to_body_frame_3d
from .features import extract_features_sequence, z_score_scale, pairwise_distances, robust_std
from .gait_cycle import extract_resampled_cycles
from ..data.joint_map import FEATURE_LANDMARKS


def build_baseline(
    normal_sequences: list[np.ndarray],
    target_cycle_length: int = 60,
    fs: float = 30.0,
) -> dict:
    """Build a motion baseline from a subject's normal gait sequences.

    Steps:
    1. SRP-normalize each sequence
    2. Detect gait cycles and resample to fixed length
    3. Extract feature vectors for each cycle
    4. Compute mean/std trajectory, z-score params, distance calibration

    Args:
        normal_sequences: list of (N, 25, 3) raw skeleton sequences (normal gait).
        target_cycle_length: frames per resampled cycle.
        fs: data sampling rate.

    Returns:
        dict with all baseline model data.
    """
    all_cycle_features = []
    all_norm_cycles = []

    for seq in normal_sequences:
        # SRP normalize the entire sequence
        norm_seq, _ = normalize_to_body_frame_3d(seq)

        # Extract resampled gait cycles from normalized data
        cycles = extract_resampled_cycles(norm_seq, target_cycle_length, fs=fs)
        for cycle in cycles:
            feats = extract_features_sequence(cycle)  # (60, 42)
            all_cycle_features.append(feats)
            all_norm_cycles.append(cycle)

    if not all_cycle_features:
        raise ValueError("No gait cycles could be extracted from the normal sequences.")

    # Stack: (num_cycles, 60, 42)
    cycle_features = np.stack(all_cycle_features, axis=0).astype(np.float32)
    norm_cycles = np.stack(all_norm_cycles, axis=0).astype(np.float32)
    n_cycles = cycle_features.shape[0]

    # Mean and std trajectory across all normal cycles
    mean_trajectory = np.mean(cycle_features, axis=0)  # (60, 42)
    std_trajectory = np.std(cycle_features, axis=0)     # (60, 42)
    std_trajectory = np.maximum(std_trajectory, 1e-6)

    # Z-score normalization params (across all frames of all cycles)
    flat_features = cycle_features.reshape(-1, cycle_features.shape[-1])  # (n_cycles*60, 42)
    feat_mean = np.mean(flat_features, axis=0).astype(np.float32)
    feat_std = robust_std(flat_features).astype(np.float32)

    # Distance calibration via leave-one-out nearest neighbor on cycle-level features
    # Flatten each cycle into a single vector for distance computation
    cycle_vectors = cycle_features.reshape(n_cycles, -1)  # (n_cycles, 60*42)
    cycle_scaled = ((cycle_vectors - cycle_vectors.mean(axis=0)) / robust_std(cycle_vectors)).astype(np.float32)

    dmat = pairwise_distances(cycle_scaled)
    n = dmat.shape[0]
    dmat[np.arange(n), np.arange(n)] = np.inf
    loo_nearest = np.min(dmat, axis=1)

    dist_p50 = float(np.percentile(loo_nearest, 50))
    dist_p90 = float(np.percentile(loo_nearest, 90))
    dist_p99 = float(np.percentile(loo_nearest, 99))

    return {
        "mean_trajectory": mean_trajectory.astype(np.float32),  # (60, 42)
        "std_trajectory": std_trajectory.astype(np.float32),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_cycles": n_cycles,
        "target_cycle_length": target_cycle_length,
        "distance_calibration": {
            "p50": dist_p50,
            "p90": dist_p90,
            "p99": dist_p99,
        },
        "norm_cycles": norm_cycles,  # (n_cycles, 60, 25, 3) for viz
        "cycle_features": cycle_features,  # (n_cycles, 60, 42)
    }


def save_baseline(baseline: dict, path: str | Path) -> None:
    """Save baseline model to .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean_trajectory=baseline["mean_trajectory"],
        std_trajectory=baseline["std_trajectory"],
        feat_mean=baseline["feat_mean"],
        feat_std=baseline["feat_std"],
        n_cycles=np.array([baseline["n_cycles"]]),
        target_cycle_length=np.array([baseline["target_cycle_length"]]),
        dist_p50=np.array([baseline["distance_calibration"]["p50"]]),
        dist_p90=np.array([baseline["distance_calibration"]["p90"]]),
        dist_p99=np.array([baseline["distance_calibration"]["p99"]]),
    )


def load_baseline(path: str | Path) -> dict:
    """Load baseline model from .npz file."""
    data = np.load(path)
    return {
        "mean_trajectory": data["mean_trajectory"],
        "std_trajectory": data["std_trajectory"],
        "feat_mean": data["feat_mean"],
        "feat_std": data["feat_std"],
        "n_cycles": int(data["n_cycles"][0]),
        "target_cycle_length": int(data["target_cycle_length"][0]),
        "distance_calibration": {
            "p50": float(data["dist_p50"][0]),
            "p90": float(data["dist_p90"][0]),
            "p99": float(data["dist_p99"][0]),
        },
    }
