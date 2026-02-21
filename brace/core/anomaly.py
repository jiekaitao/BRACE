"""Anomaly detection: score gait cycles against a baseline model."""

from __future__ import annotations

import numpy as np

from .srp import normalize_to_body_frame_3d
from .features import extract_features_sequence
from .gait_cycle import extract_resampled_cycles
from ..data.joint_map import FEATURE_LANDMARKS, FEATURE_JOINT_NAMES


def score_cycle(
    cycle_features: np.ndarray,
    baseline: dict,
) -> dict:
    """Score a single gait cycle against a baseline.

    Args:
        cycle_features: (T, 42) feature matrix for one gait cycle.
        baseline: dict from build_baseline() or load_baseline().

    Returns:
        dict with anomaly_score, joint_scores, worst_joints, phase_scores.
    """
    mean_traj = baseline["mean_trajectory"]  # (T, 42)
    std_traj = baseline["std_trajectory"]

    # Per-frame, per-feature deviation in std units
    deviation = np.abs(cycle_features - mean_traj) / std_traj  # (T, 42)

    # Per-frame RMS divergence
    frame_rms = np.sqrt(np.mean(deviation ** 2, axis=1))  # (T,)

    # Overall anomaly score: mean RMS across the cycle
    anomaly_score = float(np.mean(frame_rms))

    # Per-joint deviation: group by joint (each joint = 3 consecutive features)
    n_joints = len(FEATURE_LANDMARKS)
    joint_scores = {}
    for j_idx, joint_id in enumerate(FEATURE_LANDMARKS):
        feat_start = j_idx * 3
        feat_end = feat_start + 3
        joint_dev = float(np.mean(deviation[:, feat_start:feat_end]))
        joint_name = FEATURE_JOINT_NAMES[joint_id]
        joint_scores[joint_name] = joint_dev

    # Top-3 worst joints
    sorted_joints = sorted(joint_scores.items(), key=lambda x: x[1], reverse=True)
    worst_joints = sorted_joints[:3]

    # Phase scores: divide cycle into 4 phases (25% each)
    T = frame_rms.shape[0]
    quarter = max(1, T // 4)
    phase_scores = {
        "0-25%": float(np.mean(frame_rms[:quarter])),
        "25-50%": float(np.mean(frame_rms[quarter:2 * quarter])),
        "50-75%": float(np.mean(frame_rms[2 * quarter:3 * quarter])),
        "75-100%": float(np.mean(frame_rms[3 * quarter:])),
    }

    return {
        "anomaly_score": anomaly_score,
        "frame_rms": frame_rms,
        "joint_scores": joint_scores,
        "worst_joints": worst_joints,
        "phase_scores": phase_scores,
    }


def score_sequence(
    skeleton_seq: np.ndarray,
    baseline: dict,
    target_cycle_length: int = 60,
    fs: float = 30.0,
) -> list[dict]:
    """Score an entire skeleton sequence against a baseline.

    Segments into gait cycles, normalizes, extracts features, and scores each.

    Returns:
        List of score dicts, one per detected gait cycle.
    """
    norm_seq, _ = normalize_to_body_frame_3d(skeleton_seq)
    cycles = extract_resampled_cycles(norm_seq, target_cycle_length, fs=fs)

    results = []
    for cycle in cycles:
        feats = extract_features_sequence(cycle)  # (60, 42)
        result = score_cycle(feats, baseline)
        results.append(result)

    return results


def score_sequence_aggregate(
    skeleton_seq: np.ndarray,
    baseline: dict,
    target_cycle_length: int = 60,
    fs: float = 30.0,
) -> dict:
    """Score a sequence and return aggregate statistics.

    Returns:
        dict with mean_anomaly_score, median_anomaly_score, max_anomaly_score,
        n_cycles, per_cycle_scores, aggregate_joint_scores, aggregate_worst_joints.
    """
    cycle_results = score_sequence(skeleton_seq, baseline, target_cycle_length, fs)

    if not cycle_results:
        return {
            "mean_anomaly_score": float("nan"),
            "median_anomaly_score": float("nan"),
            "max_anomaly_score": float("nan"),
            "n_cycles": 0,
            "per_cycle_scores": [],
            "aggregate_joint_scores": {},
            "aggregate_worst_joints": [],
        }

    scores = [r["anomaly_score"] for r in cycle_results]

    # Aggregate joint scores across cycles
    all_joint_names = list(cycle_results[0]["joint_scores"].keys())
    agg_joints = {}
    for name in all_joint_names:
        agg_joints[name] = float(np.mean([r["joint_scores"][name] for r in cycle_results]))
    sorted_joints = sorted(agg_joints.items(), key=lambda x: x[1], reverse=True)

    return {
        "mean_anomaly_score": float(np.mean(scores)),
        "median_anomaly_score": float(np.median(scores)),
        "max_anomaly_score": float(np.max(scores)),
        "n_cycles": len(cycle_results),
        "per_cycle_scores": scores,
        "aggregate_joint_scores": agg_joints,
        "aggregate_worst_joints": sorted_joints[:3],
    }
