#!/usr/bin/env python3
"""Comprehensive clustering parameter sweep.

Tests 30+ strategies on synthetic dribbling+dunking data and reports:
- Number of clusters produced
- Whether dribbling and dunking are separated
- Per-strategy timing

Usage:
    python experiments/clustering_sweep.py [--output results.json]
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
    _segment_distance,
    _resample_segment,
    _merge_adjacent_clusters,
    feature_vector,
)
from brace.core.pose import FEATURE_INDICES


# ---------------------------------------------------------------------------
# Synthetic motion generators
# ---------------------------------------------------------------------------

def _make_landmarks(positions_14: np.ndarray) -> np.ndarray:
    """Convert 14 joint positions to full 33-joint landmark array.

    positions_14: (14, 2) or (14, 3) array of feature joint positions.
    Returns (33, 4) array with xyzv.
    """
    ndim = positions_14.shape[1]
    landmarks = np.zeros((33, 4), dtype=np.float32)
    # Set all visibility to 1.0
    landmarks[:, 3] = 1.0
    # Set feature joints
    for i, idx in enumerate(FEATURE_INDICES):
        landmarks[idx, :ndim] = positions_14[i]
    # Set anchor joints (hips + shoulders) for SRP normalization
    # Hips at indices 23, 24 (positions 6, 7 in FEATURE_INDICES)
    # Shoulders at indices 11, 12 (positions 0, 1 in FEATURE_INDICES)
    return landmarks


def generate_dribbling(n_frames: int = 120, fps: float = 30.0) -> np.ndarray:
    """Generate synthetic dribbling motion: repetitive up-down arm motion.

    Returns (n_frames, 28) feature array.
    """
    t = np.linspace(0, n_frames / fps, n_frames)
    features = np.zeros((n_frames, 28), dtype=np.float32)

    for i in range(n_frames):
        # Base standing pose (14 joints × 2D)
        joints = np.zeros((14, 2), dtype=np.float32)

        # Shoulders (0,1) - stable
        joints[0] = [-1.0, 2.0]   # L shoulder
        joints[1] = [1.0, 2.0]    # R shoulder

        # Elbows (2,3) - right arm dribbles
        joints[2] = [-1.0, 1.0]   # L elbow (stable)
        phase = 2 * np.pi * 2.0 * t[i]  # 2 Hz dribbling
        joints[3] = [0.8, 1.0 + 0.3 * np.sin(phase)]  # R elbow bobs

        # Wrists (4,5) - right wrist has larger amplitude
        joints[4] = [-1.0, 0.3]   # L wrist (stable)
        joints[5] = [0.8, 0.3 + 0.5 * np.sin(phase)]  # R wrist dribbles

        # Hips (6,7) - mostly stable, slight bob
        joints[6] = [-0.5, 0.0]   # L hip
        joints[7] = [0.5, 0.0]    # R hip

        # Knees (8,9) - slight flex with dribble
        joints[8] = [-0.5, -1.5 + 0.1 * np.sin(phase)]
        joints[9] = [0.5, -1.5 + 0.1 * np.sin(phase)]

        # Ankles (10,11)
        joints[10] = [-0.5, -3.0]
        joints[11] = [0.5, -3.0]

        # Feet (12,13)
        joints[12] = [-0.5, -3.2]
        joints[13] = [0.5, -3.2]

        features[i] = joints.reshape(-1)

    return features


def generate_dunking(n_frames: int = 60, fps: float = 30.0) -> np.ndarray:
    """Generate synthetic dunking motion: explosive jump + arm raise.

    Returns (n_frames, 28) feature array.
    """
    t = np.linspace(0, n_frames / fps, n_frames)
    features = np.zeros((n_frames, 28), dtype=np.float32)

    for i in range(n_frames):
        # Phase: 0-0.3 = crouch, 0.3-0.6 = jump up, 0.6-1.0 = hang/slam
        progress = i / max(n_frames - 1, 1)

        if progress < 0.3:
            # Crouching phase
            crouch = progress / 0.3  # 0 to 1
            y_offset = -0.5 * crouch  # crouch down
            arm_raise = 0.0
        elif progress < 0.6:
            # Jump phase
            jump = (progress - 0.3) / 0.3  # 0 to 1
            y_offset = -0.5 + 2.5 * jump  # launch up
            arm_raise = jump * 2.0
        else:
            # Hang/slam phase
            hang = (progress - 0.6) / 0.4  # 0 to 1
            y_offset = 2.0 - 0.5 * hang  # slight descent
            arm_raise = 2.0 - 0.3 * hang

        joints = np.zeros((14, 2), dtype=np.float32)

        # Shoulders - rise with jump
        joints[0] = [-1.0, 2.0 + y_offset]
        joints[1] = [1.0, 2.0 + y_offset]

        # Elbows - raise during dunk
        joints[2] = [-0.8, 1.0 + y_offset + arm_raise * 0.5]
        joints[3] = [0.5, 1.0 + y_offset + arm_raise * 0.7]

        # Wrists - reach high during dunk
        joints[4] = [-0.6, 0.3 + y_offset + arm_raise * 0.3]
        joints[5] = [0.3, 0.3 + y_offset + arm_raise]

        # Hips
        joints[6] = [-0.5, 0.0 + y_offset]
        joints[7] = [0.5, 0.0 + y_offset]

        # Knees - extend during jump
        if progress < 0.3:
            knee_bend = -1.0  # crouched
        elif progress < 0.6:
            knee_bend = -1.5 - 0.5 * ((progress - 0.3) / 0.3)  # extending
        else:
            knee_bend = -2.0 + 0.3 * hang  # tucked in air
        joints[8] = [-0.5, knee_bend + y_offset]
        joints[9] = [0.5, knee_bend + y_offset]

        # Ankles
        joints[10] = [-0.5, knee_bend - 1.5 + y_offset]
        joints[11] = [0.5, knee_bend - 1.5 + y_offset]

        # Feet
        joints[12] = [-0.5, knee_bend - 1.7 + y_offset]
        joints[13] = [0.5, knee_bend - 1.7 + y_offset]

        features[i] = joints.reshape(-1)

    return features


def generate_running(n_frames: int = 90, fps: float = 30.0) -> np.ndarray:
    """Generate synthetic running motion: alternating leg/arm swing."""
    t = np.linspace(0, n_frames / fps, n_frames)
    features = np.zeros((n_frames, 28), dtype=np.float32)

    for i in range(n_frames):
        phase = 2 * np.pi * 1.5 * t[i]  # 1.5 Hz stride

        joints = np.zeros((14, 2), dtype=np.float32)

        joints[0] = [-1.0, 2.0]
        joints[1] = [1.0, 2.0]
        joints[2] = [-1.0 + 0.3 * np.sin(phase), 1.0]
        joints[3] = [1.0 - 0.3 * np.sin(phase), 1.0]
        joints[4] = [-1.0 + 0.5 * np.sin(phase), 0.3]
        joints[5] = [1.0 - 0.5 * np.sin(phase), 0.3]
        joints[6] = [-0.5, 0.0]
        joints[7] = [0.5, 0.0]
        joints[8] = [-0.5 + 0.4 * np.sin(phase), -1.5]
        joints[9] = [0.5 - 0.4 * np.sin(phase), -1.5]
        joints[10] = [-0.5 + 0.3 * np.sin(phase + 0.5), -3.0]
        joints[11] = [0.5 - 0.3 * np.sin(phase + 0.5), -3.0]
        joints[12] = [-0.5 + 0.2 * np.sin(phase), -3.2]
        joints[13] = [0.5 - 0.2 * np.sin(phase), -3.2]

        features[i] = joints.reshape(-1)

    return features


def generate_standing(n_frames: int = 60, fps: float = 30.0) -> np.ndarray:
    """Generate near-static standing motion with slight sway."""
    features = np.zeros((n_frames, 28), dtype=np.float32)
    t = np.linspace(0, n_frames / fps, n_frames)

    for i in range(n_frames):
        sway = 0.05 * np.sin(2 * np.pi * 0.3 * t[i])
        joints = np.zeros((14, 2), dtype=np.float32)
        joints[0] = [-1.0, 2.0 + sway]
        joints[1] = [1.0, 2.0 + sway]
        joints[2] = [-1.0, 1.0]
        joints[3] = [1.0, 1.0]
        joints[4] = [-1.0, 0.3]
        joints[5] = [1.0, 0.3]
        joints[6] = [-0.5, 0.0]
        joints[7] = [0.5, 0.0]
        joints[8] = [-0.5, -1.5]
        joints[9] = [0.5, -1.5]
        joints[10] = [-0.5, -3.0]
        joints[11] = [0.5, -3.0]
        joints[12] = [-0.5, -3.2]
        joints[13] = [0.5, -3.2]
        features[i] = joints.reshape(-1)

    return features


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def make_basketball_scenario(dribble_reps: int = 3, dunk_reps: int = 1) -> dict:
    """Create basketball scenario: dribbling sequences + dunking.

    Returns dict with 'features', 'valid_indices', 'expected_clusters', 'description'.
    """
    segments = []
    for _ in range(dribble_reps):
        segments.append(("dribbling", generate_dribbling(n_frames=90)))
    for _ in range(dunk_reps):
        segments.append(("dunking", generate_dunking(n_frames=60)))

    # Concatenate with small transition gaps
    all_features = []
    labels = []
    for label, feat in segments:
        # Add small transition (5 frames of interpolation)
        if all_features:
            prev = all_features[-1][-1]
            curr = feat[0]
            for j in range(5):
                alpha = (j + 1) / 6
                all_features.append([(1 - alpha) * prev + alpha * curr])
                labels.append("transition")
        all_features.append(feat)
        labels.extend([label] * len(feat))

    features = np.vstack([f if isinstance(f, np.ndarray) and f.ndim == 2 else np.array(f) for f in all_features])
    valid_indices = list(range(len(features)))

    return {
        "features": features,
        "valid_indices": valid_indices,
        "expected_min_clusters": 2,  # dribbling + dunking
        "expected_max_clusters": 3,  # allow transition cluster
        "description": f"Basketball: {dribble_reps}x dribbling + {dunk_reps}x dunking",
        "ground_truth_labels": labels,
    }


def make_single_exercise_scenario(name: str = "dribbling", reps: int = 5) -> dict:
    """Single exercise repeated - should produce 1 cluster."""
    gen = {"dribbling": generate_dribbling, "running": generate_running, "standing": generate_standing}
    gen_fn = gen.get(name, generate_dribbling)

    segments = []
    for _ in range(reps):
        segments.append(gen_fn(n_frames=90))

    features_list = []
    for feat in segments:
        if features_list:
            prev = features_list[-1][-1]
            curr = feat[0]
            for j in range(5):
                alpha = (j + 1) / 6
                features_list.append([(1 - alpha) * prev + alpha * curr])
        features_list.append(feat)

    features = np.vstack([f if isinstance(f, np.ndarray) and f.ndim == 2 else np.array(f) for f in features_list])
    valid_indices = list(range(len(features)))

    return {
        "features": features,
        "valid_indices": valid_indices,
        "expected_min_clusters": 1,
        "expected_max_clusters": 1,
        "description": f"Single exercise: {reps}x {name}",
    }


def make_mixed_exercise_scenario() -> dict:
    """Mixed exercises: running + dribbling + standing. Should produce 2-3 clusters."""
    parts = [
        generate_running(90),
        generate_dribbling(90),
        generate_standing(60),
        generate_running(90),
    ]

    features_list = []
    for feat in parts:
        if features_list:
            prev = features_list[-1][-1]
            curr = feat[0]
            for j in range(5):
                alpha = (j + 1) / 6
                features_list.append([(1 - alpha) * prev + alpha * curr])
        features_list.append(feat)

    features = np.vstack([f if isinstance(f, np.ndarray) and f.ndim == 2 else np.array(f) for f in features_list])
    valid_indices = list(range(len(features)))

    return {
        "features": features,
        "valid_indices": valid_indices,
        "expected_min_clusters": 2,
        "expected_max_clusters": 4,
        "description": "Mixed: running + dribbling + standing + running",
    }


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def run_strategy(features, valid_indices, fps, strategy: dict) -> dict:
    """Run a clustering strategy and return results.

    strategy dict keys:
        - name: strategy name
        - cluster_threshold: float
        - min_segment_sec: float
        - linkage_method: str (single, complete, average, ward)
        - mean_weight: float (weight for mean pose distance)
        - spec_weight: float (weight for spectral distance)
        - velocity_kernel_mult: float (multiplier for velocity smoothing kernel)
        - prominence_mult: float (multiplier for minimum prominence)
        - resample_len: int (resample target length)
    """
    name = strategy["name"]
    threshold = strategy.get("cluster_threshold", 2.5)
    min_seg = strategy.get("min_segment_sec", 1.0)
    link_method = strategy.get("linkage_method", "single")
    mean_w = strategy.get("mean_weight", 1.0)
    spec_w = strategy.get("spec_weight", 1.0)
    vel_kernel_mult = strategy.get("velocity_kernel_mult", 1.0)
    prom_mult = strategy.get("prominence_mult", 1.0)
    resample_len = strategy.get("resample_len", 30)

    t0 = time.perf_counter()

    # --- Segmentation ---
    n = features.shape[0]
    if n < 5:
        return {"name": name, "n_clusters": 0, "n_segments": 0, "time_ms": 0, "error": "too few frames"}

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    kernel_size = max(5, int(fps * 0.3 * vel_kernel_mult))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocity, kernel, mode="same")

    min_frames = max(int(fps * min_seg), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return {"name": name, "n_clusters": 0, "n_segments": 0, "time_ms": 0}

    median_vel = float(np.median(positive_vals))
    min_prominence = median_vel * 0.5 * prom_mult

    from scipy.signal import find_peaks
    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    # --- Build segments ---
    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        if end - start < 3:
            continue
        seg_features = features[start:end]
        segments.append({
            "start_valid": start,
            "end_valid": end,
            "start_frame": valid_indices[start] if start < len(valid_indices) else 0,
            "end_frame": valid_indices[end - 1] if end - 1 < len(valid_indices) else valid_indices[-1],
            "features": seg_features,
            "mean_feature": seg_features.mean(axis=0),
        })

    if not segments:
        return {"name": name, "n_clusters": 0, "n_segments": 0, "time_ms": 0}

    n_segs = len(segments)

    if n_segs == 1:
        segments[0]["cluster"] = 0
        dt = (time.perf_counter() - t0) * 1000
        return {"name": name, "n_clusters": 1, "n_segments": 1, "time_ms": round(dt, 2), "segments": segments}

    # --- Pairwise distance with custom weights ---
    dist_matrix = np.zeros((n_segs, n_segs))
    for i in range(n_segs):
        for j in range(i + 1, n_segs):
            ra = _resample_segment(segments[i]["features"], resample_len)
            rb = _resample_segment(segments[j]["features"], resample_len)
            feat_dim = ra.shape[1]

            mean_dist = float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0)))
            spec_a = np.abs(np.fft.rfft(ra, axis=0))[1:] / resample_len
            spec_b = np.abs(np.fft.rfft(rb, axis=0))[1:] / resample_len
            spec_dist = float(np.linalg.norm(spec_a - spec_b))

            d = (mean_w * mean_dist + spec_w * spec_dist) / np.sqrt(feat_dim)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # --- Clustering ---
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=link_method)
    labels = fcluster(Z, t=threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Merge adjacent same-cluster segments
    segments = _merge_adjacent_clusters(segments)

    n_clusters = len(set(s["cluster"] for s in segments))
    dt = (time.perf_counter() - t0) * 1000

    return {
        "name": name,
        "n_clusters": n_clusters,
        "n_segments": len(segments),
        "time_ms": round(dt, 2),
        "segments": segments,
        "dist_matrix": dist_matrix.tolist() if n_segs <= 10 else None,
    }


def define_strategies() -> list[dict]:
    """Define 30+ strategies to test."""
    strategies = []

    # === GROUP 1: Threshold sweep (10 strategies) ===
    for thresh in [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0]:
        strategies.append({
            "name": f"threshold_{thresh}",
            "group": "threshold_sweep",
            "cluster_threshold": thresh,
            "min_segment_sec": 1.0,
        })

    # === GROUP 2: Min segment duration sweep (5 strategies) ===
    for ms in [0.3, 0.5, 0.75, 1.0, 1.5]:
        strategies.append({
            "name": f"min_seg_{ms}s",
            "group": "min_segment_sweep",
            "cluster_threshold": 2.5,
            "min_segment_sec": ms,
        })

    # === GROUP 3: Linkage method (4 strategies) ===
    for method in ["single", "complete", "average", "ward"]:
        strategies.append({
            "name": f"linkage_{method}",
            "group": "linkage_sweep",
            "cluster_threshold": 2.5,
            "linkage_method": method,
        })

    # === GROUP 4: Distance weight ratios (6 strategies) ===
    for mw, sw in [(0.5, 1.0), (1.0, 0.5), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0), (1.0, 2.0)]:
        strategies.append({
            "name": f"weights_mean{mw}_spec{sw}",
            "group": "weight_sweep",
            "cluster_threshold": 2.5,
            "mean_weight": mw,
            "spec_weight": sw,
        })

    # === GROUP 5: Velocity kernel size (4 strategies) ===
    for km in [0.5, 1.0, 1.5, 2.0]:
        strategies.append({
            "name": f"vel_kernel_{km}x",
            "group": "velocity_sweep",
            "cluster_threshold": 2.5,
            "velocity_kernel_mult": km,
        })

    # === GROUP 6: Prominence threshold (4 strategies) ===
    for pm in [0.3, 0.5, 0.8, 1.2]:
        strategies.append({
            "name": f"prominence_{pm}x",
            "group": "prominence_sweep",
            "cluster_threshold": 2.5,
            "prominence_mult": pm,
        })

    # === GROUP 7: Resample length (3 strategies) ===
    for rl in [15, 30, 60]:
        strategies.append({
            "name": f"resample_{rl}",
            "group": "resample_sweep",
            "cluster_threshold": 2.5,
            "resample_len": rl,
        })

    # === GROUP 8: Combined best candidates (5 strategies) ===
    strategies.append({
        "name": "combo_aggressive",
        "group": "combined",
        "cluster_threshold": 1.5,
        "min_segment_sec": 0.5,
        "linkage_method": "complete",
        "mean_weight": 1.5,
        "spec_weight": 1.0,
    })
    strategies.append({
        "name": "combo_balanced",
        "group": "combined",
        "cluster_threshold": 2.0,
        "min_segment_sec": 0.75,
        "linkage_method": "average",
        "mean_weight": 1.0,
        "spec_weight": 1.0,
    })
    strategies.append({
        "name": "combo_conservative",
        "group": "combined",
        "cluster_threshold": 3.0,
        "min_segment_sec": 1.0,
        "linkage_method": "single",
        "mean_weight": 1.0,
        "spec_weight": 1.5,
    })
    strategies.append({
        "name": "combo_mean_heavy",
        "group": "combined",
        "cluster_threshold": 2.2,
        "min_segment_sec": 0.75,
        "linkage_method": "average",
        "mean_weight": 2.0,
        "spec_weight": 0.5,
    })
    strategies.append({
        "name": "combo_spec_heavy",
        "group": "combined",
        "cluster_threshold": 2.2,
        "min_segment_sec": 0.75,
        "linkage_method": "average",
        "mean_weight": 0.5,
        "spec_weight": 2.0,
    })

    return strategies


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def evaluate_strategy(strategy: dict, scenarios: list[dict], fps: float = 30.0) -> dict:
    """Evaluate a strategy across all scenarios."""
    results = {"strategy": strategy["name"], "group": strategy.get("group", ""), "details": []}
    total_score = 0

    for scenario in scenarios:
        result = run_strategy(
            scenario["features"],
            scenario["valid_indices"],
            fps,
            strategy,
        )

        n_clust = result["n_clusters"]
        exp_min = scenario["expected_min_clusters"]
        exp_max = scenario["expected_max_clusters"]

        in_range = exp_min <= n_clust <= exp_max

        # Score: 2 points for in range, -1 per cluster off
        if in_range:
            score = 2
        else:
            score = -abs(n_clust - exp_min) if n_clust < exp_min else -abs(n_clust - exp_max)

        total_score += score

        results["details"].append({
            "scenario": scenario["description"],
            "n_clusters": n_clust,
            "n_segments": result["n_segments"],
            "expected": f"{exp_min}-{exp_max}",
            "in_range": in_range,
            "score": score,
            "time_ms": result["time_ms"],
        })

    results["total_score"] = total_score
    results["all_pass"] = all(d["in_range"] for d in results["details"])
    results["total_time_ms"] = round(sum(d["time_ms"] for d in results["details"]), 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Clustering parameter sweep")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Build scenarios
    scenarios = [
        make_basketball_scenario(dribble_reps=3, dunk_reps=1),
        make_basketball_scenario(dribble_reps=5, dunk_reps=2),
        make_single_exercise_scenario("dribbling", reps=5),
        make_single_exercise_scenario("running", reps=4),
        make_single_exercise_scenario("standing", reps=3),
        make_mixed_exercise_scenario(),
    ]

    strategies = define_strategies()

    print(f"Testing {len(strategies)} strategies across {len(scenarios)} scenarios...")
    print(f"{'='*100}")

    all_results = []

    for strategy in strategies:
        result = evaluate_strategy(strategy, scenarios)
        all_results.append(result)

        status = "PASS" if result["all_pass"] else "FAIL"
        print(f"[{status}] {result['strategy']:30s} | score={result['total_score']:+3d} | time={result['total_time_ms']:6.1f}ms | ", end="")
        for d in result["details"]:
            mark = "✓" if d["in_range"] else "✗"
            print(f"{mark}{d['n_clusters']}/{d['expected']} ", end="")
        print()

    # Sort by score
    all_results.sort(key=lambda r: (-r["total_score"], r["total_time_ms"]))

    print(f"\n{'='*100}")
    print("TOP 10 STRATEGIES:")
    print(f"{'='*100}")
    for i, r in enumerate(all_results[:10]):
        status = "PASS" if r["all_pass"] else "FAIL"
        print(f"  #{i+1} [{status}] {r['strategy']:30s} score={r['total_score']:+3d} time={r['total_time_ms']:6.1f}ms")
        for d in r["details"]:
            mark = "✓" if d["in_range"] else "✗"
            print(f"       {mark} {d['scenario']:50s} clusters={d['n_clusters']} (expect {d['expected']})")

    print(f"\n{'='*100}")
    print("PASSING STRATEGIES (all scenarios in range):")
    passing = [r for r in all_results if r["all_pass"]]
    if passing:
        for r in passing:
            print(f"  ✓ {r['strategy']:30s} score={r['total_score']:+3d} time={r['total_time_ms']:6.1f}ms")
    else:
        print("  No strategy passed all scenarios!")
        print("  Closest strategies:")
        for r in all_results[:5]:
            fails = [d for d in r["details"] if not d["in_range"]]
            print(f"  ~ {r['strategy']:30s} score={r['total_score']:+3d} fails={len(fails)}")
            for f in fails:
                print(f"       ✗ {f['scenario']:50s} got={f['n_clusters']} expect={f['expected']}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(__file__).parent / "clustering_sweep_results.json")

    with open(output_path, "w") as f:
        # Remove numpy arrays before serializing
        serializable = []
        for r in all_results:
            sr = {k: v for k, v in r.items()}
            for d in sr["details"]:
                d.pop("segments", None)
            serializable.append(sr)
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
