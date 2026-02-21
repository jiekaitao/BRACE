#!/usr/bin/env python3
"""Test SOTA temporal action segmentation approaches on 14 demo videos.

Methods tested:
1. BASELINE: Current velocity-based segmentation + agglomerative clustering
2. RUPTURES-RBF: Kernel change point detection (RBF kernel + PELT)
3. RUPTURES-L2: L2 cost change point detection (PELT)
4. RUPTURES-CLINEAR: Clinear cost change point detection (dynamic programming)
5. OT-SEGMENT: Optimal Transport pseudo-label generation (ASOT-inspired)
6. SPECTRAL-CPD: Self-similarity matrix + novelty detection (Foote method)

All methods use pre-cached 28D SRP-normalized pose features from
experiments/.feature_cache/*.npz

Each method produces segment boundaries and cluster assignments, which are
compared against ground truth expected_clusters from video_ground_truth.json.

Research findings summary (see bottom of script for detailed analysis).
"""

import sys
import os
import json
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import ruptures as rpt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans, SpectralClustering
import ot

from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    detect_motion_boundaries,
    _segment_distance,
    _resample_segment,
)

CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
OUTPUT_PATH = "/mnt/Data/GitHub/BRACE/experiments/action_segmentation_results.json"


def load_ground_truth():
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt.pop("_metadata", None)
    return gt


def load_cached_features(video_name):
    path = os.path.join(CACHE_DIR, f"{video_name}.feats.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "features": data["features"],
        "valid_indices": data["valid_indices"].tolist(),
        "fps": float(data["fps"]),
    }


# ---------------------------------------------------------------------------
# Method 1: BASELINE (current system)
# ---------------------------------------------------------------------------
def method_baseline(features, valid_indices, fps, n_expected=None):
    """Current velocity-based segmentation + agglomerative clustering."""
    segments = segment_motions(features, valid_indices, fps=fps, min_segment_sec=2.0)
    segments = cluster_segments(segments, distance_threshold=2.0, fps=fps)
    n_clusters = len(set(s["cluster"] for s in segments))
    boundaries = [s["start_valid"] for s in segments]
    labels = [s["cluster"] for s in segments]
    return {
        "n_clusters": n_clusters,
        "n_segments": len(segments),
        "boundaries": boundaries,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Method 2: RUPTURES - RBF kernel + PELT
# ---------------------------------------------------------------------------
def method_ruptures_rbf(features, valid_indices, fps, n_expected=None):
    """Kernel change point detection with RBF kernel and PELT algorithm.

    The RBF kernel maps features into a high-dimensional RKHS, enabling
    detection of distributional changes in the pose feature sequence.
    PELT provides exact segmentation with a penalty parameter.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    # Try multiple penalty values and pick the one closest to expected
    best_result = None
    best_diff = float("inf")

    for pen in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        algo = rpt.KernelCPD(kernel="rbf", min_size=max(int(fps * 1.5), 5)).fit(features)
        try:
            bkps = algo.predict(pen=pen)
        except Exception:
            continue

        # bkps includes the last index (n), filter it out for boundaries
        boundaries = [0] + [b for b in bkps if b < n]

        if len(boundaries) < 2:
            seg_count = 1
        else:
            seg_count = len(boundaries)

        # Cluster the segments using agglomerative clustering
        seg_features_list = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n
            if end - start < 3:
                continue
            seg_feat = features[start:end]
            seg_features_list.append({
                "start_valid": start,
                "end_valid": end,
                "features": seg_feat,
                "mean_feature": seg_feat.mean(axis=0),
            })

        if not seg_features_list:
            continue

        clustered = cluster_segments(seg_features_list, distance_threshold=2.0, fps=fps)
        n_clusters = len(set(s["cluster"] for s in clustered))

        if n_expected is not None:
            diff = abs(n_clusters - n_expected)
        else:
            diff = abs(n_clusters - 2)  # default: prefer moderate segmentation

        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff
            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(clustered),
                "boundaries": [s["start_valid"] for s in clustered],
                "labels": [s["cluster"] for s in clustered],
                "penalty": pen,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


# ---------------------------------------------------------------------------
# Method 3: RUPTURES - L2 cost + PELT
# ---------------------------------------------------------------------------
def method_ruptures_l2(features, valid_indices, fps, n_expected=None):
    """L2 cost change point detection with PELT.

    Detects shifts in mean of the pose feature signal.
    Simpler than RBF but very fast and often effective.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    best_result = None
    best_diff = float("inf")

    for pen in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        algo = rpt.Pelt(model="l2", min_size=max(int(fps * 1.5), 5)).fit(features)
        try:
            bkps = algo.predict(pen=pen)
        except Exception:
            continue

        boundaries = [0] + [b for b in bkps if b < n]

        seg_features_list = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n
            if end - start < 3:
                continue
            seg_feat = features[start:end]
            seg_features_list.append({
                "start_valid": start,
                "end_valid": end,
                "features": seg_feat,
                "mean_feature": seg_feat.mean(axis=0),
            })

        if not seg_features_list:
            continue

        clustered = cluster_segments(seg_features_list, distance_threshold=2.0, fps=fps)
        n_clusters = len(set(s["cluster"] for s in clustered))

        if n_expected is not None:
            diff = abs(n_clusters - n_expected)
        else:
            diff = abs(n_clusters - 2)

        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff
            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(clustered),
                "boundaries": [s["start_valid"] for s in clustered],
                "labels": [s["cluster"] for s in clustered],
                "penalty": pen,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


# ---------------------------------------------------------------------------
# Method 4: RUPTURES - Dynp with known K
# ---------------------------------------------------------------------------
def method_ruptures_dynp(features, valid_indices, fps, n_expected=None):
    """Dynamic programming segmentation with known number of change points.

    When we know the expected number of clusters, we can use DP to find
    the optimal K-1 breakpoints. Uses RBF kernel cost.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    # Use expected clusters to set number of breakpoints
    if n_expected is None or n_expected < 1:
        n_expected = 2
    n_bkps = max(n_expected - 1, 0)

    if n_bkps == 0:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    # Try with more breakpoints too (activities may repeat)
    best_result = None
    best_diff = float("inf")

    for n_try in range(max(1, n_bkps), min(n_bkps + 4, n // 5)):
        algo = rpt.KernelCPD(kernel="rbf", min_size=max(int(fps * 1.0), 5)).fit(features)
        try:
            bkps = algo.predict(n_bkps=n_try)
        except Exception:
            continue

        boundaries = [0] + [b for b in bkps if b < n]

        seg_features_list = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n
            if end - start < 3:
                continue
            seg_feat = features[start:end]
            seg_features_list.append({
                "start_valid": start,
                "end_valid": end,
                "features": seg_feat,
                "mean_feature": seg_feat.mean(axis=0),
            })

        if not seg_features_list:
            continue

        clustered = cluster_segments(seg_features_list, distance_threshold=2.0, fps=fps)
        n_clusters = len(set(s["cluster"] for s in clustered))

        diff = abs(n_clusters - n_expected) if n_expected else 0

        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff
            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(clustered),
                "boundaries": [s["start_valid"] for s in clustered],
                "labels": [s["cluster"] for s in clustered],
                "n_bkps_tried": n_try,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


# ---------------------------------------------------------------------------
# Method 5: Optimal Transport pseudo-labeling (ASOT-inspired)
# ---------------------------------------------------------------------------
def method_ot_segment(features, valid_indices, fps, n_expected=None):
    """Optimal Transport-based action segmentation (ASOT-inspired).

    Steps:
    1. Build frame-to-frame affinity matrix using cosine similarity
    2. Compute soft cluster assignments via Sinkhorn OT
    3. Add temporal consistency via a temporal prior
    4. Hard assignment via argmax

    This is a simplified version of ASOT (CVPR 2024) that works directly
    on pose features without the full training pipeline.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    K = n_expected if n_expected and n_expected > 0 else 3

    best_result = None
    best_diff = float("inf")

    for K_try in range(max(1, K - 1), K + 3):
        # Step 1: K-Means to get initial cluster centers
        if K_try >= n:
            continue
        kmeans = KMeans(n_clusters=K_try, n_init=10, random_state=42).fit(features)
        centers = kmeans.cluster_centers_  # (K, D)

        # Step 2: Compute cost matrix (frame-to-cluster distances)
        cost = cdist(features, centers, metric="euclidean")  # (N, K)
        # Normalize to [0, 1]
        cost = cost / (cost.max() + 1e-8)

        # Step 3: Add temporal consistency prior
        # Penalize rapid cluster switching: frames close in time should
        # prefer the same cluster
        temporal_reg = np.zeros_like(cost)
        window = max(int(fps * 0.5), 3)
        for i in range(n):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            # Average cost in temporal neighborhood
            temporal_reg[i] = cost[start:end].mean(axis=0)

        # Blend: original cost + temporal smoothing
        blended_cost = 0.6 * cost + 0.4 * temporal_reg

        # Step 4: Sinkhorn OT to get soft assignments
        # Uniform marginals
        a = np.ones(n) / n
        b = np.ones(K_try) / K_try

        try:
            transport_plan = ot.sinkhorn(a, b, blended_cost, reg=0.05, numItermax=200)
        except Exception:
            continue

        # Step 5: Hard assignment
        frame_labels = transport_plan.argmax(axis=1)

        # Step 6: Temporal smoothing - remove brief cluster flickers
        min_frames = max(int(fps * 1.0), 5)
        smoothed_labels = _temporal_smooth_labels(frame_labels, min_frames)

        # Count clusters
        n_clusters = len(set(smoothed_labels))

        diff = abs(n_clusters - (n_expected or 2))
        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff

            # Extract boundaries
            boundaries = [0]
            for i in range(1, len(smoothed_labels)):
                if smoothed_labels[i] != smoothed_labels[i - 1]:
                    boundaries.append(i)

            # Build segments
            seg_labels = []
            for i in range(len(boundaries)):
                start = boundaries[i]
                end = boundaries[i + 1] if i + 1 < len(boundaries) else n
                seg_labels.append(smoothed_labels[start])

            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(boundaries),
                "boundaries": boundaries,
                "labels": seg_labels,
                "K_tried": K_try,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


def _temporal_smooth_labels(labels, min_duration):
    """Remove brief cluster assignments shorter than min_duration frames."""
    labels = list(labels)
    n = len(labels)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and labels[j] == labels[i]:
                j += 1
            duration = j - i
            if duration < min_duration and i > 0:
                # Merge this short segment into the previous one
                for k in range(i, j):
                    labels[k] = labels[i - 1]
                changed = True
            i = j
    return labels


# ---------------------------------------------------------------------------
# Method 6: Self-Similarity Matrix + Novelty Detection (Foote method)
# ---------------------------------------------------------------------------
def method_ssm_novelty(features, valid_indices, fps, n_expected=None):
    """Self-Similarity Matrix novelty-based segmentation (Foote 2000).

    Classic audio/music segmentation method adapted for pose features:
    1. Compute self-similarity matrix (cosine similarity)
    2. Apply checkerboard kernel for novelty detection
    3. Find peaks in novelty curve as segment boundaries
    4. Cluster segments with agglomerative clustering

    This method is widely used in music structure analysis and has been
    adapted for action segmentation in several works.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    # Step 1: Self-similarity matrix (cosine similarity)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feat_norm = features / norms
    ssm = feat_norm @ feat_norm.T  # (N, N), values in [-1, 1]

    # Step 2: Checkerboard kernel novelty detection
    best_result = None
    best_diff = float("inf")

    for kernel_sec in [1.0, 1.5, 2.0, 3.0, 4.0]:
        kernel_size = max(int(fps * kernel_sec), 3)
        if kernel_size * 2 >= n:
            continue

        novelty = _checkerboard_novelty(ssm, kernel_size)

        # Step 3: Find peaks in novelty curve
        min_frames = max(int(fps * 1.5), 5)
        # Adaptive threshold based on novelty distribution
        if np.max(novelty) < 1e-8:
            continue
        threshold = np.median(novelty[novelty > 0]) * 0.5
        peaks, _ = find_peaks(novelty, distance=min_frames, height=threshold)

        boundaries = [0] + sorted(peaks.tolist())

        # Build segments
        seg_features_list = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n
            if end - start < 3:
                continue
            seg_feat = features[start:end]
            seg_features_list.append({
                "start_valid": start,
                "end_valid": end,
                "features": seg_feat,
                "mean_feature": seg_feat.mean(axis=0),
            })

        if not seg_features_list:
            continue

        clustered = cluster_segments(seg_features_list, distance_threshold=2.0, fps=fps)
        n_clusters = len(set(s["cluster"] for s in clustered))

        diff = abs(n_clusters - (n_expected or 2))
        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff
            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(clustered),
                "boundaries": [s["start_valid"] for s in clustered],
                "labels": [s["cluster"] for s in clustered],
                "kernel_sec": kernel_sec,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


def _checkerboard_novelty(ssm, kernel_size):
    """Compute novelty curve using checkerboard kernel on SSM (Foote 2000).

    The checkerboard kernel detects block-diagonal transitions in the SSM,
    which correspond to activity boundaries.
    """
    n = ssm.shape[0]
    novelty = np.zeros(n)
    k = kernel_size

    # Build checkerboard kernel: +1 in diagonal blocks, -1 in off-diagonal
    checker = np.ones((2 * k, 2 * k))
    checker[:k, k:] = -1
    checker[k:, :k] = -1

    for i in range(k, n - k):
        patch = ssm[i - k:i + k, i - k:i + k]
        if patch.shape == checker.shape:
            novelty[i] = np.sum(patch * checker)

    # Normalize
    max_val = np.max(np.abs(novelty))
    if max_val > 0:
        novelty = novelty / max_val

    return np.maximum(novelty, 0)  # Only positive novelty (transitions)


# ---------------------------------------------------------------------------
# Method 7: Spectral Clustering on temporal features
# ---------------------------------------------------------------------------
def method_spectral_temporal(features, valid_indices, fps, n_expected=None):
    """Spectral clustering with temporal affinity.

    Build an affinity matrix that combines feature similarity and temporal
    proximity, then use spectral clustering for frame-level labeling.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    K = n_expected if n_expected and n_expected > 0 else 2

    best_result = None
    best_diff = float("inf")

    for K_try in range(max(1, K - 1), K + 3):
        if K_try >= n:
            continue

        # Feature affinity (RBF kernel)
        feat_dists = cdist(features, features, metric="sqeuclidean")
        sigma_feat = np.median(feat_dists[feat_dists > 0]) if np.any(feat_dists > 0) else 1.0
        feat_affinity = np.exp(-feat_dists / (2 * sigma_feat))

        # Temporal affinity (Gaussian in time)
        time_idx = np.arange(n).reshape(-1, 1)
        time_dists = cdist(time_idx, time_idx, metric="sqeuclidean")
        sigma_time = (fps * 2.0) ** 2  # ~2 sec temporal scale
        time_affinity = np.exp(-time_dists / (2 * sigma_time))

        # Combined affinity
        affinity = feat_affinity * time_affinity

        try:
            sc = SpectralClustering(
                n_clusters=K_try,
                affinity="precomputed",
                random_state=42,
                n_init=10,
            ).fit(affinity)
            frame_labels = sc.labels_
        except Exception:
            continue

        # Temporal smoothing
        min_frames = max(int(fps * 1.0), 5)
        smoothed = _temporal_smooth_labels(frame_labels.tolist(), min_frames)

        n_clusters = len(set(smoothed))

        diff = abs(n_clusters - (n_expected or 2))
        if diff < best_diff or (diff == best_diff and best_result is None):
            best_diff = diff

            boundaries = [0]
            for i in range(1, len(smoothed)):
                if smoothed[i] != smoothed[i - 1]:
                    boundaries.append(i)

            seg_labels = []
            for i in range(len(boundaries)):
                start = boundaries[i]
                seg_labels.append(smoothed[start])

            best_result = {
                "n_clusters": n_clusters,
                "n_segments": len(boundaries),
                "boundaries": boundaries,
                "labels": seg_labels,
                "K_tried": K_try,
            }

    if best_result is None:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}
    return best_result


# ---------------------------------------------------------------------------
# Method 8: Fixed-penalty ruptures (no oracle K)
# ---------------------------------------------------------------------------
def method_ruptures_rbf_fixed(features, valid_indices, fps, n_expected=None):
    """RBF kernel PELT with a single fixed penalty (no oracle tuning).

    This tests whether a single penalty value works well across all videos,
    which is what a real-time system would need.
    """
    n = features.shape[0]
    if n < 10:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    pen = 5.0  # Fixed penalty
    algo = rpt.KernelCPD(kernel="rbf", min_size=max(int(fps * 1.5), 5)).fit(features)
    try:
        bkps = algo.predict(pen=pen)
    except Exception:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    boundaries = [0] + [b for b in bkps if b < n]

    seg_features_list = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        if end - start < 3:
            continue
        seg_feat = features[start:end]
        seg_features_list.append({
            "start_valid": start,
            "end_valid": end,
            "features": seg_feat,
            "mean_feature": seg_feat.mean(axis=0),
        })

    if not seg_features_list:
        return {"n_clusters": 1, "n_segments": 1, "boundaries": [0], "labels": [0]}

    clustered = cluster_segments(seg_features_list, distance_threshold=2.0, fps=fps)
    n_clusters = len(set(s["cluster"] for s in clustered))

    return {
        "n_clusters": n_clusters,
        "n_segments": len(clustered),
        "boundaries": [s["start_valid"] for s in clustered],
        "labels": [s["cluster"] for s in clustered],
        "penalty": pen,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_method(method_func, method_name, gt, use_oracle_k=False):
    """Run a method on all videos and compare to ground truth."""
    results = {}
    total_correct = 0
    total_in_range = 0
    total_videos = 0
    total_time = 0.0

    for video_name, info in sorted(gt.items()):
        data = load_cached_features(video_name)
        if data is None:
            continue

        features = data["features"]
        valid_indices = data["valid_indices"]
        fps_val = data["fps"]
        expected = info["expected_clusters"]
        expected_range = info.get("expected_clusters_range", [expected, expected])

        n_expected = expected if use_oracle_k else None

        t0 = time.time()
        result = method_func(features, valid_indices, fps_val, n_expected=n_expected)
        elapsed = time.time() - t0
        total_time += elapsed

        n_clusters = result["n_clusters"]
        correct = n_clusters == expected
        in_range = expected_range[0] <= n_clusters <= expected_range[1]

        total_videos += 1
        if correct:
            total_correct += 1
        if in_range:
            total_in_range += 1

        results[video_name] = {
            "expected": expected,
            "expected_range": expected_range,
            "got": n_clusters,
            "n_segments": result["n_segments"],
            "correct": correct,
            "in_range": in_range,
            "time_sec": round(elapsed, 4),
        }
        # Copy extra info
        for k in ["penalty", "kernel_sec", "K_tried", "n_bkps_tried"]:
            if k in result:
                results[video_name][k] = result[k]

    summary = {
        "method": method_name,
        "exact_match": f"{total_correct}/{total_videos}",
        "in_range": f"{total_in_range}/{total_videos}",
        "exact_pct": round(100 * total_correct / max(total_videos, 1), 1),
        "range_pct": round(100 * total_in_range / max(total_videos, 1), 1),
        "total_time_sec": round(total_time, 3),
        "per_video": results,
    }
    return summary


def print_summary(summary):
    """Print a concise summary table."""
    method = summary["method"]
    exact = summary["exact_match"]
    in_range = summary["in_range"]
    t = summary["total_time_sec"]

    print(f"\n{'=' * 70}")
    print(f"  {method}")
    print(f"  Exact match: {exact} ({summary['exact_pct']}%)")
    print(f"  In range:    {in_range} ({summary['range_pct']}%)")
    print(f"  Total time:  {t:.3f}s")
    print(f"{'=' * 70}")

    for video, info in sorted(summary["per_video"].items()):
        status = "PASS" if info["in_range"] else "FAIL"
        mark = "x" if info["correct"] else (" " if info["in_range"] else "!")
        extras = ""
        if "penalty" in info:
            extras += f" pen={info['penalty']}"
        if "kernel_sec" in info:
            extras += f" k={info['kernel_sec']}s"
        if "K_tried" in info:
            extras += f" K={info['K_tried']}"
        print(
            f"  [{mark}] {video:<35s} expected={info['expected']}  "
            f"got={info['got']}  segs={info['n_segments']}  "
            f"[{status}]{extras}"
        )


def main():
    gt = load_ground_truth()
    print(f"Loaded ground truth for {len(gt)} videos")
    print(f"Feature cache: {CACHE_DIR}")
    print()

    all_results = {}

    # Method 1: Baseline (current system)
    print("Running: BASELINE (velocity + agglomerative)...")
    s = evaluate_method(method_baseline, "BASELINE (velocity + agglomerative)", gt)
    print_summary(s)
    all_results["baseline"] = s

    # Method 2: Ruptures RBF (oracle K for penalty tuning)
    print("\nRunning: RUPTURES-RBF (oracle penalty tuning)...")
    s = evaluate_method(method_ruptures_rbf, "RUPTURES-RBF (oracle penalty)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["ruptures_rbf_oracle"] = s

    # Method 3: Ruptures RBF (fixed penalty, no oracle)
    print("\nRunning: RUPTURES-RBF-FIXED (pen=5.0, no oracle)...")
    s = evaluate_method(method_ruptures_rbf_fixed, "RUPTURES-RBF-FIXED (pen=5.0)", gt)
    print_summary(s)
    all_results["ruptures_rbf_fixed"] = s

    # Method 4: Ruptures L2 (oracle penalty)
    print("\nRunning: RUPTURES-L2 (oracle penalty)...")
    s = evaluate_method(method_ruptures_l2, "RUPTURES-L2 (oracle penalty)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["ruptures_l2_oracle"] = s

    # Method 5: Ruptures Dynp (oracle K)
    print("\nRunning: RUPTURES-DYNP (oracle K breakpoints)...")
    s = evaluate_method(method_ruptures_dynp, "RUPTURES-DYNP (oracle K)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["ruptures_dynp_oracle"] = s

    # Method 6: Optimal Transport (oracle K)
    print("\nRunning: OT-SEGMENT (Sinkhorn OT, oracle K)...")
    s = evaluate_method(method_ot_segment, "OT-SEGMENT (Sinkhorn, oracle K)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["ot_segment_oracle"] = s

    # Method 7: SSM Novelty (Foote)
    print("\nRunning: SSM-NOVELTY (Foote checkerboard, oracle kernel)...")
    s = evaluate_method(method_ssm_novelty, "SSM-NOVELTY (Foote, oracle kernel)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["ssm_novelty_oracle"] = s

    # Method 8: Spectral clustering (oracle K)
    print("\nRunning: SPECTRAL-TEMPORAL (oracle K)...")
    s = evaluate_method(method_spectral_temporal, "SPECTRAL-TEMPORAL (oracle K)", gt, use_oracle_k=True)
    print_summary(s)
    all_results["spectral_temporal_oracle"] = s

    # Final comparison table
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<45s} {'Exact':>8s} {'InRange':>8s} {'Time':>8s}")
    print(f"  {'-' * 45} {'-' * 8} {'-' * 8} {'-' * 8}")
    for key, s in all_results.items():
        print(
            f"  {s['method']:<45s} {s['exact_match']:>8s} {s['in_range']:>8s} "
            f"{s['total_time_sec']:>7.2f}s"
        )

    # Save results
    # Remove non-serializable data
    for key, s in all_results.items():
        for video, info in s["per_video"].items():
            info.pop("boundaries", None)
            info.pop("labels", None)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
