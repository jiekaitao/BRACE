#!/usr/bin/env python3
"""Investigate temporal patterns in clustering and test temporal coherence post-processing.

Part A: Analyze temporal cluster sequences (uniform/sequential/alternating/mixed)
Part B: Test temporal adjacency distance bonus (alpha parameter)
Part C: Test aggressive adjacent merge (merge_fraction parameter)
Part D: Test hierarchical clustering at multiple thresholds (fine vs coarse)

Uses cached features from .feature_cache/ — no GPU required.
"""

import sys
import os
import json
import copy
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    _segment_distance,
    _merge_adjacent_clusters,
    _merge_small_clusters,
    _resample_segment,
)

CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"


def load_ground_truth():
    with open(GT_PATH) as f:
        gt = json.load(f)
    # Remove metadata entry
    gt.pop("_metadata", None)
    return gt


def load_cached_features(video_name):
    """Load features from cache. Returns (features, valid_indices, fps) or None."""
    cache_path = os.path.join(CACHE_DIR, f"{video_name}.feats.npz")
    if not os.path.exists(cache_path):
        return None
    d = np.load(cache_path)
    return d["features"], d["valid_indices"].tolist(), float(d["fps"])


def run_pipeline(features, valid_indices, fps, distance_threshold=2.0, min_segment_sec=2.0):
    """Run standard segmentation + clustering. Returns clustered segments."""
    segments = segment_motions(features, valid_indices, fps, min_segment_sec=min_segment_sec)
    if len(segments) < 2:
        for s in segments:
            s["cluster"] = 0
        return segments
    clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=distance_threshold, fps=fps)
    return clustered


def check_pass(video_name, clustered, gt):
    """Check if clustering result is within expected_clusters_range."""
    gt_info = gt.get(video_name, {})
    expected_range = gt_info.get("expected_clusters_range")
    if expected_range is None:
        return None  # no ground truth
    n_clusters = len(set(s["cluster"] for s in clustered))
    return expected_range[0] <= n_clusters <= expected_range[1]


def classify_temporal_pattern(cluster_sequence):
    """Classify the temporal pattern of a cluster sequence.

    Returns one of: "uniform", "sequential", "alternating", "mixed"
    """
    if not cluster_sequence:
        return "empty"

    unique = set(cluster_sequence)
    n_unique = len(unique)

    if n_unique == 1:
        return "uniform"

    # Check sequential: monotonic changes (each cluster appears contiguously)
    seen = []
    for c in cluster_sequence:
        if not seen or seen[-1] != c:
            seen.append(c)
    # Sequential if each cluster appears only once in the run-length encoding
    if len(seen) == n_unique:
        return "sequential"

    # Check alternating: A-B-A-B or A-B-C-A-B-C periodic patterns
    # Count transitions back to a previously-seen cluster
    revisits = 0
    seen_set = set()
    prev = None
    for c in cluster_sequence:
        if c != prev:
            if c in seen_set:
                revisits += 1
            seen_set.add(c)
            prev = c

    # Alternating if there are regular revisits
    if revisits >= 2:
        return "alternating"
    if revisits == 1 and n_unique == 2:
        return "alternating"

    return "mixed"


# ============================================================================
# Part B: Temporal adjacency clustering (modified distance matrix)
# ============================================================================

def cluster_with_temporal_bonus(segments, distance_threshold=2.0, alpha=0.0, fps=30.0):
    """Cluster segments with temporal adjacency bonus on distance.

    adjusted_dist = dist * (1 + alpha * temporal_gap)
    where temporal_gap is normalized time distance between segment midpoints.
    """
    if not segments or len(segments) < 2:
        for s in segments:
            s["cluster"] = 0
        return segments

    n = len(segments)

    # Compute midpoint times for each segment (in seconds)
    midpoints = []
    for seg in segments:
        mid_frame = (seg["start_frame"] + seg["end_frame"]) / 2.0
        midpoints.append(mid_frame / max(fps, 1.0))

    total_duration = max(midpoints[-1] - midpoints[0], 1.0) if len(midpoints) > 1 else 1.0

    # Build pairwise distance matrix with temporal adjustment
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            base_dist = _segment_distance(segments[i], segments[j])
            temporal_gap = abs(midpoints[i] - midpoints[j]) / total_duration
            adjusted = base_dist * (1.0 + alpha * temporal_gap)
            dist_matrix[i, j] = adjusted
            dist_matrix[j, i] = adjusted

    # Cluster using average linkage
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    segments = _merge_adjacent_clusters(segments)
    segments = _merge_small_clusters(segments, fps=fps)
    return segments


# ============================================================================
# Part C: Aggressive adjacent merge (post-clustering distance-based merge)
# ============================================================================

def cluster_with_aggressive_merge(segments, distance_threshold=2.0, merge_fraction=0.5, fps=30.0):
    """Standard clustering followed by aggressive adjacent merge.

    After clustering, for each pair of adjacent segments in different clusters:
    if distance(seg_i, seg_j) < merge_fraction * distance_threshold: merge them.
    """
    if not segments or len(segments) < 2:
        for s in segments:
            s["cluster"] = 0
        return segments

    n = len(segments)

    # Standard clustering first
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Standard adjacent merge
    segments = _merge_adjacent_clusters(segments)

    # Aggressive merge: merge adjacent segments if they're close enough
    merge_dist = merge_fraction * distance_threshold
    changed = True
    while changed:
        changed = False
        new_segs = [segments[0]]
        for seg in segments[1:]:
            prev = new_segs[-1]
            if prev["cluster"] != seg["cluster"]:
                d = _segment_distance(prev, seg)
                if d < merge_dist:
                    # Merge into the larger segment's cluster
                    combined = np.vstack([prev["features"], seg["features"]])
                    prev["end_valid"] = seg.get("end_valid", seg["features"].shape[0])
                    prev["end_frame"] = seg["end_frame"]
                    prev["features"] = combined
                    prev["mean_feature"] = combined.mean(axis=0)
                    changed = True
                    continue
            elif prev["cluster"] == seg["cluster"]:
                # Same cluster — merge as usual
                combined = np.vstack([prev["features"], seg["features"]])
                prev["end_valid"] = seg.get("end_valid", seg["features"].shape[0])
                prev["end_frame"] = seg["end_frame"]
                prev["features"] = combined
                prev["mean_feature"] = combined.mean(axis=0)
                changed = True
                continue
            new_segs.append(seg)
        segments = new_segs

    # Re-merge adjacents and absorb small
    segments = _merge_adjacent_clusters(segments)
    segments = _merge_small_clusters(segments, fps=fps)

    # Relabel to 0-indexed contiguous
    unique = sorted(set(s["cluster"] for s in segments))
    remap = {old: new for new, old in enumerate(unique)}
    for seg in segments:
        seg["cluster"] = remap[seg["cluster"]]

    return segments


# ============================================================================
# Main
# ============================================================================

def main():
    gt = load_ground_truth()

    # Load all cached features
    videos = {}
    for fname in sorted(gt.keys()):
        data = load_cached_features(fname)
        if data is None:
            print(f"  SKIP {fname}: no cached features")
            continue
        features, valid_indices, fps = data
        if len(features) < 10:
            print(f"  SKIP {fname}: only {len(features)} features")
            continue
        videos[fname] = (features, valid_indices, fps)

    print(f"Loaded cached features for {len(videos)} videos\n")

    # ========================================================================
    # Part A: Analyze temporal cluster sequences
    # ========================================================================
    print("=" * 70)
    print("PART A: Temporal Cluster Sequence Analysis (threshold=2.0)")
    print("=" * 70)

    part_a_results = {}
    for fname, (features, valid_indices, fps) in sorted(videos.items()):
        clustered = run_pipeline(features, valid_indices, fps, distance_threshold=2.0)
        n_clusters = len(set(s["cluster"] for s in clustered))

        # Extract temporal cluster sequence
        cluster_seq = [s["cluster"] for s in clustered]
        pattern = classify_temporal_pattern(cluster_seq)
        is_pass = check_pass(fname, clustered, gt)

        # Duration per segment
        seg_details = []
        for s in clustered:
            dur = (s["end_frame"] - s["start_frame"]) / max(fps, 1)
            seg_details.append(f"C{s['cluster']}({dur:.1f}s)")

        part_a_results[fname] = {
            "n_clusters": n_clusters,
            "n_segments": len(clustered),
            "pattern": pattern,
            "sequence": cluster_seq,
            "pass": is_pass,
            "segment_details": seg_details,
        }

        status = "PASS" if is_pass else ("FAIL" if is_pass is False else "N/A")
        gt_range = gt.get(fname, {}).get("expected_clusters_range", "?")
        print(f"  {status:4s} {fname:35s} | {n_clusters} clusters | {pattern:12s} | seq={cluster_seq} | {', '.join(seg_details)}")

    # Pattern distribution
    patterns = Counter(r["pattern"] for r in part_a_results.values())
    print(f"\n  Pattern distribution: {dict(patterns)}")
    pass_count = sum(1 for r in part_a_results.values() if r["pass"] is True)
    total_count = sum(1 for r in part_a_results.values() if r["pass"] is not None)
    print(f"  Pass rate: {pass_count}/{total_count}")

    # ========================================================================
    # Part B: Temporal adjacency distance bonus
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("PART B: Temporal Adjacency Distance Bonus")
    print("=" * 70)
    print("  Idea: adjusted_dist = dist * (1 + alpha * normalized_temporal_gap)")
    print("  This penalizes merging temporally distant segments.\n")

    alphas = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    part_b_results = {a: {} for a in alphas}

    for alpha in alphas:
        pass_count = 0
        total_count = 0
        for fname, (features, valid_indices, fps) in sorted(videos.items()):
            segments = segment_motions(features, valid_indices, fps, min_segment_sec=2.0)
            if len(segments) < 2:
                for s in segments:
                    s["cluster"] = 0
                clustered = segments
            else:
                clustered = cluster_with_temporal_bonus(
                    copy.deepcopy(segments), distance_threshold=2.0, alpha=alpha, fps=fps
                )

            is_pass = check_pass(fname, clustered, gt)
            n_clusters = len(set(s["cluster"] for s in clustered))
            part_b_results[alpha][fname] = {"n_clusters": n_clusters, "pass": is_pass}

            if is_pass is not None:
                total_count += 1
                if is_pass:
                    pass_count += 1

        print(f"  alpha={alpha:.1f}: {pass_count}/{total_count} pass", end="")
        # Show per-video cluster counts
        counts = {f.replace('.mp4', ''): r["n_clusters"] for f, r in sorted(part_b_results[alpha].items())}
        print(f"  | clusters: {counts}")

    # ========================================================================
    # Part C: Aggressive adjacent merge
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("PART C: Aggressive Adjacent Merge (post-clustering)")
    print("=" * 70)
    print("  Idea: merge adjacent different-cluster segments if distance < fraction * threshold\n")

    fractions = [0.3, 0.5, 0.7, 0.9]
    part_c_results = {f: {} for f in fractions}

    for frac in fractions:
        pass_count = 0
        total_count = 0
        for fname, (features, valid_indices, fps) in sorted(videos.items()):
            segments = segment_motions(features, valid_indices, fps, min_segment_sec=2.0)
            if len(segments) < 2:
                for s in segments:
                    s["cluster"] = 0
                clustered = segments
            else:
                clustered = cluster_with_aggressive_merge(
                    copy.deepcopy(segments), distance_threshold=2.0, merge_fraction=frac, fps=fps
                )

            is_pass = check_pass(fname, clustered, gt)
            n_clusters = len(set(s["cluster"] for s in clustered))
            part_c_results[frac][fname] = {"n_clusters": n_clusters, "pass": is_pass}

            if is_pass is not None:
                total_count += 1
                if is_pass:
                    pass_count += 1

        print(f"  merge_fraction={frac:.1f}: {pass_count}/{total_count} pass", end="")
        counts = {f.replace('.mp4', ''): r["n_clusters"] for f, r in sorted(part_c_results[frac].items())}
        print(f"  | clusters: {counts}")

    # ========================================================================
    # Part D: Hierarchical clustering at multiple thresholds
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("PART D: Hierarchical Clustering — Fine (t=2.0) vs Coarse (t=4.0)")
    print("=" * 70)

    # Identify failing videos at t=2.0
    failing_videos = []
    for fname, res in part_a_results.items():
        if res["pass"] is False:
            failing_videos.append(fname)

    thresholds = [1.5, 2.0, 3.0, 4.0, 6.0]

    print(f"\n  All videos across thresholds:")
    print(f"  {'Video':35s}", end="")
    for t in thresholds:
        print(f" | t={t:.1f}", end="")
    print(f" | expected")
    print(f"  {'-'*35}", end="")
    for t in thresholds:
        print(f"-+------", end="")
    print(f"-+--------")

    part_d_results = {}
    for fname, (features, valid_indices, fps) in sorted(videos.items()):
        part_d_results[fname] = {}
        gt_range = gt.get(fname, {}).get("expected_clusters_range", "?")
        is_failing = fname in failing_videos
        marker = " ***" if is_failing else ""
        print(f"  {fname:35s}", end="")
        for t in thresholds:
            clustered = run_pipeline(features, valid_indices, fps, distance_threshold=t)
            n_clusters = len(set(s["cluster"] for s in clustered))
            seq = [s["cluster"] for s in clustered]
            part_d_results[fname][t] = {"n_clusters": n_clusters, "sequence": seq}
            is_pass = check_pass(fname, clustered, gt)
            flag = "*" if is_pass else " " if is_pass is False else "?"
            print(f" | {flag}{n_clusters:3d} ", end="")
        print(f" | {gt_range}{marker}")

    # Detailed view for failing videos
    if failing_videos:
        print(f"\n  Detailed hierarchy for failing videos:")
        for fname in sorted(failing_videos):
            features, valid_indices, fps = videos[fname]
            gt_info = gt.get(fname, {})
            print(f"\n  --- {fname} (expected: {gt_info.get('expected_clusters_range')}, activities: {gt_info.get('distinct_activities')}) ---")

            for t in [2.0, 3.0, 4.0]:
                clustered = run_pipeline(features, valid_indices, fps, distance_threshold=t)
                n_clusters = len(set(s["cluster"] for s in clustered))
                seg_details = []
                for s in clustered:
                    dur = (s["end_frame"] - s["start_frame"]) / max(fps, 1)
                    seg_details.append(f"C{s['cluster']}({dur:.1f}s)")
                pattern = classify_temporal_pattern([s["cluster"] for s in clustered])
                print(f"    t={t:.1f}: {n_clusters} clusters, {pattern:12s}, segments: {', '.join(seg_details)}")

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Part A — Temporal pattern types at t=2.0:")
    for pattern in ["uniform", "sequential", "alternating", "mixed"]:
        vids = [f.replace('.mp4', '') for f, r in part_a_results.items() if r["pattern"] == pattern]
        if vids:
            print(f"    {pattern:12s}: {', '.join(vids)}")

    print(f"\n  Part B — Temporal adjacency bonus pass rates:")
    for alpha in alphas:
        pc = sum(1 for r in part_b_results[alpha].values() if r.get("pass") is True)
        tc = sum(1 for r in part_b_results[alpha].values() if r.get("pass") is not None)
        print(f"    alpha={alpha:.1f}: {pc}/{tc}")

    print(f"\n  Part C — Aggressive adjacent merge pass rates:")
    for frac in fractions:
        pc = sum(1 for r in part_c_results[frac].values() if r.get("pass") is True)
        tc = sum(1 for r in part_c_results[frac].values() if r.get("pass") is not None)
        print(f"    merge_fraction={frac:.1f}: {pc}/{tc}")

    print(f"\n  Part D — Best threshold per video:")
    for fname in sorted(part_d_results.keys()):
        gt_range = gt.get(fname, {}).get("expected_clusters_range")
        if gt_range is None:
            continue
        best_t = None
        for t in thresholds:
            nc = part_d_results[fname][t]["n_clusters"]
            if gt_range[0] <= nc <= gt_range[1]:
                best_t = t
                break
        if best_t:
            print(f"    {fname:35s}: first pass at t={best_t:.1f}")
        else:
            counts = {t: part_d_results[fname][t]["n_clusters"] for t in thresholds}
            print(f"    {fname:35s}: NO threshold works, counts={counts}")


if __name__ == "__main__":
    main()
