"""Experiment: DTW-based clustering vs current resampled-L2 clustering.

Benchmarks Dynamic Time Warping as a distance metric for motion segment
clustering compared to the current resampled trajectory L2 approach.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
    _resample_segment,
    _segment_distance,
)
from brace.core.pose import FEATURE_INDICES


# ---------------------------------------------------------------------------
# DTW distance function
# ---------------------------------------------------------------------------

def dtw_segment_distance(seg_a, seg_b):
    """Compute DTW distance between two segments (resampled to 30 frames)."""
    ra = _resample_segment(seg_a["features"], 30)
    rb = _resample_segment(seg_b["features"], 30)
    distance, _ = fastdtw(ra, rb, dist=euclidean)
    return distance / 30  # normalize by length


def dtw_cluster_segments(segments, distance_threshold=2.0):
    """Cluster segments using DTW distance + agglomerative clustering."""
    if not segments:
        return segments
    n = len(segments)
    if n == 1:
        for s in segments:
            s["cluster_dtw"] = 0
        return segments

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_segment_distance(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster_dtw"] = label_map[labels[i]]

    return segments


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

N_FEATURES = len(FEATURE_INDICES) * 2  # 28

# Each rep is a burst of motion + a rest pause so detect_motion_boundaries
# can find the velocity dips between repetitions.
REP_FRAMES = 30   # active motion per rep
REST_FRAMES = 8   # near-zero velocity pause between reps


def _build_reps(rep_list):
    """Concatenate reps with rest pauses. rep_list: list of (rep_frames, N_FEATURES) arrays."""
    parts = []
    for i, rep in enumerate(rep_list):
        parts.append(rep)
        if i < len(rep_list) - 1:
            # Rest: hold last frame value with tiny noise
            rest = np.tile(rep[-1:], (REST_FRAMES, 1))
            rest += np.random.default_rng(42 + i).normal(0, 0.001, rest.shape).astype(np.float32)
            parts.append(rest)
    return np.concatenate(parts, axis=0)


def make_repetitive_simple():
    """10 reps of identical sine wave. Expected: 1 cluster."""
    reps = []
    for r in range(10):
        t = np.arange(REP_FRAMES, dtype=np.float32)
        rep = np.zeros((REP_FRAMES, N_FEATURES), dtype=np.float32)
        for d in range(N_FEATURES):
            phase = d * 0.3
            rep[:, d] = np.sin(2 * np.pi * t / REP_FRAMES + phase)
        reps.append(rep)
    return _build_reps(reps)


def make_two_distinct():
    """Alternate between type-A (slow, small) and type-B (fast, large).
    5 of each, interleaved. Expected: 2 clusters."""
    reps = []
    for r in range(10):
        t = np.arange(REP_FRAMES, dtype=np.float32)
        rep = np.zeros((REP_FRAMES, N_FEATURES), dtype=np.float32)
        if r % 2 == 0:
            # Type A: period=REP_FRAMES, amplitude=1
            for d in range(N_FEATURES):
                rep[:, d] = np.sin(2 * np.pi * t / REP_FRAMES + d * 0.3)
        else:
            # Type B: period=REP_FRAMES/2, amplitude=2
            for d in range(N_FEATURES):
                rep[:, d] = 2.0 * np.sin(2 * np.pi * t / (REP_FRAMES / 2) + d * 0.3)
        reps.append(rep)
    return _build_reps(reps)


def make_repetitive_with_drift():
    """10 reps with amplitude drifting 1.0 -> 1.3. Expected: 1-2 clusters."""
    reps = []
    for r in range(10):
        amp = 1.0 + 0.3 * r / 9.0
        t = np.arange(REP_FRAMES, dtype=np.float32)
        rep = np.zeros((REP_FRAMES, N_FEATURES), dtype=np.float32)
        for d in range(N_FEATURES):
            phase = d * 0.3
            rep[:, d] = amp * np.sin(2 * np.pi * t / REP_FRAMES + phase)
        reps.append(rep)
    return _build_reps(reps)


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

DATASETS = [
    ("REPETITIVE_SIMPLE", make_repetitive_simple, "1 cluster"),
    ("TWO_DISTINCT", make_two_distinct, "2 clusters"),
    ("REPETITIVE_WITH_DRIFT", make_repetitive_with_drift, "1-2 clusters"),
]


def run_experiment():
    for name, gen_fn, expected in DATASETS:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {name}")
        print(f"  Expected: {expected}")
        print(f"{'=' * 70}")

        features = gen_fn()
        valid_indices = list(range(features.shape[0]))

        # Segment the motion
        segments = segment_motions(features, valid_indices, fps=24.0)
        print(f"  Segments detected: {len(segments)}")
        if len(segments) < 2:
            print("  WARNING: Fewer than 2 segments detected, clustering skipped.")
            continue

        # Print segment lengths
        lengths = [s["features"].shape[0] for s in segments]
        print(f"  Segment lengths: {lengths}")

        # Header
        print(f"\n  {'Threshold':>10}  {'L2 Clusters':>12}  {'DTW Clusters':>12}")
        print(f"  {'-' * 10}  {'-' * 12}  {'-' * 12}")

        for thresh in THRESHOLDS:
            # Current approach: resampled L2
            import copy
            segs_l2 = copy.deepcopy(segments)
            cluster_segments(segs_l2, distance_threshold=thresh)
            n_l2 = len(set(s["cluster"] for s in segs_l2))

            # DTW approach
            segs_dtw = copy.deepcopy(segments)
            dtw_cluster_segments(segs_dtw, distance_threshold=thresh)
            n_dtw = len(set(s["cluster_dtw"] for s in segs_dtw))

            print(f"  {thresh:>10.1f}  {n_l2:>12}  {n_dtw:>12}")

        # Detailed pairwise distances at default threshold for comparison
        print(f"\n  Pairwise distances (first 6 segments):")
        show_n = min(6, len(segments))
        print(f"  {'':>6}", end="")
        for j in range(show_n):
            print(f"  seg{j:>2}", end="")
        print()
        for i in range(show_n):
            print(f"  L2 {i:>1}:", end="")
            for j in range(show_n):
                if i == j:
                    print(f"  {'--':>5}", end="")
                else:
                    d = _segment_distance(segments[i], segments[j])
                    print(f"  {d:>5.2f}", end="")
            print()
        for i in range(show_n):
            print(f"  DTW{i:>1}:", end="")
            for j in range(show_n):
                if i == j:
                    print(f"  {'--':>5}", end="")
                else:
                    d = dtw_segment_distance(segments[i], segments[j])
                    print(f"  {d:>5.2f}", end="")
            print()


if __name__ == "__main__":
    run_experiment()
