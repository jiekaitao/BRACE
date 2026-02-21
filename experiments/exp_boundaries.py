"""Experiment: boundary detection strategies for repetitive exercise.

Problem: For simple repetitive exercise (like leg press), too many motion
boundaries are detected, creating too many segments. Even with agglomerative
clustering, having 50+ segments that are all basically the same motion creates noise.

Tests four approaches:
  A) Autocorrelation period detection
  B) Post-clustering merge of adjacent segments
  C) Adaptive re-segmentation
  D) Parameter sweep on current approach
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
    _resample_segment,
    _segment_distance,
    normalize_frame,
    feature_vector,
)
from brace.core.pose import FEATURE_INDICES, LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_landmarks(n_frames: int, period: int, amplitude: float, noise_std: float,
                             phase_offset: float = 0.0) -> list[np.ndarray]:
    """Generate synthetic MediaPipe-format landmarks with sinusoidal limb motion.

    Produces (33, 4) arrays [x, y, z, visibility] per frame.
    Core body stays fixed, knees and ankles oscillate sinusoidally to simulate
    repetitive lower-body exercise (like leg press).
    """
    landmarks_list = []
    for f in range(n_frames):
        lm = np.zeros((33, 4), dtype=np.float32)
        # All joints visible
        lm[:, 3] = 1.0

        # Fixed torso landmarks
        lm[LEFT_SHOULDER] = [0.4, 0.3, 0.0, 1.0]
        lm[RIGHT_SHOULDER] = [0.6, 0.3, 0.0, 1.0]
        lm[LEFT_HIP] = [0.4, 0.6, 0.0, 1.0]
        lm[RIGHT_HIP] = [0.6, 0.6, 0.0, 1.0]

        # Sinusoidal motion on lower body
        t = 2.0 * np.pi * f / period + phase_offset
        osc = amplitude * np.sin(t) * 0.01  # scale to reasonable pixel offsets

        # Left knee (25), right knee (26)
        lm[25] = [0.4 + osc, 0.75 + osc * 0.5, 0.0, 1.0]
        lm[26] = [0.6 + osc, 0.75 + osc * 0.5, 0.0, 1.0]

        # Left ankle (27), right ankle (28)
        lm[27] = [0.4 + osc * 1.2, 0.9 + osc * 0.8, 0.0, 1.0]
        lm[28] = [0.6 + osc * 1.2, 0.9 + osc * 0.8, 0.0, 1.0]

        # Left foot (31), right foot (32)
        lm[31] = [0.4 + osc * 1.3, 0.95 + osc * 0.9, 0.0, 1.0]
        lm[32] = [0.6 + osc * 1.3, 0.95 + osc * 0.9, 0.0, 1.0]

        # Elbows and wrists (minor motion)
        lm[13] = [0.35 + osc * 0.2, 0.45, 0.0, 1.0]
        lm[14] = [0.65 + osc * 0.2, 0.45, 0.0, 1.0]
        lm[15] = [0.3 + osc * 0.3, 0.55, 0.0, 1.0]
        lm[16] = [0.7 + osc * 0.3, 0.55, 0.0, 1.0]

        # Add noise
        lm[:, :2] += np.random.randn(33, 2).astype(np.float32) * noise_std * 0.01

        landmarks_list.append(lm)

    return landmarks_list


def landmarks_to_features(landmarks_list: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Convert landmarks to SRP-normalized feature vectors."""
    features = []
    valid_indices = []
    for i, lm in enumerate(landmarks_list):
        norm = normalize_frame(lm)
        if norm is None:
            continue
        feat = feature_vector(norm)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            continue
        features.append(feat)
        valid_indices.append(i)
    return np.stack(features), valid_indices


# ---------------------------------------------------------------------------
# Generate test datasets
# ---------------------------------------------------------------------------

np.random.seed(42)
FPS = 30.0

print("=" * 70)
print("GENERATING SYNTHETIC DATA")
print("=" * 70)

# 1. SMOOTH_REPS: 600 frames, period=40, 15 reps, low noise
smooth_lm = make_synthetic_landmarks(600, period=40, amplitude=2.0, noise_std=0.01)
smooth_feat, smooth_vi = landmarks_to_features(smooth_lm)
print(f"SMOOTH_REPS: {len(smooth_lm)} frames, {smooth_feat.shape[0]} valid, "
      f"expected 15 reps (period=40, 600/40=15)")

# 2. NOISY_REPS: same params, high noise
noisy_lm = make_synthetic_landmarks(600, period=40, amplitude=2.0, noise_std=0.1)
noisy_feat, noisy_vi = landmarks_to_features(noisy_lm)
print(f"NOISY_REPS:  {len(noisy_lm)} frames, {noisy_feat.shape[0]} valid, "
      f"expected 15 reps (period=40, noisy)")

# 3. TWO_PHASE: 300 frames slow + 300 frames fast (different amplitude)
phase1_lm = make_synthetic_landmarks(300, period=40, amplitude=2.0, noise_std=0.01)
phase2_lm = make_synthetic_landmarks(300, period=20, amplitude=3.0, noise_std=0.01)
twophase_lm = phase1_lm + phase2_lm
twophase_feat, twophase_vi = landmarks_to_features(twophase_lm)
print(f"TWO_PHASE:   {len(twophase_lm)} frames, {twophase_feat.shape[0]} valid, "
      f"expected 2 distinct motion groups")


# ---------------------------------------------------------------------------
# Current baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE: Current pipeline (default params)")
print("=" * 70)

for name, feat, vi in [("SMOOTH", smooth_feat, smooth_vi),
                        ("NOISY", noisy_feat, noisy_vi),
                        ("TWO_PHASE", twophase_feat, twophase_vi)]:
    boundaries = detect_motion_boundaries(feat, fps=FPS)
    segments = segment_motions(feat, vi, fps=FPS)
    clustered = cluster_segments(segments, distance_threshold=2.0)
    n_clusters = len(set(s["cluster"] for s in clustered)) if clustered else 0
    print(f"  {name:12s}: {len(boundaries):3d} boundaries, {len(segments):3d} segments, "
          f"{n_clusters} clusters")


# ===========================================================================
# APPROACH A: Autocorrelation period detection
# ===========================================================================

print("\n" + "=" * 70)
print("APPROACH A: Autocorrelation period detection")
print("=" * 70)


def autocorrelation_boundaries(features: np.ndarray, fps: float = 30.0) -> tuple[list[int], float]:
    """Detect dominant period via autocorrelation and place boundaries at period multiples.

    Returns (boundaries, detected_period_frames).
    """
    # Use first feature dimension as representative signal
    signal = features[:, 0].copy()
    signal -= signal.mean()

    # Full autocorrelation
    acf = np.correlate(signal, signal, mode="full")
    acf = acf[len(acf) // 2:]  # keep positive lags only
    acf /= acf[0] + 1e-12  # normalize

    # Find first peak after lag 0 (skip first few frames to avoid trivial peak)
    min_lag = max(5, int(fps * 0.3))
    peaks_idx = []
    for i in range(min_lag, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
            peaks_idx.append(i)
            break

    if not peaks_idx:
        # Fallback: no clear period found
        return [0], 0.0

    period = peaks_idx[0]

    # Place boundaries at period multiples
    boundaries = list(range(0, features.shape[0], period))

    return boundaries, float(period)


for name, feat in [("SMOOTH", smooth_feat), ("NOISY", noisy_feat), ("TWO_PHASE", twophase_feat)]:
    bounds, period = autocorrelation_boundaries(feat, fps=FPS)
    expected = 15 if name != "TWO_PHASE" else "7+15=22"
    print(f"  {name:12s}: detected period={period:.1f} frames ({period/FPS:.2f}s), "
          f"{len(bounds)} boundaries (expected ~{expected})")


# ===========================================================================
# APPROACH B: Post-clustering merge of adjacent segments
# ===========================================================================

print("\n" + "=" * 70)
print("APPROACH B: Post-clustering merge of adjacent segments")
print("=" * 70)


def merge_adjacent_same_cluster(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that belong to the same cluster.

    Returns a new list of merged segments.
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["cluster"] == prev["cluster"]:
            # Merge: extend the previous segment
            prev["end_valid"] = seg["end_valid"]
            prev["end_frame"] = seg["end_frame"]
            prev["features"] = np.vstack([prev["features"], seg["features"]])
            prev["mean_feature"] = prev["features"].mean(axis=0)
        else:
            merged.append(seg.copy())

    return merged


for name, feat, vi in [("SMOOTH", smooth_feat, smooth_vi),
                        ("NOISY", noisy_feat, noisy_vi),
                        ("TWO_PHASE", twophase_feat, twophase_vi)]:
    segments = segment_motions(feat, vi, fps=FPS)
    clustered = cluster_segments(segments, distance_threshold=2.0)
    n_clusters_before = len(set(s["cluster"] for s in clustered)) if clustered else 0
    n_before = len(clustered)

    merged = merge_adjacent_same_cluster(clustered)
    n_clusters_after = len(set(s["cluster"] for s in merged)) if merged else 0

    print(f"  {name:12s}: {n_before:3d} segments -> {len(merged):3d} merged, "
          f"clusters: {n_clusters_before} -> {n_clusters_after}")


# ===========================================================================
# APPROACH C: Adaptive re-segmentation
# ===========================================================================

print("\n" + "=" * 70)
print("APPROACH C: Adaptive re-segmentation (if >80% in one cluster, re-run)")
print("=" * 70)


def adaptive_resegment(features: np.ndarray, valid_indices: list[int], fps: float = 30.0,
                       min_segment_sec: float = 0.8, smoothing_mult: float = 0.3,
                       distance_threshold: float = 2.0) -> tuple[list[dict], str]:
    """Run segmentation, and if >80% of segments end up in one cluster,
    re-run with doubled min_segment_sec and smoothing.

    Returns (segments, info_string).
    """
    # Initial pass
    segments = segment_motions(features, valid_indices, fps=fps, min_segment_sec=min_segment_sec)
    if len(segments) < 2:
        return segments, "too few segments"

    clustered = cluster_segments(segments, distance_threshold=distance_threshold)
    cluster_counts = {}
    for s in clustered:
        c = s["cluster"]
        cluster_counts[c] = cluster_counts.get(c, 0) + 1

    # Check if dominant cluster has >80%
    dominant_count = max(cluster_counts.values())
    total = len(clustered)
    dominant_pct = dominant_count / total

    info = f"pass1: {total} segs, {len(cluster_counts)} clusters, dominant={dominant_pct:.0%}"

    if dominant_pct > 0.8 and total > 10:
        # Re-run with doubled params
        new_min_sec = min_segment_sec * 2
        segments2 = segment_motions(features, valid_indices, fps=fps, min_segment_sec=new_min_sec)
        if len(segments2) >= 2:
            clustered2 = cluster_segments(segments2, distance_threshold=distance_threshold)
            cluster_counts2 = {}
            for s in clustered2:
                c = s["cluster"]
                cluster_counts2[c] = cluster_counts2.get(c, 0) + 1
            dominant2 = max(cluster_counts2.values())
            info += f" -> pass2 (min_sec={new_min_sec}): {len(clustered2)} segs, {len(cluster_counts2)} clusters"
            return clustered2, info

    return clustered, info


for name, feat, vi in [("SMOOTH", smooth_feat, smooth_vi),
                        ("NOISY", noisy_feat, noisy_vi),
                        ("TWO_PHASE", twophase_feat, twophase_vi)]:
    result, info = adaptive_resegment(feat, vi, fps=FPS)
    n_clusters = len(set(s["cluster"] for s in result)) if result else 0
    print(f"  {name:12s}: {info}, final={len(result)} segs, {n_clusters} clusters")


# ===========================================================================
# APPROACH D: Parameter sweep on current approach
# ===========================================================================

print("\n" + "=" * 70)
print("APPROACH D: Parameter sweep")
print("=" * 70)

from scipy.signal import find_peaks


def detect_motion_boundaries_custom(
    features: np.ndarray,
    fps: float = 30.0,
    min_segment_sec: float = 0.8,
    smoothing_mult: float = 0.3,
    prominence_mult: float = 0.5,
) -> list[int]:
    """Custom boundary detection with tunable smoothing and prominence."""
    n = features.shape[0]
    if n < 5:
        return [0]

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    kernel_size = max(5, int(fps * smoothing_mult))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocity, kernel, mode="same")

    min_frames = max(int(fps * min_segment_sec), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return [0]

    median_vel = float(np.median(positive_vals))
    min_prominence = median_vel * prominence_mult

    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    return boundaries


min_seg_values = [0.5, 0.8, 1.0, 1.5, 2.0]
smooth_values = [0.2, 0.3, 0.5, 0.7, 1.0]
prom_values = [0.3, 0.5, 0.7, 1.0]

print("\n  --- SMOOTH_REPS (target: ~15 segments, 1 cluster) ---")
best_smooth = None
best_smooth_score = float("inf")

for min_sec in min_seg_values:
    for sm in smooth_values:
        for pm in prom_values:
            bounds = detect_motion_boundaries_custom(smooth_feat, FPS, min_sec, sm, pm)
            n_segs = len(bounds)
            # Score: distance from 15 segments
            score = abs(n_segs - 15)
            if score < best_smooth_score:
                best_smooth_score = score
                best_smooth = (min_sec, sm, pm, n_segs)

print(f"  Best params: min_seg={best_smooth[0]}s, smooth={best_smooth[1]}, "
      f"prom={best_smooth[2]} -> {best_smooth[3]} segments (target 15)")

# Show top 5
print("  Top 5 combinations:")
results = []
for min_sec in min_seg_values:
    for sm in smooth_values:
        for pm in prom_values:
            bounds = detect_motion_boundaries_custom(smooth_feat, FPS, min_sec, sm, pm)
            n_segs = len(bounds)
            score = abs(n_segs - 15)
            results.append((score, min_sec, sm, pm, n_segs))
results.sort()
for score, ms, sm, pm, ns in results[:5]:
    print(f"    min_seg={ms}, smooth={sm}, prom={pm} -> {ns} segments (off by {score})")


print("\n  --- TWO_PHASE (target: 2 cluster groups) ---")
best_twophase = None
best_twophase_score = float("inf")

for min_sec in min_seg_values:
    for sm in smooth_values:
        for pm in prom_values:
            bounds = detect_motion_boundaries_custom(twophase_feat, FPS, min_sec, sm, pm)
            # Build segments manually
            segs = []
            for i in range(len(bounds)):
                start = bounds[i]
                end = bounds[i + 1] if i + 1 < len(bounds) else twophase_feat.shape[0]
                if end - start < 3:
                    continue
                seg_features = twophase_feat[start:end]
                segs.append({
                    "start_valid": start,
                    "end_valid": end,
                    "start_frame": twophase_vi[start] if start < len(twophase_vi) else 0,
                    "end_frame": twophase_vi[end - 1] if end - 1 < len(twophase_vi) else twophase_vi[-1],
                    "features": seg_features,
                    "mean_feature": seg_features.mean(axis=0),
                })
            if len(segs) >= 2:
                clustered = cluster_segments(segs, distance_threshold=2.0)
                n_clusters = len(set(s["cluster"] for s in clustered))
            else:
                n_clusters = len(segs)

            # Score: want exactly 2 clusters, and reasonable segment count
            cluster_score = abs(n_clusters - 2) * 10 + abs(len(segs) - 20) * 0.1
            if cluster_score < best_twophase_score:
                best_twophase_score = cluster_score
                best_twophase = (min_sec, sm, pm, len(segs), n_clusters)

print(f"  Best params: min_seg={best_twophase[0]}s, smooth={best_twophase[1]}, "
      f"prom={best_twophase[2]} -> {best_twophase[3]} segments, {best_twophase[4]} clusters (target 2)")

# Show top 5
print("  Top 5 combinations:")
results2 = []
for min_sec in min_seg_values:
    for sm in smooth_values:
        for pm in prom_values:
            bounds = detect_motion_boundaries_custom(twophase_feat, FPS, min_sec, sm, pm)
            segs = []
            for i in range(len(bounds)):
                start = bounds[i]
                end = bounds[i + 1] if i + 1 < len(bounds) else twophase_feat.shape[0]
                if end - start < 3:
                    continue
                seg_features = twophase_feat[start:end]
                segs.append({
                    "start_valid": start,
                    "end_valid": end,
                    "start_frame": twophase_vi[start] if start < len(twophase_vi) else 0,
                    "end_frame": twophase_vi[end - 1] if end - 1 < len(twophase_vi) else twophase_vi[-1],
                    "features": seg_features,
                    "mean_feature": seg_features.mean(axis=0),
                })
            if len(segs) >= 2:
                clustered = cluster_segments(segs, distance_threshold=2.0)
                n_clusters = len(set(s["cluster"] for s in clustered))
            else:
                n_clusters = len(segs)
            cluster_score = abs(n_clusters - 2) * 10 + abs(len(segs) - 20) * 0.1
            results2.append((cluster_score, min_sec, sm, pm, len(segs), n_clusters))
results2.sort()
for score, ms, sm, pm, ns, nc in results2[:5]:
    print(f"    min_seg={ms}, smooth={sm}, prom={pm} -> {ns} segments, {nc} clusters")


# ===========================================================================
# NOISY_REPS through all approaches
# ===========================================================================

print("\n" + "=" * 70)
print("NOISY_REPS through all approaches (target: ~15 segments, 1 cluster)")
print("=" * 70)

# Baseline
bounds_base = detect_motion_boundaries(noisy_feat, fps=FPS)
segs_base = segment_motions(noisy_feat, noisy_vi, fps=FPS)
clust_base = cluster_segments(segs_base, distance_threshold=2.0)
nc_base = len(set(s["cluster"] for s in clust_base)) if clust_base else 0
print(f"  Baseline:      {len(segs_base):3d} segments, {nc_base} clusters")

# A) Autocorrelation
bounds_a, period_a = autocorrelation_boundaries(noisy_feat, fps=FPS)
print(f"  Autocorrelation: period={period_a:.1f}, {len(bounds_a)} boundaries")

# B) Post-merge
merged_b = merge_adjacent_same_cluster(clust_base)
nc_b = len(set(s["cluster"] for s in merged_b)) if merged_b else 0
print(f"  Post-merge:    {len(segs_base):3d} -> {len(merged_b):3d} merged, {nc_b} clusters")

# C) Adaptive
result_c, info_c = adaptive_resegment(noisy_feat, noisy_vi, fps=FPS)
nc_c = len(set(s["cluster"] for s in result_c)) if result_c else 0
print(f"  Adaptive:      {info_c}")

# D) Best params from smooth sweep
if best_smooth:
    ms, sm, pm, _ = best_smooth
    bounds_d = detect_motion_boundaries_custom(noisy_feat, FPS, ms, sm, pm)
    segs_d = []
    for i in range(len(bounds_d)):
        start = bounds_d[i]
        end = bounds_d[i + 1] if i + 1 < len(bounds_d) else noisy_feat.shape[0]
        if end - start < 3:
            continue
        seg_features = noisy_feat[start:end]
        segs_d.append({
            "start_valid": start,
            "end_valid": end,
            "start_frame": noisy_vi[start] if start < len(noisy_vi) else 0,
            "end_frame": noisy_vi[end - 1] if end - 1 < len(noisy_vi) else noisy_vi[-1],
            "features": seg_features,
            "mean_feature": seg_features.mean(axis=0),
        })
    if len(segs_d) >= 2:
        clust_d = cluster_segments(segs_d, distance_threshold=2.0)
        nc_d = len(set(s["cluster"] for s in clust_d))
    else:
        nc_d = len(segs_d)
    print(f"  Best-D params: {len(segs_d):3d} segments, {nc_d} clusters "
          f"(min_seg={ms}, smooth={sm}, prom={pm})")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
For simple repetitive exercise (many identical reps):

A) Autocorrelation: Good at finding the dominant period, but only works
   for perfectly periodic motion. Cannot handle transitions between
   different exercises. Best for: confirming the expected rep count.

B) Post-clustering merge: Simple post-processing that reduces segment
   count without changing the underlying detection. Adjacent segments
   in the same cluster become one long segment. Best for: cleaning up
   over-segmentation after the fact.

C) Adaptive re-segmentation: If the first pass shows >80% of segments
   in one cluster (clear sign of over-segmentation), re-run with more
   aggressive smoothing. Best for: automatic parameter adjustment.

D) Parameter sweep: Finding the right min_segment_sec and smoothing
   multiplier is the most impactful. Larger min_segment_sec prevents
   tiny segments; larger smoothing prevents false boundary detection
   on noisy velocity signals. Best for: tuning for specific use cases.

RECOMMENDATION: Combine B + D. Use tuned parameters to reduce initial
over-segmentation, then apply post-clustering merge to clean up any
remaining adjacent same-cluster segments.
""")
