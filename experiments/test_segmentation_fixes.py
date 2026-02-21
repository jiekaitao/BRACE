#!/usr/bin/env python3
"""Diagnose and test fixes for velocity segmentation over-segmentation.

Loads cached pose features from .feature_cache/ and tests segment_motions()
with various parameter changes to reduce the excessive number of segments.
"""
import sys
import os
import numpy as np

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
)

CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"


def load_cached_features(cache_dir: str) -> dict:
    """Load all cached feature files. Returns dict: filename -> (features, valid_indices, fps)."""
    data = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".feats.npz"):
            continue
        path = os.path.join(cache_dir, fname)
        npz = np.load(path)
        video_name = fname.replace(".feats.npz", "")
        data[video_name] = (
            npz["features"],
            npz["valid_indices"].tolist(),
            float(npz["fps"]),
        )
    return data


def run_current_baseline(video_data: dict) -> dict:
    """Run segment_motions() with current default settings."""
    results = {}
    for name, (feats, vi, fps) in sorted(video_data.items()):
        segments = segment_motions(feats, vi, fps, min_segment_sec=1.0)
        n_seg = len(segments)
        seg_lengths = [s["end_valid"] - s["start_valid"] for s in segments]
        # Also cluster to see how many clusters result
        if len(segments) >= 2:
            import copy
            clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
            n_clust = len(set(s["cluster"] for s in clustered))
        else:
            n_clust = max(1, len(segments))
        results[name] = {
            "n_segments": n_seg,
            "n_clusters": n_clust,
            "seg_lengths": seg_lengths,
            "total_frames": feats.shape[0],
        }
    return results


def test_min_segment_sec(video_data: dict, values: list[float]) -> dict:
    """Test different min_segment_sec values."""
    results = {}
    for val in values:
        results[val] = {}
        for name, (feats, vi, fps) in sorted(video_data.items()):
            segments = segment_motions(feats, vi, fps, min_segment_sec=val)
            results[val][name] = len(segments)
    return results


def detect_boundaries_smoothed(
    features: np.ndarray,
    fps: float = 24.0,
    min_segment_sec: float = 1.0,
    velocity_percentile: float = 85,
    smooth_factor: float = 1.0,
    prominence_factor: float = 0.5,
) -> list[int]:
    """Modified detect_motion_boundaries with tunable smoothing and prominence."""
    from scipy.signal import find_peaks

    n = features.shape[0]
    if n < 5:
        return [0]

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    # Tunable smoothing: smooth_factor multiplies the default kernel width
    kernel_size = max(5, int(fps * 0.3 * smooth_factor))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocity, kernel, mode="same")

    min_frames = max(int(fps * min_segment_sec), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return [0]

    median_vel = float(np.median(positive_vals))
    min_prominence = median_vel * prominence_factor

    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    return boundaries


def detect_boundaries_gaussian(
    features: np.ndarray,
    fps: float = 24.0,
    min_segment_sec: float = 1.0,
    sigma_sec: float = 0.5,
    prominence_factor: float = 0.75,
) -> list[int]:
    """Detect boundaries using Gaussian smoothing instead of box filter."""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d

    n = features.shape[0]
    if n < 5:
        return [0]

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    sigma = fps * sigma_sec
    smoothed = gaussian_filter1d(velocity, sigma=sigma)

    min_frames = max(int(fps * min_segment_sec), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return [0]

    median_vel = float(np.median(positive_vals))
    min_prominence = median_vel * prominence_factor

    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    return boundaries


def detect_boundaries_velocity_ratio(
    features: np.ndarray,
    fps: float = 24.0,
    min_segment_sec: float = 2.0,
    low_velocity_ratio: float = 0.3,
) -> list[int]:
    """Detect boundaries by finding frames where velocity drops below a ratio of local max.

    Instead of finding peaks in inverted velocity, find sustained low-velocity
    regions relative to surrounding activity.
    """
    from scipy.ndimage import gaussian_filter1d

    n = features.shape[0]
    if n < 5:
        return [0]

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    # Heavy smoothing
    sigma = fps * 0.5
    smoothed = gaussian_filter1d(velocity, sigma=sigma)

    # Compute running max over a wide window (capture local activity level)
    window = int(fps * 3.0)
    running_max = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        running_max[i] = np.max(smoothed[lo:hi])

    # A boundary candidate is where smoothed velocity < low_velocity_ratio * running_max
    min_frames = max(int(fps * min_segment_sec), 5)
    threshold = running_max * low_velocity_ratio

    # Find the lowest velocity points within each low-velocity valley
    in_low = smoothed < threshold
    boundaries = [0]

    i = 0
    while i < n:
        if in_low[i] and i - boundaries[-1] >= min_frames:
            # Find the minimum within this low-velocity region
            j = i
            while j < n and in_low[j]:
                j += 1
            valley_start = i
            valley_end = j
            min_idx = valley_start + int(np.argmin(smoothed[valley_start:valley_end]))
            if min_idx - boundaries[-1] >= min_frames:
                boundaries.append(min_idx)
            i = j
        else:
            i += 1

    return boundaries


def test_custom_detectors(video_data: dict) -> dict:
    """Test various custom boundary detection strategies."""

    configs = {
        # Higher min_segment_sec (simple approach)
        "min_seg_1.5s": lambda f, fps: detect_motion_boundaries(f, fps, min_segment_sec=1.5),
        "min_seg_2.0s": lambda f, fps: detect_motion_boundaries(f, fps, min_segment_sec=2.0),
        "min_seg_3.0s": lambda f, fps: detect_motion_boundaries(f, fps, min_segment_sec=3.0),

        # Higher prominence (require deeper velocity dips)
        "prom_0.75": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, prominence_factor=0.75),
        "prom_1.0": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, prominence_factor=1.0),
        "prom_1.5": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, prominence_factor=1.5),

        # Wider smoothing (suppresses minor dips)
        "smooth_2x": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, smooth_factor=2.0),
        "smooth_3x": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, smooth_factor=3.0),
        "smooth_4x": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, smooth_factor=4.0),

        # Gaussian smoothing + higher prominence
        "gauss_0.5s_p0.75": lambda f, fps: detect_boundaries_gaussian(f, fps, sigma_sec=0.5, prominence_factor=0.75),
        "gauss_1.0s_p0.75": lambda f, fps: detect_boundaries_gaussian(f, fps, sigma_sec=1.0, prominence_factor=0.75),
        "gauss_1.0s_p1.0": lambda f, fps: detect_boundaries_gaussian(f, fps, sigma_sec=1.0, prominence_factor=1.0),

        # Combined: wider smoothing + higher prominence + longer min segment
        "combo_s2x_p0.75_ms2.0": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=2.0, smooth_factor=2.0, prominence_factor=0.75),
        "combo_s3x_p1.0_ms2.0": lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=2.0, smooth_factor=3.0, prominence_factor=1.0),
        "combo_gauss1s_p1.0_ms2.0": lambda f, fps: detect_boundaries_gaussian(f, fps, min_segment_sec=2.0, sigma_sec=1.0, prominence_factor=1.0),

        # Velocity ratio approach (completely different strategy)
        "vel_ratio_0.3": lambda f, fps: detect_boundaries_velocity_ratio(f, fps, min_segment_sec=2.0, low_velocity_ratio=0.3),
        "vel_ratio_0.2": lambda f, fps: detect_boundaries_velocity_ratio(f, fps, min_segment_sec=2.0, low_velocity_ratio=0.2),
    }

    results = {}
    for config_name, detector_fn in configs.items():
        results[config_name] = {}
        for name, (feats, vi, fps) in sorted(video_data.items()):
            boundaries = detector_fn(feats, fps)
            # Convert boundaries to segments (same logic as segment_motions)
            segments = []
            for i in range(len(boundaries)):
                start = boundaries[i]
                end = boundaries[i + 1] if i + 1 < len(boundaries) else feats.shape[0]
                if end - start >= 3:
                    segments.append({"start_valid": start, "end_valid": end})
            results[config_name][name] = len(segments)

    return results


def analyze_velocity_profile(video_data: dict, target_videos: list[str]) -> None:
    """Print detailed velocity analysis for specific videos to understand the problem."""
    for name in target_videos:
        if name not in video_data:
            continue
        feats, vi, fps = video_data[name]
        n = feats.shape[0]

        velocity = np.zeros(n)
        for i in range(1, n):
            velocity[i] = float(np.linalg.norm(feats[i] - feats[i - 1]))

        kernel_size = max(5, int(fps * 0.3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(velocity, kernel, mode="same")

        print(f"\n{'='*70}")
        print(f"VELOCITY PROFILE: {name}")
        print(f"{'='*70}")
        print(f"  Total frames: {n} ({n/fps:.1f} sec at {fps:.1f} fps)")
        print(f"  Velocity stats: min={velocity.min():.3f}, max={velocity.max():.3f}, "
              f"mean={velocity.mean():.3f}, median={np.median(velocity):.3f}")
        print(f"  Smoothed stats: min={smoothed.min():.3f}, max={smoothed.max():.3f}, "
              f"mean={smoothed.mean():.3f}, median={np.median(smoothed[smoothed>0]):.3f}")
        print(f"  Smoothing kernel: {kernel_size} frames ({kernel_size/fps:.2f} sec)")

        # Current boundaries
        boundaries = detect_motion_boundaries(feats, fps, min_segment_sec=1.0)
        print(f"  Current boundaries (min_seg=1.0s): {len(boundaries)} -> {len(boundaries)} segments")
        if len(boundaries) > 1:
            dists = np.diff(boundaries)
            print(f"    Segment lengths (frames): min={dists.min()}, max={dists.max()}, "
                  f"mean={dists.mean():.0f}, median={np.median(dists):.0f}")
            print(f"    Segment lengths (sec): min={dists.min()/fps:.1f}, max={dists.max()/fps:.1f}, "
                  f"mean={dists.mean()/fps:.1f}")

        # Prominence info
        positive_vals = smoothed[smoothed > 0]
        median_vel = float(np.median(positive_vals)) if len(positive_vals) > 0 else 0
        print(f"  median velocity: {median_vel:.4f}")
        print(f"  current prominence threshold (0.5 * median): {median_vel * 0.5:.4f}")
        print(f"  0.75 * median: {median_vel * 0.75:.4f}")
        print(f"  1.0 * median: {median_vel * 1.0:.4f}")


def main():
    print("Loading cached features...", flush=True)
    video_data = load_cached_features(CACHE_DIR)
    print(f"Loaded {len(video_data)} videos\n", flush=True)

    for name, (feats, vi, fps) in sorted(video_data.items()):
        duration = feats.shape[0] / fps
        print(f"  {name:35s} {feats.shape[0]:5d} frames  {duration:6.1f}s  {fps:.1f}fps  {feats.shape[1]}D")

    # =========================================================================
    # 1. Current baseline
    # =========================================================================
    print(f"\n{'='*70}")
    print("CURRENT BASELINE (min_segment_sec=1.0, default params)")
    print(f"{'='*70}")

    baseline = run_current_baseline(video_data)
    for name, info in sorted(baseline.items()):
        dur = info["total_frames"] / video_data[name][2]
        print(f"  {name:35s} segs={info['n_segments']:3d}  clusters={info['n_clusters']:3d}  "
              f"frames={info['total_frames']:5d}  dur={dur:.0f}s")

    total_segs = sum(v["n_segments"] for v in baseline.values())
    total_clusts = sum(v["n_clusters"] for v in baseline.values())
    print(f"\n  TOTALS: {total_segs} segments, {total_clusts} clusters across {len(baseline)} videos")

    # =========================================================================
    # 2. Detailed velocity analysis for worst offenders
    # =========================================================================
    worst = sorted(baseline.items(), key=lambda x: x[1]["n_segments"], reverse=True)[:5]
    print(f"\n{'='*70}")
    print(f"DETAILED VELOCITY ANALYSIS (top 5 over-segmented)")
    print(f"{'='*70}")
    analyze_velocity_profile(video_data, [name for name, _ in worst])

    # =========================================================================
    # 3. Test all custom detector configurations
    # =========================================================================
    print(f"\n{'='*70}")
    print("TESTING CUSTOM BOUNDARY DETECTION CONFIGURATIONS")
    print(f"{'='*70}")

    custom_results = test_custom_detectors(video_data)

    # Print as a table
    video_names = sorted(video_data.keys())
    short_names = {n: n[:20] for n in video_names}

    # Print header
    header = f"{'Config':35s}"
    for n in video_names:
        header += f" {short_names[n]:>6s}"
    header += f" {'TOTAL':>6s} {'MEAN':>6s}"
    print(f"\n{header}")
    print("-" * len(header))

    # Print baseline first
    row = f"{'BASELINE (current)':35s}"
    for n in video_names:
        row += f" {baseline[n]['n_segments']:6d}"
    row += f" {total_segs:6d} {total_segs/len(video_names):6.1f}"
    print(row)
    print("-" * len(header))

    # Print each config
    for config_name, vresults in sorted(custom_results.items()):
        row = f"{config_name:35s}"
        total = 0
        for n in video_names:
            count = vresults.get(n, 0)
            row += f" {count:6d}"
            total += count
        row += f" {total:6d} {total/len(video_names):6.1f}"
        print(row)

    # =========================================================================
    # 4. Cluster counts for promising configurations
    # =========================================================================
    print(f"\n{'='*70}")
    print("CLUSTER COUNTS FOR PROMISING CONFIGURATIONS (threshold=2.0)")
    print(f"{'='*70}")

    import copy

    # Pick the most promising configs (those that substantially reduce segments)
    promising = [
        ("BASELINE", lambda f, vi, fps: segment_motions(f, vi, fps, min_segment_sec=1.0)),
        ("min_seg_2.0s", lambda f, vi, fps: segment_motions(f, vi, fps, min_segment_sec=2.0)),
        ("min_seg_3.0s", lambda f, vi, fps: segment_motions(f, vi, fps, min_segment_sec=3.0)),
    ]

    # Add some custom detector configs that create segments from custom boundaries
    def make_segment_fn(detector_fn):
        def fn(feats, vi, fps):
            boundaries = detector_fn(feats, fps)
            segments = []
            for i in range(len(boundaries)):
                start = boundaries[i]
                end = boundaries[i + 1] if i + 1 < len(boundaries) else feats.shape[0]
                if end - start < 3:
                    continue
                seg_features = feats[start:end]
                segments.append({
                    "start_valid": start,
                    "end_valid": end,
                    "start_frame": vi[start] if start < len(vi) else 0,
                    "end_frame": vi[end - 1] if end - 1 < len(vi) else vi[-1],
                    "features": seg_features,
                    "mean_feature": seg_features.mean(axis=0),
                })
            return segments
        return fn

    promising.extend([
        ("combo_s3x_p1.0_ms2.0",
         make_segment_fn(lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=2.0, smooth_factor=3.0, prominence_factor=1.0))),
        ("combo_gauss1s_p1.0_ms2.0",
         make_segment_fn(lambda f, fps: detect_boundaries_gaussian(f, fps, min_segment_sec=2.0, sigma_sec=1.0, prominence_factor=1.0))),
        ("vel_ratio_0.3",
         make_segment_fn(lambda f, fps: detect_boundaries_velocity_ratio(f, fps, min_segment_sec=2.0, low_velocity_ratio=0.3))),
        ("vel_ratio_0.2",
         make_segment_fn(lambda f, fps: detect_boundaries_velocity_ratio(f, fps, min_segment_sec=2.0, low_velocity_ratio=0.2))),
        ("prom_1.0",
         make_segment_fn(lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, prominence_factor=1.0))),
        ("prom_1.5",
         make_segment_fn(lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, prominence_factor=1.5))),
        ("smooth_3x_prom1.0",
         make_segment_fn(lambda f, fps: detect_boundaries_smoothed(f, fps, min_segment_sec=1.0, smooth_factor=3.0, prominence_factor=1.0))),
    ])

    header2 = f"{'Config':35s}"
    for n in video_names:
        header2 += f"  {short_names[n][:8]:>8s}"
    header2 += f"  {'Avg_seg':>7s}  {'Avg_clust':>9s}"
    print(f"\n{header2}")
    print("-" * len(header2))

    for config_name, segment_fn in promising:
        row = f"{config_name:35s}"
        total_seg = 0
        total_clust = 0
        for n in video_names:
            feats, vi, fps = video_data[n]
            segments = segment_fn(feats, vi, fps)
            n_seg = len(segments)
            if len(segments) >= 2:
                clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
                n_clust = len(set(s["cluster"] for s in clustered))
            else:
                n_clust = max(1, n_seg)
            row += f"  {n_seg:2d}/{n_clust:<5d}"
            total_seg += n_seg
            total_clust += n_clust
        avg_seg = total_seg / len(video_names)
        avg_clust = total_clust / len(video_names)
        row += f"  {avg_seg:7.1f}  {avg_clust:9.1f}"
        print(row)

    # =========================================================================
    # 5. Summary & Recommendation
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY & ANALYSIS")
    print(f"{'='*70}")
    print("""
The over-segmentation problem stems from detect_motion_boundaries():
  1. Smoothing window is too narrow (fps*0.3 ~= 7 frames at 24fps) -- minor
     velocity dips during exercise reps trigger boundaries
  2. Prominence threshold (median_vel * 0.5) is too low -- even small pauses
     between reps create "significant" enough velocity valleys
  3. min_segment_sec=1.0 allows very short segments (24 frames at 24fps)

The fix should combine:
  - Wider smoothing to suppress intra-rep velocity fluctuations
  - Higher prominence threshold to require real activity transitions
  - Longer minimum segment duration to prevent tiny fragments
    """)


if __name__ == "__main__":
    main()
