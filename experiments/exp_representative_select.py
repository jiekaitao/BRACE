"""Experiment: representative segment selection methods.

Given a cluster of motion segments, find the BEST representative segment
to use as a reference for correction arrows.

Tests 4 methods: Mean Trajectory, Medoid, Best-Fit-to-Mean, Smoothest.
Evaluates on 3 synthetic datasets: clean, varied-amplitude, noisy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from brace.core.motion_segments import (
    segment_motions, cluster_segments, _resample_segment, _segment_distance,
)
from brace.core.pose import (
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER,
    FEATURE_INDICES,
)

RESAMPLE_LEN = 30
N_FRAMES = 40
FEAT_DIM = 28  # 14 joints * 2 (x, y)

# Knee joints in the 14-joint feature array are at positions 8 (left_knee=25)
# and 9 (right_knee=26). In 28D features: dims 16-17 (left knee x,y)
# and 18-19 (right knee x,y).
KNEE_FEAT_DIMS = [16, 17, 18, 19]


# --- Synthetic data generation ---

def make_base_motion(n_frames: int, amplitude: float = 1.0) -> np.ndarray:
    """Create base sinusoidal motion on knee joints. Returns (n_frames, 28)."""
    t = np.linspace(0, 2 * np.pi, n_frames)
    features = np.zeros((n_frames, FEAT_DIM), dtype=np.float64)
    for d in KNEE_FEAT_DIMS:
        features[:, d] = amplitude * np.sin(t)
    return features


def make_segment(features: np.ndarray, idx: int) -> dict:
    """Wrap a feature array as a segment dict."""
    return {
        "features": features.astype(np.float32),
        "start_valid": idx * N_FRAMES,
        "end_valid": (idx + 1) * N_FRAMES,
        "cluster": 0,
        "mean_feature": features.mean(axis=0).astype(np.float32),
    }


def generate_clean_reps(rng: np.random.Generator) -> list[dict]:
    """CLEAN_REPS: 10 segments, one (idx=3) with very low noise."""
    segments = []
    for i in range(10):
        base = make_base_motion(N_FRAMES)
        sigma = 0.005 if i == 3 else 0.02
        noise = rng.normal(0, sigma, base.shape)
        segments.append(make_segment(base + noise, i))
    return segments


def generate_varied_reps(rng: np.random.Generator) -> list[dict]:
    """VARIED_REPS: 8 segments, 6 with amplitude 1.0, 2 with amplitude 0.7."""
    segments = []
    for i in range(8):
        amp = 0.7 if i in (2, 5) else 1.0
        base = make_base_motion(N_FRAMES, amplitude=amp)
        noise = rng.normal(0, 0.02, base.shape)
        segments.append(make_segment(base + noise, i))
    return segments


def generate_noisy_reps(rng: np.random.Generator) -> list[dict]:
    """NOISY_REPS: 10 segments, noise levels from 0.01 to 0.15."""
    segments = []
    noise_levels = np.linspace(0.01, 0.15, 10)
    for i, sigma in enumerate(noise_levels):
        base = make_base_motion(N_FRAMES)
        noise = rng.normal(0, sigma, base.shape)
        segments.append(make_segment(base + noise, i))
    return segments


# --- Representative selection methods ---

def method_mean_trajectory(segments: list[dict]) -> np.ndarray:
    """A) MEAN TRAJECTORY: average of all resampled segments."""
    resampled = [_resample_segment(seg["features"], RESAMPLE_LEN) for seg in segments]
    return np.mean(resampled, axis=0)


def method_medoid(segments: list[dict]) -> tuple[int, np.ndarray]:
    """B) MEDOID: segment with minimum total DTW distance to all others."""
    n = len(segments)
    total_dists = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                total_dists[i] += _segment_distance(segments[i], segments[j])
    best = int(np.argmin(total_dists))
    return best, _resample_segment(segments[best]["features"], RESAMPLE_LEN)


def method_best_fit_to_mean(segments: list[dict]) -> tuple[int, np.ndarray]:
    """C) BEST FIT TO MEAN: segment closest (L2) to the mean trajectory."""
    resampled = [_resample_segment(seg["features"], RESAMPLE_LEN) for seg in segments]
    mean_traj = np.mean(resampled, axis=0)
    dists = [float(np.linalg.norm(r - mean_traj)) for r in resampled]
    best = int(np.argmin(dists))
    return best, resampled[best]


def method_smoothest(segments: list[dict]) -> tuple[int, np.ndarray]:
    """D) SMOOTHEST: segment with lowest mean absolute jerk."""
    jerks = []
    for seg in segments:
        jerk = np.diff(seg["features"], n=3, axis=0)
        jerks.append(float(np.mean(np.abs(jerk))))
    best = int(np.argmin(jerks))
    return best, _resample_segment(segments[best]["features"], RESAMPLE_LEN)


# --- Evaluation ---

def compute_mean_dtw_to_all(representative: np.ndarray, segments: list[dict]) -> float:
    """Mean DTW distance from representative to all segments."""
    rep_seg = {"features": representative}
    dists = [_segment_distance(rep_seg, seg) for seg in segments]
    return float(np.mean(dists))


def compute_smoothness(trajectory: np.ndarray) -> float:
    """Mean absolute jerk of a trajectory (lower = smoother)."""
    if trajectory.shape[0] < 4:
        return 0.0
    jerk = np.diff(trajectory, n=3, axis=0)
    return float(np.mean(np.abs(jerk)))


def evaluate_dataset(name: str, segments: list[dict]):
    """Run all 4 methods on a dataset and print comparison table."""
    print(f"\n{'='*70}")
    print(f"  Dataset: {name} ({len(segments)} segments, {N_FRAMES} frames each)")
    print(f"{'='*70}")

    results = []

    # A) Mean trajectory
    mean_repr = method_mean_trajectory(segments)
    dtw_a = compute_mean_dtw_to_all(mean_repr, segments)
    smooth_a = compute_smoothness(mean_repr)
    results.append(("A) Mean Trajectory", dtw_a, smooth_a, "synthetic", "-"))

    # B) Medoid
    idx_b, repr_b = method_medoid(segments)
    dtw_b = compute_mean_dtw_to_all(repr_b, segments)
    smooth_b = compute_smoothness(repr_b)
    results.append(("B) Medoid", dtw_b, smooth_b, "real", f"seg {idx_b}"))

    # C) Best fit to mean
    idx_c, repr_c = method_best_fit_to_mean(segments)
    dtw_c = compute_mean_dtw_to_all(repr_c, segments)
    smooth_c = compute_smoothness(repr_c)
    results.append(("C) Best-Fit-to-Mean", dtw_c, smooth_c, "real", f"seg {idx_c}"))

    # D) Smoothest
    idx_d, repr_d = method_smoothest(segments)
    dtw_d = compute_mean_dtw_to_all(repr_d, segments)
    smooth_d = compute_smoothness(repr_d)
    results.append(("D) Smoothest", dtw_d, smooth_d, "real", f"seg {idx_d}"))

    # Print table
    print(f"\n  {'Method':<22} {'DTW dist':>10} {'Smoothness':>12} {'Type':>10} {'Selected':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    best_dtw = min(r[1] for r in results)
    best_smooth = min(r[2] for r in results)

    for method, dtw, smooth, typ, sel in results:
        dtw_marker = " *" if dtw == best_dtw else ""
        smooth_marker = " *" if smooth == best_smooth else ""
        print(f"  {method:<22} {dtw:>9.4f}{dtw_marker} {smooth:>11.6f}{smooth_marker} {typ:>10} {sel:>10}")

    print(f"\n  (* = best in column)")


def main():
    rng = np.random.default_rng(42)

    print("Representative Segment Selection Experiment")
    print("=" * 70)
    print(f"Feature dim: {FEAT_DIM}D, Frames/seg: {N_FRAMES}, Resample: {RESAMPLE_LEN}")

    # Dataset 1: Clean reps
    clean = generate_clean_reps(rng)
    evaluate_dataset("CLEAN_REPS (10 segs, seg 3 has lowest noise sigma=0.005)", clean)

    # Dataset 2: Varied amplitude
    varied = generate_varied_reps(rng)
    evaluate_dataset("VARIED_REPS (8 segs, 2 outliers with amp=0.7)", varied)

    # Dataset 3: Noisy reps
    noisy = generate_noisy_reps(rng)
    evaluate_dataset("NOISY_REPS (10 segs, noise 0.01 to 0.15)", noisy)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print("""
  Mean Trajectory: Synthetic average, always central but may not be physically
  plausible (blurs timing details). Smoothest by construction.

  Medoid: Real segment closest to all others (DTW). Preserves natural timing
  and is always a real motion. Good all-around choice.

  Best-Fit-to-Mean: Real segment closest to the mean. Similar to medoid but
  uses L2 on resampled data instead of DTW across all pairs. Faster to compute.

  Smoothest: Real segment with lowest jerk. Best for correction reference
  since it represents the cleanest execution. May not be most central.

  RECOMMENDATION: Use Best-Fit-to-Mean as the primary method:
  - Always returns a real segment (physically plausible)
  - Nearly as central as the medoid
  - Much faster to compute (O(n) vs O(n^2) pairwise DTW)
  - Good smoothness when the cluster is consistent
  For additional quality, use Smoothest as a tiebreaker or secondary filter.
""")


if __name__ == "__main__":
    main()
