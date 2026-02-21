"""Experiment: Frame-to-Reference Alignment Methods for Real-Time Correction.

Given a current live frame and a representative motion segment, find which frame
in the representative best corresponds to the current moment. This enables
real-time correction arrows (showing the difference between current pose and
the ideal pose at the same phase of motion).

Test scenario:
- Representative: 60 frames, one full sin cycle on knee joints
- Live stream: 80 frames, same motion but 33% slower, slight noise and amplitude diff
- Ground truth: live_frame[i] maps to representative_frame[i * 60/80]

Methods compared:
  A) Phase Detection - velocity-based boundary detection + phase interpolation
  B) Nearest Neighbor - brute-force L2 distance to all representative frames
  C) Sequential Pointer - online DTW-lite with forward-only pointer
  D) Sliding Window DTW - fastdtw on recent buffer vs representative window
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def generate_representative(n_frames=60, n_features=28):
    """60-frame segment: sin wave on knee joints, one full cycle."""
    rep = np.zeros((n_frames, n_features))
    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    # 14 joints x 2 (x, y). Knee joints are indices 8 and 9 in the 14-joint list.
    # In 28D: joint k -> features [2k, 2k+1]
    # Knee L = joint 8 -> features 16, 17
    # Knee R = joint 9 -> features 18, 19
    for knee_feat in [16, 17, 18, 19]:
        rep[:, knee_feat] = np.sin(t) * 1.0
    # Add small baseline motion on other joints so they aren't all zero
    for j in range(n_features):
        if j not in [16, 17, 18, 19]:
            rep[:, j] = np.sin(t) * 0.1
    return rep


def generate_live_stream(n_frames=80, n_features=28, noise_sigma=0.03, amplitude=0.95):
    """80-frame stream: same motion, 33% slower, noise, slight amplitude diff."""
    live = np.zeros((n_frames, n_features))
    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    for knee_feat in [16, 17, 18, 19]:
        live[:, knee_feat] = np.sin(t) * amplitude
    for j in range(n_features):
        if j not in [16, 17, 18, 19]:
            live[:, j] = np.sin(t) * 0.1 * amplitude
    # Add gaussian noise
    live += np.random.default_rng(42).normal(0, noise_sigma, live.shape)
    return live


def ground_truth_mapping(n_live=80, n_rep=60):
    """Ground truth: live frame i -> representative frame int(i * 60/80), clamped."""
    return np.array([min(int(i * n_rep / n_live), n_rep - 1) for i in range(n_live)])


# ---------------------------------------------------------------------------
# Method A: Phase Detection
# ---------------------------------------------------------------------------

def align_phase_detection(live, representative):
    """Velocity-based phase detection.

    Track velocity magnitude per frame. Detect motion boundaries where velocity
    crosses a threshold (high -> low). Phase = frames_since_boundary / expected_length.
    """
    n_rep = len(representative)
    n_live = len(live)
    predictions = np.zeros(n_live, dtype=int)

    # Compute velocity magnitudes for live frames
    velocities = np.zeros(n_live)
    for i in range(1, n_live):
        velocities[i] = np.linalg.norm(live[i] - live[i - 1])

    # Threshold: median of representative velocities
    rep_velocities = np.array([
        np.linalg.norm(representative[i] - representative[i - 1])
        for i in range(1, n_rep)
    ])
    threshold = np.median(rep_velocities) * 0.3

    # Detect boundaries: velocity drops below threshold (motion pauses)
    expected_rep_length = n_rep  # Use representative length as expected cycle length
    boundary_frame = 0

    for i in range(n_live):
        # Detect boundary: velocity was above threshold and now below
        if i > 0 and velocities[i - 1] > threshold and velocities[i] <= threshold:
            # Only treat as boundary if enough frames have passed
            if i - boundary_frame > expected_rep_length * 0.5:
                boundary_frame = i

        frames_since = i - boundary_frame
        phase = frames_since / expected_rep_length
        phase = min(phase, 1.0)
        predictions[i] = min(int(phase * n_rep), n_rep - 1)

    return predictions


# ---------------------------------------------------------------------------
# Method B: Nearest Neighbor
# ---------------------------------------------------------------------------

def align_nearest_neighbor(live, representative):
    """Brute-force L2 distance to every frame in representative, pick argmin."""
    n_live = len(live)
    n_rep = len(representative)
    predictions = np.zeros(n_live, dtype=int)

    for i in range(n_live):
        dists = np.linalg.norm(representative - live[i], axis=1)
        predictions[i] = np.argmin(dists)

    return predictions


# ---------------------------------------------------------------------------
# Method C: Sequential Pointer (Online DTW-lite)
# ---------------------------------------------------------------------------

def align_sequential_pointer(live, representative):
    """Forward-only pointer that advances to the best match within a look-ahead window."""
    n_live = len(live)
    n_rep = len(representative)
    predictions = np.zeros(n_live, dtype=int)
    ptr = 0

    for i in range(n_live):
        # Compare to representative[ptr] through representative[min(ptr+5, n_rep-1)]
        end = min(ptr + 6, n_rep)
        candidates = representative[ptr:end]
        dists = np.linalg.norm(candidates - live[i], axis=1)
        best_offset = np.argmin(dists)
        ptr = ptr + best_offset  # Never goes backward

        # When pointer reaches end, reset to 0
        if ptr >= n_rep - 1:
            ptr = 0

        predictions[i] = min(ptr, n_rep - 1)

    return predictions


# ---------------------------------------------------------------------------
# Method D: Sliding Window DTW
# ---------------------------------------------------------------------------

def align_sliding_window_dtw(live, representative):
    """Keep buffer of last 15 frames, run fastdtw against corresponding representative window."""
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    n_live = len(live)
    n_rep = len(representative)
    window_size = 15
    predictions = np.zeros(n_live, dtype=int)

    buffer = []

    for i in range(n_live):
        buffer.append(live[i])
        if len(buffer) > window_size:
            buffer.pop(0)

        buf_arr = np.array(buffer)
        buf_len = len(buf_arr)

        # Estimate where we are in the representative based on progress
        estimated_pos = int(i * n_rep / n_live)
        # Window in representative centered around estimated position
        ref_start = max(0, estimated_pos - window_size)
        ref_end = min(n_rep, estimated_pos + window_size)
        ref_window = representative[ref_start:ref_end]

        if len(ref_window) < 2 or buf_len < 2:
            predictions[i] = estimated_pos
            continue

        # Run fastdtw
        _, path = fastdtw(buf_arr, ref_window, dist=euclidean)

        # Find what the last live frame maps to
        last_live_idx = buf_len - 1
        mapped_ref_indices = [p[1] for p in path if p[0] == last_live_idx]
        if mapped_ref_indices:
            ref_idx = mapped_ref_indices[-1]  # Take the last mapping
            predictions[i] = min(ref_start + ref_idx, n_rep - 1)
        else:
            predictions[i] = estimated_pos

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions, gt):
    """Compute MAE, smoothness, max jump, and return as dict."""
    mae = np.mean(np.abs(predictions.astype(float) - gt.astype(float)))
    diffs = np.abs(np.diff(predictions.astype(float)))
    smoothness = np.mean(diffs) if len(diffs) > 0 else 0.0
    max_jump = np.max(diffs) if len(diffs) > 0 else 0.0
    return {
        "mae": mae,
        "smoothness": smoothness,
        "max_jump": int(max_jump),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    representative = generate_representative()
    live = generate_live_stream()
    gt = ground_truth_mapping()

    methods = {
        "A) Phase Detection": align_phase_detection,
        "B) Nearest Neighbor": align_nearest_neighbor,
        "C) Sequential Pointer": align_sequential_pointer,
        "D) Sliding Window DTW": align_sliding_window_dtw,
    }

    results = {}
    all_predictions = {}

    for name, method in methods.items():
        t0 = time.perf_counter()
        preds = method(live, representative)
        elapsed = time.perf_counter() - t0
        metrics = compute_metrics(preds, gt)
        metrics["time_per_frame_us"] = (elapsed / 80) * 1e6
        metrics["total_time_ms"] = elapsed * 1e3
        results[name] = metrics
        all_predictions[name] = preds

    # Print comparison table
    print("=" * 90)
    print("FRAME ALIGNMENT EXPERIMENT")
    print("=" * 90)
    print(f"Representative: 60 frames, Live: 80 frames (33% slower, noise=0.03, amp=0.95)")
    print(f"Ground truth rate: 0.75 representative frames per live frame")
    print()
    print(f"{'Method':<28} {'MAE (frames)':>13} {'Smoothness':>11} {'Max Jump':>9} {'us/frame':>10}")
    print("-" * 75)
    for name, m in results.items():
        print(
            f"{name:<28} {m['mae']:>13.2f} {m['smoothness']:>11.2f} "
            f"{m['max_jump']:>9d} {m['time_per_frame_us']:>10.1f}"
        )
    print("-" * 75)
    print(f"{'(ideal)':28} {'0.00':>13} {'0.75':>11} {'1':>9} {'--':>10}")
    print()

    # Print first 20 mappings
    print("FIRST 20 FRAME MAPPINGS (live -> representative)")
    print("-" * 90)
    header = f"{'Live':>5} {'GT':>4}"
    for name in methods:
        short = name.split(")")[0] + ")"
        header += f" {short:>8}"
    print(header)
    print("-" * 90)

    for i in range(20):
        row = f"{i:>5} {gt[i]:>4}"
        for name in methods:
            pred = all_predictions[name][i]
            marker = " " if abs(pred - gt[i]) <= 2 else "*"
            row += f" {pred:>7d}{marker}"
        print(row)

    print()
    print("* = error > 2 frames from ground truth")
    print()

    # Summary
    best_method = min(results, key=lambda k: results[k]["mae"])
    smoothest = min(results, key=lambda k: abs(results[k]["smoothness"] - 0.75))
    fastest = min(results, key=lambda k: results[k]["time_per_frame_us"])
    print("SUMMARY:")
    print(f"  Lowest MAE:       {best_method} ({results[best_method]['mae']:.2f} frames)")
    print(f"  Best smoothness:  {smoothest} (rate={results[smoothest]['smoothness']:.2f}, ideal=0.75)")
    print(f"  Fastest:          {fastest} ({results[fastest]['time_per_frame_us']:.1f} us/frame)")


if __name__ == "__main__":
    main()
