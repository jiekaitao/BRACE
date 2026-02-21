"""Detect, cluster, and analyze repeated motion patterns from pose sequences."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

from .pose import FEATURE_INDICES, LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER


def normalize_frame(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """SRP-normalize a single frame's landmarks to body frame (2D).

    Uses hip center as origin, hip width as scale, hip-shoulder axes for rotation.

    Returns (33, 2) normalized xy or None if landmarks are bad.
    """
    xy = landmarks_xyzv[:, :2].astype(np.float64)
    vis = landmarks_xyzv[:, 3]

    # Check key joints are visible
    key_joints = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
    if any(vis[j] < 0.3 for j in key_joints):
        return None

    lhip = xy[LEFT_HIP]
    rhip = xy[RIGHT_HIP]
    lsh = xy[LEFT_SHOULDER]
    rsh = xy[RIGHT_SHOULDER]

    pelvis = (lhip + rhip) * 0.5
    hip_vec = lhip - rhip
    hip_width = float(np.linalg.norm(hip_vec))
    if hip_width < 1e-4:
        return None

    # Body frame axes
    x_axis = hip_vec / hip_width
    shoulder_center = (lsh + rsh) * 0.5
    up_guess = shoulder_center - pelvis
    up_proj = up_guess - np.dot(up_guess, x_axis) * x_axis
    if np.linalg.norm(up_proj) < 1e-6:
        up_proj = np.array([-x_axis[1], x_axis[0]])
    y_axis = up_proj / np.linalg.norm(up_proj)

    # Transform
    rel = xy - pelvis
    out = np.zeros((xy.shape[0], 2), dtype=np.float32)
    out[:, 0] = (rel @ x_axis) / hip_width
    out[:, 1] = (rel @ y_axis) / hip_width
    return out


# Anthropometric limb lengths in hip-width units (population averages)
_LIMB_LENGTHS_HW = {
    (11, 13): 1.20,  # shoulder -> elbow (upper arm)
    (12, 14): 1.20,
    (13, 15): 1.05,  # elbow -> wrist (forearm)
    (14, 16): 1.05,
    (23, 25): 1.60,  # hip -> knee (thigh)
    (24, 26): 1.60,
    (25, 27): 1.50,  # knee -> ankle (shin)
    (26, 28): 1.50,
}

# Kinematic chain order: process parent -> child so Z propagates correctly
_CHAIN_ORDER = [
    (11, 13), (12, 14),  # shoulders -> elbows
    (13, 15), (14, 16),  # elbows -> wrists
    (23, 25), (24, 26),  # hips -> knees
    (25, 27), (26, 28),  # knees -> ankles
]


def normalize_frame_visual(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """Position+scale normalize a frame for visualization (NO rotation removal).

    Unlike full SRP, this preserves the body's orientation relative to the camera.
    Used for the skeleton graph display so a person leaning on a rowing machine
    still appears leaned, not standing upright.

    Returns (33, 2) normalized xy or None if landmarks are bad.
    """
    xy = landmarks_xyzv[:, :2].astype(np.float64)
    vis = landmarks_xyzv[:, 3]

    key_joints = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
    if any(vis[j] < 0.3 for j in key_joints):
        return None

    lhip = xy[LEFT_HIP]
    rhip = xy[RIGHT_HIP]
    pelvis = (lhip + rhip) * 0.5
    hip_width = float(np.linalg.norm(lhip - rhip))
    if hip_width < 1e-4:
        return None

    rel = xy - pelvis
    return (rel / hip_width).astype(np.float32)


def normalize_frame_3d_visual(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """Position+scale normalize with Z estimation, preserving body orientation.

    Like normalize_frame_3d but without rotation to body frame.
    Used for skeleton visualization.

    Returns (33, 3) array or None.
    """
    norm2d = normalize_frame_visual(landmarks_xyzv)
    if norm2d is None:
        return None

    out = np.zeros((33, 3), dtype=np.float32)
    out[:, :2] = norm2d

    for parent, child in _CHAIN_ORDER:
        expected = _LIMB_LENGTHS_HW.get((parent, child), 0)
        if expected <= 0:
            continue
        dx = out[child, 0] - out[parent, 0]
        dy = out[child, 1] - out[parent, 1]
        d2d_sq = dx * dx + dy * dy
        exp_sq = expected * expected
        if d2d_sq < exp_sq:
            dz = np.sqrt(exp_sq - d2d_sq)
            out[child, 2] = out[parent, 2] - dz * 0.5
        else:
            out[child, 2] = out[parent, 2]

    return out


def normalize_frame_3d_visual_real(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """Position+scale normalize real 3D data, preserving body orientation.

    For RTMW3D data with real depth values.

    Returns (33, 3) array or None.
    """
    xyz = landmarks_xyzv[:, :3].astype(np.float64)
    vis = landmarks_xyzv[:, 3]

    key_joints = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
    if any(vis[j] < 0.3 for j in key_joints):
        return None

    lhip = xyz[LEFT_HIP]
    rhip = xyz[RIGHT_HIP]
    pelvis = (lhip + rhip) * 0.5
    hip_width = float(np.linalg.norm(lhip - rhip))
    if hip_width < 1e-4:
        return None

    rel = xyz - pelvis
    return (rel / hip_width).astype(np.float32)


def normalize_frame_3d_real(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """SRP-normalize a frame with real 3D depth from RTMW3D.

    Full 3D Gram-Schmidt body frame using real XYZ coordinates.
    Same math as srp.py:normalize_to_body_frame_3d() but with MediaPipe indices.

    Returns (33, 3) array [x, y, z] in hip-width units, or None.
    """
    xyz = landmarks_xyzv[:, :3].astype(np.float64)
    vis = landmarks_xyzv[:, 3]

    # Check key joints are visible
    key_joints = [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]
    if any(vis[j] < 0.3 for j in key_joints):
        return None

    lhip = xyz[LEFT_HIP]    # (3,)
    rhip = xyz[RIGHT_HIP]
    lsh = xyz[LEFT_SHOULDER]
    rsh = xyz[RIGHT_SHOULDER]

    pelvis = (lhip + rhip) * 0.5
    hip_vec = lhip - rhip
    hip_width = float(np.linalg.norm(hip_vec))
    if hip_width < 1e-4:
        return None

    # X-axis: hip direction
    x_axis = hip_vec / hip_width

    # Y-axis: Gram-Schmidt orthogonalize shoulder-pelvis against x-axis
    shoulder_center = (lsh + rsh) * 0.5
    up_guess = shoulder_center - pelvis
    up_proj = up_guess - np.dot(up_guess, x_axis) * x_axis
    up_norm = float(np.linalg.norm(up_proj))
    if up_norm < 1e-6:
        # Fallback: use world up direction
        up_proj = np.array([0.0, 1.0, 0.0])
        up_proj = up_proj - np.dot(up_proj, x_axis) * x_axis
        up_norm = float(np.linalg.norm(up_proj))
        if up_norm < 1e-6:
            return None
    y_axis = up_proj / up_norm

    # Z-axis: cross product for right-handed frame
    z_axis = np.cross(x_axis, y_axis)

    # Transform all joints into body frame
    rel = xyz - pelvis  # (33, 3)
    out = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    out[:, 0] = (rel @ x_axis) / hip_width
    out[:, 1] = (rel @ y_axis) / hip_width
    out[:, 2] = (rel @ z_axis) / hip_width
    return out


def normalize_frame_3d(landmarks_xyzv: np.ndarray) -> np.ndarray | None:
    """SRP-normalize a frame and estimate Z depth from limb foreshortening.

    Returns (33, 3) array [x, y, z] in hip-width units, or None.
    """
    norm2d = normalize_frame(landmarks_xyzv)
    if norm2d is None:
        return None

    out = np.zeros((33, 3), dtype=np.float32)
    out[:, :2] = norm2d

    for parent, child in _CHAIN_ORDER:
        expected = _LIMB_LENGTHS_HW.get((parent, child), 0)
        if expected <= 0:
            continue
        dx = out[child, 0] - out[parent, 0]
        dy = out[child, 1] - out[parent, 1]
        d2d_sq = dx * dx + dy * dy
        exp_sq = expected * expected
        if d2d_sq < exp_sq:
            dz = np.sqrt(exp_sq - d2d_sq)
            # Sign: extremities tend forward (negative Z = closer to camera)
            out[child, 2] = out[parent, 2] - dz * 0.5  # damped
        else:
            out[child, 2] = out[parent, 2]

    return out


def feature_vector(norm_frame: np.ndarray) -> np.ndarray:
    """Extract feature vector from normalized frame. Returns (28,) float32."""
    return norm_frame[FEATURE_INDICES].reshape(-1).astype(np.float32)


def compute_feature_trajectory(landmarks_list: list[np.ndarray | None]) -> tuple[np.ndarray, list[int]]:
    """Compute SRP-normalized feature vectors for all valid frames.

    Returns:
        features: (N_valid, 28) feature matrix.
        valid_indices: which original frame indices are valid.
    """
    features = []
    valid_indices = []

    for i, lm in enumerate(landmarks_list):
        if lm is None:
            continue
        norm = normalize_frame(lm)
        if norm is None:
            continue
        feat = feature_vector(norm)
        if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
            continue
        features.append(feat)
        valid_indices.append(i)

    if not features:
        return np.zeros((0, len(FEATURE_INDICES) * 2)), []

    return np.stack(features, axis=0), valid_indices


def compute_self_similarity(features: np.ndarray, window: int = 15) -> np.ndarray:
    """Compute windowed self-similarity curve.

    For each frame, compare the window around it to all other windows.
    Returns a 1D curve where peaks indicate the start of repeated motions.
    """
    n = features.shape[0]
    if n < window * 2:
        return np.zeros(n)

    # Compute windowed feature means for efficiency
    half_w = window // 2
    similarity = np.zeros(n)

    for i in range(half_w, n - half_w):
        seg_i = features[i - half_w:i + half_w + 1]  # (window, D)
        seg_mean = seg_i.mean(axis=0)

        # Compare to all other positions
        dists = []
        for j in range(half_w, n - half_w):
            if abs(i - j) < window:
                continue
            seg_j = features[j - half_w:j + half_w + 1]
            seg_j_mean = seg_j.mean(axis=0)
            d = float(np.linalg.norm(seg_mean - seg_j_mean))
            dists.append(d)

        if dists:
            # Similarity = inverse of min distance to any other window
            min_dist = min(dists)
            similarity[i] = 1.0 / (min_dist + 0.01)

    return similarity


def detect_motion_boundaries(
    features: np.ndarray,
    fps: float = 24.0,
    min_segment_sec: float = 2.0,
    velocity_percentile: float = 85,
    use_savgol: bool = False,
) -> list[int]:
    """Detect motion segment boundaries using velocity-based segmentation.

    A "boundary" is where the body pauses or transitions between movements
    (low velocity points). Uses adaptive prominence based on the overall
    motion amplitude to avoid over-segmentation.

    Args:
        features: (N, D) feature matrix.
        fps: Frames per second.
        min_segment_sec: Minimum segment duration.
        velocity_percentile: Unused (kept for backwards compat).
        use_savgol: If True, use Savitzky-Golay derivative for velocity
            instead of simple finite difference. Gives cleaner peaks with
            better SNR for noisy data.

    Returns list of frame indices where segments start.
    """
    n = features.shape[0]
    if n < 5:
        return [0]

    # Compute velocity
    if use_savgol and n >= 9:
        # Savitzky-Golay: window_length=7, polyorder=2, deriv=1
        window_length = min(7, n if n % 2 == 1 else n - 1)
        if window_length >= 5:
            sg_deriv = savgol_filter(features, window_length, 2, deriv=1, axis=0)
            velocity = np.linalg.norm(sg_deriv, axis=1)
        else:
            # Fallback to finite difference
            velocity = np.zeros(n)
            for i in range(1, n):
                velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))
    else:
        # Standard finite difference
        velocity = np.zeros(n)
        for i in range(1, n):
            velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    # Smoothing window scales with min_segment_sec to avoid splitting within reps
    kernel_size = max(5, int(fps * min_segment_sec * 0.45))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocity, kernel, mode="same")

    # Adaptive prominence: use median velocity as base, require significant dips
    min_frames = max(int(fps * min_segment_sec), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return [0]

    median_vel = float(np.median(positive_vals))
    # Prominence must be at least 100% of median velocity for a real boundary
    min_prominence = median_vel * 1.0

    # Invert to find peaks (which are valleys in velocity)
    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    # Always include frame 0 and filter out too-close boundaries
    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    return boundaries


def _resample_segment(features: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a segment's feature trajectory to a fixed length."""
    src_x = np.linspace(0, 1, features.shape[0])
    tgt_x = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, features.shape[1]), dtype=np.float32)
    for d in range(features.shape[1]):
        out[:, d] = np.interp(tgt_x, src_x, features[:, d])
    return out


def segment_motions(
    features: np.ndarray,
    valid_indices: list[int],
    fps: float = 24.0,
    min_segment_sec: float = 2.0,
    forced_boundaries: list[int] | None = None,
) -> list[dict]:
    """Segment the feature trajectory into individual motion segments.

    Args:
        forced_boundaries: Additional boundary indices (e.g. from video loop
            points) that must always appear as segment splits, regardless of
            velocity.  These prevent chimera segments that straddle a
            discontinuity.

    Returns list of dicts: {start_frame, end_frame, start_valid, end_valid, features}
    """
    boundaries = detect_motion_boundaries(features, fps, min_segment_sec)

    # Merge forced boundaries (e.g. video loop points) into the list
    if forced_boundaries:
        boundary_set = set(boundaries)
        for fb in forced_boundaries:
            if 0 <= fb < features.shape[0]:
                boundary_set.add(fb)
        boundaries = sorted(boundary_set)

    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else features.shape[0]

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

    return segments


def _segment_distance(seg_a: dict, seg_b: dict, resample_len: int = 30) -> float:
    """Compute distance between two segments using mean pose + spectral content.

    Phase-invariant: the power spectrum captures motion frequency content
    without depending on where in the exercise cycle the segment starts.
    Combined with mean feature distance for overall pose similarity.

    Normalized by sqrt(feat_dim) so threshold scale is invariant to 2D/3D.
    """
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]

    # Mean pose distance (different exercises have different average poses)
    mean_dist = float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0)))

    # Power spectrum distance (phase-invariant, captures motion frequency/amplitude)
    # Normalize FFT by N so magnitudes are amplitude-scale, skip DC (bin 0)
    spec_a = np.abs(np.fft.rfft(ra, axis=0))[1:] / resample_len
    spec_b = np.abs(np.fft.rfft(rb, axis=0))[1:] / resample_len
    spec_dist = float(np.linalg.norm(spec_a - spec_b))

    return (mean_dist + spec_dist) / np.sqrt(feat_dim)


def cluster_segments(
    segments: list[dict],
    distance_threshold: float = 2.0,
    fps: float = 30.0,
) -> list[dict]:
    """Cluster motion segments using agglomerative clustering on spectral distance.

    Uses mean pose + FFT power spectrum distance with average linkage, then
    merges adjacent same-cluster segments and absorbs small clusters into their
    nearest large neighbor.

    Returns segments with added 'cluster' field.
    """
    if not segments:
        return segments

    n = len(segments)

    if n == 1:
        segments[0]["cluster"] = 0
        return segments

    # Build pairwise distance matrix using spectral + mean pose comparison
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Average linkage: considers mean distance between all pairs in two clusters,
    # avoids chaining artifacts from single linkage
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")

    # Cut the dendrogram at distance_threshold
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Relabel to 0-indexed contiguous
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}

    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Merge adjacent segments with the same cluster ID
    segments = _merge_adjacent_clusters(segments)

    # Absorb small clusters into nearest large neighbor
    segments = _merge_small_clusters(segments, fps=fps)

    return segments


def _merge_adjacent_clusters(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments that share the same cluster ID.

    This reduces noise from over-segmentation: if boundary detection split
    a single motion into 3 parts that all cluster together, merge them back.
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.get("cluster") == prev.get("cluster"):
            # Merge: extend the previous segment
            combined_features = np.vstack([prev["features"], seg["features"]])
            if "end_valid" in seg:
                prev["end_valid"] = seg["end_valid"]
            if "end_frame" in seg:
                prev["end_frame"] = seg["end_frame"]
            prev["features"] = combined_features
            prev["mean_feature"] = combined_features.mean(axis=0)
        else:
            merged.append(seg)

    return merged


def _merge_small_clusters(
    segments: list[dict],
    min_segments: int = 2,
    min_seconds: float = 3.0,
    fps: float = 30.0,
) -> list[dict]:
    """Absorb small clusters into their nearest large neighbor.

    A cluster is "small" if it has fewer than min_segments segments AND less
    than min_seconds total duration.  Small clusters are merged into the
    nearest large cluster by centroid distance.
    """
    if len(segments) <= 1:
        return segments

    # Gather per-cluster info
    cluster_info: dict[int, dict] = {}
    for seg in segments:
        cid = seg["cluster"]
        if cid not in cluster_info:
            cluster_info[cid] = {"count": 0, "total_frames": 0, "segments": []}
        cluster_info[cid]["count"] += 1
        cluster_info[cid]["total_frames"] += seg.get("end_frame", seg["features"].shape[0]) - seg.get("start_frame", 0)
        cluster_info[cid]["segments"].append(seg)

    small_cids = set()
    for cid, info in cluster_info.items():
        duration_sec = info["total_frames"] / max(fps, 1)
        if info["count"] < min_segments and duration_sec < min_seconds:
            small_cids.add(cid)

    if not small_cids or len(small_cids) >= len(cluster_info):
        return segments

    large_cids = [cid for cid in cluster_info if cid not in small_cids]
    large_centroids = {}
    for cid in large_cids:
        feats = [s["mean_feature"] for s in cluster_info[cid]["segments"]]
        large_centroids[cid] = np.mean(feats, axis=0)

    merge_map = {}
    for small_cid in small_cids:
        small_centroid = np.mean(
            [s["mean_feature"] for s in cluster_info[small_cid]["segments"]], axis=0
        )
        best_cid, best_dist = None, float("inf")
        for large_cid, centroid in large_centroids.items():
            d = float(np.linalg.norm(small_centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_cid = large_cid
        if best_cid is not None:
            merge_map[small_cid] = best_cid

    for seg in segments:
        if seg["cluster"] in merge_map:
            seg["cluster"] = merge_map[seg["cluster"]]

    # Re-merge adjacent same-cluster segments
    segments = _merge_adjacent_clusters(segments)

    # Relabel to 0-indexed contiguous
    unique = sorted(set(s["cluster"] for s in segments))
    remap = {old: new for new, old in enumerate(unique)}
    for seg in segments:
        seg["cluster"] = remap[seg["cluster"]]

    return segments


def analyze_consistency(segments: list[dict]) -> dict:
    """Analyze consistency within each motion cluster.

    For each cluster, compute a baseline from the mean and score each
    repetition for deviation. Returns per-cluster analysis.

    Returns:
        dict mapping cluster_id -> {
            count, mean_trajectory, scores, baseline_std,
            anomaly_segments (indices of high-deviation reps)
        }
    """
    clusters = defaultdict(list)
    for i, seg in enumerate(segments):
        clusters[seg.get("cluster", 0)].append((i, seg))

    analysis = {}
    for c_id, members in clusters.items():
        if len(members) < 2:
            # Need at least 2 repetitions to analyze consistency
            analysis[c_id] = {
                "count": len(members),
                "scores": [0.0] * len(members),
                "anomaly_segments": [],
                "mean_score": 0.0,
            }
            continue

        # Resample all segments to same length for comparison
        target_len = int(np.median([m[1]["features"].shape[0] for m in members]))
        target_len = max(target_len, 5)

        resampled = []
        for _, seg in members:
            feat = seg["features"]
            src_x = np.linspace(0, 1, feat.shape[0])
            tgt_x = np.linspace(0, 1, target_len)
            r = np.zeros((target_len, feat.shape[1]), dtype=np.float32)
            for d in range(feat.shape[1]):
                r[:, d] = np.interp(tgt_x, src_x, feat[:, d])
            resampled.append(r)

        resampled = np.stack(resampled, axis=0)  # (n_members, target_len, D)
        mean_traj = np.mean(resampled, axis=0)
        std_traj = np.std(resampled, axis=0)
        std_traj = np.maximum(std_traj, 1e-6)

        # Score each repetition
        scores = []
        for r in resampled:
            deviation = np.abs(r - mean_traj) / std_traj
            rms = float(np.sqrt(np.mean(deviation ** 2)))
            scores.append(rms)

        # Flag anomalies: > 1.5 std above mean score
        mean_score = float(np.mean(scores))
        score_std = float(np.std(scores)) if len(scores) > 1 else 1.0
        score_std = max(score_std, 0.3)  # prevent tight distributions from flagging noise
        threshold = mean_score + 1.5 * score_std

        anomaly_indices = [members[i][0] for i, s in enumerate(scores) if s > threshold]

        # Don't flag single outlier in large consistent set
        if len(members) >= 5 and len(anomaly_indices) <= 1:
            anomaly_indices = []

        analysis[c_id] = {
            "count": len(members),
            "scores": scores,
            "anomaly_segments": anomaly_indices,
            "mean_score": mean_score,
            "threshold": threshold,
        }

        # Write scores back to segments
        for i, (seg_idx, seg) in enumerate(members):
            seg["consistency_score"] = scores[i]
            seg["is_anomaly"] = seg_idx in anomaly_indices

    return analysis
