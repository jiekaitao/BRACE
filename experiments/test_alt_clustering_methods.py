#!/usr/bin/env python3
"""Test alternative clustering algorithms on all 5 demo videos.

Extracts YOLO-pose features, then tests 20+ strategies covering:
- Alternative clustering algorithms (HDBSCAN, Spectral, GMM, etc.)
- Alternative distance metrics on agglomerative (DTW, cosine, correlation)
- Different linkage methods with tuned thresholds
- Alternative feature representations (raw, FFT-only, wavelet, statistical)

Key constraint: basketball_solo.mp4 MUST get 2 clusters.
"""

import sys
import time
import warnings

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import (
    SpectralClustering,
    AffinityPropagation,
    OPTICS,
    MeanShift,
)
from sklearn.mixture import GaussianMixture
import hdbscan

from brace.core.motion_segments import (
    normalize_frame,
    segment_motions,
    cluster_segments,
    _segment_distance,
    _resample_segment,
    detect_motion_boundaries,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

# ─── Video paths ───────────────────────────────────────────────────────────────
VIDEOS = {
    "basketball": "/mnt/Data/GitHub/BRACE/data/sports_videos/basketball_solo.mp4",
    "exercise": "/mnt/Data/GitHub/BRACE/data/sports_videos/exercise.mp4",
    "crossfit": "/mnt/Data/GitHub/BRACE/data/sports_videos/gym_crossfit.mp4",
    "soccer": "/mnt/Data/GitHub/BRACE/data/sports_videos/soccer_match2.mp4",
    "mma": "/mnt/Data/GitHub/BRACE/data/sports_videos/mma_spar.mp4",
}


# ─── Feature extraction ───────────────────────────────────────────────────────
def extract_features(video_path: str):
    """Extract YOLO-pose features -> segments with various feature representations."""
    from ultralytics import YOLO

    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            xy = kp.xy.cpu().numpy()[0]
            conf = kp.conf.cpu().numpy()[0]
            kp_with_conf = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_with_conf, img_w, img_h)
            landmarks_list.append(mp33)
        else:
            landmarks_list.append(None)
    cap.release()

    # Normalize features
    features = []
    valid_indices = []
    for i, lm in enumerate(landmarks_list):
        if lm is None:
            continue
        feat = normalize_frame(lm)
        if feat is None:
            continue
        feat_vec = feat[FEATURE_INDICES, :2].flatten()
        if np.any(np.isnan(feat_vec)) or np.any(np.isinf(feat_vec)):
            continue
        features.append(feat_vec)
        valid_indices.append(i)

    features_arr = np.stack(features) if features else np.zeros((0, 28))
    return features_arr, valid_indices, fps


# ─── Segment extraction ───────────────────────────────────────────────────────
def get_segments(features_arr, valid_indices, fps, min_segment_sec=1.0):
    """Get segments from feature array."""
    segments = segment_motions(features_arr, valid_indices, fps, min_segment_sec=min_segment_sec)
    return segments


# ─── Feature representation builders ──────────────────────────────────────────
RESAMPLE_LEN = 30


def build_spectral_features(segments):
    """Default: mean pose + FFT power spectrum (current method)."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        mean_pose = r.mean(axis=0)
        spec = np.abs(np.fft.rfft(r, axis=0))[1:] / RESAMPLE_LEN
        spec_flat = spec.flatten()
        vectors.append(np.concatenate([mean_pose, spec_flat]))
    return np.array(vectors, dtype=np.float32)


def build_raw_features(segments):
    """Raw trajectory: mean + std per joint (no FFT)."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        mean_pose = r.mean(axis=0)
        std_pose = r.std(axis=0)
        vectors.append(np.concatenate([mean_pose, std_pose]))
    return np.array(vectors, dtype=np.float32)


def build_fft_only_features(segments):
    """FFT-only: power spectrum without mean pose."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        spec = np.abs(np.fft.rfft(r, axis=0))[1:] / RESAMPLE_LEN
        vectors.append(spec.flatten())
    return np.array(vectors, dtype=np.float32)


def build_wavelet_features(segments):
    """Wavelet features: DWT coefficients."""
    import pywt

    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        coeffs_list = []
        for d in range(r.shape[1]):
            coeffs = pywt.wavedec(r[:, d], "db4", level=3)
            flat = np.concatenate([c for c in coeffs])
            coeffs_list.append(flat)
        vectors.append(np.concatenate(coeffs_list))
    return np.array(vectors, dtype=np.float32)


def build_statistical_features(segments):
    """Statistical: mean, std, skew, kurtosis per joint."""
    from scipy.stats import skew, kurtosis

    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        mean = r.mean(axis=0)
        std = r.std(axis=0)
        sk = skew(r, axis=0)
        kurt = kurtosis(r, axis=0)
        vectors.append(np.concatenate([mean, std, sk, kurt]))
    return np.array(vectors, dtype=np.float32)


def build_mean_only_features(segments):
    """Just the mean pose vector (simplest possible)."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        vectors.append(r.mean(axis=0))
    return np.array(vectors, dtype=np.float32)


def build_flattened_features(segments):
    """Full flattened resampled trajectory."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], RESAMPLE_LEN)
        vectors.append(r.flatten())
    return np.array(vectors, dtype=np.float32)


# ─── Clustering strategies ─────────────────────────────────────────────────────

def current_method(segments, threshold=2.0):
    """Current: agglomerative + spectral distance + single linkage."""
    segs_copy = [dict(s) for s in segments]
    clustered = cluster_segments(segs_copy, distance_threshold=threshold)
    labels = [s["cluster"] for s in clustered]
    return labels


def agglo_with_custom_distance(segments, feat_builder, metric="euclidean",
                                linkage_method="single", threshold=2.0):
    """Agglomerative with custom feature representation and distance metric."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    feat_dim = segments[0]["features"].shape[1]

    if metric == "dtw":
        # DTW distance on resampled trajectories
        n = len(segments)
        dist_matrix = np.zeros((n, n))
        from dtaidistance import dtw

        for i in range(n):
            ri = _resample_segment(segments[i]["features"], RESAMPLE_LEN)
            for j in range(i + 1, n):
                rj = _resample_segment(segments[j]["features"], RESAMPLE_LEN)
                # Average DTW across dimensions
                dtw_dists = []
                for d in range(ri.shape[1]):
                    dd = dtw.distance(ri[:, d].astype(np.double),
                                      rj[:, d].astype(np.double))
                    dtw_dists.append(dd)
                avg_dtw = np.mean(dtw_dists) / np.sqrt(feat_dim)
                dist_matrix[i, j] = avg_dtw
                dist_matrix[j, i] = avg_dtw
        condensed = squareform(dist_matrix)
    else:
        condensed = pdist(X, metric=metric)
        condensed = condensed / np.sqrt(feat_dim)

    condensed = np.nan_to_num(condensed, nan=0.0, posinf=1e6, neginf=0.0)
    if np.all(condensed == 0):
        return [0] * len(segments)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    # Relabel 0-indexed
    unique = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique)}
    return [label_map[l] for l in labels]


def hdbscan_cluster(segments, feat_builder, min_cluster_size=2, min_samples=1):
    """HDBSCAN: density-based, auto cluster count."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    # HDBSCAN uses -1 for noise; relabel
    unique = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique)}
    return [label_map[l] for l in labels]


def spectral_cluster(segments, feat_builder, n_clusters=2):
    """Spectral Clustering with fixed n_clusters."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_clusters = min(n_clusters, len(segments))
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=min(5, len(segments)),
        random_state=42,
    )
    labels = sc.fit_predict(X)
    return list(labels)


def gmm_cluster(segments, feat_builder, n_components=None):
    """Gaussian Mixture Model with BIC selection."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if n_components is not None:
        n_comp = min(n_components, len(segments))
        gmm = GaussianMixture(n_components=n_comp, random_state=42, covariance_type="diag")
        labels = gmm.fit_predict(X)
        return list(labels)

    # BIC selection
    best_bic = np.inf
    best_labels = [0] * len(segments)
    max_k = min(8, len(segments))
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="diag")
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_labels = list(gmm.predict(X))
    return best_labels


def affinity_prop_cluster(segments, feat_builder, damping=0.7):
    """Affinity Propagation: auto exemplars."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ap = AffinityPropagation(damping=damping, random_state=42, max_iter=500)
    labels = ap.fit_predict(X)
    if len(set(labels)) == 0:
        return [0] * len(segments)
    return list(labels)


def optics_cluster(segments, feat_builder, min_samples=2, xi=0.05):
    """OPTICS: variable density DBSCAN."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    opt = OPTICS(min_samples=min_samples, xi=xi, metric="euclidean")
    labels = opt.fit_predict(X)
    # OPTICS uses -1 for noise
    unique = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique)}
    return [label_map[l] for l in labels]


def meanshift_cluster(segments, feat_builder):
    """Mean Shift: mode-seeking."""
    if len(segments) < 2:
        return [0] * len(segments)
    X = feat_builder(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    ms = MeanShift()
    try:
        labels = ms.fit_predict(X)
    except Exception:
        return [0] * len(segments)
    return list(labels)


# ─── Strategy definitions ──────────────────────────────────────────────────────

def define_strategies():
    """Return list of (name, function) strategies."""
    strategies = []

    # 1. Current method (baseline)
    strategies.append(("1_current_spectral_single_t2.0",
        lambda segs: current_method(segs, threshold=2.0)))

    # 2. Current method with different thresholds
    strategies.append(("2_current_t3.5",
        lambda segs: current_method(segs, threshold=3.5)))
    strategies.append(("3_current_t1.5",
        lambda segs: current_method(segs, threshold=1.5)))

    # 4. Agglomerative + ward linkage + spectral features
    strategies.append(("4_agglo_ward_spectral_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "euclidean", "ward", 2.0)))
    strategies.append(("5_agglo_ward_spectral_t3.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "euclidean", "ward", 3.0)))

    # 6. Agglomerative + complete linkage
    strategies.append(("6_agglo_complete_spectral_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "euclidean", "complete", 2.0)))

    # 7. Agglomerative + average linkage
    strategies.append(("7_agglo_average_spectral_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "euclidean", "average", 2.0)))

    # 8. Cosine distance + agglomerative
    strategies.append(("8_agglo_cosine_single_t0.3",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "cosine", "single", 0.3)))
    strategies.append(("9_agglo_cosine_average_t0.3",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "cosine", "average", 0.3)))

    # 10. Correlation distance
    strategies.append(("10_agglo_correlation_single_t0.3",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "correlation", "single", 0.3)))

    # 11. DTW distance
    strategies.append(("11_agglo_dtw_single_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "dtw", "single", 2.0)))

    # 12. HDBSCAN
    strategies.append(("12_hdbscan_spectral_mcs2",
        lambda segs: hdbscan_cluster(segs, build_spectral_features,
            min_cluster_size=2, min_samples=1)))
    strategies.append(("13_hdbscan_spectral_mcs3",
        lambda segs: hdbscan_cluster(segs, build_spectral_features,
            min_cluster_size=3, min_samples=1)))

    # 14. Spectral Clustering (k=2 and k=3)
    strategies.append(("14_spectral_k2",
        lambda segs: spectral_cluster(segs, build_spectral_features, n_clusters=2)))
    strategies.append(("15_spectral_k3",
        lambda segs: spectral_cluster(segs, build_spectral_features, n_clusters=3)))

    # 16. GMM with BIC
    strategies.append(("16_gmm_bic_spectral",
        lambda segs: gmm_cluster(segs, build_spectral_features)))

    # 17. Affinity Propagation
    strategies.append(("17_affinity_prop_spectral",
        lambda segs: affinity_prop_cluster(segs, build_spectral_features)))

    # 18. OPTICS
    strategies.append(("18_optics_spectral",
        lambda segs: optics_cluster(segs, build_spectral_features, min_samples=2)))

    # 19. Mean Shift
    strategies.append(("19_meanshift_spectral",
        lambda segs: meanshift_cluster(segs, build_spectral_features)))

    # === Alternative feature representations ===

    # 20. Raw features (mean + std, no FFT) + agglomerative
    strategies.append(("20_agglo_raw_single_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_raw_features,
            "euclidean", "single", 2.0)))

    # 21. FFT-only (no mean pose)
    strategies.append(("21_agglo_fft_only_single_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_fft_only_features,
            "euclidean", "single", 2.0)))

    # 22. Wavelet features
    strategies.append(("22_agglo_wavelet_single_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_wavelet_features,
            "euclidean", "single", 2.0)))

    # 23. Statistical features (mean, std, skew, kurtosis)
    strategies.append(("23_agglo_stats_single_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_statistical_features,
            "euclidean", "single", 2.0)))

    # 24. Mean-only features + various methods
    strategies.append(("24_agglo_meanonly_ward_t2.0",
        lambda segs: agglo_with_custom_distance(segs, build_mean_only_features,
            "euclidean", "ward", 2.0)))

    # 25. Flattened trajectory + cosine
    strategies.append(("25_agglo_flat_cosine_single_t0.3",
        lambda segs: agglo_with_custom_distance(segs, build_flattened_features,
            "cosine", "single", 0.3)))

    # 26. HDBSCAN on raw features
    strategies.append(("26_hdbscan_raw_mcs2",
        lambda segs: hdbscan_cluster(segs, build_raw_features,
            min_cluster_size=2, min_samples=1)))

    # 27. HDBSCAN on statistical features
    strategies.append(("27_hdbscan_stats_mcs2",
        lambda segs: hdbscan_cluster(segs, build_statistical_features,
            min_cluster_size=2, min_samples=1)))

    # 28. GMM on raw features
    strategies.append(("28_gmm_bic_raw",
        lambda segs: gmm_cluster(segs, build_raw_features)))

    # 29. Agglomerative + spectral features + ward + t=4.0
    strategies.append(("29_agglo_ward_spectral_t4.0",
        lambda segs: agglo_with_custom_distance(segs, build_spectral_features,
            "euclidean", "ward", 4.0)))

    # 30. Agglomerative + raw features + complete + t=3.0
    strategies.append(("30_agglo_raw_complete_t3.0",
        lambda segs: agglo_with_custom_distance(segs, build_raw_features,
            "euclidean", "complete", 3.0)))

    return strategies


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("ALTERNATIVE CLUSTERING METHODS BENCHMARK")
    print("=" * 100)

    # Step 1: Extract features from all videos
    video_data = {}
    for name, path in VIDEOS.items():
        print(f"\nExtracting features from {name}...")
        t0 = time.time()
        features, valid_idx, fps = extract_features(path)
        elapsed = time.time() - t0
        segments = get_segments(features, valid_idx, fps, min_segment_sec=1.0)
        video_data[name] = {
            "features": features,
            "valid_indices": valid_idx,
            "fps": fps,
            "segments": segments,
        }
        print(f"  {name}: {features.shape[0]} frames, {len(segments)} segments, "
              f"FPS={fps:.1f}, extracted in {elapsed:.1f}s")

    # Step 2: Run all strategies
    strategies = define_strategies()
    print(f"\n{'=' * 100}")
    print(f"Running {len(strategies)} strategies across {len(VIDEOS)} videos...")
    print(f"{'=' * 100}")

    results = []
    for strat_name, strat_fn in strategies:
        row = {"name": strat_name}
        total_time = 0

        for vid_name in VIDEOS:
            segs = video_data[vid_name]["segments"]
            # Deep copy segments to avoid mutation
            segs_copy = []
            for s in segs:
                sc = dict(s)
                sc["features"] = s["features"].copy()
                sc["mean_feature"] = s["mean_feature"].copy()
                segs_copy.append(sc)

            try:
                t0 = time.time()
                labels = strat_fn(segs_copy)
                elapsed = time.time() - t0
                total_time += elapsed

                n_clusters = len(set(labels))
                row[vid_name] = n_clusters
                row[f"{vid_name}_labels"] = labels
                row[f"{vid_name}_time"] = elapsed
            except Exception as e:
                row[vid_name] = f"ERR"
                row[f"{vid_name}_labels"] = []
                row[f"{vid_name}_time"] = 0
                print(f"  ERROR in {strat_name} on {vid_name}: {e}")

        row["total_time"] = total_time
        results.append(row)

    # Step 3: Print results table
    vid_names = list(VIDEOS.keys())
    print(f"\n{'=' * 100}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 100}")

    header = f"{'Strategy':<45} | "
    header += " | ".join(f"{v:>10}" for v in vid_names)
    header += f" | {'Time':>6}"
    print(header)
    print("-" * len(header))

    for row in results:
        line = f"{row['name']:<45} | "
        vals = []
        for v in vid_names:
            val = row.get(v, "?")
            if isinstance(val, int):
                # Highlight basketball == 2
                if v == "basketball" and val == 2:
                    vals.append(f"  {val:>2} [OK]")
                elif v == "basketball":
                    vals.append(f"  {val:>2} [!!]")
                else:
                    vals.append(f"  {val:>8}")
            else:
                vals.append(f"  {str(val):>8}")
        line += " | ".join(vals)
        line += f" | {row['total_time']:>5.2f}s"
        print(line)

    # Step 4: Rank strategies
    print(f"\n{'=' * 100}")
    print("RANKING (basketball=2 required, then by cluster diversity)")
    print(f"{'=' * 100}")

    scored = []
    for row in results:
        basketball_ok = row.get("basketball") == 2
        if not basketball_ok:
            continue

        # Quality score: variety of clusters across videos
        # Basketball should have 2, others should have reasonable counts (not all 1)
        cluster_counts = []
        for v in vid_names:
            val = row.get(v)
            if isinstance(val, int):
                cluster_counts.append(val)

        # Penalize if everything collapses to 1 cluster
        n_single = sum(1 for c in cluster_counts if c == 1)
        # Penalize extreme fragmentation (>15 clusters)
        n_extreme = sum(1 for c in cluster_counts if c > 15)
        # Reward variety
        variety = len(set(cluster_counts))

        quality = variety * 10 - n_single * 5 - n_extreme * 3
        scored.append((quality, row))

    scored.sort(key=lambda x: -x[0])

    for rank, (quality, row) in enumerate(scored, 1):
        vals = ", ".join(f"{v}={row.get(v)}" for v in vid_names)
        print(f"  #{rank}: {row['name']:<45} | {vals} | quality={quality}")

    if not scored:
        print("  No strategy achieved basketball=2!")
        print("\n  Strategies closest to basketball=2:")
        for row in results:
            bb = row.get("basketball", "?")
            if bb in (2, 3):
                vals = ", ".join(f"{v}={row.get(v)}" for v in vid_names)
                print(f"    {row['name']:<45} | {vals}")

    # Step 5: Segment details per video
    print(f"\n{'=' * 100}")
    print("SEGMENT DETAILS PER VIDEO")
    print(f"{'=' * 100}")
    for vid_name in vid_names:
        segs = video_data[vid_name]["segments"]
        fps = video_data[vid_name]["fps"]
        print(f"\n{vid_name}: {len(segs)} segments")
        for i, seg in enumerate(segs):
            dur = (seg["end_frame"] - seg["start_frame"]) / fps
            print(f"  Seg {i}: frames {seg['start_frame']}-{seg['end_frame']} "
                  f"({dur:.1f}s), feat shape={seg['features'].shape}")

    # Step 6: Pairwise distances for basketball
    print(f"\n{'=' * 100}")
    print("PAIRWISE DISTANCES - BASKETBALL")
    print(f"{'=' * 100}")
    bb_segs = video_data["basketball"]["segments"]
    n_bb = len(bb_segs)
    for i in range(n_bb):
        for j in range(i + 1, n_bb):
            d = _segment_distance(bb_segs[i], bb_segs[j])
            print(f"  d({i},{j}) = {d:.4f}")


if __name__ == "__main__":
    main()
