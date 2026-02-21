#!/usr/bin/env python3
"""Test advanced clustering algorithms on 14 demo videos with CLIP validation.

Compares multiple clustering methods against the current production agglomerative
clustering (average linkage, spectral distance, threshold=2.0). Each method uses
the same segmentation pipeline (segment_motions) and the same pairwise spectral
distance matrix (_segment_distance). Where a method requires feature vectors
instead of a distance matrix, segment mean features are used.

Validation: CLIP ViT-L/14 + 61 expanded sport-specific labels. A video "passes"
if all clusters receive distinct CLIP labels.

Methods tested:
  1. Production agglomerative (baseline)
  2. HDBSCAN (density-based, auto K)
  3. Spectral Clustering with eigengap K selection
  4. Affinity Propagation (message passing, auto K)
  5. OPTICS (density ordering)
  6. Mean Shift (mode seeking)
  7. GMM with BIC model selection (auto K)
  8. Bayesian GMM with automatic relevance determination (auto K)
  9. HDBSCAN on precomputed distance matrix
  10. X-means approximation (BIC-based K splitting)
  11. Ensemble / consensus clustering

Usage:
    .venv/bin/python experiments/test_advanced_clustering.py
"""

import sys
import os
import json
import copy
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import numpy as np
import torch
from PIL import Image
import cv2

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from sklearn.cluster import (
    SpectralClustering,
    AffinityPropagation,
    OPTICS,
    MeanShift,
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import hdbscan

from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    _segment_distance,
    _resample_segment,
    _merge_adjacent_clusters,
)

# ── Paths ────────────────────────────────────────────────────────────────────

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
OUTPUT_PATH = "/mnt/Data/GitHub/BRACE/experiments/advanced_clustering_results.json"

# ── Expanded 61-label vocabulary (from final_validation.py) ──────────────────

EXPANDED_LABELS = [
    "a person running at full speed", "a person jogging at a moderate pace",
    "a person walking", "a person sprinting",
    "a person dribbling a basketball while stationary",
    "a person dribbling a basketball while moving",
    "a person shooting a basketball jump shot",
    "a person doing a basketball layup", "a person dunking a basketball",
    "a person passing a basketball", "a person defending in basketball",
    "a person doing pushups", "a person doing squats",
    "a person doing pullups", "a person doing burpees",
    "a person doing jumping jacks", "a person doing lunges",
    "a person doing mountain climbers", "a person doing high knees",
    "a person doing box jumps", "a person doing planks",
    "a person doing crunches", "a person doing stretching",
    "a person lifting weights overhead",
    "a person doing bench press", "a person doing deadlifts",
    "a person doing barbell curls", "a person doing rows",
    "a person doing kettlebell swings", "a person doing clean and jerk",
    "a person doing lateral raises", "a person doing tricep extensions",
    "a person shadowboxing", "a person punching a heavy bag",
    "a person doing boxing footwork", "a person in a fighting stance",
    "a person jumping rope with both feet",
    "a person doing double unders with a jump rope",
    "a person swimming freestyle", "a person swimming backstroke",
    "a person swimming breaststroke", "a person doing a flip turn",
    "a person hitting a tennis forehand", "a person hitting a tennis backhand",
    "a person serving in tennis", "a person doing a tennis volley",
    "a person doing a tennis overhead smash",
    "a person doing a tennis split step ready position",
    "a person dribbling a soccer ball",
    "a person kicking a soccer ball",
    "a person doing soccer juggling tricks",
    "a person doing soccer footwork drills",
    "a person doing a yoga standing pose",
    "a person doing a yoga balance pose",
    "a person doing a yoga seated pose",
    "a person doing a yoga inversion",
    "a person transitioning between yoga poses",
    "a person doing bodyweight dips",
    "a person doing handstand practice",
    "a person doing muscle ups",
    "a person standing still", "a person resting between exercises",
    "a person warming up",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_video_data():
    """Load cached features for all videos."""
    video_data = {}
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".feats.npz"):
            continue
        video_name = fname.replace(".feats.npz", "")
        cache_path = os.path.join(CACHE_DIR, fname)
        data = np.load(cache_path)
        feats = data["features"]
        vi = data["valid_indices"].tolist()
        fps = float(data["fps"])
        if len(feats) >= 10:
            video_path = os.path.join(VIDEO_DIR, video_name)
            video_data[video_name] = {
                "path": video_path,
                "features": feats,
                "valid_indices": vi,
                "fps": fps,
            }
    return video_data


def build_distance_matrix(segments):
    """Build pairwise spectral distance matrix for segments."""
    n = len(segments)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def build_segment_feature_vectors(segments, resample_len=30):
    """Build feature vectors for each segment: mean pose + FFT power spectrum."""
    vectors = []
    for seg in segments:
        r = _resample_segment(seg["features"], resample_len)
        mean_pose = r.mean(axis=0)
        spec = np.abs(np.fft.rfft(r, axis=0))[1:] / resample_len
        spec_flat = spec.flatten()
        vectors.append(np.concatenate([mean_pose, spec_flat]))
    return np.array(vectors, dtype=np.float32)


def build_mean_features(segments):
    """Build simple mean feature vectors for each segment."""
    return np.array([seg["mean_feature"] for seg in segments], dtype=np.float32)


def relabel_contiguous(labels):
    """Relabel to 0-indexed contiguous integers."""
    unique = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique)}
    return [label_map[l] for l in labels]


def apply_labels_and_postprocess(segments, labels, fps=30.0):
    """Apply cluster labels and run standard post-processing (merge adjacent + absorb small)."""
    for i, seg in enumerate(segments):
        seg["cluster"] = labels[i]
    segments = _merge_adjacent_clusters(segments)
    segments = _postprocess_small_clusters(segments, fps=fps)
    return segments


def _postprocess_small_clusters(segments, min_segments=2, min_seconds=3.0, fps=30.0):
    """Absorb small clusters into nearest large neighbor (same as production)."""
    if len(segments) <= 1:
        return segments

    cluster_info = {}
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

    segments = _merge_adjacent_clusters(segments)

    unique = sorted(set(s["cluster"] for s in segments))
    remap = {old: new for new, old in enumerate(unique)}
    for seg in segments:
        seg["cluster"] = remap[seg["cluster"]]

    return segments


# ── Clustering Methods ───────────────────────────────────────────────────────

def method_production(segments, fps=30.0):
    """Production agglomerative clustering (baseline)."""
    segs = copy.deepcopy(segments)
    clustered = cluster_segments(segs, distance_threshold=2.0, fps=fps)
    return clustered


def method_hdbscan_precomputed(segments, fps=30.0, min_cluster_size=2, min_samples=1):
    """HDBSCAN on precomputed spectral distance matrix."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    dist_matrix = build_distance_matrix(segments)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # HDBSCAN -1 = noise; assign noise points to nearest cluster
    labels = list(labels)
    valid_clusters = set(l for l in labels if l >= 0)
    if not valid_clusters:
        # All noise -> single cluster
        labels = [0] * n
    else:
        for i in range(n):
            if labels[i] == -1:
                best_cid, best_dist = None, float("inf")
                for j in range(n):
                    if labels[j] >= 0 and dist_matrix[i, j] < best_dist:
                        best_dist = dist_matrix[i, j]
                        best_cid = labels[j]
                labels[i] = best_cid if best_cid is not None else 0

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_hdbscan_features(segments, fps=30.0, min_cluster_size=2, min_samples=1):
    """HDBSCAN on segment feature vectors (mean + FFT)."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_segment_feature_vectors(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    labels = list(labels)

    valid_clusters = set(l for l in labels if l >= 0)
    if not valid_clusters:
        labels = [0] * n
    else:
        # Assign noise points to nearest cluster centroid
        cluster_centroids = {}
        for cid in valid_clusters:
            members = [X[i] for i in range(n) if labels[i] == cid]
            cluster_centroids[cid] = np.mean(members, axis=0)
        for i in range(n):
            if labels[i] == -1:
                best_cid, best_dist = 0, float("inf")
                for cid, centroid in cluster_centroids.items():
                    d = float(np.linalg.norm(X[i] - centroid))
                    if d < best_dist:
                        best_dist = d
                        best_cid = cid
                labels[i] = best_cid

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def _eigengap_k(affinity_matrix, max_k=10):
    """Estimate optimal K for spectral clustering via eigengap heuristic."""
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    n = affinity_matrix.shape[0]
    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 1

    # Compute normalized Laplacian
    D = np.diag(affinity_matrix.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(affinity_matrix.sum(axis=1), 1e-10)))
    L = np.eye(n) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt

    try:
        eigenvalues = np.sort(np.linalg.eigvalsh(L))
    except np.linalg.LinAlgError:
        return 2

    # Find largest eigengap
    eigenvalues = eigenvalues[:max_k + 1]
    gaps = np.diff(eigenvalues)
    if len(gaps) == 0:
        return 1

    # Skip first gap (between 0 and first nonzero eigenvalue)
    best_k = int(np.argmax(gaps[1:]) + 2) if len(gaps) > 1 else 2
    return max(1, min(best_k, n))


def method_spectral_eigengap(segments, fps=30.0):
    """Spectral clustering with automatic K via eigengap heuristic."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    dist_matrix = build_distance_matrix(segments)

    # Convert distance to affinity (RBF kernel)
    sigma = np.median(dist_matrix[dist_matrix > 0]) if np.any(dist_matrix > 0) else 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 1.0)

    # Determine K
    k = _eigengap_k(affinity, max_k=min(8, n))
    k = max(1, min(k, n))

    if k == 1:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    try:
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=42,
            n_init=10,
        )
        labels = list(sc.fit_predict(affinity))
    except Exception:
        labels = [0] * n

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_spectral_fixed(segments, fps=30.0, n_clusters=2):
    """Spectral clustering with fixed K on precomputed affinity."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    n_clusters = min(n_clusters, n)
    dist_matrix = build_distance_matrix(segments)

    sigma = np.median(dist_matrix[dist_matrix > 0]) if np.any(dist_matrix > 0) else 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 1.0)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            n_init=10,
        )
        labels = list(sc.fit_predict(affinity))
    except Exception:
        labels = [0] * n

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_affinity_propagation(segments, fps=30.0, damping=0.7, preference_quantile=0.5):
    """Affinity Propagation on precomputed similarity matrix."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    dist_matrix = build_distance_matrix(segments)
    # Convert to similarity (negative distance)
    similarity = -dist_matrix

    # Set preference (controls number of clusters)
    pref = np.quantile(similarity[similarity < 0], preference_quantile) if np.any(similarity < 0) else -1.0

    try:
        ap = AffinityPropagation(
            damping=damping,
            preference=pref,
            affinity="precomputed",
            random_state=42,
            max_iter=500,
        )
        labels = list(ap.fit_predict(similarity))
    except Exception:
        labels = [0] * n

    if len(set(labels)) == 0 or all(l == -1 for l in labels):
        labels = [0] * n

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_optics(segments, fps=30.0, min_samples=2, xi=0.05):
    """OPTICS on precomputed distance matrix."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    dist_matrix = build_distance_matrix(segments)

    min_s = min(min_samples, n - 1)

    try:
        opt = OPTICS(
            min_samples=max(min_s, 2),
            xi=xi,
            metric="precomputed",
        )
        labels = list(opt.fit_predict(dist_matrix))
    except Exception:
        labels = [0] * n

    # OPTICS -1 = noise; assign to nearest labeled point
    valid_clusters = set(l for l in labels if l >= 0)
    if not valid_clusters:
        labels = [0] * n
    else:
        for i in range(n):
            if labels[i] == -1:
                best_cid, best_dist = 0, float("inf")
                for j in range(n):
                    if labels[j] >= 0 and dist_matrix[i, j] < best_dist:
                        best_dist = dist_matrix[i, j]
                        best_cid = labels[j]
                labels[i] = best_cid

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_meanshift(segments, fps=30.0):
    """Mean Shift on segment feature vectors."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_segment_feature_vectors(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        ms = MeanShift()
        labels = list(ms.fit_predict(X))
    except Exception:
        labels = [0] * n

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_gmm_bic(segments, fps=30.0, max_k=8, cov_type="diag"):
    """Gaussian Mixture Model with BIC-based model selection."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_mean_features(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # BIC selection over K
    best_bic = np.inf
    best_labels = [0] * n
    max_k = min(max_k, n)

    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=42,
                max_iter=200,
                n_init=3,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_labels = list(gmm.predict(X))
        except Exception:
            continue

    labels = relabel_contiguous(best_labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_bayesian_gmm(segments, fps=30.0, max_k=8):
    """Bayesian GMM with automatic relevance determination (auto K)."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_mean_features(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    max_k = min(max_k, n)

    try:
        bgmm = BayesianGaussianMixture(
            n_components=max_k,
            covariance_type="diag",
            random_state=42,
            max_iter=300,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.1,
        )
        labels = list(bgmm.fit_predict(X))
    except Exception:
        labels = [0] * n

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_xmeans(segments, fps=30.0, max_k=8):
    """X-means approximation: recursive BIC-based splitting from K-means."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_mean_features(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    max_k = min(max_k, n)

    from sklearn.cluster import KMeans

    # Start with k=1, try splitting
    best_k = 1
    best_bic = np.inf
    best_labels = [0] * n

    for k in range(1, max_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            # Compute BIC: -2 * log_likelihood + k * log(n)
            labels = list(km.predict(X))
            # Use GMM BIC as proxy
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                   random_state=42, max_iter=100)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_labels = labels
                best_k = k
        except Exception:
            continue

    labels = relabel_contiguous(best_labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_ensemble(segments, fps=30.0):
    """Ensemble clustering: majority vote from multiple methods.

    Runs several base clusterers and creates a co-association matrix
    (how often pairs are in the same cluster), then uses agglomerative
    clustering on the co-association distance.
    """
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    dist_matrix = build_distance_matrix(segments)
    X = build_segment_feature_vectors(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Co-association matrix
    coassoc = np.zeros((n, n))
    n_runs = 0

    # Base method 1: agglomerative at various thresholds
    for t in [1.5, 2.0, 2.5, 3.0]:
        try:
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=t, criterion="distance")
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        coassoc[i, j] += 1
                        coassoc[j, i] += 1
            n_runs += 1
        except Exception:
            pass

    # Base method 2: HDBSCAN on features
    for mcs in [2, 3]:
        try:
            h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="euclidean")
            labels = h.fit_predict(X)
            labels = [l if l >= 0 else -1 for l in labels]
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] >= 0 and labels[i] == labels[j]:
                        coassoc[i, j] += 1
                        coassoc[j, i] += 1
            n_runs += 1
        except Exception:
            pass

    # Base method 3: GMM with various K
    X_mean = build_mean_features(segments)
    X_mean = np.nan_to_num(X_mean, nan=0.0)
    for k in range(1, min(5, n) + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="diag",
                                   random_state=42, max_iter=100)
            labels = list(gmm.fit_predict(X_mean))
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        coassoc[i, j] += 1
                        coassoc[j, i] += 1
            n_runs += 1
        except Exception:
            pass

    if n_runs == 0:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    # Normalize to [0, 1]
    coassoc /= n_runs
    np.fill_diagonal(coassoc, 1.0)

    # Convert co-association to distance
    coassoc_dist = 1.0 - coassoc

    # Cluster the co-association distance matrix
    condensed = squareform(coassoc_dist)
    Z = linkage(condensed, method="average")
    labels = list(fcluster(Z, t=0.5, criterion="distance"))

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


def method_agglo_ward(segments, fps=30.0, threshold=2.0):
    """Agglomerative with Ward linkage on segment feature vectors."""
    n = len(segments)
    if n < 2:
        segs = copy.deepcopy(segments)
        for s in segs:
            s["cluster"] = 0
        return segs

    X = build_segment_feature_vectors(segments)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    from scipy.spatial.distance import pdist
    feat_dim = segments[0]["features"].shape[1]
    condensed = pdist(X, metric="euclidean") / np.sqrt(feat_dim)

    Z = linkage(condensed, method="ward")
    labels = list(fcluster(Z, t=threshold, criterion="distance"))

    labels = relabel_contiguous(labels)
    segs = copy.deepcopy(segments)
    return apply_labels_and_postprocess(segs, labels, fps)


# ── CLIP validation ──────────────────────────────────────────────────────────

def classify_cluster_with_clip(video_path, cluster_frame_ranges, clip_model, preprocess,
                                text_features, device, labels):
    """Classify cluster frames with CLIP. Memory-efficient: reads on-demand."""
    total_frames = sum(e - s + 1 for s, e in cluster_frame_ranges)
    n_sample = min(8, total_frames)

    all_frame_idxs = []
    for s, e in cluster_frame_ranges:
        all_frame_idxs.extend(range(s, e + 1))

    if len(all_frame_idxs) == 0:
        return "unknown", 0.0

    sample_idxs = sorted(set(
        all_frame_idxs[i]
        for i in np.linspace(0, len(all_frame_idxs) - 1, n_sample, dtype=int)
    ))

    cap = cv2.VideoCapture(video_path)
    sampled_frames = []
    frame_idx = 0
    sample_set = set(sample_idxs)
    max_idx = max(sample_idxs)

    while cap.isOpened() and frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(rgb)
        frame_idx += 1
    cap.release()

    if not sampled_frames:
        return "unknown", 0.0

    images = torch.stack([preprocess(Image.fromarray(f)) for f in sampled_frames]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    avg_feat = image_features.mean(dim=0, keepdim=True)
    avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
    similarity = (avg_feat @ text_features.T).squeeze()

    best_idx = similarity.argmax().item()
    return labels[best_idx], float(similarity[best_idx])


def validate_with_clip(video_path, clustered_segments, clip_model, preprocess,
                        text_features, device, labels):
    """Validate clustering result using CLIP labels.

    Returns dict with n_clusters, cluster_labels, all_distinct, etc.
    """
    n_clusters = len(set(s["cluster"] for s in clustered_segments))

    cluster_ranges = {}
    for seg in clustered_segments:
        cid = seg["cluster"]
        cluster_ranges.setdefault(cid, []).append(
            (seg["start_frame"], seg["end_frame"])
        )

    cluster_labels = {}
    for cid, ranges in sorted(cluster_ranges.items()):
        label, conf = classify_cluster_with_clip(
            video_path, ranges, clip_model, preprocess, text_features, device, labels
        )
        n_frames = sum(e - s + 1 for s, e in ranges)
        cluster_labels[cid] = {"label": label, "confidence": round(conf, 3), "n_frames": n_frames}

    labels_only = [v["label"] for v in cluster_labels.values()]
    unique_labels_set = set(labels_only)
    all_distinct = len(labels_only) == len(unique_labels_set)

    # Also compute post-merge result (merge clusters with same CLIP label)
    n_after_merge = n_clusters
    if not all_distinct and n_clusters > 1:
        label_to_cids = {}
        for cid, info in cluster_labels.items():
            label_to_cids.setdefault(info["label"], []).append(cid)
        merge_map = {}
        for label, cids in label_to_cids.items():
            if len(cids) > 1:
                target = min(cids)
                for cid in cids:
                    merge_map[cid] = target
        merged_segs = copy.deepcopy(clustered_segments)
        for seg in merged_segs:
            if seg["cluster"] in merge_map:
                seg["cluster"] = merge_map[seg["cluster"]]
        merged_segs = _merge_adjacent_clusters(merged_segs)
        n_after_merge = len(set(s["cluster"] for s in merged_segs))

    return {
        "n_segments": len(clustered_segments),
        "n_clusters": n_clusters,
        "n_clusters_after_clip_merge": n_after_merge,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "all_distinct": all_distinct,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

METHODS = {
    "1_production_agglo": method_production,
    "2_hdbscan_precomputed": method_hdbscan_precomputed,
    "3_hdbscan_features": method_hdbscan_features,
    "4_spectral_eigengap": method_spectral_eigengap,
    "5_spectral_k2": lambda segs, fps: method_spectral_fixed(segs, fps, n_clusters=2),
    "6_spectral_k3": lambda segs, fps: method_spectral_fixed(segs, fps, n_clusters=3),
    "7_affinity_prop": method_affinity_propagation,
    "8_affinity_prop_low": lambda segs, fps: method_affinity_propagation(segs, fps, preference_quantile=0.3),
    "9_optics": method_optics,
    "10_optics_xi01": lambda segs, fps: method_optics(segs, fps, xi=0.1),
    "11_meanshift": method_meanshift,
    "12_gmm_bic": method_gmm_bic,
    "13_gmm_bic_full": lambda segs, fps: method_gmm_bic(segs, fps, cov_type="full"),
    "14_bayesian_gmm": method_bayesian_gmm,
    "15_xmeans": method_xmeans,
    "16_ensemble": method_ensemble,
    "17_agglo_ward": method_agglo_ward,
    "18_agglo_ward_t3": lambda segs, fps: method_agglo_ward(segs, fps, threshold=3.0),
}


def main():
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load CLIP ViT-L/14
    print("Loading CLIP ViT-L/14...")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    text_tokens = clip.tokenize(EXPANDED_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load ground truth
    gt = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH) as f:
            gt = json.load(f)

    # Load cached features
    print("Loading cached features...")
    video_data = load_video_data()
    print(f"Loaded {len(video_data)} videos\n")

    # Pre-segment all videos (shared across methods)
    video_segments = {}
    for vname, vdata in sorted(video_data.items()):
        segs = segment_motions(
            vdata["features"], vdata["valid_indices"], vdata["fps"],
            min_segment_sec=2.0,
        )
        video_segments[vname] = segs
        print(f"  {vname:35s}: {len(segs):2d} segments")

    print()

    # Run all methods
    all_results = {}

    for method_name, method_fn in METHODS.items():
        print(f"{'=' * 80}")
        print(f"METHOD: {method_name}")
        print(f"{'=' * 80}")

        method_results = {}
        pass_count = 0
        total_time = 0

        for vname in sorted(video_data.keys()):
            vdata = video_data[vname]
            segs = video_segments[vname]

            if len(segs) < 1:
                method_results[vname] = {
                    "n_segments": 0, "n_clusters": 0,
                    "all_distinct": True, "error": "no segments",
                }
                pass_count += 1
                continue

            try:
                t0 = time.time()
                clustered = method_fn(segs, fps=vdata["fps"])
                elapsed = time.time() - t0
                total_time += elapsed

                # Validate with CLIP
                result = validate_with_clip(
                    vdata["path"], clustered, clip_model, preprocess,
                    text_features, device, EXPANDED_LABELS,
                )
                result["time_sec"] = round(elapsed, 4)

                method_results[vname] = result
                if result["all_distinct"]:
                    pass_count += 1

                status = "PASS" if result["all_distinct"] else "FAIL"
                labels_short = [
                    v["label"].replace("a person ", "").replace("doing ", "")[:35]
                    for v in result["cluster_labels"].values()
                ]
                print(f"  {status:4s} {vname:35s} segs={result['n_segments']:2d} "
                      f"clust={result['n_clusters']:2d} {labels_short}")

            except Exception as e:
                method_results[vname] = {
                    "n_segments": len(segs), "n_clusters": 0,
                    "all_distinct": False, "error": str(e),
                }
                print(f"  ERR  {vname:35s} {e}")

        total = len(video_data)
        print(f"\n  {method_name}: {pass_count}/{total} pass ({100 * pass_count / total:.0f}%) "
              f"[{total_time:.2f}s total]\n")

        all_results[method_name] = {
            "pass_count": pass_count,
            "total": total,
            "total_time_sec": round(total_time, 3),
            "videos": method_results,
        }

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: ALL METHODS")
    print(f"{'=' * 80}")
    print(f"{'Method':<30s} | {'Pass':>4s} | {'Total':>5s} | {'Rate':>5s} | {'Time':>6s}")
    print("-" * 60)

    ranked = sorted(all_results.items(), key=lambda x: (-x[1]["pass_count"], x[1]["total_time_sec"]))
    for method_name, result in ranked:
        pc = result["pass_count"]
        total = result["total"]
        rate = f"{100 * pc / total:.0f}%"
        t = f"{result['total_time_sec']:.2f}s"
        print(f"  {method_name:<28s} | {pc:>4d} | {total:>5d} | {rate:>5s} | {t:>6s}")

    # Per-video cluster count comparison
    print(f"\n{'=' * 80}")
    print("PER-VIDEO CLUSTER COUNTS")
    print(f"{'=' * 80}")

    video_names = sorted(video_data.keys())
    header = f"{'Method':<28s} | " + " | ".join(f"{v[:12]:>12s}" for v in video_names)
    print(header)
    print("-" * len(header))

    for method_name, result in ranked:
        vals = []
        for vn in video_names:
            vr = result["videos"].get(vn, {})
            nc = vr.get("n_clusters", "?")
            distinct = vr.get("all_distinct", False)
            marker = "" if distinct else "*"
            vals.append(f"{nc}{marker:>11s}")
        line = f"  {method_name:<26s} | " + " | ".join(vals)
        print(line)

    print("\n  (* = clusters share duplicate CLIP labels)")

    # GT comparison
    print(f"\n{'=' * 80}")
    print("GROUND TRUTH COMPARISON (expected cluster counts)")
    print(f"{'=' * 80}")

    gt_videos = {k: v for k, v in gt.items() if k != "_metadata" and "expected_clusters" in v}
    for vname, gt_info in sorted(gt_videos.items()):
        expected = gt_info["expected_clusters"]
        exp_range = gt_info.get("expected_clusters_range", [expected, expected])
        print(f"\n  {vname}: expected={expected} (range {exp_range[0]}-{exp_range[1]})")
        for method_name, result in ranked:
            vr = result["videos"].get(vname, {})
            nc = vr.get("n_clusters", "?")
            match = ""
            if isinstance(nc, int):
                if nc == expected:
                    match = "EXACT"
                elif exp_range[0] <= nc <= exp_range[1]:
                    match = "IN RANGE"
                else:
                    match = "MISS"
            print(f"    {method_name:<28s}: {nc:>2} clusters  {match}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
