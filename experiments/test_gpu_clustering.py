#!/usr/bin/env python3
"""Test GPU-accelerated and alternative clustering methods on real video data.

Compares: HDBSCAN, DBSCAN, Spectral Clustering, KMeans, and the current
agglomerative approach. Uses segment-level features (mean pose + FFT spectrum)
extracted from all 5 demo videos via YOLO-pose + SRP normalization.

Key constraint: basketball_solo must produce 2 clusters (dribbling + dunking).
"""

import sys
import os
import time
import copy
import json

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import hdbscan

from brace.core.motion_segments import (
    normalize_frame,
    segment_motions,
    cluster_segments,
    _segment_distance,
    _resample_segment,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEOS_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
RESAMPLE_LEN = 30  # match _segment_distance default

# Expected cluster counts per video (ground truth from manual inspection)
EXPECTED_CLUSTERS = {
    "basketball_solo.mp4": 2,   # dribbling + dunking
    "exercise.mp4": 2,          # two exercise types
    "gym_crossfit.mp4": 5,      # multiple different exercises
    "soccer_match2.mp4": 3,     # different play phases
    "mma_spar.mp4": 11,         # many different fighting moves
}

# ---------------------------------------------------------------------------
# Feature extraction (cached)
# ---------------------------------------------------------------------------

model = None

def get_model():
    global model
    if model is None:
        model = YOLO("yolo11n-pose.pt")
    return model


def extract_features(video_path: str) -> tuple[np.ndarray | None, list[int], float]:
    """Extract SRP-normalized pose features from video."""
    m = get_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    features = []
    valid_indices = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = m(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            xy = kp.xy.cpu().numpy()[0]
            conf = kp.conf.cpu().numpy()[0]
            kp_with_conf = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_with_conf, img_w, img_h)
            feat = normalize_frame(mp33)
            if feat is not None:
                feat_vec = feat[FEATURE_INDICES, :2].flatten()
                features.append(feat_vec)
                valid_indices.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not features:
        return None, [], fps
    return np.stack(features), valid_indices, fps


def build_segment_features(segments: list[dict], resample_len: int = RESAMPLE_LEN) -> np.ndarray:
    """Build segment-level feature matrix: [mean_pose | FFT_spectrum] per segment.

    This is what _segment_distance uses internally; we extract it explicitly
    so clustering methods can work on segment-level features directly.
    """
    seg_features = []
    for seg in segments:
        resampled = _resample_segment(seg["features"], resample_len)
        mean_pose = resampled.mean(axis=0)
        spec = np.abs(np.fft.rfft(resampled, axis=0))[1:] / resample_len
        spec_flat = spec.flatten()
        combined = np.concatenate([mean_pose, spec_flat])
        seg_features.append(combined)
    return np.stack(seg_features)


def build_distance_matrix(segments: list[dict]) -> np.ndarray:
    """Build pairwise distance matrix using the project's spectral distance."""
    n = len(segments)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def cluster_agglomerative(segments: list[dict], threshold: float = 3.5) -> dict:
    """Current production method: agglomerative + single linkage + spectral distance."""
    t0 = time.perf_counter()
    segs = copy.deepcopy(segments)
    segs = cluster_segments(segs, distance_threshold=threshold)
    elapsed = time.perf_counter() - t0

    labels = [s["cluster"] for s in segs]
    n_clusters = len(set(labels))

    # Note: after merging adjacent segments, segment count may differ
    return {
        "method": f"Agglomerative(t={threshold})",
        "n_clusters": n_clusters,
        "labels": labels,
        "n_segments_after_merge": len(segs),
        "time_ms": elapsed * 1000,
    }


def cluster_hdbscan(dist_matrix: np.ndarray, min_cluster_size: int = 2,
                     min_samples: int = 1) -> dict:
    """HDBSCAN: auto-selects cluster count, handles noise."""
    t0 = time.perf_counter()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(dist_matrix)
    elapsed = time.perf_counter() - t0

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    return {
        "method": f"HDBSCAN(mcs={min_cluster_size},ms={min_samples})",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "labels": labels.tolist(),
        "time_ms": elapsed * 1000,
    }


def cluster_dbscan(dist_matrix: np.ndarray, eps: float = 1.0,
                    min_samples: int = 1) -> dict:
    """DBSCAN with precomputed distance matrix."""
    t0 = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)
    elapsed = time.perf_counter() - t0

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    return {
        "method": f"DBSCAN(eps={eps},ms={min_samples})",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "labels": labels.tolist(),
        "time_ms": elapsed * 1000,
    }


def cluster_spectral(dist_matrix: np.ndarray, n_clusters: int = 2) -> dict:
    """Spectral Clustering with Gaussian kernel on distance matrix."""
    t0 = time.perf_counter()

    # Convert distances to affinity using Gaussian kernel
    sigma = np.median(dist_matrix[dist_matrix > 0])
    if sigma < 1e-6:
        sigma = 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(affinity)
    elapsed = time.perf_counter() - t0

    sil = -1.0
    if len(set(labels)) > 1:
        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

    return {
        "method": f"Spectral(k={n_clusters})",
        "n_clusters": int(len(set(labels))),
        "labels": labels.tolist(),
        "silhouette": round(sil, 4),
        "time_ms": elapsed * 1000,
    }


def cluster_spectral_auto(dist_matrix: np.ndarray, max_k: int = 8) -> dict:
    """Spectral Clustering with automatic k selection via silhouette score."""
    t0 = time.perf_counter()

    sigma = np.median(dist_matrix[dist_matrix > 0])
    if sigma < 1e-6:
        sigma = 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    best_k = 1
    best_sil = -2.0
    best_labels = np.zeros(dist_matrix.shape[0], dtype=int)

    for k in range(2, min(max_k + 1, dist_matrix.shape[0])):
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=42,
                assign_labels="kmeans",
            )
            labels = sc.fit_predict(affinity)
            if len(set(labels)) > 1:
                sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_labels = labels
        except Exception:
            continue

    elapsed = time.perf_counter() - t0

    return {
        "method": f"Spectral-auto(k={best_k})",
        "n_clusters": best_k,
        "labels": best_labels.tolist(),
        "silhouette": round(best_sil, 4),
        "time_ms": elapsed * 1000,
    }


def cluster_kmeans(seg_features: np.ndarray, n_clusters: int = 2) -> dict:
    """KMeans on segment-level features (not distance matrix)."""
    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(seg_features)
    elapsed = time.perf_counter() - t0

    sil = -1.0
    ch = -1.0
    if len(set(labels)) > 1 and len(labels) > n_clusters:
        sil = float(silhouette_score(seg_features, labels))
        ch = float(calinski_harabasz_score(seg_features, labels))

    return {
        "method": f"KMeans(k={n_clusters})",
        "n_clusters": int(len(set(labels))),
        "labels": labels.tolist(),
        "silhouette": round(sil, 4),
        "calinski_harabasz": round(ch, 2),
        "time_ms": elapsed * 1000,
    }


def cluster_kmeans_auto(seg_features: np.ndarray, max_k: int = 8) -> dict:
    """KMeans with automatic k selection via silhouette score."""
    t0 = time.perf_counter()
    best_k = 1
    best_sil = -2.0
    best_labels = np.zeros(seg_features.shape[0], dtype=int)

    for k in range(2, min(max_k + 1, seg_features.shape[0])):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(seg_features)
        if len(set(labels)) > 1 and len(labels) > k:
            sil = float(silhouette_score(seg_features, labels))
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

    elapsed = time.perf_counter() - t0

    return {
        "method": f"KMeans-auto(k={best_k})",
        "n_clusters": best_k,
        "labels": best_labels.tolist(),
        "silhouette": round(best_sil, 4),
        "time_ms": elapsed * 1000,
    }


def cluster_kmeans_gpu(seg_features: np.ndarray, n_clusters: int = 2) -> dict:
    """KMeans implemented on GPU via PyTorch for speed comparison."""
    import torch

    t0 = time.perf_counter()

    X = torch.tensor(seg_features, dtype=torch.float32, device="cuda")
    n = X.shape[0]

    # Initialize centroids with k-means++
    centroids_idx = [np.random.randint(n)]
    for _ in range(1, n_clusters):
        dists = torch.cdist(X, X[centroids_idx]).min(dim=1).values
        probs = (dists ** 2) / (dists ** 2).sum()
        centroids_idx.append(int(torch.multinomial(probs, 1).item()))
    centroids = X[centroids_idx].clone()

    # Iterate
    for _ in range(100):
        dists = torch.cdist(X, centroids)
        assignments = dists.argmin(dim=1)
        new_centroids = torch.stack([
            X[assignments == k].mean(dim=0) if (assignments == k).any()
            else centroids[k]
            for k in range(n_clusters)
        ])
        if torch.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids

    labels = assignments.cpu().numpy().tolist()
    elapsed = time.perf_counter() - t0

    return {
        "method": f"KMeans-GPU(k={n_clusters})",
        "n_clusters": int(len(set(labels))),
        "labels": labels,
        "time_ms": elapsed * 1000,
    }


# ---------------------------------------------------------------------------
# PyTorch GPU spectral clustering
# ---------------------------------------------------------------------------

def cluster_spectral_gpu(dist_matrix: np.ndarray, n_clusters: int = 2) -> dict:
    """Spectral clustering using PyTorch GPU for eigendecomposition."""
    import torch

    t0 = time.perf_counter()

    # Convert to affinity
    sigma = float(np.median(dist_matrix[dist_matrix > 0]))
    if sigma < 1e-6:
        sigma = 1.0
    W = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    W_t = torch.tensor(W, dtype=torch.float64, device="cuda")

    # Normalized Laplacian on GPU
    D = W_t.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-10))
    L_norm = torch.eye(W_t.shape[0], device="cuda", dtype=torch.float64) - D_inv_sqrt @ W_t @ D_inv_sqrt

    # Eigendecomposition on GPU
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
    embedding = eigenvectors[:, :n_clusters].cpu().numpy()

    # KMeans on the embedding (small data, CPU fine)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embedding)
    elapsed = time.perf_counter() - t0

    sil = -1.0
    if len(set(labels)) > 1:
        sil = float(silhouette_score(dist_matrix, labels, metric="precomputed"))

    return {
        "method": f"Spectral-GPU(k={n_clusters})",
        "n_clusters": int(len(set(labels))),
        "labels": labels.tolist(),
        "silhouette": round(sil, 4),
        "time_ms": elapsed * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_methods(segments: list[dict], dist_matrix: np.ndarray,
                    seg_features: np.ndarray) -> list[dict]:
    """Run all clustering methods and collect results."""
    results = []
    n_segs = len(segments)

    # 1. Current agglomerative (production) at multiple thresholds
    for t in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        results.append(cluster_agglomerative(segments, threshold=t))

    # 2. HDBSCAN variants
    for mcs in [2, 3]:
        for ms in [1, 2]:
            if n_segs >= mcs:
                results.append(cluster_hdbscan(dist_matrix, min_cluster_size=mcs, min_samples=ms))

    # 3. DBSCAN variants
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        results.append(cluster_dbscan(dist_matrix, eps=eps))

    # 4. Spectral clustering (fixed k)
    for k in range(2, min(6, n_segs)):
        results.append(cluster_spectral(dist_matrix, n_clusters=k))

    # 5. Spectral auto-k
    results.append(cluster_spectral_auto(dist_matrix))

    # 6. KMeans (fixed k)
    for k in range(2, min(6, n_segs)):
        results.append(cluster_kmeans(seg_features, n_clusters=k))

    # 7. KMeans auto-k
    results.append(cluster_kmeans_auto(seg_features))

    # 8. GPU KMeans
    for k in range(2, min(6, n_segs)):
        try:
            results.append(cluster_kmeans_gpu(seg_features, n_clusters=k))
        except Exception as e:
            results.append({"method": f"KMeans-GPU(k={k})", "error": str(e)})

    # 9. GPU Spectral
    for k in range(2, min(6, n_segs)):
        try:
            results.append(cluster_spectral_gpu(dist_matrix, n_clusters=k))
        except Exception as e:
            results.append({"method": f"Spectral-GPU(k={k})", "error": str(e)})

    return results


def print_results_table(video_name: str, results: list[dict], expected: int | None):
    """Print a formatted table of clustering results."""
    print(f"\n{'='*90}")
    print(f"VIDEO: {video_name}  (expected clusters: {expected or '?'})")
    print(f"{'='*90}")
    print(f"  {'Method':<35} {'Clusters':>8} {'Noise':>6} {'Silh':>7} {'Time':>8} {'Match':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")

    for r in results:
        if "error" in r:
            print(f"  {r['method']:<35} ERROR: {r['error'][:40]}")
            continue

        nc = r.get("n_clusters", "?")
        noise = r.get("n_noise", "-")
        sil = r.get("silhouette", "-")
        t_ms = r.get("time_ms", 0)
        match = "YES" if expected and nc == expected else ("" if not expected else "no")

        sil_str = f"{sil:.3f}" if isinstance(sil, float) and sil > -1.5 else "-"
        noise_str = str(noise) if noise != "-" else "-"

        print(f"  {r['method']:<35} {nc:>8} {noise_str:>6} {sil_str:>7} {t_ms:>7.1f}ms {match:>6}")


def main():
    print("GPU Clustering Benchmark for BRACE")
    print(f"Videos directory: {VIDEOS_DIR}")
    print()

    # Process each video
    all_video_results = {}
    video_files = sorted(f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4"))

    for fname in video_files:
        path = os.path.join(VIDEOS_DIR, fname)
        expected = EXPECTED_CLUSTERS.get(fname)

        print(f"\nExtracting features from {fname}...")
        t0 = time.perf_counter()
        feats, vi, fps = extract_features(path)
        extract_time = time.perf_counter() - t0

        if feats is None or feats.shape[0] < 10:
            print(f"  Not enough features ({feats.shape[0] if feats is not None else 0}). Skipping.")
            continue

        print(f"  Feature frames: {feats.shape[0]}, dim: {feats.shape[1]}, "
              f"FPS: {fps:.0f}, extraction: {extract_time:.1f}s")

        # Segment
        segments = segment_motions(feats, vi, fps)
        n_segs = len(segments)
        print(f"  Segments: {n_segs}")

        if n_segs < 2:
            print(f"  Only {n_segs} segment(s), trying min_segment_sec=0.5...")
            segments = segment_motions(feats, vi, fps, min_segment_sec=0.5)
            n_segs = len(segments)
            print(f"  Segments with min_segment_sec=0.5: {n_segs}")

        if n_segs < 2:
            print(f"  Still not enough segments. Skipping.")
            continue

        # Print segment info
        for i, seg in enumerate(segments):
            dur = (seg["end_frame"] - seg["start_frame"]) / fps
            print(f"    Seg {i}: frames {seg['start_frame']}-{seg['end_frame']} "
                  f"({dur:.1f}s, {seg['features'].shape[0]} feat frames)")

        # Build distance matrix and segment-level features
        dist_matrix = build_distance_matrix(segments)
        seg_features = build_segment_features(segments)
        print(f"  Segment feature dim: {seg_features.shape[1]}")
        print(f"  Distance range: {dist_matrix[dist_matrix > 0].min():.3f} - {dist_matrix.max():.3f}")

        # Run all clustering methods
        results = run_all_methods(segments, dist_matrix, seg_features)
        print_results_table(fname, results, expected)

        all_video_results[fname] = {
            "n_features": feats.shape[0],
            "n_segments": n_segs,
            "expected_clusters": expected,
            "results": results,
        }

    # ---------------------------------------------------------------------------
    # Summary: which methods get basketball_solo right?
    # ---------------------------------------------------------------------------
    print("\n\n")
    print("=" * 90)
    print("BASKETBALL_SOLO FOCUS: Which methods get 2 clusters?")
    print("=" * 90)

    bb = all_video_results.get("basketball_solo.mp4", {})
    if bb:
        for r in bb.get("results", []):
            if r.get("n_clusters") == 2:
                sil = r.get("silhouette", "-")
                sil_str = f"{sil:.3f}" if isinstance(sil, float) and sil > -1.5 else "-"
                print(f"  PASS: {r['method']:<35} clusters=2  silhouette={sil_str}  "
                      f"time={r.get('time_ms', 0):.1f}ms")

    # ---------------------------------------------------------------------------
    # Summary: cross-video accuracy
    # ---------------------------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("CROSS-VIDEO SUMMARY: Methods that match expected cluster counts")
    print("=" * 90)

    # Collect all unique method names
    method_names = set()
    for vr in all_video_results.values():
        for r in vr.get("results", []):
            method_names.add(r.get("method", ""))

    # For each method, count hits across videos
    method_scores = {}
    for mname in sorted(method_names):
        hits = 0
        total = 0
        for fname, vr in all_video_results.items():
            expected = vr.get("expected_clusters")
            if expected is None:
                continue
            total += 1
            for r in vr.get("results", []):
                if r.get("method") == mname and r.get("n_clusters") == expected:
                    hits += 1
                    break
        method_scores[mname] = (hits, total)

    # Sort by score descending
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1][0], reverse=True)
    print(f"  {'Method':<35} {'Correct':>8} {'Total':>6} {'Rate':>6}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6}")
    for mname, (hits, total) in sorted_methods:
        if total > 0:
            pct = f"{100 * hits / total:.0f}%"
        else:
            pct = "N/A"
        print(f"  {mname:<35} {hits:>8} {total:>6} {pct:>6}")

    # ---------------------------------------------------------------------------
    # Speed comparison
    # ---------------------------------------------------------------------------
    print("\n")
    print("=" * 90)
    print("SPEED COMPARISON (average across videos)")
    print("=" * 90)

    method_times = {}
    for vr in all_video_results.values():
        for r in vr.get("results", []):
            mname = r.get("method", "")
            t = r.get("time_ms", 0)
            if mname and t > 0:
                method_times.setdefault(mname, []).append(t)

    sorted_by_time = sorted(method_times.items(), key=lambda x: np.mean(x[1]))
    print(f"  {'Method':<35} {'Avg ms':>8} {'Min ms':>8} {'Max ms':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for mname, times in sorted_by_time:
        print(f"  {mname:<35} {np.mean(times):>7.2f} {np.min(times):>7.2f} {np.max(times):>7.2f}")


if __name__ == "__main__":
    main()
