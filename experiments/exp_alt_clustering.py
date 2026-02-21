"""Experiment: Compare DBSCAN, Spectral Clustering, and Agglomerative clustering.

Generates 3 synthetic datasets and evaluates each clustering method's ability
to recover the correct number of clusters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score

from brace.core.motion_segments import (
    detect_motion_boundaries,
    segment_motions,
    cluster_segments,
    _resample_segment,
    _segment_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sine_features(n_frames: int, period: int, amplitude: float, phase: float = 0.0,
                       noise: float = 0.02, dim: int = 28) -> np.ndarray:
    """Generate a sine-wave feature trajectory (n_frames, dim)."""
    t = np.linspace(0, n_frames / period * 2 * np.pi, n_frames) + phase
    base = amplitude * np.sin(t)
    features = np.zeros((n_frames, dim), dtype=np.float32)
    for d in range(dim):
        # Each dimension gets a phase-shifted version
        features[:, d] = base * np.cos(d * 0.3) + noise * np.random.randn(n_frames)
    return features


def build_distance_matrix(segments: list[dict]) -> np.ndarray:
    """Build pairwise distance matrix from segments using _segment_distance."""
    n = len(segments)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def generate_dataset_repetitive(seed: int = 42) -> tuple[np.ndarray, list[int], str, int]:
    """300 frames, 10 reps of sine wave. Expected: 1 cluster."""
    np.random.seed(seed)
    features = make_sine_features(300, period=30, amplitude=1.5)
    valid_indices = list(range(300))
    return features, valid_indices, "REPETITIVE (10 reps, expect 1 cluster)", 1


def generate_dataset_two_distinct(seed: int = 43) -> tuple[np.ndarray, list[int], str, int]:
    """300 frames, alternating 2 motion patterns. Expected: 2 clusters."""
    np.random.seed(seed)
    features = np.zeros((300, 28), dtype=np.float32)
    for i in range(10):
        start = i * 30
        end = start + 30
        if i % 2 == 0:
            features[start:end] = make_sine_features(30, period=30, amplitude=1.5, noise=0.02)
        else:
            features[start:end] = make_sine_features(30, period=15, amplitude=3.0, phase=1.0, noise=0.02)
    valid_indices = list(range(300))
    return features, valid_indices, "TWO_DISTINCT (alternating, expect 2 clusters)", 2


def generate_dataset_one_outlier(seed: int = 44) -> tuple[np.ndarray, list[int], str, int]:
    """270 frames repetitive + 30 frames outlier. Expected: 1 main + 1 outlier."""
    np.random.seed(seed)
    main = make_sine_features(270, period=30, amplitude=1.5)
    outlier = make_sine_features(30, period=10, amplitude=5.0, phase=2.0, noise=0.1)
    features = np.vstack([main, outlier])
    valid_indices = list(range(300))
    return features, valid_indices, "ONE_OUTLIER (9 reps + 1 outlier, expect 2 clusters)", 2


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def run_dbscan(dist_matrix: np.ndarray) -> list[dict]:
    """Run DBSCAN with various eps/min_samples. Returns list of result dicts."""
    results = []
    for eps in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        for min_samples in [1, 2]:
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            labels = db.fit_predict(dist_matrix)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "labels": labels.tolist(),
            })
    return results


def run_spectral(dist_matrix: np.ndarray) -> list[dict]:
    """Run Spectral Clustering with various n_clusters. Returns list of result dicts."""
    results = []
    sigma = np.median(dist_matrix[dist_matrix > 0])
    if sigma < 1e-6:
        sigma = 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    for n_clusters_try in [1, 2, 3, 4]:
        if n_clusters_try >= dist_matrix.shape[0]:
            continue
        try:
            if n_clusters_try == 1:
                # Spectral clustering needs at least 2 clusters
                labels = np.zeros(dist_matrix.shape[0], dtype=int)
                sil = -1.0
            else:
                sc = SpectralClustering(
                    n_clusters=n_clusters_try, affinity='precomputed',
                    random_state=42, assign_labels='kmeans',
                )
                labels = sc.fit_predict(affinity)
                if len(set(labels)) > 1:
                    sil = float(silhouette_score(dist_matrix, labels, metric='precomputed'))
                else:
                    sil = -1.0
            results.append({
                "n_clusters": n_clusters_try,
                "silhouette": round(sil, 4),
                "labels": labels.tolist(),
            })
        except Exception as e:
            results.append({
                "n_clusters": n_clusters_try,
                "silhouette": -1.0,
                "labels": [],
                "error": str(e),
            })
    return results


def run_agglomerative(segments: list[dict]) -> list[dict]:
    """Run current agglomerative clustering at various thresholds."""
    import copy
    results = []
    for threshold in [1.0, 1.5, 2.0, 3.0, 5.0]:
        segs_copy = copy.deepcopy(segments)
        clustered = cluster_segments(segs_copy, distance_threshold=threshold)
        labels = [s["cluster"] for s in clustered]
        n_clusters = len(set(labels))
        results.append({
            "threshold": threshold,
            "n_clusters": n_clusters,
            "labels": labels,
        })
    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 80):
    print(char * width)


def print_dbscan_table(results: list[dict], expected: int):
    print(f"\n  A) DBSCAN (expected: {expected} cluster(s))")
    print(f"  {'eps':>5}  {'min_s':>5}  {'n_clust':>7}  {'n_noise':>7}  {'correct':>7}  labels")
    print(f"  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
    for r in results:
        correct = "YES" if r["n_clusters"] == expected else "no"
        label_str = str(r["labels"][:15])
        if len(r["labels"]) > 15:
            label_str += "..."
        print(f"  {r['eps']:5.1f}  {r['min_samples']:5d}  {r['n_clusters']:7d}  "
              f"{r['n_noise']:7d}  {correct:>7}  {label_str}")


def print_spectral_table(results: list[dict], expected: int):
    print(f"\n  B) Spectral Clustering (expected: {expected} cluster(s))")
    print(f"  {'n_clust':>7}  {'silhouette':>10}  {'correct':>7}  labels")
    print(f"  {'-'*7}  {'-'*10}  {'-'*7}  {'-'*20}")
    best_sil = -2.0
    best_k = -1
    for r in results:
        if r["silhouette"] > best_sil and r["n_clusters"] > 1:
            best_sil = r["silhouette"]
            best_k = r["n_clusters"]
    for r in results:
        pick = "<-- best" if r["n_clusters"] == best_k and r["n_clusters"] > 1 else ""
        correct = "YES" if r["n_clusters"] == expected else "no"
        label_str = str(r.get("labels", [])[:15])
        print(f"  {r['n_clusters']:7d}  {r['silhouette']:10.4f}  {correct:>7}  {label_str} {pick}")
    if best_k > 0:
        auto_correct = "YES" if best_k == expected else "no"
        print(f"  --> Auto-selected k={best_k} (silhouette={best_sil:.4f}) correct={auto_correct}")


def print_agglom_table(results: list[dict], expected: int):
    print(f"\n  C) Agglomerative / current (expected: {expected} cluster(s))")
    print(f"  {'threshold':>9}  {'n_clust':>7}  {'correct':>7}  labels")
    print(f"  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*20}")
    for r in results:
        correct = "YES" if r["n_clusters"] == expected else "no"
        label_str = str(r["labels"][:15])
        if len(r["labels"]) > 15:
            label_str += "..."
        print(f"  {r['threshold']:9.1f}  {r['n_clusters']:7d}  {correct:>7}  {label_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    datasets = [
        generate_dataset_repetitive(),
        generate_dataset_two_distinct(),
        generate_dataset_one_outlier(),
    ]

    # Track correctness across methods
    summary = {"DBSCAN": 0, "Spectral": 0, "Agglomerative": 0}
    total_datasets = len(datasets)

    for features, valid_indices, name, expected_clusters in datasets:
        print_separator()
        print(f"DATASET: {name}")
        print(f"  Shape: {features.shape}, Expected clusters: {expected_clusters}")
        print_separator("-")

        # Segment
        segments = segment_motions(features, valid_indices, fps=30.0, min_segment_sec=0.8)
        print(f"  Segmented into {len(segments)} segments")

        if len(segments) < 2:
            print("  WARNING: Too few segments for clustering comparison. Skipping.")
            continue

        # Build distance matrix
        dist_matrix = build_distance_matrix(segments)
        print(f"  Distance matrix: min={dist_matrix[dist_matrix > 0].min():.3f}, "
              f"max={dist_matrix.max():.3f}, median={np.median(dist_matrix[dist_matrix > 0]):.3f}")

        # A) DBSCAN
        dbscan_results = run_dbscan(dist_matrix)
        print_dbscan_table(dbscan_results, expected_clusters)
        # Count DBSCAN as correct if any parameter combo works
        if any(r["n_clusters"] == expected_clusters for r in dbscan_results):
            summary["DBSCAN"] += 1

        # B) Spectral
        spectral_results = run_spectral(dist_matrix)
        print_spectral_table(spectral_results, expected_clusters)
        # Count spectral as correct if auto-selected k matches
        best_sil = -2.0
        best_k = -1
        for r in spectral_results:
            if r["silhouette"] > best_sil and r["n_clusters"] > 1:
                best_sil = r["silhouette"]
                best_k = r["n_clusters"]
        if best_k == expected_clusters:
            summary["Spectral"] += 1

        # C) Agglomerative
        agglom_results = run_agglomerative(segments)
        print_agglom_table(agglom_results, expected_clusters)
        # Count agglom as correct if default threshold (2.0) works
        default_result = next((r for r in agglom_results if r["threshold"] == 2.0), None)
        if default_result and default_result["n_clusters"] == expected_clusters:
            summary["Agglomerative"] += 1

    # Final summary
    print("\n")
    print_separator("=")
    print("SUMMARY: Which method gets CORRECT cluster count?")
    print_separator("=")
    print(f"  {'Method':<20}  {'Correct':<10}  {'Total':<10}  {'Rate':<10}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    for method, correct in summary.items():
        rate = f"{correct}/{total_datasets}"
        pct = f"{100*correct/total_datasets:.0f}%"
        print(f"  {method:<20}  {correct:<10}  {total_datasets:<10}  {rate} ({pct})")

    print(f"\n  Notes:")
    print(f"  - DBSCAN: counted as correct if ANY (eps, min_samples) combo gives expected count")
    print(f"  - Spectral: counted as correct if silhouette-selected k matches expected count")
    print(f"  - Agglomerative: counted as correct if default threshold=2.0 gives expected count")
    print()


if __name__ == "__main__":
    main()
