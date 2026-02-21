"""Unsupervised motion type clustering using SRP-normalized gait features."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE

from .srp import normalize_to_body_frame_3d
from .features import extract_features_sequence, z_score_scale
from .gait_cycle import extract_resampled_cycles


def prepare_cycle_vectors(
    sequences: list[np.ndarray],
    labels: list[str],
    subjects: list[str],
    target_cycle_length: int = 60,
    fs: float = 30.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Convert raw skeleton sequences into flat feature vectors for clustering.

    For each sequence:
    1. SRP-normalize
    2. Detect and resample gait cycles
    3. Extract features and flatten

    Args:
        sequences: list of (N, 25, 3) skeleton arrays.
        labels: gait type label for each sequence.
        subjects: subject ID for each sequence.
        target_cycle_length: frames per resampled cycle.
        fs: sampling rate.

    Returns:
        vectors: (M, D) feature matrix where M = total cycles, D = 60*42.
        cycle_labels: gait type for each cycle.
        cycle_subjects: subject ID for each cycle.
    """
    all_vectors = []
    cycle_labels = []
    cycle_subjects = []

    for seq, label, subj in zip(sequences, labels, subjects):
        norm_seq, _ = normalize_to_body_frame_3d(seq)
        cycles = extract_resampled_cycles(norm_seq, target_cycle_length, fs=fs)
        for cycle in cycles:
            feats = extract_features_sequence(cycle)  # (60, 42)
            all_vectors.append(feats.reshape(-1))  # flatten to 2520D
            cycle_labels.append(label)
            cycle_subjects.append(subj)

    if not all_vectors:
        return np.zeros((0, target_cycle_length * 42)), [], []

    vectors = np.stack(all_vectors, axis=0).astype(np.float32)
    return vectors, cycle_labels, cycle_subjects


def cluster_kmeans(
    vectors: np.ndarray,
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans]:
    """Run K-means clustering.

    Returns:
        cluster_assignments: (M,) int array.
        model: fitted KMeans object.
    """
    # Z-score scale before clustering, replacing any NaN/Inf with 0
    scaled, _, _ = z_score_scale(vectors)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    assignments = km.fit_predict(scaled)
    return assignments, km


def cluster_dbscan(
    vectors: np.ndarray,
    eps: float = 10.0,
    min_samples: int = 5,
) -> tuple[np.ndarray, DBSCAN]:
    """Run DBSCAN clustering.

    Returns:
        cluster_assignments: (M,) int array (-1 = noise).
        model: fitted DBSCAN object.
    """
    scaled, _, _ = z_score_scale(vectors)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    assignments = db.fit_predict(scaled)
    return assignments, db


def compute_tsne(vectors: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE 2D embedding of cycle vectors.

    Returns:
        (M, 2) array of 2D coordinates.
    """
    scaled, _, _ = z_score_scale(vectors)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = scaled.shape[0]
    perp = min(perplexity, max(5.0, n_samples / 4.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
    return tsne.fit_transform(scaled)


def evaluate_clustering(
    assignments: np.ndarray,
    true_labels: list[str],
) -> dict:
    """Evaluate clustering quality against ground truth.

    Returns dict with adjusted_rand_index and label-to-cluster mapping.
    """
    # Convert string labels to int
    unique_labels = sorted(set(true_labels))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    true_int = np.array([label_to_int[l] for l in true_labels])

    ari = adjusted_rand_score(true_int, assignments)

    # Build label-to-cluster mapping (majority vote)
    n_clusters = max(assignments.max() + 1, 1)
    label_cluster_map = {}
    for label in unique_labels:
        mask = np.array([l == label for l in true_labels])
        if mask.sum() == 0:
            continue
        cluster_counts = np.bincount(assignments[mask].clip(min=0), minlength=n_clusters)
        label_cluster_map[label] = int(np.argmax(cluster_counts))

    return {
        "adjusted_rand_index": float(ari),
        "label_to_cluster": label_cluster_map,
        "n_clusters_found": int(len(set(assignments) - {-1})),
    }
