"""Tests for unsupervised motion type clustering."""

import numpy as np
import pytest

from brace.core.clustering import (
    prepare_cycle_vectors,
    cluster_kmeans,
    compute_tsne,
    evaluate_clustering,
)
from brace.data.joint_map import NUM_KINECT_JOINTS


def _make_gait_sequence(gait_style="normal", n_frames=120, period=30, seed=42):
    """Generate synthetic gait with different characteristics per style."""
    rng = np.random.RandomState(seed)
    seq = np.zeros((n_frames, NUM_KINECT_JOINTS, 3), dtype=np.float32)
    t = np.arange(n_frames, dtype=np.float32)

    seq[:, 12, 0] = -0.1
    seq[:, 16, 0] = 0.1
    seq[:, 4, :] = [-0.15, -0.3, 0]
    seq[:, 8, :] = [0.15, -0.3, 0]

    amp = 0.1
    if gait_style == "normal":
        seq[:, 14, 1] = amp * np.sin(2 * np.pi * t / period)
        seq[:, 18, 1] = amp * np.sin(2 * np.pi * t / period + np.pi)
    elif gait_style == "antalgic":
        # Asymmetric amplitude (limping)
        seq[:, 14, 1] = amp * 0.3 * np.sin(2 * np.pi * t / period)
        seq[:, 18, 1] = amp * 1.5 * np.sin(2 * np.pi * t / period + np.pi)
        seq[:, 13, 0] += 0.1
    elif gait_style == "stiff":
        # Reduced range of motion
        seq[:, 14, 1] = amp * 0.2 * np.sin(2 * np.pi * t / period)
        seq[:, 18, 1] = amp * 0.2 * np.sin(2 * np.pi * t / period + np.pi)
        seq[:, 13, 1] = 0.01 * np.sin(2 * np.pi * t / period)

    seq += rng.randn(*seq.shape).astype(np.float32) * 0.005
    return seq


def test_prepare_cycle_vectors():
    """Should produce feature vectors with correct dimensions."""
    seqs = [_make_gait_sequence(seed=i) for i in range(5)]
    labels = ["normal"] * 5
    subjects = ["human1"] * 5

    vectors, cycle_labels, cycle_subjects = prepare_cycle_vectors(seqs, labels, subjects)

    assert vectors.ndim == 2
    assert vectors.shape[1] == 60 * 42  # 60 frames * 42 features
    assert len(cycle_labels) == vectors.shape[0]
    assert len(cycle_subjects) == vectors.shape[0]


def test_cluster_kmeans_returns_correct_shape():
    """K-Means should return assignments for all input vectors."""
    seqs = []
    labels = []
    for style in ["normal", "antalgic", "stiff"]:
        for i in range(5):
            seqs.append(_make_gait_sequence(style, seed=i + hash(style) % 1000))
            labels.append(style)

    subjects = ["human1"] * len(seqs)
    vectors, cycle_labels, _ = prepare_cycle_vectors(seqs, labels, subjects)
    assignments, model = cluster_kmeans(vectors, n_clusters=3)

    assert assignments.shape[0] == vectors.shape[0]
    assert set(assignments).issubset({0, 1, 2})


def test_clustering_separates_distinct_gaits():
    """Clustering should achieve above-random ARI for clearly distinct gaits."""
    seqs = []
    labels = []
    for style in ["normal", "antalgic", "stiff"]:
        for i in range(8):
            seqs.append(_make_gait_sequence(style, seed=i + hash(style) % 1000))
            labels.append(style)

    subjects = ["human1"] * len(seqs)
    vectors, cycle_labels, _ = prepare_cycle_vectors(seqs, labels, subjects)

    if vectors.shape[0] < 6:
        pytest.skip("Not enough cycles extracted")

    assignments, _ = cluster_kmeans(vectors, n_clusters=3)
    eval_result = evaluate_clustering(assignments, cycle_labels)

    # ARI > 0 means better than random
    assert eval_result["adjusted_rand_index"] > 0.0, \
        f"ARI={eval_result['adjusted_rand_index']:.3f} not above random"


def test_compute_tsne():
    """t-SNE should produce 2D embedding with correct shape."""
    seqs = [_make_gait_sequence(seed=i) for i in range(10)]
    labels = ["normal"] * 10
    subjects = ["human1"] * 10

    vectors, _, _ = prepare_cycle_vectors(seqs, labels, subjects)

    if vectors.shape[0] < 5:
        pytest.skip("Not enough cycles")

    embedding = compute_tsne(vectors)
    assert embedding.shape == (vectors.shape[0], 2)


def test_evaluate_clustering():
    """Evaluation should produce valid metrics."""
    assignments = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]

    result = evaluate_clustering(assignments, labels)
    assert result["adjusted_rand_index"] == 1.0  # Perfect clustering
    assert result["n_clusters_found"] == 3
