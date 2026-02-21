#!/usr/bin/env python3
"""Cluster all gait types unsupervised and evaluate against ground truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from brace.data.kinect_loader import list_sequences, load_sequence
from brace.core.clustering import (
    prepare_cycle_vectors,
    cluster_kmeans,
    cluster_dbscan,
    compute_tsne,
    evaluate_clustering,
)
from brace.viz.plots import clustering_scatter, confusion_matrix_plot, cross_subject_scatter

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    print("Loading all gait sequences...")
    all_sequences = list_sequences(DATA_ROOT)

    # Sample: take first 5 instances per subject per gait type to keep manageable
    from collections import defaultdict
    counts = defaultdict(int)
    selected = []
    for seq_info in all_sequences:
        key = (seq_info["subject"], seq_info["gait_type"])
        if counts[key] < 5:
            counts[key] += 1
            selected.append(seq_info)

    print(f"Selected {len(selected)} sequences from {len(set(s['subject'] for s in selected))} subjects")

    # Load raw skeleton data
    raw_sequences = []
    labels = []
    subjects = []
    for seq_info in selected:
        data = load_sequence(seq_info)
        if data.shape[0] > 0:
            raw_sequences.append(data)
            labels.append(seq_info["gait_type"])
            subjects.append(seq_info["subject"])

    print(f"Loaded {len(raw_sequences)} valid sequences")
    print("Extracting SRP-normalized cycle features...")

    vectors, cycle_labels, cycle_subjects = prepare_cycle_vectors(
        raw_sequences, labels, subjects, target_cycle_length=60
    )

    print(f"Extracted {vectors.shape[0]} gait cycles, each {vectors.shape[1]}D")
    print()

    # K-Means (k=6)
    print("Running K-Means (k=6)...")
    km_assignments, km_model = cluster_kmeans(vectors, n_clusters=6)
    km_eval = evaluate_clustering(km_assignments, cycle_labels)
    print(f"  Adjusted Rand Index: {km_eval['adjusted_rand_index']:.4f}")
    print(f"  Label-to-cluster map: {km_eval['label_to_cluster']}")
    print()

    # DBSCAN
    print("Running DBSCAN...")
    db_assignments, db_model = cluster_dbscan(vectors, eps=15.0, min_samples=3)
    n_noise = int(np.sum(db_assignments == -1))
    db_eval = evaluate_clustering(db_assignments, cycle_labels)
    print(f"  Clusters found: {db_eval['n_clusters_found']}")
    print(f"  Noise points: {n_noise}")
    print(f"  Adjusted Rand Index: {db_eval['adjusted_rand_index']:.4f}")
    print()

    # t-SNE embedding
    print("Computing t-SNE embedding...")
    embedding = compute_tsne(vectors)

    # Generate plots
    print("Generating plots...")

    clustering_scatter(
        embedding, cycle_labels,
        "t-SNE of Gait Cycles (colored by gait type)",
        OUTPUT_DIR / "clustering" / "tsne_by_gait_type.png",
    )

    km_cluster_labels = [f"C{a}" for a in km_assignments]
    clustering_scatter(
        embedding, km_cluster_labels,
        "t-SNE of Gait Cycles (colored by K-Means cluster)",
        OUTPUT_DIR / "clustering" / "tsne_by_kmeans_cluster.png",
    )

    confusion_matrix_plot(
        km_assignments, cycle_labels,
        OUTPUT_DIR / "clustering" / "kmeans_confusion_matrix.png",
    )

    cross_subject_scatter(
        embedding, cycle_labels, cycle_subjects,
        OUTPUT_DIR / "clustering" / "cross_subject_invariance.png",
    )

    print(f"\nPlots saved to {OUTPUT_DIR / 'clustering'}/")
    print("Done.")


if __name__ == "__main__":
    main()
