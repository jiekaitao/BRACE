"""Matplotlib visualizations for BRACE results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def anomaly_dashboard(
    subject: str,
    gait_scores: dict[str, float],
    output_path: str | Path,
) -> None:
    """Bar chart of anomaly scores across gait types for one subject.

    Args:
        subject: subject ID (e.g. "human1").
        gait_scores: {gait_type: mean_anomaly_score}.
        output_path: where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    types = sorted(gait_scores.keys())
    scores = [gait_scores[t] for t in types]

    colors = []
    for t in types:
        if t == "normal":
            colors.append("#58CC02")
        else:
            colors.append("#E74C3C")

    bars = ax.bar(types, scores, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Anomaly Score (std units)", fontsize=12)
    ax.set_title(f"Anomaly Scores — {subject}", fontsize=14)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Threshold = 1.0")
    ax.legend()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{score:.2f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def joint_deviation_heatmap(
    joint_scores_by_gait: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Heatmap of joint deviations across gait types.

    Args:
        joint_scores_by_gait: {gait_type: {joint_name: deviation}}.
        output_path: where to save the figure.
    """
    gait_types = sorted(joint_scores_by_gait.keys())
    if not gait_types:
        return

    joint_names = sorted(joint_scores_by_gait[gait_types[0]].keys())
    matrix = np.zeros((len(gait_types), len(joint_names)))

    for i, gt in enumerate(gait_types):
        for j, jn in enumerate(joint_names):
            matrix[i, j] = joint_scores_by_gait[gt].get(jn, 0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(joint_names)))
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(gait_types)))
    ax.set_yticklabels(gait_types, fontsize=10)
    ax.set_title("Joint Deviation Heatmap by Gait Type", fontsize=14)
    fig.colorbar(im, ax=ax, label="Mean deviation (std units)")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def gait_cycle_overlay(
    normal_features: np.ndarray,
    pathological_features: np.ndarray,
    joint_idx: int,
    joint_name: str,
    gait_type: str,
    output_path: str | Path,
) -> None:
    """Overlay normal vs pathological stride trajectories for a specific joint.

    Args:
        normal_features: (T, 42) mean normal cycle features.
        pathological_features: (T, 42) mean pathological cycle features.
        joint_idx: index into FEATURE_LANDMARKS (0-13).
        joint_name: display name.
        gait_type: pathological gait label.
        output_path: where to save.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    feat_start = joint_idx * 3
    axis_labels = ["X (lateral)", "Y (vertical)", "Z (depth)"]

    phase = np.linspace(0, 100, normal_features.shape[0])

    for d in range(3):
        ax = axes[d]
        ax.plot(phase, normal_features[:, feat_start + d], "g-", linewidth=2, label="Normal")
        ax.plot(phase, pathological_features[:, feat_start + d], "r-", linewidth=2, label=gait_type)
        ax.set_xlabel("Gait cycle %")
        ax.set_ylabel(axis_labels[d])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{joint_name} Trajectory — Normal vs {gait_type}", fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def clustering_scatter(
    embedding: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str | Path,
) -> None:
    """t-SNE scatter plot colored by label.

    Args:
        embedding: (M, 2) t-SNE coordinates.
        labels: label for each point.
        title: plot title.
        output_path: where to save.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique))

    for i, label in enumerate(unique):
        mask = np.array([l == label for l in labels])
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], label=label, alpha=0.6, s=20)

    ax.legend(fontsize=9, markerscale=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def confusion_matrix_plot(
    assignments: np.ndarray,
    true_labels: list[str],
    output_path: str | Path,
) -> None:
    """Plot confusion matrix: cluster assignments vs ground truth gait types."""
    unique_labels = sorted(set(true_labels))
    n_labels = len(unique_labels)
    n_clusters = max(assignments.max() + 1, 1)

    matrix = np.zeros((n_labels, n_clusters), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    for a, l in zip(assignments, true_labels):
        if a >= 0:
            matrix[label_to_idx[l], a] += 1

    fig, ax = plt.subplots(figsize=(max(8, n_clusters), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{i}" for i in range(n_clusters)], fontsize=9)
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(unique_labels, fontsize=10)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("True Gait Type")
    ax.set_title("Confusion Matrix: Clusters vs Ground Truth", fontsize=13)

    for i in range(n_labels):
        for j in range(n_clusters):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cross_subject_scatter(
    embedding: np.ndarray,
    gait_labels: list[str],
    subject_labels: list[str],
    output_path: str | Path,
) -> None:
    """Two side-by-side scatter plots: colored by gait type and by subject."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: by gait type
    unique_gaits = sorted(set(gait_labels))
    cmap1 = plt.cm.get_cmap("tab10", len(unique_gaits))
    for i, g in enumerate(unique_gaits):
        mask = np.array([l == g for l in gait_labels])
        ax1.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap1(i)], label=g, alpha=0.5, s=15)
    ax1.legend(fontsize=8, markerscale=2)
    ax1.set_title("Colored by Gait Type", fontsize=12)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Right: by subject
    unique_subjs = sorted(set(subject_labels))
    cmap2 = plt.cm.get_cmap("tab10", len(unique_subjs))
    for i, s in enumerate(unique_subjs):
        mask = np.array([l == s for l in subject_labels])
        ax2.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap2(i)], label=s, alpha=0.5, s=15)
    ax2.legend(fontsize=7, markerscale=2, ncol=2)
    ax2.set_title("Colored by Subject (shows subject-invariance)", fontsize=12)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    fig.suptitle("Cross-Subject Gait Clustering", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
