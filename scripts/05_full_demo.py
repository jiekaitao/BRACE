#!/usr/bin/env python3
"""End-to-end BRACE demo: baseline → anomaly detection → clustering → plots."""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from brace.data.kinect_loader import list_sequences, load_sequence, GAIT_TYPES
from brace.core.baseline import build_baseline
from brace.core.anomaly import score_sequence_aggregate
from brace.core.features import extract_features_sequence
from brace.core.srp import normalize_to_body_frame_3d
from brace.core.gait_cycle import extract_resampled_cycles
from brace.core.clustering import (
    prepare_cycle_vectors,
    cluster_kmeans,
    compute_tsne,
    evaluate_clustering,
)
from brace.viz.plots import (
    anomaly_dashboard,
    joint_deviation_heatmap,
    gait_cycle_overlay,
    clustering_scatter,
    confusion_matrix_plot,
    cross_subject_scatter,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    print("=" * 70)
    print("  BRACE — Full Demo")
    print("  SRP-based gait anomaly detection on pathological gait data")
    print("=" * 70)
    print()

    # ── Step 1: Load dataset ──
    print("[1/5] Loading dataset...")
    all_seqs = list_sequences(DATA_ROOT)
    subjects = sorted(set(s["subject"] for s in all_seqs))
    print(f"  {len(all_seqs)} sequences, {len(subjects)} subjects, {len(GAIT_TYPES)} gait types")

    # ── Step 2: Build baselines and score anomalies ──
    print("\n[2/5] Building baselines and scoring anomalies...")
    all_results = {}
    all_joint_scores = {}

    for subject in subjects:
        # Get normal sequences
        normal_seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == "normal"]
        if not normal_seqs:
            continue

        raw_normal = []
        for seq_info in normal_seqs[:10]:  # up to 10 sequences for baseline
            data = load_sequence(seq_info)
            if data.shape[0] > 0:
                raw_normal.append(data)

        if not raw_normal:
            continue

        try:
            baseline = build_baseline(raw_normal)
        except ValueError:
            continue

        gait_scores = {}
        joint_scores_by_gait = {}

        for gait_type in GAIT_TYPES:
            gait_seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == gait_type]
            scores = []
            agg_joints = None
            for seq_info in gait_seqs[:5]:
                data = load_sequence(seq_info)
                if data.shape[0] == 0:
                    continue
                result = score_sequence_aggregate(data, baseline)
                if not np.isnan(result["mean_anomaly_score"]):
                    scores.append(result["mean_anomaly_score"])
                    if agg_joints is None:
                        agg_joints = result["aggregate_joint_scores"]

            if scores:
                gait_scores[gait_type] = float(np.mean(scores))
                if agg_joints:
                    joint_scores_by_gait[gait_type] = agg_joints

        all_results[subject] = gait_scores
        all_joint_scores[subject] = joint_scores_by_gait
        print(f"  {subject}: scored {len(gait_scores)} gait types")

    # ── Step 3: Generate anomaly plots ──
    print("\n[3/5] Generating anomaly plots...")
    for subject, gait_scores in all_results.items():
        if gait_scores:
            anomaly_dashboard(
                subject, gait_scores,
                OUTPUT_DIR / "anomaly" / f"{subject}_dashboard.png",
            )
            if subject in all_joint_scores and all_joint_scores[subject]:
                joint_deviation_heatmap(
                    all_joint_scores[subject],
                    OUTPUT_DIR / "anomaly" / f"{subject}_joint_heatmap.png",
                )

    # Generate gait cycle overlays for first subject with data
    for subject in subjects:
        if subject not in all_results or "normal" not in all_results[subject]:
            continue

        normal_seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == "normal"]
        antalgic_seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == "antalgic"]
        if not normal_seqs or not antalgic_seqs:
            continue

        # Get mean cycle features for normal and antalgic
        normal_data = load_sequence(normal_seqs[0])
        antalgic_data = load_sequence(antalgic_seqs[0])

        norm_normal, _ = normalize_to_body_frame_3d(normal_data)
        norm_antalgic, _ = normalize_to_body_frame_3d(antalgic_data)

        normal_cycles = extract_resampled_cycles(norm_normal, 60)
        antalgic_cycles = extract_resampled_cycles(norm_antalgic, 60)

        if normal_cycles and antalgic_cycles:
            normal_feats = np.mean([extract_features_sequence(c) for c in normal_cycles], axis=0)
            antalgic_feats = np.mean([extract_features_sequence(c) for c in antalgic_cycles], axis=0)

            # Plot knee (joint index 7 in FEATURE_LANDMARKS = KneeLeft at index 13)
            gait_cycle_overlay(
                normal_feats, antalgic_feats,
                joint_idx=7, joint_name="KneeLeft",
                gait_type="Antalgic",
                output_path=OUTPUT_DIR / "anomaly" / f"{subject}_knee_overlay.png",
            )
            # Plot ankle (joint index 8 = AnkleLeft at index 14)
            gait_cycle_overlay(
                normal_feats, antalgic_feats,
                joint_idx=8, joint_name="AnkleLeft",
                gait_type="Antalgic",
                output_path=OUTPUT_DIR / "anomaly" / f"{subject}_ankle_overlay.png",
            )
        break  # Only do for first subject

    print(f"  Saved to {OUTPUT_DIR / 'anomaly'}/")

    # ── Step 4: Clustering ──
    print("\n[4/5] Running unsupervised clustering...")

    # Sample sequences for clustering
    counts = defaultdict(int)
    selected = []
    for seq_info in all_seqs:
        key = (seq_info["subject"], seq_info["gait_type"])
        if counts[key] < 3:
            counts[key] += 1
            selected.append(seq_info)

    raw_seqs, seq_labels, seq_subjects = [], [], []
    for seq_info in selected:
        data = load_sequence(seq_info)
        if data.shape[0] > 0:
            raw_seqs.append(data)
            seq_labels.append(seq_info["gait_type"])
            seq_subjects.append(seq_info["subject"])

    vectors, cycle_labels, cycle_subjects = prepare_cycle_vectors(
        raw_seqs, seq_labels, seq_subjects
    )
    print(f"  {vectors.shape[0]} gait cycles extracted ({vectors.shape[1]}D)")

    km_assignments, _ = cluster_kmeans(vectors, n_clusters=6)
    km_eval = evaluate_clustering(km_assignments, cycle_labels)
    print(f"  K-Means ARI: {km_eval['adjusted_rand_index']:.4f}")

    embedding = compute_tsne(vectors)

    clustering_scatter(
        embedding, cycle_labels,
        "Gait Cycles by Type (t-SNE)",
        OUTPUT_DIR / "clustering" / "tsne_gait_type.png",
    )
    confusion_matrix_plot(
        km_assignments, cycle_labels,
        OUTPUT_DIR / "clustering" / "confusion_matrix.png",
    )
    cross_subject_scatter(
        embedding, cycle_labels, cycle_subjects,
        OUTPUT_DIR / "clustering" / "cross_subject.png",
    )
    print(f"  Saved to {OUTPUT_DIR / 'clustering'}/")

    # ── Step 5: Summary ──
    print("\n[5/5] Results Summary")
    print("=" * 80)
    header = f"{'Subject':>10s}"
    for gt in GAIT_TYPES:
        header += f"  {gt:>14s}"
    print(header)
    print("-" * len(header))

    normal_scores = []
    patho_scores = []

    for subject in sorted(all_results.keys()):
        row = f"{subject:>10s}"
        for gt in GAIT_TYPES:
            if gt in all_results[subject]:
                score = all_results[subject][gt]
                row += f"  {score:>14.3f}"
                if gt == "normal":
                    normal_scores.append(score)
                else:
                    patho_scores.append(score)
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    print()
    if normal_scores and patho_scores:
        print(f"Normal gait mean score:       {np.mean(normal_scores):.3f} +/- {np.std(normal_scores):.3f}")
        print(f"Pathological gait mean score: {np.mean(patho_scores):.3f} +/- {np.std(patho_scores):.3f}")
        print(f"Separation ratio:             {np.mean(patho_scores) / max(np.mean(normal_scores), 1e-6):.1f}x")
    print(f"K-Means clustering ARI:       {km_eval['adjusted_rand_index']:.4f}")
    print(f"  (Random baseline = 0.0, perfect = 1.0)")
    print()

    # Save results JSON
    results_json = {
        "anomaly_scores": {s: {k: round(v, 4) for k, v in gs.items()} for s, gs in all_results.items()},
        "clustering": {
            "adjusted_rand_index": round(km_eval["adjusted_rand_index"], 4),
            "n_cycles": int(vectors.shape[0]),
            "label_to_cluster": km_eval["label_to_cluster"],
        },
        "summary": {
            "normal_mean": round(float(np.mean(normal_scores)), 4) if normal_scores else None,
            "pathological_mean": round(float(np.mean(patho_scores)), 4) if patho_scores else None,
        },
    }
    results_path = OUTPUT_DIR / "demo_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_json, indent=2))
    print(f"Results saved to {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
