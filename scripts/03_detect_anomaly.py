#!/usr/bin/env python3
"""Run anomaly detection: score pathological gaits against each subject's baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from brace.data.kinect_loader import load_all_for_subject, list_sequences, GAIT_TYPES
from brace.core.baseline import build_baseline
from brace.core.anomaly import score_sequence_aggregate
from brace.viz.plots import anomaly_dashboard, joint_deviation_heatmap

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    sequences = list_sequences(DATA_ROOT)
    subjects = sorted(set(s["subject"] for s in sequences))

    all_results = {}

    for subject in subjects:
        print(f"\n{'='*60}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")

        # Build baseline from normal gait
        normal_data = load_all_for_subject(DATA_ROOT, subject, gait_type="normal")
        if not normal_data:
            print("  No normal gait data, skipping.")
            continue

        raw_normal = [d for _, d in normal_data]
        try:
            baseline = build_baseline(raw_normal)
        except ValueError as e:
            print(f"  Baseline error: {e}")
            continue

        print(f"  Baseline: {baseline['n_cycles']} cycles")

        gait_scores = {}
        joint_scores_by_gait = {}

        for gait_type in GAIT_TYPES:
            gait_data = load_all_for_subject(DATA_ROOT, subject, gait_type=gait_type)
            if not gait_data:
                continue

            # Score first 5 sequences to keep it fast
            scores = []
            agg_joints = None
            for seq_info, raw_seq in gait_data[:5]:
                result = score_sequence_aggregate(raw_seq, baseline)
                if not np.isnan(result["mean_anomaly_score"]):
                    scores.append(result["mean_anomaly_score"])
                    if agg_joints is None:
                        agg_joints = result["aggregate_joint_scores"]

            if scores:
                mean_score = float(np.mean(scores))
                gait_scores[gait_type] = mean_score
                if agg_joints:
                    joint_scores_by_gait[gait_type] = agg_joints
                print(f"  {gait_type:20s}: anomaly_score = {mean_score:.3f} (n={len(scores)} sequences)")

        if gait_scores:
            all_results[subject] = gait_scores

            # Generate per-subject plots
            anomaly_dashboard(
                subject, gait_scores,
                OUTPUT_DIR / "anomaly" / f"{subject}_anomaly_dashboard.png",
            )
            if joint_scores_by_gait:
                joint_deviation_heatmap(
                    joint_scores_by_gait,
                    OUTPUT_DIR / "anomaly" / f"{subject}_joint_heatmap.png",
                )

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Anomaly Scores (subject x gait type)")
    print(f"{'='*80}")

    header = f"{'Subject':>10s}"
    for gt in GAIT_TYPES:
        header += f"  {gt:>14s}"
    print(header)
    print("-" * len(header))

    for subject in sorted(all_results.keys()):
        row = f"{subject:>10s}"
        for gt in GAIT_TYPES:
            if gt in all_results[subject]:
                row += f"  {all_results[subject][gt]:>14.3f}"
            else:
                row += f"  {'N/A':>14s}"
        print(row)

    print()


if __name__ == "__main__":
    main()
