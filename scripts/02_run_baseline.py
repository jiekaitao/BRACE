#!/usr/bin/env python3
"""Build baseline models from each subject's normal gait data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brace.data.kinect_loader import load_all_for_subject, list_sequences
from brace.core.baseline import build_baseline, save_baseline

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    sequences = list_sequences(DATA_ROOT)
    subjects = sorted(set(s["subject"] for s in sequences))

    print(f"Found {len(subjects)} subjects: {subjects}")
    print(f"Total sequences: {len(sequences)}")
    print()

    for subject in subjects:
        print(f"Building baseline for {subject}...")
        normal_data = load_all_for_subject(DATA_ROOT, subject, gait_type="normal")

        if not normal_data:
            print(f"  No normal gait data for {subject}, skipping.")
            continue

        raw_sequences = [d for _, d in normal_data]
        print(f"  Loaded {len(raw_sequences)} normal gait sequences")

        try:
            baseline = build_baseline(raw_sequences)
            out_path = OUTPUT_DIR / "baselines" / f"{subject}_baseline.npz"
            save_baseline(baseline, out_path)
            print(f"  Baseline saved: {out_path}")
            print(f"  Cycles extracted: {baseline['n_cycles']}")
            cal = baseline["distance_calibration"]
            print(f"  Distance calibration: p50={cal['p50']:.3f}, p90={cal['p90']:.3f}, p99={cal['p99']:.3f}")
        except ValueError as e:
            print(f"  Error: {e}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
