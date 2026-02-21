"""Parse kooksung pathological gait skeleton data from Kinect v2 CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .joint_map import NUM_KINECT_JOINTS

# Gait types in the dataset
GAIT_TYPES = ["normal", "antalgic", "lurch", "steppage", "stiff_legged", "trendelenburg"]


def parse_skeleton_csv(csv_path: str | Path) -> np.ndarray:
    """Parse a single Kinect skeleton CSV file.

    Format: tab-separated rows, each row is:
        timestamp  0  x  y  z  1  x  y  z  ...  24  x  y  z

    Returns:
        (num_frames, 25, 3) float32 array of joint positions.
    """
    frames = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split("\t")
            # First value is timestamp, then 25 joints * 4 values (idx, x, y, z) = 100
            # Total expected: 1 + 100 = 101
            if len(vals) < 1 + NUM_KINECT_JOINTS * 4:
                continue
            joints = np.zeros((NUM_KINECT_JOINTS, 3), dtype=np.float64)
            valid = True
            for j in range(NUM_KINECT_JOINTS):
                base = 1 + j * 4  # skip timestamp, then each joint is (idx, x, y, z)
                # vals[base] is the joint index (0-24), vals[base+1..3] are x,y,z
                x, y, z = float(vals[base + 1]), float(vals[base + 2]), float(vals[base + 3])
                if abs(x) > 1e6 or abs(y) > 1e6 or abs(z) > 1e6:
                    valid = False
                    break
                joints[j, 0] = x
                joints[j, 1] = y
                joints[j, 2] = z
            if valid:
                frames.append(joints.astype(np.float32))
    if not frames:
        return np.zeros((0, NUM_KINECT_JOINTS, 3), dtype=np.float32)
    return np.stack(frames, axis=0)


def _parse_dir_name(dirname: str) -> tuple[str, str, int] | None:
    """Parse 'human3_antalgic12' -> ('human3', 'antalgic', 12)."""
    # Handle stiff_legged specially since it has underscore in the gait name
    m = re.match(r"(human\d+)_(stiff_legged|antalgic|lurch|steppage|normal|trendelenburg)(\d+)$", dirname)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def list_sequences(data_root: str | Path) -> list[dict]:
    """Enumerate all sequences in the kooksung dataset.

    Returns list of dicts: {subject, gait_type, instance, dir_path, csv_files}
    """
    data_root = Path(data_root)
    gaits_dir = data_root / "pathological_gait_datasets" / "Pathological_Gaits"
    if not gaits_dir.exists():
        # Try alternate layout
        gaits_dir = data_root / "Pathological_Gaits"
    if not gaits_dir.exists():
        raise FileNotFoundError(f"Cannot find Pathological_Gaits directory under {data_root}")

    sequences = []
    for subdir in sorted(gaits_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parsed = _parse_dir_name(subdir.name)
        if parsed is None:
            continue
        subject, gait_type, instance = parsed
        csvs = sorted(subdir.glob("*.csv"))
        if csvs:
            sequences.append({
                "subject": subject,
                "gait_type": gait_type,
                "instance": instance,
                "dir_path": subdir,
                "csv_files": csvs,
            })
    return sequences


def load_sequence(seq_info: dict, sensor_idx: int = 0) -> np.ndarray:
    """Load skeleton data for one sequence from a specific sensor.

    Args:
        seq_info: dict from list_sequences()
        sensor_idx: which sensor CSV to use (0-based). Default 0 = first sensor.

    Returns:
        (num_frames, 25, 3) float32 array.
    """
    csvs = seq_info["csv_files"]
    if sensor_idx >= len(csvs):
        sensor_idx = 0
    return parse_skeleton_csv(csvs[sensor_idx])


def load_all_for_subject(
    data_root: str | Path,
    subject: str,
    gait_type: str | None = None,
    sensor_idx: int = 0,
) -> list[tuple[dict, np.ndarray]]:
    """Load all sequences for a subject, optionally filtered by gait type.

    Returns list of (seq_info, skeleton_array) tuples.
    """
    sequences = list_sequences(data_root)
    results = []
    for seq in sequences:
        if seq["subject"] != subject:
            continue
        if gait_type is not None and seq["gait_type"] != gait_type:
            continue
        data = load_sequence(seq, sensor_idx)
        if data.shape[0] > 0:
            results.append((seq, data))
    return results
