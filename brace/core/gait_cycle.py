"""Gait cycle detection via ankle oscillation analysis."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

from ..data.joint_map import ANKLE_LEFT, ANKLE_RIGHT


def _lowpass_filter(signal: np.ndarray, cutoff_hz: float = 3.0, fs: float = 30.0, order: int = 2) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    nyq = fs / 2.0
    norm_cutoff = cutoff_hz / nyq
    if norm_cutoff >= 1.0:
        norm_cutoff = 0.95
    b, a = butter(order, norm_cutoff, btype="low")
    if len(signal) <= 3 * max(len(a), len(b)):
        return signal
    return filtfilt(b, a, signal)


def detect_heel_strikes(
    skeleton_seq: np.ndarray,
    ankle_joint: int = ANKLE_LEFT,
    axis: int = 1,
    fs: float = 30.0,
    min_cycle_frames: int = 15,
) -> np.ndarray:
    """Detect heel strike frames from ankle vertical position.

    Heel strikes correspond to local minima (valleys) of the ankle y-coordinate
    (in Kinect space, y is vertical with lower = closer to ground).

    Args:
        skeleton_seq: (N, 25, 3) raw or normalized skeleton sequence.
        ankle_joint: which ankle to track.
        axis: which axis is vertical (1=y for Kinect).
        fs: sampling rate in Hz.
        min_cycle_frames: minimum frames between heel strikes.

    Returns:
        Array of frame indices where heel strikes occur.
    """
    ankle_y = skeleton_seq[:, ankle_joint, axis].astype(np.float64)

    # Low-pass filter to remove noise
    smoothed = _lowpass_filter(ankle_y, cutoff_hz=3.0, fs=fs)

    # Find valleys (heel strikes = minima of ankle height)
    # Negate to find peaks (since find_peaks finds maxima)
    peaks, _ = find_peaks(-smoothed, distance=min_cycle_frames)

    return peaks


def segment_gait_cycles(
    skeleton_seq: np.ndarray,
    heel_strikes: np.ndarray | None = None,
    fs: float = 30.0,
    min_cycle_frames: int = 15,
) -> list[np.ndarray]:
    """Segment a skeleton sequence into individual gait cycles.

    Each cycle spans from one heel strike to the next.

    Args:
        skeleton_seq: (N, 25, 3) skeleton sequence.
        heel_strikes: pre-computed heel strike indices. If None, auto-detected.
        fs: sampling rate.
        min_cycle_frames: minimum frames between strikes.

    Returns:
        List of (cycle_length, 25, 3) arrays, one per gait cycle.
    """
    if heel_strikes is None:
        heel_strikes = detect_heel_strikes(skeleton_seq, fs=fs, min_cycle_frames=min_cycle_frames)

    if len(heel_strikes) < 2:
        # Not enough strikes to form a cycle; return entire sequence as one cycle
        return [skeleton_seq]

    cycles = []
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        if end - start >= min_cycle_frames:
            cycles.append(skeleton_seq[start:end])

    return cycles


def resample_cycle(cycle: np.ndarray, target_length: int = 60) -> np.ndarray:
    """Resample a gait cycle to a fixed number of frames via linear interpolation.

    Args:
        cycle: (L, 25, 3) one gait cycle.
        target_length: desired output length.

    Returns:
        (target_length, 25, 3) resampled cycle.
    """
    src_len = cycle.shape[0]
    if src_len == target_length:
        return cycle.copy()

    n_joints = cycle.shape[1]
    n_dims = cycle.shape[2]
    out = np.zeros((target_length, n_joints, n_dims), dtype=np.float32)

    src_x = np.linspace(0, 1, src_len)
    tgt_x = np.linspace(0, 1, target_length)

    for j in range(n_joints):
        for d in range(n_dims):
            out[:, j, d] = np.interp(tgt_x, src_x, cycle[:, j, d])

    return out


def extract_resampled_cycles(
    skeleton_seq: np.ndarray,
    target_length: int = 60,
    fs: float = 30.0,
    min_cycle_frames: int = 15,
) -> list[np.ndarray]:
    """Full pipeline: detect cycles, segment, and resample to fixed length.

    Returns:
        List of (target_length, 25, 3) resampled cycles.
    """
    cycles = segment_gait_cycles(skeleton_seq, fs=fs, min_cycle_frames=min_cycle_frames)
    return [resample_cycle(c, target_length) for c in cycles]
