"""Movement quality assessment: biomechanical metrics, fatigue detection, form scoring.

Synthesizes methods from biomechanics (joint angles, bilateral asymmetry, ROM),
signal processing (SPARC, spectral entropy, cross-correlation),
topology/geometry (curvature, jerk, loop spread, path efficiency),
statistics (EWMA control charts, CUSUM change-point detection),
and ML (Mahalanobis distance, PCA reconstruction error).

All computations are designed for real-time streaming at 30fps.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kendalltau

from brace.core.pose import FEATURE_INDICES

try:
    from risk_profile import RiskModifiers, apply_modifiers
except ImportError:
    try:
        from backend.risk_profile import RiskModifiers, apply_modifiers
    except ImportError:
        RiskModifiers = None
        apply_modifiers = None


# --- Joint chain definitions for angle computation ---
# Maps joint names to (proximal, middle, distal) indices in FEATURE_INDICES order
# FEATURE_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]
#                    0  1  2  3  4  5  6  7  8  9 10 11 12 13
JOINT_CHAINS = {
    "left_knee": (6, 8, 10),    # hip23, knee25, ankle27
    "right_knee": (7, 9, 11),   # hip24, knee26, ankle28
    "left_elbow": (0, 2, 4),    # shoulder11, elbow13, wrist15
    "right_elbow": (1, 3, 5),   # shoulder12, elbow14, wrist16
    "left_hip": (0, 6, 8),      # shoulder11, hip23, knee25
    "right_hip": (1, 7, 9),     # shoulder12, hip24, knee26
}

# Bilateral pairs for asymmetry tracking
BILATERAL_PAIRS = [
    ("left_knee", "right_knee"),
    ("left_elbow", "right_elbow"),
    ("left_hip", "right_hip"),
]

# Joint names for display (indexed by position in FEATURE_INDICES)
JOINT_NAMES = [
    "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
    "L Wrist", "R Wrist", "L Hip", "R Hip",
    "L Knee", "R Knee", "L Ankle", "R Ankle",
    "L Foot", "R Foot",
]


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by segments p1-p2 and p3-p2. Returns degrees [0, 180]."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def compute_fppa(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float:
    """Frontal Plane Projection Angle for knee valgus (Hewett 2005).

    Measures the lateral deviation of the knee from the hip-ankle line,
    expressed as an angle in degrees. This isolates valgus/varus from
    normal knee flexion.

    Returns degrees. Negative = valgus (knee medial to hip-ankle line).
    Positive = varus (knee lateral).
    """
    # Vector from hip to ankle (the reference line)
    ha = ankle - hip
    ha_len = float(np.linalg.norm(ha))
    if ha_len < 1e-8:
        return 0.0
    ha_unit = ha / ha_len
    # Vector from hip to knee
    hk = knee - hip
    # Project knee onto the hip-ankle line
    proj_len = float(np.dot(hk, ha_unit))
    proj = ha_unit * proj_len
    # Perpendicular deviation of knee from the hip-ankle line
    perp = hk - proj
    perp_dist = float(np.linalg.norm(perp))
    if perp_dist < 1e-8:
        return 0.0
    # Use half the hip-ankle distance as the reference length for the angle
    # (approximation of thigh length for angular conversion)
    ref_len = max(ha_len * 0.5, 1e-8)
    angle = math.degrees(math.atan2(perp_dist, ref_len))
    # Determine sign: use X-component of perpendicular to determine
    # medial vs lateral. In image coords (Y-down), for the LEFT leg,
    # medial (valgus) means the knee deviates to the RIGHT (+X).
    # We use the 2D cross product of ha and hk to determine side.
    if ha.shape[0] >= 2:
        cross = float(ha[0] * hk[1] - ha[1] * hk[0])
    else:
        cross = 0.0
    return -angle if cross > 0 else angle


def compute_hip_drop(left_hip: np.ndarray, right_hip: np.ndarray,
                     y_up: bool = False) -> float:
    """Pelvic obliquity angle from horizontal in degrees.

    0 = level pelvis. Positive = left hip higher. Negative = left hip lower.
    Measures the Y-difference relative to the hip vector length.

    Args:
        y_up: If False (default), coordinates are image-space (Y-down).
    """
    delta = left_hip - right_hip
    hip_width = np.linalg.norm(delta)
    if hip_width < 1e-8 or delta.shape[0] < 2:
        return 0.0
    # In Y-down image space, lower Y = higher position, so negate.
    y_diff = -delta[1] if not y_up else delta[1]
    return float(math.degrees(math.asin(np.clip(y_diff / hip_width, -1.0, 1.0))))


def compute_trunk_lean(
    left_shoulder: np.ndarray, right_shoulder: np.ndarray,
    left_hip: np.ndarray, right_hip: np.ndarray,
    ndim: int = 2,
    y_up: bool = False,
) -> float:
    """Trunk lateral lean from vertical in degrees.

    Args:
        y_up: If False (default), coordinates are image-space (Y-down),
              so "up" is [0, -1]. If True, Y increases upward.
    """
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    trunk_vec = shoulder_mid - hip_mid
    # Vertical direction depends on coordinate system
    y_dir = 1.0 if y_up else -1.0
    if ndim == 2:
        vertical = np.array([0.0, y_dir])
    else:
        vertical = np.array([0.0, y_dir, 0.0])
    trunk_norm = np.linalg.norm(trunk_vec)
    if trunk_norm < 1e-8:
        return 0.0
    cos_angle = np.dot(trunk_vec[:len(vertical)], vertical) / trunk_norm
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    sign = 1.0 if trunk_vec[0] > 0 else -1.0
    return sign * angle


def bilateral_asymmetry_index(left_val: float, right_val: float) -> float:
    """BAI as percentage (Bishop 2018). 0% = perfect symmetry."""
    denom = max(abs(left_val), abs(right_val))
    if denom < 1e-8:
        return 0.0
    return abs(left_val - right_val) / denom * 100.0


def sparc(speed_profile: np.ndarray, fps: float = 30.0,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length (Balasubramanian 2012). More negative = less smooth."""
    if len(speed_profile) < 4:
        return 0.0
    nfft = int(2 ** np.ceil(np.log2(len(speed_profile)) + 4))
    freq = np.fft.rfftfreq(nfft, d=1.0 / fps)
    Mf = np.abs(np.fft.rfft(speed_profile, n=nfft))
    mx = Mf.max()
    if mx < 1e-10:
        return 0.0
    Mf = Mf / mx

    mask_fc = freq <= fc
    freq_sel = freq[mask_fc]
    Mf_sel = Mf[mask_fc]

    above_th = np.where(Mf_sel > amp_th)[0]
    if len(above_th) == 0:
        return 0.0
    idx_cut = above_th[-1]
    freq_sel = freq_sel[:idx_cut + 1]
    Mf_sel = Mf_sel[:idx_cut + 1]
    if len(freq_sel) < 2:
        return 0.0

    df = np.diff(freq_sel) / max(freq_sel[-1] - freq_sel[0], 1e-10)
    dMf = np.diff(Mf_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dMf ** 2)))


def log_dimensionless_jerk(positions: np.ndarray, fps: float = 30.0) -> float:
    """LDLJ smoothness metric (Balasubramanian 2015).

    More negative = jerkier (higher jerk integral). Less negative = smoother.
    """
    if len(positions) < 4:
        return -10.0  # default "very smooth"
    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    v_peak = np.max(speed)
    if v_peak < 1e-8:
        return -10.0
    jerk = np.diff(positions, n=3, axis=0) / (dt ** 3)
    jerk_sq = np.sum(jerk ** 2, axis=1)
    T = len(positions) / fps
    integral = np.sum(jerk_sq) * dt
    dj = (T ** 3 / v_peak ** 2) * integral
    return float(-np.log(max(dj, 1e-10)))


def spectral_entropy(signal: np.ndarray) -> float:
    """Shannon entropy of normalized PSD. 0 = pure sinusoid, 1 = white noise."""
    if len(signal) < 4:
        return 0.0
    signal = signal - np.mean(signal)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    psd = psd[1:]  # remove DC
    psd_sum = np.sum(psd)
    if psd_sum < 1e-10:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)))
    return H / max(np.log(len(p)), 1e-10)


def rep_cross_correlation(rep_i: np.ndarray, rep_j: np.ndarray,
                          resample_len: int = 60) -> float:
    """Pearson correlation between two resampled reps (flattened)."""
    def _resample(rep: np.ndarray, L: int) -> np.ndarray:
        N, D = rep.shape
        x_old = np.linspace(0, 1, N)
        x_new = np.linspace(0, 1, L)
        out = np.zeros((L, D), dtype=np.float32)
        for d in range(D):
            out[:, d] = np.interp(x_new, x_old, rep[:, d])
        return out

    r_i = _resample(rep_i, resample_len).flatten()
    r_j = _resample(rep_j, resample_len).flatten()
    cc = np.corrcoef(r_i, r_j)
    if cc.shape == (2, 2):
        return float(cc[0, 1])
    return 0.0


# --- Bone pairs for bone-length projection (indices into FEATURE_INDICES 14-joint layout) ---
# FEATURE_INDICES = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]
#                    0  1  2  3  4  5  6  7  8  9 10 11 12 13
BONE_PAIRS = [
    (0, 2),   # L shoulder -> L elbow
    (1, 3),   # R shoulder -> R elbow
    (2, 4),   # L elbow -> L wrist
    (3, 5),   # R elbow -> R wrist
    (6, 8),   # L hip -> L knee
    (7, 9),   # R hip -> R knee
    (8, 10),  # L knee -> L ankle
    (9, 11),  # R knee -> R ankle
]

# Anthropometric segment mass fractions (Winter 1990) for CoM estimation
# Maps (parent_idx, child_idx) -> (mass_fraction, com_position_from_proximal)
SEGMENT_MASSES = {
    (0, 2): (0.028, 0.43),   # L upper arm
    (1, 3): (0.028, 0.43),   # R upper arm
    (2, 4): (0.016, 0.43),   # L forearm
    (3, 5): (0.016, 0.43),   # R forearm
    (6, 8): (0.100, 0.43),   # L thigh
    (7, 9): (0.100, 0.43),   # R thigh
    (8, 10): (0.0465, 0.43), # L shank
    (9, 11): (0.0465, 0.43), # R shank
    (10, 12): (0.0145, 0.43), # L foot
    (11, 13): (0.0145, 0.43), # R foot
}
TRUNK_MASS = 0.497  # trunk mass fraction


def project_bone_lengths(
    joints: np.ndarray, target_lengths: dict[tuple[int, int], float]
) -> np.ndarray:
    """Project joints to enforce target bone lengths while preserving direction.

    Args:
        joints: (14, ndim) joint positions.
        target_lengths: dict mapping (parent_idx, child_idx) -> target length.

    Returns:
        (14, ndim) corrected joint positions.
    """
    out = joints.copy()
    for parent, child in BONE_PAIRS:
        if (parent, child) not in target_lengths:
            continue
        target_len = target_lengths[(parent, child)]
        direction = out[child] - out[parent]
        current_len = float(np.linalg.norm(direction))
        if current_len < 1e-8:
            continue
        unit = direction / current_len
        out[child] = out[parent] + unit * target_len
    return out


class BoneLengthFilter:
    """EMA-based bone length stabilizer. Enforces consistent bone lengths."""

    def __init__(self, warmup: int = 30, alpha: float = 0.05):
        self._warmup = warmup
        self._alpha = alpha
        self._frame_count = 0
        self._ema_lengths: dict[tuple[int, int], float] = {}

    def update(self, joints: np.ndarray) -> np.ndarray:
        """Measure bone lengths, update EMA, project to enforce mean lengths.

        Returns corrected joints (or raw joints during warmup).
        """
        self._frame_count += 1

        # Measure current bone lengths
        for parent, child in BONE_PAIRS:
            length = float(np.linalg.norm(joints[child] - joints[parent]))
            key = (parent, child)
            if key not in self._ema_lengths:
                self._ema_lengths[key] = length
            else:
                self._ema_lengths[key] = (
                    self._alpha * length + (1 - self._alpha) * self._ema_lengths[key]
                )

        # Only project after warmup
        if self._frame_count < self._warmup:
            return joints

        return project_bone_lengths(joints, self._ema_lengths)

    def reset(self) -> None:
        self._frame_count = 0
        self._ema_lengths.clear()


def evaluate_injury_risks(
    biomechanics: dict[str, float],
    angular_velocities: dict[str, float] | None = None,
    profile: Any | None = None,
    modifiers: Any = None,
) -> list[dict[str, Any]]:
    """Evaluate injury risks from biomechanical metrics using clinical thresholds.

    Args:
        biomechanics: Dict of metric name -> value.
        angular_velocities: Dict of joint name -> angular velocity (deg/s).
        profile: Optional MovementProfile with movement-specific thresholds.
                 Falls back to hardcoded generic thresholds when None.
        modifiers: Optional RiskModifiers for personalized threshold scaling.
                   Only used when profile is None.

    Returns list of risk alerts with joint, risk type, severity, value, threshold.
    """
    # When a profile is provided, use its thresholds (profile takes precedence)
    if profile is not None:
        return _evaluate_with_profile(biomechanics, angular_velocities, profile)

    # If we have risk modifiers but no movement profile, use modified generic thresholds
    if modifiers is not None and apply_modifiers is not None:
        modified_thresholds = apply_modifiers(modifiers)
        return _evaluate_with_modified_thresholds(biomechanics, angular_velocities, modified_thresholds)

    # --- Hardcoded generic fallback (backward-compatible) ---
    risks: list[dict[str, Any]] = []

    # ACL risk: |FPPA| thresholds (Hewett 2005)
    for side, key in [("left_knee", "fppa_left"), ("right_knee", "fppa_right")]:
        val = abs(biomechanics.get(key, 0.0))
        if val > 25.0:
            risks.append({"joint": side, "risk": "acl_valgus", "severity": "high",
                          "value": round(val, 1), "threshold": 25.0})
        elif val > 15.0:
            risks.append({"joint": side, "risk": "acl_valgus", "severity": "medium",
                          "value": round(val, 1), "threshold": 15.0})

    # Hip drop risk
    val = abs(biomechanics.get("hip_drop", 0.0))
    if val > 12.0:
        risks.append({"joint": "pelvis", "risk": "hip_drop", "severity": "high",
                      "value": round(val, 1), "threshold": 12.0})
    elif val > 8.0:
        risks.append({"joint": "pelvis", "risk": "hip_drop", "severity": "medium",
                      "value": round(val, 1), "threshold": 8.0})

    # Trunk lean risk
    val = abs(biomechanics.get("trunk_lean", 0.0))
    if val > 25.0:
        risks.append({"joint": "trunk", "risk": "trunk_lean", "severity": "high",
                      "value": round(val, 1), "threshold": 25.0})
    elif val > 15.0:
        risks.append({"joint": "trunk", "risk": "trunk_lean", "severity": "medium",
                      "value": round(val, 1), "threshold": 15.0})

    # Bilateral asymmetry risk
    val = biomechanics.get("asymmetry", 0.0)
    if val > 25.0:
        risks.append({"joint": "bilateral", "risk": "asymmetry", "severity": "high",
                      "value": round(val, 1), "threshold": 25.0})
    elif val > 15.0:
        risks.append({"joint": "bilateral", "risk": "asymmetry", "severity": "medium",
                      "value": round(val, 1), "threshold": 15.0})

    # Knee angular velocity spike
    if angular_velocities:
        for side in ["left_knee", "right_knee"]:
            vel = angular_velocities.get(side, 0.0)
            if vel > 500.0:
                risks.append({"joint": side, "risk": "angular_velocity_spike",
                              "severity": "medium", "value": round(vel, 1),
                              "threshold": 500.0})

    return risks


def _evaluate_with_profile(
    biomechanics: dict[str, float],
    angular_velocities: dict[str, float] | None,
    profile: Any,
) -> list[dict[str, Any]]:
    """Evaluate injury risks using a MovementProfile's thresholds."""
    risks: list[dict[str, Any]] = []

    # Map metric names to how we read values from biomechanics/angular_velocities
    _METRIC_READERS = {
        "fppa": lambda joint, bio, av: abs(bio.get(
            "fppa_left" if "left" in joint else "fppa_right", 0.0)),
        "hip_drop": lambda joint, bio, av: abs(bio.get("hip_drop", 0.0)),
        "trunk_lean": lambda joint, bio, av: abs(bio.get("trunk_lean", 0.0)),
        "asymmetry": lambda joint, bio, av: bio.get("asymmetry", 0.0),
        "angular_velocity": lambda joint, bio, av: (
            av.get(joint, 0.0) if av else 0.0),
    }

    for threshold in profile.thresholds:
        if not threshold.enabled:
            continue

        reader = _METRIC_READERS.get(threshold.metric)
        if reader is None:
            continue

        val = reader(threshold.joint, biomechanics, angular_velocities)

        if val > threshold.high:
            risks.append({
                "joint": threshold.joint,
                "risk": threshold.risk_name,
                "severity": "high",
                "value": round(val, 1),
                "threshold": threshold.high,
            })
        elif val > threshold.medium:
            risks.append({
                "joint": threshold.joint,
                "risk": threshold.risk_name,
                "severity": "medium",
                "value": round(val, 1),
                "threshold": threshold.medium,
            })

    return risks


def _evaluate_with_modified_thresholds(
    biomechanics: dict[str, float],
    angular_velocities: dict[str, float] | None,
    thresholds: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Evaluate injury risks using modified threshold values."""
    risks: list[dict[str, Any]] = []

    fppa_t = thresholds.get("fppa", {"medium": 15.0, "high": 25.0})
    for side, key in [("left_knee", "fppa_left"), ("right_knee", "fppa_right")]:
        val = abs(biomechanics.get(key, 0.0))
        if val > fppa_t["high"]:
            risks.append({"joint": side, "risk": "acl_valgus", "severity": "high",
                          "value": round(val, 1), "threshold": fppa_t["high"]})
        elif val > fppa_t["medium"]:
            risks.append({"joint": side, "risk": "acl_valgus", "severity": "medium",
                          "value": round(val, 1), "threshold": fppa_t["medium"]})

    hip_t = thresholds.get("hip_drop", {"medium": 8.0, "high": 12.0})
    val = abs(biomechanics.get("hip_drop", 0.0))
    if val > hip_t["high"]:
        risks.append({"joint": "pelvis", "risk": "hip_drop", "severity": "high",
                      "value": round(val, 1), "threshold": hip_t["high"]})
    elif val > hip_t["medium"]:
        risks.append({"joint": "pelvis", "risk": "hip_drop", "severity": "medium",
                      "value": round(val, 1), "threshold": hip_t["medium"]})

    trunk_t = thresholds.get("trunk_lean", {"medium": 15.0, "high": 25.0})
    val = abs(biomechanics.get("trunk_lean", 0.0))
    if val > trunk_t["high"]:
        risks.append({"joint": "trunk", "risk": "trunk_lean", "severity": "high",
                      "value": round(val, 1), "threshold": trunk_t["high"]})
    elif val > trunk_t["medium"]:
        risks.append({"joint": "trunk", "risk": "trunk_lean", "severity": "medium",
                      "value": round(val, 1), "threshold": trunk_t["medium"]})

    asym_t = thresholds.get("asymmetry", {"medium": 15.0, "high": 25.0})
    val = biomechanics.get("asymmetry", 0.0)
    if val > asym_t["high"]:
        risks.append({"joint": "bilateral", "risk": "asymmetry", "severity": "high",
                      "value": round(val, 1), "threshold": asym_t["high"]})
    elif val > asym_t["medium"]:
        risks.append({"joint": "bilateral", "risk": "asymmetry", "severity": "medium",
                      "value": round(val, 1), "threshold": asym_t["medium"]})

    av_t = thresholds.get("angular_velocity", {"medium": 500.0})
    if angular_velocities:
        for side in ["left_knee", "right_knee"]:
            vel = angular_velocities.get(side, 0.0)
            if vel > av_t["medium"]:
                risks.append({"joint": side, "risk": "angular_velocity_spike",
                              "severity": "medium", "value": round(vel, 1),
                              "threshold": av_t["medium"]})

    return risks


def sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy (SampEn) of a 1D signal.

    Low SampEn = regular/predictable, high SampEn = complex/irregular.
    Fatigue typically increases entropy (more erratic movement).

    Args:
        signal: 1D array.
        m: Embedding dimension.
        r_factor: Tolerance as fraction of signal std.

    Returns:
        SampEn value (float). Returns 0.0 for signals too short or constant.
    """
    N = len(signal)
    if N < m + 2:
        return 0.0
    std = float(np.std(signal))
    if std < 1e-10:
        return 0.0
    r = r_factor * std

    def _count_matches(dim: int) -> int:
        templates = np.array([signal[i:i + dim] for i in range(N - dim)])
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return 0.0
    return float(-np.log(max(A, 1) / B))


def estimate_center_of_mass(joints_14: np.ndarray) -> np.ndarray:
    """Estimate whole-body center of mass from 14 feature joints.

    Uses Winter (1990) anthropometric segment mass fractions.
    Each segment's CoM is at ~43% from the proximal end.

    Args:
        joints_14: (14, ndim) joint positions.

    Returns:
        (ndim,) weighted CoM position.
    """
    ndim = joints_14.shape[1]
    total_mass = 0.0
    weighted_pos = np.zeros(ndim, dtype=np.float64)

    # Trunk: midpoint of shoulders to midpoint of hips
    shoulder_mid = (joints_14[0] + joints_14[1]) / 2.0
    hip_mid = (joints_14[6] + joints_14[7]) / 2.0
    trunk_com = hip_mid + 0.43 * (shoulder_mid - hip_mid)
    weighted_pos += TRUNK_MASS * trunk_com
    total_mass += TRUNK_MASS

    # Limb segments
    for (parent, child), (mass, com_frac) in SEGMENT_MASSES.items():
        if parent < joints_14.shape[0] and child < joints_14.shape[0]:
            seg_com = joints_14[parent] + com_frac * (joints_14[child] - joints_14[parent])
            weighted_pos += mass * seg_com
            total_mass += mass

    if total_mass < 1e-8:
        return np.zeros(ndim, dtype=np.float64)
    return weighted_pos / total_mass


def compute_kinematic_sequence(
    angular_velocity_history: dict[str, list[float]],
    ideal_order: list[str] | None = None,
) -> dict[str, Any]:
    """Compute kinematic chain sequencing score.

    Ideal proximal-to-distal: hip peaks first, then trunk, shoulder, elbow.
    Uses Kendall's tau correlation of peak timing with ideal order.

    Args:
        angular_velocity_history: joint_name -> list of angular velocities over recent frames.
        ideal_order: Expected proximal-to-distal joint order.

    Returns:
        dict with sequence_score (0-1), peak_times, timing_gaps.
    """
    if ideal_order is None:
        ideal_order = ["left_hip", "right_hip", "left_knee", "right_knee",
                       "left_elbow", "right_elbow"]

    # Find peak timing for each joint
    peak_times: dict[str, int] = {}
    for name in ideal_order:
        if name not in angular_velocity_history:
            continue
        vals = angular_velocity_history[name]
        if len(vals) < 3:
            continue
        peak_idx = int(np.argmax(vals))
        peak_times[name] = peak_idx

    if len(peak_times) < 3:
        return {"sequence_score": None, "peak_times": peak_times, "timing_gaps": []}

    # Build actual order from peak times
    ordered_joints = sorted(peak_times.keys(), key=lambda j: peak_times[j])
    # Map to ideal rank
    ideal_ranks = {name: i for i, name in enumerate(ideal_order) if name in peak_times}
    actual_ranks = [ideal_ranks[j] for j in ordered_joints]
    expected_ranks = sorted(actual_ranks)

    # Kendall's tau
    tau, _ = kendalltau(actual_ranks, expected_ranks)
    # Normalize to 0-1 (tau is -1 to 1)
    score = max(0.0, (tau + 1.0) / 2.0)

    # Timing gaps between successive peaks
    times_ordered = [peak_times[j] for j in ordered_joints]
    gaps = [times_ordered[i + 1] - times_ordered[i] for i in range(len(times_ordered) - 1)]

    return {
        "sequence_score": round(score, 3),
        "peak_times": peak_times,
        "timing_gaps": gaps,
    }


def median_frequency(signal: np.ndarray, fps: float = 30.0) -> float:
    """Spectral median frequency: frequency that divides PSD area into equal halves.

    MNF decrease across reps = fatigue indicator (used with angular velocity as proxy).

    Args:
        signal: 1D array (e.g., angular velocity time series).
        fps: Sampling rate in Hz.

    Returns:
        Median frequency in Hz. Returns 0.0 for short/constant signals.
    """
    if len(signal) < 4:
        return 0.0
    signal = signal - np.mean(signal)
    if np.std(signal) < 1e-10:
        return 0.0

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    psd = np.abs(np.fft.rfft(signal)) ** 2
    psd = psd[1:]  # remove DC
    freqs = freqs[1:]

    if len(psd) == 0 or np.sum(psd) < 1e-10:
        return 0.0

    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    idx = np.searchsorted(cumulative, total / 2.0)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


class StreamingCurvature:
    """Per-frame curvature and jerk in feature space. O(D) per frame."""

    def __init__(self):
        self._buf: list[np.ndarray] = []  # last 4 frames

    def update(self, feat: np.ndarray) -> tuple[float, float]:
        """Returns (curvature, jerk_magnitude) for current frame."""
        self._buf.append(feat.copy())
        if len(self._buf) > 4:
            self._buf.pop(0)

        if len(self._buf) < 3:
            return 0.0, 0.0

        v1 = self._buf[-2] - self._buf[-3]
        v2 = self._buf[-1] - self._buf[-2]
        a = v2 - v1
        speed = np.linalg.norm(v1)

        if speed < 1e-8:
            kappa = 0.0
        else:
            v_hat = v1 / speed
            a_perp = a - np.dot(a, v_hat) * v_hat
            kappa = float(np.linalg.norm(a_perp) / (speed ** 2))

        # Jerk
        jerk_mag = 0.0
        if len(self._buf) >= 4:
            a_prev = self._buf[-2] - 2 * self._buf[-3] + (self._buf[-4] if len(self._buf) >= 4 else self._buf[-3])
            a_curr = a
            jerk_mag = float(np.linalg.norm(a_curr - a_prev))

        return kappa, jerk_mag

    def reset(self) -> None:
        self._buf.clear()


class MovementQualityTracker:
    """Per-subject movement quality tracking across multiple clusters.

    Computes per-frame biomechanical metrics and per-segment quality analysis.
    Thread-safe: results read via get_* methods are snapshots.
    """

    def __init__(self, fps: float = 30.0, n_baseline_reps: int = 5, risk_modifiers: Any = None):
        self.fps = fps
        self.n_baseline_reps = n_baseline_reps
        self._risk_modifiers = risk_modifiers

        # Per-frame streaming state
        self._curvature_tracker = StreamingCurvature()
        self._bone_filter = BoneLengthFilter()
        self._frame_count = 0

        # Per-frame metric EWMAs (smoothed for display)
        self._curvature_ema = 0.0
        self._jerk_ema = 0.0
        self._asymmetry_ema = 0.0

        # Joint angle history (for trend computation)
        self._joint_score_history: deque[np.ndarray] = deque(maxlen=120)  # ~4s

        # Angular velocity tracking (Phase 1B)
        self._prev_joint_angles: dict[str, float] = {}
        self._angular_velocities: dict[str, float] = {}
        self._peak_angular_velocities: dict[str, float] = {}
        self._angular_velocity_history: dict[str, deque] = {
            name: deque(maxlen=300) for name in JOINT_CHAINS
        }

        # Isolation Forest anomaly scoring (Phase 1C)
        self._biomech_history: deque[np.ndarray] = deque(maxlen=300)
        self._anomaly_model: Any = None
        self._anomaly_score: float | None = None
        self._anomaly_refit_counter = 0

        # Injury risk tracking (Phase 1D)
        self._current_injury_risks: list[dict[str, Any]] = []

        # Center of Mass tracking (Phase 3A)
        self._com_history: deque[np.ndarray] = deque(maxlen=300)
        self._com_velocity = 0.0
        self._com_sway = 0.0

        # Per-cluster state
        self._cluster_state: dict[int, _ClusterQualityState] = {}

        # Current frame results (updated per-frame, read by build_response)
        self._current_angles: dict[str, float] = {}
        self._current_fppa_left = 0.0
        self._current_fppa_right = 0.0
        self._current_hip_drop = 0.0
        self._current_trunk_lean = 0.0
        self._current_asymmetry = 0.0
        self._current_curvature = 0.0
        self._current_jerk = 0.0
        self._current_form_score: float | None = None
        self._current_joint_scores: np.ndarray | None = None
        self._current_joint_deviations: np.ndarray | None = None
        self._current_degrading_joints: list[int] = []
        self._current_movement_phase: dict[str, Any] | None = None

        # Concussion / Head tracking (head-specific kinematics)
        self._head_prev_pos: np.ndarray | None = None  # nose position (2D, hip-centered px)
        self._head_prev_vel: np.ndarray | None = None  # nose velocity (m/s)
        self._head_prev_ear_angle: float | None = None  # ear-to-ear angle (rad)
        self._head_vel_baseline_ema: float = 0.0  # adaptive baseline of head speed
        self._head_vel_baseline_initialized: bool = False
        self._head_linear_accel_g: float = 0.0  # current head acceleration (g, display only)
        self._head_angular_vel: float = 0.0  # current head angular velocity (rad/s)
        self._head_angular_vel_ema: float = 0.0  # EMA-smoothed angular velocity
        self._head_spike_z: float = 0.0  # z-score of current head speed vs baseline
        self._concussion_peak_value: float = 0.0  # peak-hold value
        self._concussion_peak_hold_frames: int = 0  # frames remaining in hold
        self._current_concussion_rating = 0.0
        
        # Fatigue Rating
        self._current_fatigue_rating = 0.0

        # Movement guideline profile (set from Gemini activity label)
        self._current_activity_label: str | None = None
        self._current_profile: Any = None  # MovementProfile or None

        # Fatigue timeline (sampled every 30 frames = 1/sec)
        self._fatigue_timeline_timestamps: list[float] = []
        self._fatigue_timeline_fatigue: list[float] = []
        self._fatigue_timeline_form: list[float] = []
        self._fatigue_timeline_version = 0
        self._last_sent_timeline_version = 0

    def set_activity_label(self, label: str | None) -> None:
        """Update the movement profile based on Gemini activity label."""
        try:
            from backend.movement_guidelines import match_guideline
        except ImportError:
            from movement_guidelines import match_guideline
        self._current_activity_label = label
        self._current_profile = match_guideline(label) if label else None

    def process_frame(
        self,
        srp_joints: np.ndarray,
        cluster_id: int | None,
        seg_info: dict | None,
        representative_joints: np.ndarray | None,
        fatigue_index: float,
        video_time: float = 0.0,
        raw_joints: np.ndarray | None = None,
        joint_vis: list[float] | None = None,
        head_landmarks: np.ndarray | None = None,
        shoulder_width_px: float = 0.0,
    ) -> None:
        """Per-frame quality assessment. Call after SRP normalization.

        Args:
            srp_joints: (14, 2) or (14, 3) SRP-normalized feature joints.
            cluster_id: Current cluster assignment (None if unassigned).
            seg_info: Current segment dict with start_valid, end_valid, cluster.
            representative_joints: (14, 2) or (14, 3) from cluster representative at current phase.
            fatigue_index: Current velocity-based fatigue index.
            video_time: Current video timestamp for timeline.
            raw_joints: (14, 2) or (14, 3) hip-centered un-rotated joints for biomechanics.
                        Falls back to srp_joints if not provided.
            joint_vis: Per-feature-joint visibility (0-1). Indices match FEATURE_INDICES.
                       When provided, biomechanical angles are skipped for invisible joints.
            head_landmarks: (3, 2) hip-centered [nose, left_ear, right_ear] in pixels, or None.
            shoulder_width_px: Inter-shoulder distance in pixels for scale estimation.
        """
        self._frame_count += 1
        ndim = srp_joints.shape[1] if srp_joints.ndim == 2 else 2

        # --- Bone-length projection filter (Phase 1A) ---
        srp_joints = self._bone_filter.update(srp_joints)

        # Use raw (un-rotated) joints for biomechanical angles.
        # SRP rotation destroys frontal-plane geometry needed for FPPA/trunk lean.
        bio_joints = raw_joints if raw_joints is not None else srp_joints

        # --- Biomechanical angles ---
        # Helper: check if all joints in a set are visible (vis >= 0.3)
        def _vis_ok(indices: list[int]) -> bool:
            if joint_vis is None:
                return True  # backward compat: no vis data → assume all visible
            return all(
                joint_vis[i] >= 0.3 for i in indices if i < len(joint_vis)
            )

        freshly_computed_angles: set[str] = set()
        for name, (i, j, k) in JOINT_CHAINS.items():
            if _vis_ok([i, j, k]):
                self._current_angles[name] = compute_joint_angle(
                    bio_joints[i], bio_joints[j], bio_joints[k]
                )
                freshly_computed_angles.add(name)
            # else: hold last computed value (already in self._current_angles)

        # FPPA (knee valgus) — skip when leg joints not visible
        if _vis_ok([6, 8, 10]):  # L hip, L knee, L ankle
            self._current_fppa_left = compute_fppa(
                bio_joints[6], bio_joints[8], bio_joints[10]
            )
        if _vis_ok([7, 9, 11]):  # R hip, R knee, R ankle
            self._current_fppa_right = compute_fppa(
                bio_joints[7], bio_joints[9], bio_joints[11]
            )

        # Hip drop — skip when hips not visible
        if _vis_ok([6, 7]):
            self._current_hip_drop = compute_hip_drop(bio_joints[6], bio_joints[7])

        # Trunk lean — skip when shoulders or hips not visible
        if _vis_ok([0, 1, 6, 7]):
            self._current_trunk_lean = compute_trunk_lean(
                bio_joints[0], bio_joints[1],  # L/R shoulder
                bio_joints[6], bio_joints[7],  # L/R hip
                ndim=ndim,
            )

        # --- Angular velocity (Phase 1B) ---
        # Only update for joints whose angle was freshly computed this frame
        for name in JOINT_CHAINS:
            if name not in freshly_computed_angles:
                continue  # skip stale angles to avoid false velocity spikes
            angle = self._current_angles.get(name, 0.0)
            if name in self._prev_joint_angles:
                omega = abs(angle - self._prev_joint_angles[name]) * self.fps  # deg/s
                self._angular_velocities[name] = omega
                self._angular_velocity_history[name].append(omega)
                # EMA-decaying peak
                prev_peak = self._peak_angular_velocities.get(name, 0.0)
                self._peak_angular_velocities[name] = max(omega, prev_peak * 0.99)
            self._prev_joint_angles[name] = angle

        # --- Bilateral asymmetry ---
        asym_vals = []
        for left_name, right_name in BILATERAL_PAIRS:
            if left_name in self._current_angles and right_name in self._current_angles:
                bai = bilateral_asymmetry_index(
                    self._current_angles[left_name],
                    self._current_angles[right_name],
                )
                asym_vals.append(bai)
        self._current_asymmetry = float(np.mean(asym_vals)) if asym_vals else 0.0
        alpha = 0.1
        self._asymmetry_ema = alpha * self._current_asymmetry + (1 - alpha) * self._asymmetry_ema

        # --- Curvature & jerk ---
        feat = srp_joints.flatten()
        kappa, jerk = self._curvature_tracker.update(feat)
        self._current_curvature = kappa
        self._current_jerk = jerk
        self._curvature_ema = 0.1 * kappa + 0.9 * self._curvature_ema
        self._jerk_ema = 0.1 * jerk + 0.9 * self._jerk_ema

        # --- Concussion Risk / Head Kinematics ---
        self._update_concussion_rating(head_landmarks, shoulder_width_px)

        # --- Fatigue Rating ---
        # Convert fatigue_index (0.0 - 1.0) to a 0-100 scale rating
        self._current_fatigue_rating = float(np.clip(fatigue_index * 100.0, 0.0, 100.0))

        # --- Center of Mass (Phase 3A) ---
        com = estimate_center_of_mass(srp_joints)
        self._com_history.append(com)
        if len(self._com_history) >= 2:
            self._com_velocity = float(np.linalg.norm(
                self._com_history[-1] - self._com_history[-2]
            )) * self.fps
        # CoM sway: RMS deviation from base of support (ankle midpoint)
        ankle_mid = (srp_joints[10] + srp_joints[11]) / 2.0
        self._com_sway = float(np.linalg.norm(com[:len(ankle_mid)] - ankle_mid))

        # --- Isolation Forest anomaly scoring (Phase 1C) ---
        biomech_vec = np.array([
            self._current_fppa_left, self._current_fppa_right,
            self._current_hip_drop, self._current_trunk_lean,
            self._current_asymmetry, self._current_curvature,
            self._current_jerk,
        ])
        self._biomech_history.append(biomech_vec)
        self._anomaly_refit_counter += 1

        if self._frame_count >= 60 and self._anomaly_refit_counter >= 60:
            self._anomaly_refit_counter = 0
            try:
                from sklearn.ensemble import IsolationForest
                history_arr = np.array(list(self._biomech_history))
                if history_arr.shape[0] >= 30:
                    self._anomaly_model = IsolationForest(
                        n_estimators=50, contamination=0.1, random_state=42
                    )
                    self._anomaly_model.fit(history_arr)
            except ImportError:
                self._anomaly_model = None

        if self._anomaly_model is not None:
            raw_score = self._anomaly_model.decision_function(biomech_vec.reshape(1, -1))[0]
            # Convert: negative = anomalous, map to 0-1 where 1 = most anomalous
            self._anomaly_score = float(np.clip(-raw_score, 0.0, 1.0))
        elif self._frame_count < 60:
            self._anomaly_score = None

        # --- Injury risk evaluation (Phase 1D) ---
        biomech_dict = {
            "fppa_left": self._current_fppa_left,
            "fppa_right": self._current_fppa_right,
            "hip_drop": self._current_hip_drop,
            "trunk_lean": self._current_trunk_lean,
            "asymmetry": self._asymmetry_ema,
        }
        self._current_injury_risks = evaluate_injury_risks(
            biomech_dict,
            self._angular_velocities if self._angular_velocities else None,
            profile=self._current_profile,
            modifiers=self._risk_modifiers,
        )

        # --- Form score (deviation from cluster representative) ---
        self._current_form_score = None
        self._current_joint_scores = None
        self._current_joint_deviations = None
        if representative_joints is not None and cluster_id is not None:
            rep = representative_joints
            if rep.shape == srp_joints.shape:
                deviations = np.linalg.norm(srp_joints - rep, axis=1)  # (14,)
                self._current_joint_deviations = deviations.copy()
                # Per-joint score: 0-1 where 1 = perfect match
                # Threshold: 1.0 hip-width unit = maximum expected deviation
                threshold = 1.0
                scores = np.clip(1.0 - deviations / threshold, 0.0, 1.0)
                self._current_joint_scores = scores
                # Overall form score 0-100
                self._current_form_score = float(np.mean(scores) * 100.0)

        # Store joint scores for trend computation
        if self._current_joint_scores is not None:
            self._joint_score_history.append(self._current_joint_scores.copy())

        # --- Degrading joints (recent vs early window) ---
        self._current_degrading_joints = []
        if len(self._joint_score_history) >= 60:
            history = np.array(list(self._joint_score_history))
            early = history[:30]
            recent = history[-30:]
            early_mean = np.mean(early, axis=0)
            recent_mean = np.mean(recent, axis=0)
            for j in range(14):
                if early_mean[j] > 0.1:  # only track if joint was active
                    drop = (early_mean[j] - recent_mean[j]) / early_mean[j]
                    if drop > 0.10:  # >10% drop = degrading
                        self._current_degrading_joints.append(j)

        # --- Movement phase detection ---
        self._current_movement_phase = None
        if seg_info is not None and cluster_id is not None:
            start_vi = seg_info.get("start_valid", 0)
            end_vi = seg_info.get("end_valid", 1)
            seg_len = max(end_vi - start_vi, 1)
            # Progress within segment
            valid_idx = seg_info.get("_current_valid_idx", start_vi)
            progress = min(1.0, max(0.0, (valid_idx - start_vi) / max(seg_len - 1, 1)))

            # Phase label from velocity direction
            # Simple heuristic: first half = ascending, second half = descending
            if progress < 0.1 or progress > 0.9:
                phase_label = "transition"
            elif progress < 0.5:
                phase_label = "ascending"
            else:
                phase_label = "descending"

            # Cycle count from cluster state
            cs = self._cluster_state.get(cluster_id)
            cycle_count = cs.n_reps if cs else 0

            self._current_movement_phase = {
                "label": phase_label,
                "progress": round(progress, 3),
                "cycle_count": cycle_count,
            }

        # --- Fatigue timeline sampling (every ~1 second) ---
        if self._frame_count % 30 == 0 and self._current_form_score is not None:
            self._fatigue_timeline_timestamps.append(round(video_time, 2))
            self._fatigue_timeline_fatigue.append(round(fatigue_index, 3))
            self._fatigue_timeline_form.append(round(self._current_form_score, 1))
            self._fatigue_timeline_version += 1

    def _update_concussion_rating(
        self,
        head_landmarks: np.ndarray | None,
        shoulder_width_px: float,
    ) -> None:
        """Compute concussion risk from head-specific kinematics.

        Scoring uses head **velocity** (not acceleration) as the primary signal.
        Acceleration requires double-differentiating noisy pixel positions at
        30fps, which amplifies detection jitter into phantom 10-30g readings
        that swamp real thresholds.  Velocity (single differentiation) is far
        more stable and still discriminates normal motion from impacts.

        Components:
          - Head speed (m/s) with dead zone at 5 m/s  →  0-40 pts
          - EMA-smoothed angular velocity (rad/s) at 12 rad/s  →  0-35 pts
          - Velocity z-score spike at z > 5  →  0-25 pts

        Peak-hold + exponential decay prevents flickering.

        Args:
            head_landmarks: (3, 2) hip-centered [nose, left_ear, right_ear] in pixels,
                            or None when head keypoints are not visible.
            shoulder_width_px: Inter-shoulder distance in pixels (for px→m scaling).
        """
        dt = 1.0 / self.fps
        _PEAK_HOLD_FRAMES = 45  # ~1.5s at 30fps
        _DECAY_FACTOR = 0.92

        if head_landmarks is None:
            # Graceful degradation: decay existing score, don't accumulate
            if self._concussion_peak_hold_frames > 0:
                self._concussion_peak_hold_frames -= 1
            else:
                self._concussion_peak_value *= _DECAY_FACTOR
            self._current_concussion_rating = float(np.clip(self._concussion_peak_value, 0.0, 100.0))
            return

        nose = head_landmarks[0]  # (2,)
        left_ear = head_landmarks[1]
        right_ear = head_landmarks[2]

        # --- Pixel → meter scaling ---
        # Average shoulder width ~0.40m (anthropometric reference)
        if shoulder_width_px > 1.0:
            px_to_m = 0.40 / shoulder_width_px
        else:
            px_to_m = 0.002  # fallback ~480px frame

        # --- Head linear velocity (m/s) ---
        if self._head_prev_pos is not None:
            displacement_px = nose - self._head_prev_pos
            head_vel = displacement_px * px_to_m / dt  # m/s
        else:
            head_vel = np.zeros(2)
        self._head_prev_pos = nose.copy()

        head_speed = float(np.linalg.norm(head_vel))

        # --- Head linear acceleration (g) — stored for display, NOT used in scoring ---
        if self._head_prev_vel is not None:
            accel_vec = (head_vel - self._head_prev_vel) / dt
            accel_g = float(np.linalg.norm(accel_vec)) / 9.81
        else:
            accel_g = 0.0
        self._head_prev_vel = head_vel.copy()
        self._head_linear_accel_g = accel_g

        # --- Head angular velocity (rad/s) ---
        ear_vec = right_ear - left_ear
        ear_angle = float(np.arctan2(ear_vec[1], ear_vec[0]))
        if self._head_prev_ear_angle is not None:
            # Unwrap angle difference to handle ±π crossover
            d_angle = ear_angle - self._head_prev_ear_angle
            d_angle = (d_angle + math.pi) % (2 * math.pi) - math.pi
            angular_vel = abs(d_angle) / dt  # rad/s
        else:
            angular_vel = 0.0
        self._head_prev_ear_angle = ear_angle
        self._head_angular_vel = angular_vel

        # EMA-smooth angular velocity to reject single-frame ear-detection jitter
        _ANGULAR_EMA_ALPHA = 0.3
        self._head_angular_vel_ema = (
            _ANGULAR_EMA_ALPHA * angular_vel
            + (1 - _ANGULAR_EMA_ALPHA) * self._head_angular_vel_ema
        )

        # --- Adaptive baseline (EMA of head speed) ---
        # α=0.05: adapts in ~20 frames (0.7s) so normal athletic motion
        # quickly raises the baseline, keeping z-scores low during play.
        _BASELINE_ALPHA = 0.05
        if not self._head_vel_baseline_initialized:
            self._head_vel_baseline_ema = head_speed
            self._head_vel_baseline_initialized = True
        else:
            self._head_vel_baseline_ema = (
                _BASELINE_ALPHA * head_speed
                + (1 - _BASELINE_ALPHA) * self._head_vel_baseline_ema
            )

        # Z-score of current speed vs baseline
        baseline = max(self._head_vel_baseline_ema, 0.01)
        z_score = (head_speed - baseline) / baseline
        self._head_spike_z = max(z_score, 0.0)

        # --- Composite score (0–100) ---
        #
        # Velocity-based scoring.  Acceleration is NOT used because
        # double-differentiation of 30fps pixel positions amplifies
        # detection noise into phantom 10-30g spikes during normal play.
        #
        # Typical basketball head speeds:
        #   Running/jumping: 1-3 m/s,  head turns: 5-10 rad/s
        # Concussive-level:
        #   Impact whiplash: 8-15+ m/s, angular: 20-35+ rad/s

        # Velocity component (40 pts max)
        _VEL_FLOOR_MPS = 5.0     # below 5 m/s = normal athletic motion
        _VEL_RANGE_MPS = 10.0    # 5 → 15 m/s maps to 0 → 40 pts
        vel_excess = max(head_speed - _VEL_FLOOR_MPS, 0.0)
        vel_component = min(vel_excess / _VEL_RANGE_MPS, 1.0) * 40.0

        # Angular velocity component (35 pts max) — uses EMA-smoothed value
        _ANGULAR_FLOOR_RPS = 12.0  # below 12 rad/s = normal head turns + noise
        _ANGULAR_RANGE_RPS = 23.0  # 12 → 35 rad/s maps to 0 → 35 pts
        ang_excess = max(self._head_angular_vel_ema - _ANGULAR_FLOOR_RPS, 0.0)
        angular_component = min(ang_excess / _ANGULAR_RANGE_RPS, 1.0) * 35.0

        # Spike z-score component (25 pts max)
        _SPIKE_FLOOR_Z = 5.0     # z below 5 = within normal variation
        _SPIKE_RANGE_Z = 10.0    # z 5 → 15 maps to 0 → 25 pts
        spike_excess = max(self._head_spike_z - _SPIKE_FLOOR_Z, 0.0)
        spike_component = min(spike_excess / _SPIKE_RANGE_Z, 1.0) * 25.0

        raw_score = vel_component + angular_component + spike_component

        # --- Peak hold + decay ---
        if raw_score > self._concussion_peak_value:
            self._concussion_peak_value = raw_score
            self._concussion_peak_hold_frames = _PEAK_HOLD_FRAMES
        elif self._concussion_peak_hold_frames > 0:
            self._concussion_peak_hold_frames -= 1
        else:
            self._concussion_peak_value *= _DECAY_FACTOR

        self._current_concussion_rating = float(np.clip(self._concussion_peak_value, 0.0, 100.0))

    def analyze_cluster_quality(
        self,
        cluster_id: int,
        resampled_reps: np.ndarray,
        raw_reps: list[np.ndarray],
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """Per-cluster quality analysis. Called from run_analysis().

        Args:
            cluster_id: Cluster identifier.
            resampled_reps: (N, T, D) array of N reps resampled to T frames.
            raw_reps: List of original (variable-length) rep feature arrays.
            fps: Sampling rate.

        Returns:
            Dict of quality metrics for this cluster.
        """
        N, T, D = resampled_reps.shape
        n_joints = 14
        ndim = D // n_joints  # 2 or 3

        # Get or create cluster state
        if cluster_id not in self._cluster_state:
            self._cluster_state[cluster_id] = _ClusterQualityState(
                n_baseline=self.n_baseline_reps, n_joints=n_joints, ndim=ndim
            )
        cs = self._cluster_state[cluster_id]
        cs.n_reps = N

        if N < 2:
            return {"n_reps": N, "enough_data": False}

        n_0 = max(2, min(self.n_baseline_reps, N // 2))
        baseline_reps = resampled_reps[:n_0]
        baseline_mean = np.mean(baseline_reps, axis=0)  # (T, D)

        result: dict[str, Any] = {"n_reps": N, "enough_data": True}

        # --- ROM per joint chain ---
        rom_per_rep: dict[str, list[float]] = defaultdict(list)
        for k in range(N):
            rep = resampled_reps[k]  # (T, D)
            joints_traj = rep.reshape(T, n_joints, ndim)
            for name, (i, j, m) in JOINT_CHAINS.items():
                angles = [
                    compute_joint_angle(joints_traj[t, i], joints_traj[t, j], joints_traj[t, m])
                    for t in range(T)
                ]
                rom = max(angles) - min(angles)
                rom_per_rep[name].append(rom)

        rom_decay = {}
        for name, roms in rom_per_rep.items():
            if len(roms) >= 3:
                baseline_rom = np.mean(roms[:n_0])
                current_rom = np.mean(roms[-min(2, len(roms)):])
                if baseline_rom > 5.0:
                    decay = max(0.0, (baseline_rom - current_rom) / baseline_rom)
                    rom_decay[name] = round(decay, 3)
        result["rom_decay"] = rom_decay

        # --- SPARC smoothness per rep ---
        sparc_values = []
        for k in range(N):
            vel = np.diff(resampled_reps[k], axis=0) * fps
            speed = np.linalg.norm(vel, axis=1)
            sparc_values.append(sparc(speed, fps))
        result["sparc_values"] = [round(s, 3) for s in sparc_values]
        if len(sparc_values) >= 3:
            baseline_sparc = np.mean(sparc_values[:n_0])
            current_sparc = np.mean(sparc_values[-2:])
            if abs(baseline_sparc) > 0.1:
                result["sparc_decay"] = round(
                    (baseline_sparc - current_sparc) / abs(baseline_sparc), 3
                )

        # --- Cross-correlation decay ---
        if N >= 3 and len(raw_reps) >= 3:
            correlations = []
            for i in range(min(len(raw_reps) - 1, N - 1)):
                if raw_reps[i].shape[0] >= 3 and raw_reps[i + 1].shape[0] >= 3:
                    cc = rep_cross_correlation(raw_reps[i], raw_reps[i + 1])
                    correlations.append(cc)
            if len(correlations) >= 2:
                x = np.arange(len(correlations))
                slope = float(np.polyfit(x, correlations, 1)[0])
                result["correlation_decay_rate"] = round(slope, 4)
                result["correlations"] = [round(c, 3) for c in correlations]

        # --- EWMA per-joint degradation ---
        rep_deviations = np.array([
            np.mean(np.abs(resampled_reps[k] - baseline_mean), axis=0)
            for k in range(N)
        ])  # (N, D)

        mu_0 = np.mean(rep_deviations[:n_0], axis=0)
        sigma_0 = np.std(rep_deviations[:n_0], axis=0)
        sigma_0 = np.maximum(sigma_0, 1e-6)

        lam = 0.2
        L_factor = 2.7
        ewma = np.zeros((N, D))
        ewma[0] = mu_0.copy()
        joint_alarms: dict[int, int] = {}

        for k in range(1, N):
            ewma[k] = lam * rep_deviations[k] + (1 - lam) * ewma[k - 1]
            factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (k + 1))))
            ucl_k = mu_0 + L_factor * sigma_0 * factor
            alarms = ewma[k] > ucl_k

            # Map dimensions to joints
            for j_idx in range(n_joints):
                start = j_idx * ndim
                end = start + ndim
                if np.any(alarms[start:end]) and j_idx not in joint_alarms and k >= n_0:
                    joint_alarms[j_idx] = k

        cs.ewma_alarms = joint_alarms
        result["ewma_alarming_joints"] = list(joint_alarms.keys())
        result["ewma_alarm_reps"] = joint_alarms

        # --- CUSUM on scalar quality ---
        distances = np.array([
            float(np.sqrt(np.mean((resampled_reps[k] - baseline_mean) ** 2)))
            for k in range(N)
        ])
        mu_d = np.mean(distances[:n_0])
        sigma_d = max(np.std(distances[:n_0]), 1e-6)
        delta = 0.5 * sigma_d
        h = 4.0 * sigma_d

        cusum = np.zeros(N)
        for k in range(1, N):
            cusum[k] = max(0.0, cusum[k - 1] + (distances[k] - mu_d - delta / 2))

        alarm_idx = np.where(cusum > h)[0]
        cs.cusum_onset = int(alarm_idx[0]) if len(alarm_idx) > 0 else None
        result["cusum_onset_rep"] = cs.cusum_onset
        result["cusum_values"] = [round(c, 3) for c in cusum]

        # --- Loop spread per rep ---
        loop_spreads = []
        for k in range(N):
            centroid = resampled_reps[k].mean(axis=0)
            dists = np.linalg.norm(resampled_reps[k] - centroid, axis=1)
            loop_spreads.append(float(np.std(dists)))
        result["loop_spreads"] = [round(s, 3) for s in loop_spreads]

        # --- Path efficiency per rep ---
        path_efficiencies = []
        for k in range(N):
            vel = np.diff(resampled_reps[k], axis=0)
            path_len = float(np.sum(np.linalg.norm(vel, axis=1)))
            displacement = float(np.linalg.norm(resampled_reps[k][-1] - resampled_reps[k][0]))
            eff = displacement / max(path_len, 1e-8)
            path_efficiencies.append(eff)
        result["path_efficiencies"] = [round(e, 3) for e in path_efficiencies]

        # --- LDLJ per rep (Phase 2A) ---
        ldlj_values = []
        for k in range(N):
            ldlj_val = log_dimensionless_jerk(resampled_reps[k], fps=fps)
            ldlj_values.append(ldlj_val)
        result["ldlj_values"] = [round(v, 3) for v in ldlj_values]
        if len(ldlj_values) >= 3:
            baseline_ldlj = np.mean(ldlj_values[:n_0])
            current_ldlj = np.mean(ldlj_values[-2:])
            if abs(baseline_ldlj) > 0.1:
                result["ldlj_decay"] = round(
                    (baseline_ldlj - current_ldlj) / abs(baseline_ldlj), 3
                )

        # --- Sample entropy per rep (Phase 2B) ---
        sampen_values = []
        for k in range(N):
            vel = np.diff(resampled_reps[k], axis=0) * fps
            speed = np.linalg.norm(vel, axis=1)
            sampen_values.append(sample_entropy(speed))
        result["sample_entropy_values"] = [round(s, 3) for s in sampen_values]

        # --- Angular velocity per-joint per-rep for MNF and kinematic chain ---
        joint_angular_vel_per_rep: dict[str, list[np.ndarray]] = defaultdict(list)
        for k in range(N):
            joints_traj = resampled_reps[k].reshape(T, n_joints, ndim)
            for name, (i, j, m) in JOINT_CHAINS.items():
                angles = np.array([
                    compute_joint_angle(joints_traj[t, i], joints_traj[t, j], joints_traj[t, m])
                    for t in range(T)
                ])
                ang_vel = np.abs(np.diff(angles)) * fps
                joint_angular_vel_per_rep[name].append(ang_vel)

        # --- Spectral Median Frequency per joint per rep (Phase 3C) ---
        mnf_per_joint: dict[str, list[float]] = {}
        for name, vel_list in joint_angular_vel_per_rep.items():
            mnf_vals = [median_frequency(v, fps) for v in vel_list]
            mnf_per_joint[name] = [round(m, 2) for m in mnf_vals]
        result["mnf_per_joint"] = mnf_per_joint

        # --- Kinematic chain sequencing (Phase 3B) ---
        if N >= 2:
            # Use last rep angular velocities for kinematic sequence
            last_rep_ang_vel: dict[str, list[float]] = {}
            for name, vel_list in joint_angular_vel_per_rep.items():
                last_rep_ang_vel[name] = vel_list[-1].tolist()
            kin_seq = compute_kinematic_sequence(last_rep_ang_vel)
            if kin_seq["sequence_score"] is not None:
                result["kinematic_sequence_score"] = kin_seq["sequence_score"]
                result["kinematic_timing_gaps"] = kin_seq["timing_gaps"]

        # --- Composite cluster fatigue score ---
        # Updated weights: ROM 0.25, EWMA 0.20, CUSUM 0.20, correlation 0.15,
        # SPARC 0.08, LDLJ 0.07, spread 0.05
        fatigue_components = []

        # ROM decay (average across joint chains, if available)
        if rom_decay:
            avg_rom_decay = np.mean(list(rom_decay.values()))
            fatigue_components.append(("rom_decay", min(1.0, avg_rom_decay / 0.25), 0.25))

        # EWMA (fraction of joints alarming)
        ewma_frac = len(joint_alarms) / n_joints
        fatigue_components.append(("ewma", ewma_frac, 0.20))

        # CUSUM (normalized height)
        cusum_norm = min(1.0, float(np.max(cusum)) / max(h, 1e-8))
        fatigue_components.append(("cusum", cusum_norm, 0.20))

        # Cross-correlation decay
        if "correlation_decay_rate" in result:
            cc_signal = min(1.0, max(0.0, -result["correlation_decay_rate"] * 20))
            fatigue_components.append(("correlation", cc_signal, 0.15))

        # SPARC decay
        if "sparc_decay" in result:
            sparc_signal = min(1.0, max(0.0, result["sparc_decay"]))
            fatigue_components.append(("sparc", sparc_signal, 0.08))

        # LDLJ decay (Phase 2A)
        if "ldlj_decay" in result:
            ldlj_signal = min(1.0, max(0.0, result["ldlj_decay"]))
            fatigue_components.append(("ldlj", ldlj_signal, 0.07))

        # Loop spread trend
        if len(loop_spreads) >= 3:
            spread_slope = float(np.polyfit(np.arange(len(loop_spreads)), loop_spreads, 1)[0])
            spread_signal = min(1.0, max(0.0, spread_slope * 10))
            fatigue_components.append(("spread", spread_signal, 0.05))

        # Weighted composite
        if fatigue_components:
            total_weight = sum(w for _, _, w in fatigue_components)
            composite = sum(v * w for _, v, w in fatigue_components) / total_weight
            result["composite_fatigue"] = round(composite, 3)
            result["fatigue_components"] = {
                name: round(val, 3) for name, val, _ in fatigue_components
            }
        else:
            result["composite_fatigue"] = 0.0

        cs.latest_quality = result
        return result

    def get_frame_quality(self) -> dict[str, Any]:
        """Get current per-frame quality metrics for WebSocket response."""
        result: dict[str, Any] = {}

        # Movement phase
        if self._current_movement_phase is not None:
            result["movement_phase"] = self._current_movement_phase

        # Form score
        if self._current_form_score is not None:
            result["form_score"] = round(self._current_form_score, 1)

        # Joint quality
        if self._current_joint_scores is not None:
            scores = [round(float(s) * 100, 1) for s in self._current_joint_scores]
            deviations = (
                [round(float(d), 3) for d in self._current_joint_deviations]
                if self._current_joint_deviations is not None
                else []
            )
            result["joint_quality"] = {
                "scores": scores,
                "degrading": self._current_degrading_joints,
                "deviations": deviations,
            }

        # Biomechanical angles (compact)
        biomech: dict[str, Any] = {
            "fppa_left": round(self._current_fppa_left, 1),
            "fppa_right": round(self._current_fppa_right, 1),
            "hip_drop": round(self._current_hip_drop, 1),
            "trunk_lean": round(self._current_trunk_lean, 1),
            "asymmetry": round(self._asymmetry_ema, 1),
            "curvature": round(self._curvature_ema, 4),
            "jerk": round(self._jerk_ema, 4),
        }

        # Angular velocities (Phase 1B)
        if self._angular_velocities:
            biomech["angular_velocities"] = {
                name: round(vel, 1) for name, vel in self._angular_velocities.items()
            }

        # Anomaly score (Phase 1C)
        if self._anomaly_score is not None:
            biomech["anomaly_score"] = round(self._anomaly_score, 3)

        # Center of Mass metrics (Phase 3A)
        biomech["com_velocity"] = round(self._com_velocity, 3)
        biomech["com_sway"] = round(self._com_sway, 3)

        # Head kinematics (concussion tracking)
        biomech["head_accel_g"] = round(self._head_linear_accel_g, 2)
        biomech["head_angular_vel"] = round(self._head_angular_vel, 2)
        biomech["head_spike_z"] = round(self._head_spike_z, 2)

        result["biomechanics"] = biomech

        # Injury risks (Phase 1D)
        if self._current_injury_risks:
            result["injury_risks"] = self._current_injury_risks

        # Active guideline
        if self._current_profile is not None:
            result["active_guideline"] = {
                "name": self._current_profile.name,
                "display_name": self._current_profile.display_name,
                "form_cues": self._current_profile.form_cues,
            }

        # Fatigue timeline (incremental, only when changed)
        if self._fatigue_timeline_version != self._last_sent_timeline_version:
            result["fatigue_timeline"] = {
                "timestamps": self._fatigue_timeline_timestamps,
                "fatigue": self._fatigue_timeline_fatigue,
                "form_scores": self._fatigue_timeline_form,
            }
            self._last_sent_timeline_version = self._fatigue_timeline_version

        # Ratings
        result["concussion_rating"] = round(self._current_concussion_rating, 1)
        result["fatigue_rating"] = round(self._current_fatigue_rating, 1)

        return result

    def reset(self) -> None:
        """Reset all state."""
        self._curvature_tracker.reset()
        self._bone_filter.reset()
        self._frame_count = 0
        self._curvature_ema = 0.0
        self._jerk_ema = 0.0
        self._asymmetry_ema = 0.0
        self._joint_score_history.clear()
        self._prev_joint_angles.clear()
        self._angular_velocities.clear()
        self._peak_angular_velocities.clear()
        for dq in self._angular_velocity_history.values():
            dq.clear()
        self._biomech_history.clear()
        self._anomaly_model = None
        self._anomaly_score = None
        self._anomaly_refit_counter = 0
        self._current_injury_risks = []
        self._com_history.clear()
        self._com_velocity = 0.0
        self._com_sway = 0.0
        self._cluster_state.clear()
        self._current_angles.clear()
        self._current_form_score = None
        self._current_joint_scores = None
        self._current_joint_deviations = None
        self._current_degrading_joints = []
        self._current_movement_phase = None
        self._head_prev_pos = None
        self._head_prev_vel = None
        self._head_prev_ear_angle = None
        self._head_vel_baseline_ema = 0.0
        self._head_vel_baseline_initialized = False
        self._head_linear_accel_g = 0.0
        self._head_angular_vel = 0.0
        self._head_angular_vel_ema = 0.0
        self._head_spike_z = 0.0
        self._concussion_peak_value = 0.0
        self._concussion_peak_hold_frames = 0
        self._current_concussion_rating = 0.0
        self._current_fatigue_rating = 0.0
        self._fatigue_timeline_timestamps.clear()
        self._fatigue_timeline_fatigue.clear()
        self._fatigue_timeline_form.clear()
        self._fatigue_timeline_version = 0
        self._last_sent_timeline_version = 0


class _ClusterQualityState:
    """Per-cluster quality tracking state."""

    def __init__(self, n_baseline: int = 5, n_joints: int = 14, ndim: int = 2):
        self.n_baseline = n_baseline
        self.n_joints = n_joints
        self.ndim = ndim
        self.n_reps = 0
        self.ewma_alarms: dict[int, int] = {}  # joint_idx -> first alarm rep
        self.cusum_onset: int | None = None
        self.latest_quality: dict[str, Any] = {}
