"""Experiment: Quantitative Comparison of Pose Smoothing Methods.

Generates realistic synthetic 3D pose data with body-part-specific frequencies,
adds noise (Gaussian + random spikes), then compares seven smoothing approaches
on six quantitative metrics.

Methods:
  A) RAW          - no filtering (baseline)
  B) EMA          - exponential moving average (alpha=0.3)
  C) ONE EURO     - adaptive cutoff filter (min_cutoff=1.5, beta=0.01)
  D) KALMAN       - constant-velocity Kalman filter
  E) SAVGOL       - Savitzky-Golay (window=7, order=3)
  F) DOUBLE EMA   - Holt's linear (alpha=0.3, beta_trend=0.05)
  G) MEDIAN+EMA   - median(3) then EMA(0.4)

Metrics:
  1. MSE to ground truth
  2. Jerk: mean|d^3x/dt^3|
  3. Lag: cross-correlation peak offset (frames)
  4. Bone-length variance
  5. Spike removal %
  6. Compute time (us/frame)

Also tests at noise levels: 0.5x, 1x, 2x default sigma.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from scipy.signal import savgol_filter
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_JOINTS = 14
DIMS = 3
NUM_FRAMES = 300
FPS = 30
DT = 1.0 / FPS

# Body-part groups -> (frequency_hz, amplitude, noise_sigma)
# Joints: 0=l_shoulder, 1=r_shoulder, 2=l_elbow, 3=r_elbow, 4=l_wrist, 5=r_wrist
#         6=l_hip, 7=r_hip, 8=l_knee, 9=r_knee, 10=l_ankle, 11=r_ankle,
#         12=l_foot, 13=r_foot
TORSO_JOINTS = [0, 1, 6, 7]
ARM_JOINTS = [2, 3, 4, 5]
LEG_JOINTS = [8, 9, 10, 11, 12, 13]

TORSO_PARAMS = dict(freq=1.0, amp=0.05, sigma=0.005)
ARM_PARAMS = dict(freq=2.5, amp=0.15, sigma=0.015)
LEG_PARAMS = dict(freq=1.5, amp=0.10, sigma=0.010)

SPIKE_FRACTION = 0.02
SPIKE_MULTIPLIER = 5.0

# Bone definitions: (joint_a, joint_b)
BONES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
    (7, 9), (9, 11), (10, 12), (11, 13),
]

NOISE_LEVELS = [0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_data(noise_scale: float = 1.0, seed: int = 42):
    """Return (ground_truth, noisy, spike_mask) each shape (NUM_FRAMES, NUM_JOINTS, 3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(NUM_FRAMES) * DT  # seconds

    gt = np.zeros((NUM_FRAMES, NUM_JOINTS, DIMS))
    noisy = np.zeros_like(gt)
    sigma_per_joint = np.zeros(NUM_JOINTS)

    # Base positions so skeleton has reasonable structure
    base_positions = np.array([
        [-0.15, 0.40, 0.0],   # 0 l_shoulder
        [ 0.15, 0.40, 0.0],   # 1 r_shoulder
        [-0.30, 0.25, 0.0],   # 2 l_elbow
        [ 0.30, 0.25, 0.0],   # 3 r_elbow
        [-0.40, 0.10, 0.0],   # 4 l_wrist
        [ 0.40, 0.10, 0.0],   # 5 r_wrist
        [-0.10, 0.00, 0.0],   # 6 l_hip
        [ 0.10, 0.00, 0.0],   # 7 r_hip
        [-0.10,-0.20, 0.0],   # 8 l_knee
        [ 0.10,-0.20, 0.0],   # 9 r_knee
        [-0.10,-0.40, 0.0],   # 10 l_ankle
        [ 0.10,-0.40, 0.0],   # 11 r_ankle
        [-0.10,-0.45, 0.05],  # 12 l_foot
        [ 0.10,-0.45, 0.05],  # 13 r_foot
    ])

    def assign_group(joints, params):
        freq, amp, sigma = params["freq"], params["amp"], params["sigma"]
        for j in joints:
            sigma_per_joint[j] = sigma
            # Each joint gets unique phase offsets per dimension
            for d in range(DIMS):
                phase = rng.uniform(0, 2 * np.pi)
                gt[:, j, d] = base_positions[j, d] + amp * np.sin(2 * np.pi * freq * t + phase)

    assign_group(TORSO_JOINTS, TORSO_PARAMS)
    assign_group(ARM_JOINTS, ARM_PARAMS)
    assign_group(LEG_JOINTS, LEG_PARAMS)

    # Add Gaussian noise
    for j in range(NUM_JOINTS):
        s = sigma_per_joint[j] * noise_scale
        noisy[:, j, :] = gt[:, j, :] + rng.normal(0, s, (NUM_FRAMES, DIMS))

    # Add random spikes
    num_spikes = int(NUM_FRAMES * NUM_JOINTS * SPIKE_FRACTION)
    spike_frames = rng.integers(0, NUM_FRAMES, num_spikes)
    spike_joints = rng.integers(0, NUM_JOINTS, num_spikes)
    spike_mask = np.zeros((NUM_FRAMES, NUM_JOINTS), dtype=bool)
    for f, j in zip(spike_frames, spike_joints):
        spike_sigma = sigma_per_joint[j] * noise_scale * SPIKE_MULTIPLIER
        noisy[f, j, :] += rng.normal(0, spike_sigma, DIMS)
        spike_mask[f, j] = True

    return gt, noisy, spike_mask


# ---------------------------------------------------------------------------
# Smoothing methods
# ---------------------------------------------------------------------------

def smooth_raw(data):
    """A) No filtering."""
    return data.copy()


def smooth_ema(data, alpha=0.3):
    """B) Exponential moving average."""
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


class OneEuroFilter:
    """C) One Euro Filter - from-scratch implementation.

    Reference: Casiez et al., "1 Euro Filter: A Simple Speed-based
    Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.
    """

    def __init__(self, min_cutoff=1.5, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / DT)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        # Derivative estimation
        dx = (x - self.x_prev) / DT
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


def smooth_one_euro(data, min_cutoff=1.5, beta=0.01):
    """C) One Euro filter applied per-joint per-dim."""
    out = np.empty_like(data)
    n_frames, n_joints, n_dims = data.shape
    for j in range(n_joints):
        for d in range(n_dims):
            filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
            for i in range(n_frames):
                out[i, j, d] = filt(np.array([data[i, j, d]]))[0]
    return out


class KalmanFilterCV:
    """D) Constant-velocity Kalman filter - from-scratch implementation.

    State: [x, v] per scalar signal.
    Prediction: x' = x + v*dt, v' = v
    """

    def __init__(self, process_noise=0.001, measurement_noise=0.01):
        self.dt = DT
        # State: [position, velocity]
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1.0
        # Transition matrix
        self.F = np.array([[1, self.dt], [0, 1]])
        # Observation matrix
        self.H = np.array([[1, 0]])
        # Process noise
        q = process_noise
        self.Q = np.array([
            [q * self.dt**3 / 3, q * self.dt**2 / 2],
            [q * self.dt**2 / 2, q * self.dt],
        ])
        # Measurement noise
        self.R = np.array([[measurement_noise]])
        self.initialized = False

    def __call__(self, z):
        if not self.initialized:
            self.x[0] = z
            self.initialized = True
            return z

        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x[0]


def smooth_kalman(data, process_noise=0.001, measurement_noise=0.01):
    """D) Kalman filter applied per-joint per-dim."""
    out = np.empty_like(data)
    n_frames, n_joints, n_dims = data.shape
    for j in range(n_joints):
        for d in range(n_dims):
            kf = KalmanFilterCV(process_noise, measurement_noise)
            for i in range(n_frames):
                out[i, j, d] = kf(data[i, j, d])
    return out


def smooth_savgol(data, window=7, order=3):
    """E) Savitzky-Golay filter from scipy."""
    out = np.empty_like(data)
    n_frames, n_joints, n_dims = data.shape
    for j in range(n_joints):
        for d in range(n_dims):
            out[:, j, d] = savgol_filter(data[:, j, d], window, order)
    return out


def smooth_double_ema(data, alpha=0.3, beta_trend=0.05):
    """F) Double EMA (Holt's linear method)."""
    out = np.empty_like(data)
    n_frames = len(data)
    # Level and trend
    level = data[0].copy()
    trend = np.zeros_like(data[0])
    out[0] = level

    for i in range(1, n_frames):
        new_level = alpha * data[i] + (1 - alpha) * (level + trend)
        new_trend = beta_trend * (new_level - level) + (1 - beta_trend) * trend
        level = new_level
        trend = new_trend
        out[i] = level + trend
    return out


def smooth_median_ema(data, median_window=3, alpha=0.4):
    """G) Median filter(3) + EMA(0.4)."""
    from scipy.ndimage import median_filter as _mf
    n_frames, n_joints, n_dims = data.shape
    # Apply median filter along time axis per joint/dim
    med = np.empty_like(data)
    for j in range(n_joints):
        for d in range(n_dims):
            med[:, j, d] = _mf(data[:, j, d], size=median_window)
    # Then EMA
    return smooth_ema(med, alpha=alpha)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_mse(smoothed, gt):
    """Mean squared error to ground truth."""
    return float(np.mean((smoothed - gt) ** 2))


def metric_jerk(smoothed):
    """Mean absolute jerk: |d^3x/dt^3|, averaged over joints and dims."""
    # Third derivative via finite differences
    d1 = np.diff(smoothed, axis=0) / DT
    d2 = np.diff(d1, axis=0) / DT
    d3 = np.diff(d2, axis=0) / DT
    return float(np.mean(np.abs(d3)))


def metric_lag(smoothed, gt):
    """Cross-correlation peak offset in frames, averaged over all signals."""
    n_frames, n_joints, n_dims = gt.shape
    total_lag = 0.0
    count = 0
    for j in range(n_joints):
        for d in range(n_dims):
            g = gt[:, j, d] - np.mean(gt[:, j, d])
            s = smoothed[:, j, d] - np.mean(smoothed[:, j, d])
            corr = np.correlate(g, s, mode="full")
            peak = np.argmax(corr)
            lag = peak - (n_frames - 1)  # 0 = no lag, positive = smoothed lags
            total_lag += abs(lag)
            count += 1
    return total_lag / count


def metric_bone_variance(smoothed):
    """Mean bone-length variance across frames."""
    variances = []
    for a, b in BONES:
        lengths = np.linalg.norm(smoothed[:, a, :] - smoothed[:, b, :], axis=1)
        variances.append(np.var(lengths))
    return float(np.mean(variances))


def metric_spike_removal(smoothed, gt, noisy, spike_mask):
    """Percentage of spike frames where smoothed is closer to GT than noisy.

    Only evaluates frames/joints that had spikes.
    """
    spike_frames, spike_joints = np.where(spike_mask)
    if len(spike_frames) == 0:
        return 100.0
    removed = 0
    for f, j in zip(spike_frames, spike_joints):
        err_noisy = np.linalg.norm(noisy[f, j] - gt[f, j])
        err_smooth = np.linalg.norm(smoothed[f, j] - gt[f, j])
        if err_smooth < err_noisy:
            removed += 1
    return 100.0 * removed / len(spike_frames)


def measure_compute_time(smooth_fn, data, n_runs=5):
    """Microseconds per frame, averaged over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        smooth_fn(data)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg = np.median(times)
    return 1e6 * avg / NUM_FRAMES  # us/frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    mse: float
    jerk: float
    lag: float
    bone_var: float
    spike_pct: float
    time_us: float


def run_experiment(noise_scale: float = 1.0):
    """Run all methods at given noise scale, return list of MethodResult."""
    gt, noisy, spike_mask = generate_data(noise_scale=noise_scale)

    methods = [
        ("RAW",        smooth_raw),
        ("EMA",        lambda d: smooth_ema(d, alpha=0.3)),
        ("ONE EURO",   lambda d: smooth_one_euro(d, min_cutoff=1.5, beta=0.01)),
        ("KALMAN",     lambda d: smooth_kalman(d, process_noise=0.001, measurement_noise=0.01)),
        ("SAVGOL",     lambda d: smooth_savgol(d, window=7, order=3)),
        ("DOUBLE EMA", lambda d: smooth_double_ema(d, alpha=0.3, beta_trend=0.05)),
        ("MED+EMA",    lambda d: smooth_median_ema(d, median_window=3, alpha=0.4)),
    ]

    results = []
    for name, fn in methods:
        smoothed = fn(noisy)
        mse = metric_mse(smoothed, gt)
        jerk = metric_jerk(smoothed)
        lag = metric_lag(smoothed, gt)
        bone_var = metric_bone_variance(smoothed)
        spike_pct = metric_spike_removal(smoothed, gt, noisy, spike_mask)
        time_us = measure_compute_time(fn, noisy, n_runs=3)
        results.append(MethodResult(name, mse, jerk, lag, bone_var, spike_pct, time_us))

    return results


def print_table(results: list, title: str):
    """Print a formatted comparison table."""
    header = f"{'Method':<12} {'MSE':>10} {'Jerk':>12} {'Lag':>6} {'BoneVar':>10} {'Spike%':>8} {'us/frm':>10}"
    sep = "-" * len(header)
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.name:<12} {r.mse:>10.6f} {r.jerk:>12.2f} {r.lag:>6.2f} "
            f"{r.bone_var:>10.7f} {r.spike_pct:>7.1f}% {r.time_us:>10.1f}"
        )
    print(sep)


def score_methods(results: list):
    """Rank methods. Lower is better for MSE, Jerk, Lag, BoneVar, Time.
    Higher is better for Spike%.
    Return dict of method -> total rank score (lower = better)."""
    n = len(results)
    scores = {r.name: 0 for r in results}

    # Metrics where lower is better
    for attr in ["mse", "jerk", "lag", "bone_var", "time_us"]:
        ranked = sorted(results, key=lambda r: getattr(r, attr))
        for rank, r in enumerate(ranked):
            scores[r.name] += rank

    # Spike removal: higher is better
    ranked = sorted(results, key=lambda r: r.spike_pct, reverse=True)
    for rank, r in enumerate(ranked):
        scores[r.name] += rank

    return scores


def main():
    print("=" * 78)
    print("POSE SMOOTHING METHOD COMPARISON")
    print(f"Synthetic data: {NUM_JOINTS} joints, {DIMS}D, {NUM_FRAMES} frames @ {FPS}fps")
    print(f"Bones: {len(BONES)} connections")
    print(f"Spike rate: {SPIKE_FRACTION*100:.0f}% frames, {SPIKE_MULTIPLIER:.0f}x noise")
    print("=" * 78)

    all_scores = {}

    for ns in NOISE_LEVELS:
        label = f"Noise level: {ns:.1f}x"
        results = run_experiment(noise_scale=ns)
        print_table(results, label)
        scores = score_methods(results)
        for name, s in scores.items():
            all_scores.setdefault(name, 0)
            all_scores[name] += s

    # Collect per-method aggregate MSE for tiebreaking
    agg_mse = {}
    for ns in NOISE_LEVELS:
        results = run_experiment(noise_scale=ns)
        for r in results:
            agg_mse.setdefault(r.name, 0.0)
            agg_mse[r.name] += r.mse

    # Final ranking (tiebreak by aggregate MSE - lower is better)
    print("\n" + "=" * 78)
    print("AGGREGATE RANKING (sum of rank positions across all noise levels, lower=better)")
    print("  Tiebreak: lower aggregate MSE wins")
    print("=" * 78)
    ranked = sorted(all_scores.items(), key=lambda x: (x[1], agg_mse.get(x[0], 0)))
    for i, (name, score) in enumerate(ranked):
        marker = " <-- WINNER" if i == 0 else ""
        print(f"  {i+1}. {name:<12}  score={score:>3d}  (MSE_sum={agg_mse.get(name, 0):.6f}){marker}")

    print("\n" + "=" * 78)
    print(f"WINNER: {ranked[0][0]}")
    print("=" * 78)


if __name__ == "__main__":
    main()
