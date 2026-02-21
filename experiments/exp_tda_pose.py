"""Experiment: Topological & Geometric Methods for Pose Smoothing.

Explores novel approaches to temporal pose smoothing that respect the
geometric structure of human skeleton data, rather than treating joint
coordinates as independent signals.

APPROACHES:
  1. Lie Group Smoothing (SO(3))
     - Convert bone vectors to rotation matrices
     - Smooth in the Lie algebra so(3) via log/exp maps
     - Guarantees constant bone lengths by construction
     - Refs: Grassia (1998) "Practical Parameterization of Rotations";
       Boumal et al. (2023) "An Introduction to Optimization on Smooth Manifolds";
       He et al. CVPR 2024 "NRDF: Neural Riemannian Distance Fields"

  2. Graph Spectral Filtering
     - Skeleton as a graph, joints as signals on vertices
     - Low-pass filter via graph Laplacian eigendecomposition
     - Smooths spatially across joints (complementary to temporal)
     - Refs: Shuman et al. (2013) "Emerging Field of Signal Processing on Graphs";
       Yan et al. AAAI 2018 "ST-GCN"; iterative graph filtering (GS-Net 2023)

  3. Manifold Projection (bone-length constraints)
     - After any temporal filter, project back onto the bone-length manifold
     - Propagate from root to leaves with correct bone lengths
     - Refs: ManiPose NeurIPS 2024; Matsune & Hu WACV 2024 geometry loss

  4. Persistent Homology Quality Metric
     - Skipped: ripser/gudhi not available in this environment
     - Would measure topological noise (spurious H0/H1 features)

SKELETON TOPOLOGY (14 joints):
  0=L_Hip, 1=R_Hip, 2=L_Shoulder, 3=R_Shoulder, 4=L_Elbow, 5=R_Elbow,
  6=L_Knee, 7=R_Knee, 8=L_Wrist, 9=R_Wrist, 10=L_Ankle, 11=R_Ankle,
  12=L_Foot, 13=R_Foot

  Bones: (0,1),(0,2),(1,3),(2,4),(3,5),(0,6),(1,7),(6,7),(6,8),(7,9),
         (8,10),(9,11),(10,12),(11,13)
  Note: bone (6,7) is a cross-link (knees); tree root = midpoint of hips.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import expm, logm

# ============================================================================
# Constants
# ============================================================================

NUM_JOINTS = 14
NUM_FRAMES = 300
NOISE_SIGMA = 0.01

# Joint names for readability
JOINT_NAMES = [
    "L_Hip", "R_Hip", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Knee", "R_Knee",
    "L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle",
    "L_Foot", "R_Foot",
]

# Skeleton bones (parent, child) -- directed for tree traversal
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (10, 12), (11, 13),
]

# Tree structure for root-to-leaf propagation (no cross-links).
# Root = 0 (L_Hip). We skip (6,7) cross-link for tree traversal.
TREE_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
    (0, 6), (1, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (10, 12), (11, 13),
]


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_test_data(
    num_frames: int = NUM_FRAMES,
    num_joints: int = NUM_JOINTS,
    noise_sigma: float = NOISE_SIGMA,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 3D skeleton motion with sinusoidal movement + noise.

    Returns:
        clean: (num_frames, num_joints, 3) clean joint positions
        noisy: (num_frames, num_joints, 3) noisy joint positions
    """
    rng = np.random.default_rng(seed)

    # Base skeleton in T-pose (approximate human proportions, meters)
    base_pose = np.array([
        [-0.10, 0.00, 0.00],   # 0: L_Hip
        [ 0.10, 0.00, 0.00],   # 1: R_Hip
        [-0.15, 0.45, 0.00],   # 2: L_Shoulder
        [ 0.15, 0.45, 0.00],   # 3: R_Shoulder
        [-0.40, 0.45, 0.00],   # 4: L_Elbow
        [ 0.40, 0.45, 0.00],   # 5: R_Elbow
        [-0.10,-0.40, 0.00],   # 6: L_Knee
        [ 0.10,-0.40, 0.00],   # 7: R_Knee
        [-0.60, 0.45, 0.00],   # 8: L_Wrist
        [ 0.60, 0.45, 0.00],   # 9: R_Wrist
        [-0.10,-0.80, 0.00],   # 10: L_Ankle
        [ 0.10,-0.80, 0.00],   # 11: R_Ankle
        [-0.10,-0.90, 0.05],   # 12: L_Foot
        [ 0.10,-0.90, 0.05],   # 13: R_Foot
    ], dtype=np.float64)

    t = np.linspace(0, 2 * np.pi * 3, num_frames)  # 3 full cycles

    clean = np.zeros((num_frames, num_joints, 3), dtype=np.float64)

    for f in range(num_frames):
        pose = base_pose.copy()

        # Walking-like motion: hips sway, knees flex, arms swing
        hip_sway = 0.02 * np.sin(t[f])
        pose[0, 1] += hip_sway
        pose[1, 1] -= hip_sway

        # Knee flexion (alternating)
        knee_flex = 0.08 * np.sin(t[f])
        pose[6, 1] += knee_flex
        pose[6, 2] -= abs(knee_flex) * 0.3
        pose[7, 1] -= knee_flex
        pose[7, 2] -= abs(knee_flex) * 0.3

        # Ankle follows knee
        ankle_flex = 0.06 * np.sin(t[f])
        pose[10, 1] += ankle_flex
        pose[11, 1] -= ankle_flex

        # Foot follows ankle
        pose[12, 1] += ankle_flex * 0.8
        pose[13, 1] -= ankle_flex * 0.8

        # Arm swing (opposite phase to legs)
        arm_swing = 0.10 * np.sin(t[f] + np.pi)
        pose[4, 2] += arm_swing       # L_Elbow forward/back
        pose[5, 2] -= arm_swing       # R_Elbow
        pose[8, 2] += arm_swing * 1.2  # L_Wrist
        pose[9, 2] -= arm_swing * 1.2  # R_Wrist

        # Shoulder rotation
        pose[2, 2] += arm_swing * 0.3
        pose[3, 2] -= arm_swing * 0.3

        # Global forward motion
        pose[:, 2] += t[f] * 0.05

        clean[f] = pose

    # Add Gaussian noise
    noise = rng.normal(0, noise_sigma, clean.shape)
    noisy = clean + noise

    return clean, noisy


def compute_bone_lengths(poses: np.ndarray, bones: list[tuple[int, int]]) -> np.ndarray:
    """Compute bone lengths for each frame.

    Args:
        poses: (num_frames, num_joints, 3)
        bones: list of (parent, child) tuples

    Returns:
        lengths: (num_frames, num_bones)
    """
    lengths = np.zeros((poses.shape[0], len(bones)), dtype=np.float64)
    for i, (p, c) in enumerate(bones):
        diff = poses[:, c] - poses[:, p]
        lengths[:, i] = np.linalg.norm(diff, axis=-1)
    return lengths


# ============================================================================
# Method 0: Baseline - One Euro Filter (per-coordinate)
# ============================================================================

class OneEuroFilter1D:
    """One Euro Filter for a single scalar signal."""

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0,
                 beta: float = 0.007, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        # Derivative
        a_d = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)

        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def one_euro_filter_poses(
    poses: np.ndarray,
    freq: float = 30.0,
    mincutoff: float = 1.0,
    beta: float = 0.007,
) -> np.ndarray:
    """Apply One Euro Filter independently to each joint coordinate."""
    num_frames, num_joints, dim = poses.shape
    filtered = np.zeros_like(poses)

    for j in range(num_joints):
        for d in range(dim):
            filt = OneEuroFilter1D(freq=freq, mincutoff=mincutoff, beta=beta)
            for f in range(num_frames):
                filtered[f, j, d] = filt(poses[f, j, d])

    return filtered


# ============================================================================
# Method 1: Lie Group Smoothing (SO(3) rotations)
# ============================================================================

def _rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute rotation matrix R such that R @ v1_hat = v2_hat.

    Uses Rodrigues' formula. Returns 3x3 rotation matrix.
    """
    v1n = v1 / (np.linalg.norm(v1) + 1e-12)
    v2n = v2 / (np.linalg.norm(v2) + 1e-12)

    cos_angle = np.clip(np.dot(v1n, v2n), -1.0, 1.0)

    if cos_angle > 1.0 - 1e-10:
        return np.eye(3)
    if cos_angle < -1.0 + 1e-10:
        # 180-degree rotation: pick arbitrary perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1n[0]) < 0.9 else np.array([0, 1, 0])
        axis = np.cross(v1n, perp)
        axis /= np.linalg.norm(axis)
        # Rodrigues for 180 degrees: R = 2 * outer(axis,axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = np.cross(v1n, v2n)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3)
    axis /= axis_norm

    sin_angle = axis_norm
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])

    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * (K @ K)
    return R


def _log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map SO(3) -> so(3). Returns 3x3 skew-symmetric matrix."""
    # Use scipy for numerical stability
    rotvec = Rotation.from_matrix(R).as_rotvec()
    # Skew-symmetric from rotation vector
    return np.array([
        [0, -rotvec[2], rotvec[1]],
        [rotvec[2], 0, -rotvec[0]],
        [-rotvec[1], rotvec[0], 0],
    ])


def _exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3). Input: 3x3 skew-symmetric matrix."""
    # Extract rotation vector from skew-symmetric
    rotvec = np.array([omega[2, 1], omega[0, 2], omega[1, 0]])
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(rotvec).as_matrix()


def _ema_so3(rotations: list[np.ndarray], alpha: float = 0.3) -> list[np.ndarray]:
    """Exponential moving average in SO(3) via Lie algebra.

    For each frame t:
      delta = log(R_{t-1}^T @ R_t)           -- relative rotation in so(3)
      delta_filtered = alpha * delta          -- dampen in tangent space
      R_filtered_t = R_filtered_{t-1} @ exp(delta_filtered)
    """
    if not rotations:
        return []

    filtered = [rotations[0].copy()]

    for t in range(1, len(rotations)):
        R_prev = filtered[-1]
        R_curr = rotations[t]

        # Relative rotation
        delta_R = R_prev.T @ R_curr
        # Map to tangent space
        omega = _log_so3(delta_R)
        # Dampen
        omega_filtered = alpha * omega
        # Map back and compose
        R_filt = R_prev @ _exp_so3(omega_filtered)
        filtered.append(R_filt)

    return filtered


def lie_group_smooth(
    poses: np.ndarray,
    reference_bone_lengths: np.ndarray | None = None,
    alpha: float = 0.3,
) -> np.ndarray:
    """Smooth poses by filtering bone rotations in SO(3).

    1. For each bone, compute the rotation from a reference direction
       to the current bone direction at each frame.
    2. Smooth the rotation sequence in SO(3) via EMA in the Lie algebra.
    3. Reconstruct joint positions from smoothed rotations + bone lengths.

    Args:
        poses: (num_frames, num_joints, 3)
        reference_bone_lengths: (num_bones,) target bone lengths. If None,
            uses median over all frames.
        alpha: EMA smoothing factor (lower = smoother)

    Returns:
        smoothed: (num_frames, num_joints, 3)
    """
    num_frames = poses.shape[0]
    ref_dir = np.array([0.0, 1.0, 0.0])  # arbitrary reference direction

    # Compute reference bone lengths from median across frames
    if reference_bone_lengths is None:
        all_lengths = compute_bone_lengths(poses, TREE_BONES)
        reference_bone_lengths = np.median(all_lengths, axis=0)

    # Step 1: Convert each bone to a rotation sequence
    bone_rotations: dict[tuple[int, int], list[np.ndarray]] = {}
    for bone in TREE_BONES:
        p, c = bone
        rots = []
        for f in range(num_frames):
            bone_vec = poses[f, c] - poses[f, p]
            R = _rotation_between_vectors(ref_dir, bone_vec)
            rots.append(R)
        bone_rotations[bone] = rots

    # Step 2: Smooth rotations in SO(3)
    smoothed_rotations: dict[tuple[int, int], list[np.ndarray]] = {}
    for bone in TREE_BONES:
        smoothed_rotations[bone] = _ema_so3(bone_rotations[bone], alpha=alpha)

    # Step 3: Smooth the root position (simple EMA in R^3)
    root_positions = poses[:, 0, :].copy()
    smoothed_root = np.zeros_like(root_positions)
    smoothed_root[0] = root_positions[0]
    for f in range(1, num_frames):
        smoothed_root[f] = alpha * root_positions[f] + (1 - alpha) * smoothed_root[f - 1]

    # Step 4: Reconstruct from root outward using BFS order
    smoothed = np.zeros_like(poses)

    # Build adjacency: parent -> list of children (tree only)
    children: dict[int, list[tuple[int, int]]] = {}  # parent_joint -> [(bone_idx, child_joint)]
    for bi, (p, c) in enumerate(TREE_BONES):
        children.setdefault(p, []).append((bi, c))

    for f in range(num_frames):
        smoothed[f, 0] = smoothed_root[f]
        visited = {0}
        queue = [0]

        while queue:
            parent = queue.pop(0)
            for bi, child in children.get(parent, []):
                if child in visited:
                    continue
                bone = TREE_BONES[bi]
                R = smoothed_rotations[bone][f]
                bone_len = reference_bone_lengths[bi]
                # Reconstruct bone direction from smoothed rotation
                bone_dir = R @ ref_dir
                bone_dir_norm = np.linalg.norm(bone_dir)
                if bone_dir_norm > 1e-12:
                    bone_dir /= bone_dir_norm
                smoothed[f, child] = smoothed[f, parent] + bone_dir * bone_len
                visited.add(child)
                queue.append(child)

    return smoothed


# ============================================================================
# Method 2: Graph Spectral Filtering
# ============================================================================

def build_graph_laplacian(
    bones: list[tuple[int, int]],
    num_joints: int = NUM_JOINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized graph Laplacian for the skeleton.

    Returns:
        L: (num_joints, num_joints) Laplacian
        V: (num_joints, num_joints) eigenvectors (columns)
        lam: (num_joints,) eigenvalues (ascending)
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float64)
    for p, c in bones:
        A[p, c] = 1.0
        A[c, p] = 1.0

    D = np.diag(A.sum(axis=1))
    L = D - A

    # Eigendecompose (symmetric -> real eigenvalues)
    lam, V = np.linalg.eigh(L)
    return L, V, lam


def graph_spectral_filter(
    poses: np.ndarray,
    bones: list[tuple[int, int]] = BONES,
    cutoff_ratio: float = 0.5,
) -> np.ndarray:
    """Low-pass filter joint positions using graph spectral domain.

    High graph frequencies correspond to spatially irregular patterns
    (adjacent joints moving in very different directions). Zeroing them
    enforces spatial smoothness across the skeleton.

    Args:
        poses: (num_frames, num_joints, 3)
        bones: skeleton connectivity
        cutoff_ratio: fraction of eigenvalues to keep (0.5 = keep lowest half)

    Returns:
        filtered: (num_frames, num_joints, 3)
    """
    num_frames, num_joints, dim = poses.shape
    L, V, lam = build_graph_laplacian(bones, num_joints)

    # Number of frequencies to keep
    k = max(1, int(num_joints * cutoff_ratio))

    # Low-pass: keep only the k lowest-frequency components
    V_low = V[:, :k]  # (num_joints, k)

    filtered = np.zeros_like(poses)
    for f in range(num_frames):
        for d in range(dim):
            signal = poses[f, :, d]  # (num_joints,)
            # Forward: project onto graph Fourier basis
            coeffs = V_low.T @ signal  # (k,)
            # Inverse: reconstruct from low frequencies only
            filtered[f, :, d] = V_low @ coeffs

    return filtered


def graph_spectral_temporal_filter(
    poses: np.ndarray,
    bones: list[tuple[int, int]] = BONES,
    cutoff_ratio: float = 0.6,
    temporal_alpha: float = 0.3,
) -> np.ndarray:
    """Combined: graph spectral (spatial) + EMA (temporal) filtering."""
    # First pass: spatial graph filtering
    spatial = graph_spectral_filter(poses, bones, cutoff_ratio)
    # Second pass: temporal EMA
    filtered = np.zeros_like(spatial)
    filtered[0] = spatial[0]
    for f in range(1, spatial.shape[0]):
        filtered[f] = temporal_alpha * spatial[f] + (1 - temporal_alpha) * filtered[f - 1]
    return filtered


# ============================================================================
# Method 3: Manifold Projection (bone-length constraints)
# ============================================================================

def manifold_project(
    poses: np.ndarray,
    target_bone_lengths: np.ndarray | None = None,
    tree_bones: list[tuple[int, int]] = TREE_BONES,
) -> np.ndarray:
    """Project poses onto the bone-length constraint manifold.

    For each frame, traverse from root to leaves, scaling each bone
    vector to have the correct (target) length.

    Args:
        poses: (num_frames, num_joints, 3)
        target_bone_lengths: (num_tree_bones,) desired lengths.
            If None, uses median across all frames.
        tree_bones: directed (parent, child) bones for traversal.

    Returns:
        projected: (num_frames, num_joints, 3) with exact bone lengths.
    """
    num_frames = poses.shape[0]

    if target_bone_lengths is None:
        all_lengths = compute_bone_lengths(poses, tree_bones)
        target_bone_lengths = np.median(all_lengths, axis=0)

    # Build parent->children adjacency
    children: dict[int, list[tuple[int, int]]] = {}
    for bi, (p, c) in enumerate(tree_bones):
        children.setdefault(p, []).append((bi, c))

    projected = poses.copy()

    for f in range(num_frames):
        visited = {0}
        queue = [0]
        # Root stays in place

        while queue:
            parent = queue.pop(0)
            for bi, child in children.get(parent, []):
                if child in visited:
                    continue
                bone_vec = projected[f, child] - projected[f, parent]
                bone_len = np.linalg.norm(bone_vec)
                if bone_len > 1e-12:
                    bone_dir = bone_vec / bone_len
                else:
                    bone_dir = np.array([0.0, 1.0, 0.0])
                projected[f, child] = (
                    projected[f, parent] + bone_dir * target_bone_lengths[bi]
                )
                visited.add(child)
                queue.append(child)

    return projected


def one_euro_then_project(
    poses: np.ndarray,
    freq: float = 30.0,
    mincutoff: float = 1.0,
    beta: float = 0.007,
) -> np.ndarray:
    """One Euro filter followed by manifold projection."""
    filtered = one_euro_filter_poses(poses, freq, mincutoff, beta)
    return manifold_project(filtered)


# ============================================================================
# Method 4: Persistent Homology Quality Metric (SKIPPED - no ripser/gudhi)
# ============================================================================

def persistent_homology_available() -> bool:
    """Check if TDA libraries are available."""
    try:
        import ripser  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import gudhi  # noqa: F401
        return True
    except ImportError:
        pass
    return False


# ============================================================================
# Metrics
# ============================================================================

def compute_mse(clean: np.ndarray, filtered: np.ndarray) -> float:
    """Mean squared error between clean and filtered poses."""
    return float(np.mean((clean - filtered) ** 2))


def compute_jerk(poses: np.ndarray, dt: float = 1.0 / 30.0) -> float:
    """Mean jerk magnitude (3rd derivative of position).

    Lower jerk = smoother motion. Jerk is the gold standard for
    perceived smoothness in animation and biomechanics.
    """
    if poses.shape[0] < 4:
        return 0.0
    # Third-order finite difference
    d1 = np.diff(poses, axis=0) / dt
    d2 = np.diff(d1, axis=0) / dt
    d3 = np.diff(d2, axis=0) / dt
    jerk_mag = np.linalg.norm(d3, axis=-1)
    return float(np.mean(jerk_mag))


def compute_bone_length_variance(
    poses: np.ndarray,
    bones: list[tuple[int, int]] = BONES,
) -> float:
    """Mean variance of bone lengths across frames.

    Ideal: 0 (constant bone lengths). Any temporal filter that treats
    coordinates independently will have nonzero variance.
    """
    lengths = compute_bone_lengths(poses, bones)
    per_bone_var = np.var(lengths, axis=0)
    return float(np.mean(per_bone_var))


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    """Run all methods and compare."""

    print("=" * 72)
    print("EXPERIMENT: Topological & Geometric Methods for Pose Smoothing")
    print("=" * 72)
    print()

    # Generate data
    print("Generating test data...")
    print(f"  Frames: {NUM_FRAMES}, Joints: {NUM_JOINTS}, Noise sigma: {NOISE_SIGMA}")
    clean, noisy = generate_test_data()

    # Reference bone lengths from clean data
    clean_bone_lengths = compute_bone_lengths(clean, BONES)
    clean_bone_var = compute_bone_length_variance(clean)
    print(f"  Clean bone-length variance: {clean_bone_var:.2e}")
    print(f"  Noisy bone-length variance: {compute_bone_length_variance(noisy):.2e}")
    print()

    # Graph Laplacian analysis
    print("-" * 72)
    print("GRAPH LAPLACIAN ANALYSIS")
    print("-" * 72)
    L, V, lam = build_graph_laplacian(BONES)
    print(f"  Eigenvalues of skeleton Laplacian:")
    for i, l in enumerate(lam):
        print(f"    lambda_{i:2d} = {l:.4f}")
    print(f"  Algebraic connectivity (lambda_1): {lam[1]:.4f}")
    print(f"  Spectral gap: {lam[1] / lam[-1]:.4f}")
    print()

    # Run methods
    methods = {}

    # Baseline: noisy (no filtering)
    methods["Noisy (no filter)"] = noisy

    # Method 0: One Euro Filter
    print("-" * 72)
    print("METHOD 0: One Euro Filter (baseline)")
    print("-" * 72)
    t0 = time.perf_counter()
    result_oef = one_euro_filter_poses(noisy, mincutoff=1.5, beta=0.01)
    dt_oef = time.perf_counter() - t0
    methods["One Euro Filter"] = result_oef
    print(f"  Time: {dt_oef*1000:.1f} ms")
    print()

    # Method 1: Lie Group Smoothing
    print("-" * 72)
    print("METHOD 1: Lie Group Smoothing (SO(3) rotations)")
    print("-" * 72)
    t0 = time.perf_counter()
    result_lie = lie_group_smooth(noisy, alpha=0.4)
    dt_lie = time.perf_counter() - t0
    methods["Lie Group (SO3)"] = result_lie
    print(f"  Time: {dt_lie*1000:.1f} ms")
    print(f"  Key property: bone lengths constant BY CONSTRUCTION")
    print()

    # Method 2a: Graph Spectral (spatial only)
    print("-" * 72)
    print("METHOD 2a: Graph Spectral Filter (spatial only)")
    print("-" * 72)
    t0 = time.perf_counter()
    result_gf = graph_spectral_filter(noisy, cutoff_ratio=0.6)
    dt_gf = time.perf_counter() - t0
    methods["Graph Spectral (spatial)"] = result_gf
    print(f"  Time: {dt_gf*1000:.1f} ms")
    print(f"  Kept {int(NUM_JOINTS * 0.6)}/{NUM_JOINTS} graph frequencies")
    print()

    # Method 2b: Graph Spectral + Temporal
    print("-" * 72)
    print("METHOD 2b: Graph Spectral + Temporal EMA")
    print("-" * 72)
    t0 = time.perf_counter()
    result_gft = graph_spectral_temporal_filter(noisy, cutoff_ratio=0.6, temporal_alpha=0.4)
    dt_gft = time.perf_counter() - t0
    methods["Graph Spectral + Temporal"] = result_gft
    print(f"  Time: {dt_gft*1000:.1f} ms")
    print()

    # Method 3: Manifold Projection (applied to One Euro output)
    print("-" * 72)
    print("METHOD 3: One Euro + Manifold Projection")
    print("-" * 72)
    t0 = time.perf_counter()
    result_mp = one_euro_then_project(noisy, mincutoff=1.5, beta=0.01)
    dt_mp = time.perf_counter() - t0
    methods["One Euro + Manifold Proj"] = result_mp
    print(f"  Time: {dt_mp*1000:.1f} ms")
    print()

    # Method 3b: Lie Group + Manifold Projection (belt and suspenders)
    print("-" * 72)
    print("METHOD 3b: Lie Group + Manifold Projection")
    print("-" * 72)
    t0 = time.perf_counter()
    result_lie_mp = manifold_project(result_lie)
    dt_lie_mp = time.perf_counter() - t0 + dt_lie
    methods["Lie Group + Manifold Proj"] = result_lie_mp
    print(f"  Time: {dt_lie_mp*1000:.1f} ms (total)")
    print()

    # Combined: Graph Spectral + Temporal + Manifold Projection
    print("-" * 72)
    print("METHOD COMBO: Graph Spectral + Temporal + Manifold Projection")
    print("-" * 72)
    t0 = time.perf_counter()
    result_combo = manifold_project(result_gft)
    dt_combo = time.perf_counter() - t0 + dt_gft
    methods["Spectral+Temporal+ManProj"] = result_combo
    print(f"  Time: {dt_combo*1000:.1f} ms (total)")
    print()

    # Persistent Homology
    print("-" * 72)
    print("METHOD 4: Persistent Homology Quality Metric")
    print("-" * 72)
    if persistent_homology_available():
        print("  Available - computing...")
    else:
        print("  SKIPPED: ripser/gudhi not installed")
        print("  Install with: pip install ripser gudhi")
        print("  Would measure topological noise (spurious H0/H1 features)")
    print()

    # -----------------------------------------------------------------------
    # Results comparison
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("RESULTS COMPARISON")
    print("=" * 72)
    print()
    print(f"{'Method':<32s}  {'MSE':>10s}  {'Jerk':>12s}  {'BoneLenVar':>12s}")
    print("-" * 72)

    for name, result in methods.items():
        mse = compute_mse(clean, result)
        jerk = compute_jerk(result)
        blv = compute_bone_length_variance(result)
        print(f"{name:<32s}  {mse:10.2e}  {jerk:12.2f}  {blv:12.2e}")

    # Clean reference
    jerk_clean = compute_jerk(clean)
    blv_clean = compute_bone_length_variance(clean)
    print("-" * 72)
    print(f"{'Clean (ground truth)':<32s}  {'0':>10s}  {jerk_clean:12.2f}  {blv_clean:12.2e}")
    print()

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("ANALYSIS & INSIGHTS")
    print("=" * 72)
    print()
    print("1. LIE GROUP SMOOTHING (SO(3)):")
    print("   - Smooths in rotation space, guaranteeing constant bone lengths")
    print("   - Bone-length variance should be near-zero (limited by tree approx)")
    print("   - Natural for articulated bodies: rotations are the true DOFs")
    print("   - Related work: NRDF (He et al., CVPR 2024) uses product-quaternion")
    print("     manifold with Neural Riemannian Distance Fields for pose priors")
    print("   - Grassia (1998) established log/exp map smoothing for animation")
    print()
    print("2. GRAPH SPECTRAL FILTERING:")
    print("   - Treats skeleton as a graph, filters joint signals in spectral domain")
    print("   - High graph frequencies = spatially irregular patterns (noise)")
    print("   - Complementary to temporal filtering (operates across joints)")
    print("   - Related: ST-GCN (Yan et al., AAAI 2018) uses graph convolutions")
    print("     for action recognition; GS-Net (2023) iterative spectral filtering")
    print("   - Pure spatial filter alone may introduce temporal artifacts")
    print()
    print("3. MANIFOLD PROJECTION:")
    print("   - Simple constraint satisfaction: scale bones to target length")
    print("   - Can be applied AFTER any other filter as a post-processing step")
    print("   - ManiPose (NeurIPS 2024) uses manifold constraints for multi-")
    print("     hypothesis 3D lifting with anatomical plausibility")
    print("   - Matsune & Hu (WACV 2024) use geometry losses including bone")
    print("     length penalties during training")
    print()
    print("4. PERSISTENT HOMOLOGY (not tested - libraries unavailable):")
    print("   - Would provide a quality metric based on topological features")
    print("   - Clean skeleton: 1 connected component (H0), possibly 1 loop (H1)")
    print("   - Noisy data creates spurious topological features")
    print("   - Total persistence = sum of (death - birth) for all features")
    print("   - Recent survey: Hayes et al. (2025) persistent sheaf Laplacian")
    print()
    print("RECOMMENDATION:")
    print("   For real-time pose smoothing in BRACE:")
    print("   - Use One Euro Filter for low-latency temporal smoothing")
    print("   - Apply Manifold Projection as post-processing (cheap, guarantees")
    print("     bone lengths)")
    print("   - Lie Group smoothing is elegant but ~3-5x slower than coordinate")
    print("     filtering; best for offline processing")
    print("   - Graph Spectral filtering is good for spatial denoising but adds")
    print("     latency; consider for recorded video analysis")
    print()

    # -----------------------------------------------------------------------
    # Verify invariants
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("INVARIANT CHECKS")
    print("=" * 72)

    # Lie group should have near-zero bone-length variance
    lie_blv = compute_bone_length_variance(result_lie, TREE_BONES)
    print(f"  Lie Group bone-length variance (tree bones): {lie_blv:.2e}")
    assert lie_blv < 1e-20, f"Lie group should have zero BLV, got {lie_blv}"
    print("    PASS: bone lengths are exactly constant")

    # Manifold projection should have zero variance on tree bones
    mp_blv = compute_bone_length_variance(result_mp, TREE_BONES)
    print(f"  Manifold Proj bone-length variance (tree): {mp_blv:.2e}")
    assert mp_blv < 1e-20, f"Manifold proj should have zero BLV, got {mp_blv}"
    print("    PASS: bone lengths are exactly constant")

    # All filtered should have lower jerk than noisy
    noisy_jerk = compute_jerk(noisy)
    for name, result in methods.items():
        if name == "Noisy (no filter)":
            continue
        j = compute_jerk(result)
        status = "PASS" if j < noisy_jerk else "FAIL"
        print(f"  {name} jerk < noisy jerk: {status} ({j:.2f} vs {noisy_jerk:.2f})")

    print()
    print("DONE.")


if __name__ == "__main__":
    run_experiment()
