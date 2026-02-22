"""Collision detection between two tracked skeletons from YOLO-pose output.

Provides a coarse-to-fine collision pipeline:
1. Bounding-box IoU as a fast rejection filter.
2. Head-to-body proximity measurement in pixel space.
3. Metric-scale estimation from shoulder width.
4. Closing-velocity computation via central difference on head position history.
5. Contact-zone classification with head-coupling factors for downstream
   concussion severity estimation.

Designed to plug into the BRACE streaming pipeline where each tracked person
is represented by COCO-17 keypoints with per-keypoint confidence scores.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# COCO-17 keypoint indices
# ---------------------------------------------------------------------------

# Individual keypoint names for readability
NOSE = 0
L_EYE = 1
R_EYE = 2
L_EAR = 3
R_EAR = 4
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

# Semantic groups used throughout the module
HEAD_INDICES = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]  # [0, 1, 2, 3, 4]
BODY_INDICES = [                                      # [5..16]
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST, L_HIP, R_HIP,
    L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
]

# Minimum keypoint confidence to treat a detection as visible
CONF_THRESHOLD = 0.3

# Maximum plausible closing speed in field sports (m/s).
# Speeds above this are almost certainly calibration/tracking artefacts.
MAX_CLOSING_SPEED_MS = 10.0

# Average adult bi-acromial (shoulder) width used as a metric ruler
SHOULDER_WIDTH_METERS = 0.45


# ---------------------------------------------------------------------------
# 1. Bounding-box IoU — coarse spatial overlap filter
# ---------------------------------------------------------------------------

def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two axis-aligned bounding boxes.

    This is the coarse collision filter: if the players' bounding boxes do not
    overlap at all (IoU == 0), there is no need to run the more expensive
    keypoint-level proximity checks.

    Parameters
    ----------
    box_a : array-like, shape (4,)
        Bounding box in (x1, y1, x2, y2) format (top-left, bottom-right).
    box_b : array-like, shape (4,)
        Second bounding box in the same format.

    Returns
    -------
    float
        IoU value in [0.0, 1.0].  Returns 0.0 when the boxes do not overlap
        or when either box has zero area.
    """
    # Unpack coordinates for clarity
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    # Compute the intersection rectangle
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Width and height of the intersection (clamped to zero if no overlap)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Individual box areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Union = sum of areas minus the overlap counted twice
    union = area_a + area_b - inter_area

    # Guard against degenerate boxes with zero union area
    if union <= 0.0:
        return 0.0

    return inter_area / union


# ---------------------------------------------------------------------------
# 2. Head-to-body minimum distance
# ---------------------------------------------------------------------------

def head_to_body_min_distance(
    kpts_a: np.ndarray,
    conf_a: np.ndarray,
    kpts_b: np.ndarray,
    conf_b: np.ndarray,
    vis_threshold: float = CONF_THRESHOLD,
) -> tuple[float, int]:
    """Minimum pixel distance from person A's head centroid to person B's body.

    Measures how close person A's head is to any visible body keypoint on
    person B.  This is the core proximity metric for detecting a head-contact
    collision.

    Parameters
    ----------
    kpts_a : ndarray, shape (17, 2)
        (x, y) keypoint coordinates for person A.
    conf_a : ndarray, shape (17,)
        Per-keypoint confidence for person A.
    kpts_b : ndarray, shape (17, 2)
        (x, y) keypoint coordinates for person B.
    conf_b : ndarray, shape (17,)
        Per-keypoint confidence for person B.
    vis_threshold : float
        Minimum confidence to treat a keypoint as visible.

    Returns
    -------
    tuple[float, int]
        (min_distance_px, nearest_body_keypoint_idx).
        Returns (inf, -1) if person A has no visible head keypoints or person B
        has no visible body keypoints.
    """
    # Compute person A's head centroid from visible head keypoints
    head_center = compute_head_center(kpts_a, conf_a)
    if head_center is None:
        # Cannot compute head position — bail out
        return (float("inf"), -1)

    # Gather person B's visible body keypoints (indices 5-16)
    visible_body_idxs = [
        idx for idx in BODY_INDICES
        if idx < len(conf_b) and conf_b[idx] >= vis_threshold
    ]
    if not visible_body_idxs:
        # Person B has no visible body keypoints — no valid proximity
        return (float("inf"), -1)

    # Compute Euclidean distance from A's head centroid to each visible B body keypoint
    body_pts = kpts_b[visible_body_idxs]  # shape (N, 2)
    diffs = body_pts - head_center        # broadcast head_center (2,) over rows
    distances = np.linalg.norm(diffs, axis=1)

    # Find the nearest body keypoint
    min_idx_local = int(np.argmin(distances))
    min_distance = float(distances[min_idx_local])
    nearest_body_kpt = visible_body_idxs[min_idx_local]

    # Bug 3 fix: Check synthetic torso center (midpoint of visible shoulders
    # and hips).  The torso center is not a real COCO keypoint but captures
    # impacts to the middle of the trunk that fall between existing keypoints.
    torso_kpt_idxs = [
        idx for idx in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        if idx < len(conf_b) and conf_b[idx] >= vis_threshold
    ]
    if len(torso_kpt_idxs) >= 2:
        torso_center = np.mean(kpts_b[torso_kpt_idxs], axis=0)
        torso_dist = float(np.linalg.norm(head_center - torso_center))
        # If the synthetic torso center is closer, classify as torso contact
        if torso_dist < min_distance:
            min_distance = torso_dist
            # Use L_HIP as proxy index to trigger "head_to_torso" in
            # determine_contact_zone
            nearest_body_kpt = L_HIP

    return (min_distance, nearest_body_kpt)


# ---------------------------------------------------------------------------
# 2b. Head-to-head distance (Bug 1 fix: head-to-head collisions were never
#     detected because head_to_body_min_distance only searches BODY_INDICES
#     5-16, so HEAD_INDICES 0-4 could never be the nearest_body_kpt)
# ---------------------------------------------------------------------------

def head_to_head_distance(
    kpts_a: np.ndarray,
    conf_a: np.ndarray,
    kpts_b: np.ndarray,
    conf_b: np.ndarray,
    vis_threshold: float = CONF_THRESHOLD,
) -> float:
    """Distance between two persons' head centroids in pixels.

    Parameters
    ----------
    kpts_a : ndarray, shape (17, 2)
        (x, y) keypoint coordinates for person A.
    conf_a : ndarray, shape (17,)
        Per-keypoint confidence for person A.
    kpts_b : ndarray, shape (17, 2)
        (x, y) keypoint coordinates for person B.
    conf_b : ndarray, shape (17,)
        Per-keypoint confidence for person B.
    vis_threshold : float
        Minimum confidence to treat a keypoint as visible.

    Returns
    -------
    float
        Euclidean distance in pixels between the two head centroids.
        Returns inf if either person has no visible head keypoints.
    """
    head_a = compute_head_center(kpts_a, conf_a)
    head_b = compute_head_center(kpts_b, conf_b)
    if head_a is None or head_b is None:
        return float("inf")
    return float(np.linalg.norm(head_a - head_b))


# ---------------------------------------------------------------------------
# 3. Metric scale estimation from shoulder width
# ---------------------------------------------------------------------------

def estimate_meters_per_pixel(
    kpts: np.ndarray,
    conf: np.ndarray,
) -> tuple[float, str]:
    """Derive a meters-per-pixel scale factor from visible anatomical landmarks.

    Priority order:
    1. Shoulder width (L_SHOULDER to R_SHOULDER) — most reliable because
       bi-acromial width is fairly constant across adults (~0.45 m).
    2. Torso length (midpoint of shoulders to midpoint of hips) — used as
       fallback when one shoulder is occluded.  Assumed ~0.50 m.
    3. Hard-coded fallback of 0.004 m/px — roughly correct for a person
       filling half a 1080p frame at ~3 m distance.

    A height cross-check (head-to-ankle distance) validates the scale:
    if the implied height falls outside [1.5, 2.1] m, the scale is
    overridden and confidence is set to "low".

    Parameters
    ----------
    kpts : ndarray, shape (17, 2)
        (x, y) keypoint coordinates.
    conf : ndarray, shape (17,)
        Per-keypoint confidence scores.

    Returns
    -------
    tuple[float, str]
        (meters_per_pixel, calibration_confidence).
        calibration_confidence is one of "high", "medium", "low".
    """
    m_per_px: float | None = None
    source: str = "fallback"

    # --- Priority 1: shoulder width ---
    if (
        L_SHOULDER < len(conf)
        and R_SHOULDER < len(conf)
        and conf[L_SHOULDER] >= CONF_THRESHOLD
        and conf[R_SHOULDER] >= CONF_THRESHOLD
    ):
        shoulder_px = float(np.linalg.norm(kpts[L_SHOULDER] - kpts[R_SHOULDER]))
        if shoulder_px > 1.0:
            m_per_px = SHOULDER_WIDTH_METERS / shoulder_px
            source = "shoulder"

    # --- Priority 2: torso length ---
    if m_per_px is None:
        shoulder_vis = [
            idx for idx in (L_SHOULDER, R_SHOULDER)
            if idx < len(conf) and conf[idx] >= CONF_THRESHOLD
        ]
        hip_vis = [
            idx for idx in (L_HIP, R_HIP)
            if idx < len(conf) and conf[idx] >= CONF_THRESHOLD
        ]
        if shoulder_vis and hip_vis:
            shoulder_mid = np.mean(kpts[shoulder_vis], axis=0)
            hip_mid = np.mean(kpts[hip_vis], axis=0)
            torso_px = float(np.linalg.norm(shoulder_mid - hip_mid))
            if torso_px > 1.0:
                m_per_px = 0.50 / torso_px
                source = "torso"

    # --- Priority 3: fallback ---
    if m_per_px is None:
        m_per_px = 0.004
        source = "fallback"

    # --- Height cross-check ---
    # If head (NOSE) and at least one ankle are visible, verify the implied
    # standing height is plausible (1.5–2.1 m for adults).
    head_pt = None
    if NOSE < len(conf) and conf[NOSE] >= CONF_THRESHOLD:
        head_pt = kpts[NOSE]

    ankle_pts = []
    for aidx in (L_ANKLE, R_ANKLE):
        if aidx < len(conf) and conf[aidx] >= CONF_THRESHOLD:
            ankle_pts.append(kpts[aidx])

    height_checked = False
    if head_pt is not None and ankle_pts:
        # Use the lowest (largest-Y) ankle
        lowest_ankle = max(ankle_pts, key=lambda p: p[1])
        height_px = float(np.linalg.norm(head_pt - lowest_ankle))
        if height_px > 1.0:
            height_m = height_px * m_per_px
            height_checked = True
            if height_m < 1.5 or height_m > 2.1:
                # Override to put height at 1.75 m
                m_per_px = 1.75 / height_px
                source = "height_override"

    # --- Determine confidence ---
    if source == "fallback" or source == "height_override":
        calibration_confidence = "low"
    elif source == "shoulder":
        calibration_confidence = "high" if height_checked else "medium"
    else:
        # torso-based
        calibration_confidence = "medium"

    return (m_per_px, calibration_confidence)


# ---------------------------------------------------------------------------
# 4. Head center computation
# ---------------------------------------------------------------------------

def compute_head_center(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
) -> np.ndarray | None:
    """Average of visible head keypoints (nose, eyes, ears) above threshold.

    Parameters
    ----------
    kpts_xy : ndarray, shape (17, 2) or (N, 2)
        (x, y) coordinates of all keypoints.
    kpts_conf : ndarray, shape (17,) or (N,)
        Per-keypoint confidence scores.

    Returns
    -------
    ndarray, shape (2,) or None
        Mean (x, y) position of visible head keypoints, or None when fewer
        than one head keypoint passes the confidence threshold.
    """
    # Filter to head keypoints that are both in range and above confidence
    visible_head = [
        idx for idx in HEAD_INDICES
        if idx < len(kpts_conf) and kpts_conf[idx] >= CONF_THRESHOLD
    ]

    if not visible_head:
        # No visible head keypoints — cannot determine head position
        return None

    # Arithmetic mean of visible head keypoint positions
    return np.mean(kpts_xy[visible_head], axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# 5. Contact-zone classification
# ---------------------------------------------------------------------------

def determine_contact_zone(nearest_body_idx: int) -> tuple[str, float]:
    """Classify the contact zone from the nearest body keypoint on the struck person.

    The head-coupling factor represents the fraction of kinetic energy
    transmitted to the struck person's head based on where the impact lands:
    - Head-to-head contacts transmit the most energy (0.7).
    - Shoulder impacts are partially absorbed by the neck musculature (0.4).
    - Torso impacts are cushioned by the large mass of the trunk (0.2).
    - Limb contacts are glancing blows with minimal head coupling (0.08).

    Parameters
    ----------
    nearest_body_idx : int
        COCO keypoint index (0-16) of the struck person's body part closest to
        the incoming head.

    Returns
    -------
    tuple[str, float]
        (zone_name, head_coupling_factor).
    """
    # Head-to-head: incoming head collides with the struck person's head region
    if nearest_body_idx in HEAD_INDICES:
        return ("head_to_head", 0.7)

    # Shoulder zone: direct shoulder contact
    if nearest_body_idx in (L_SHOULDER, R_SHOULDER):
        return ("head_to_shoulder", 0.4)

    # Bug 2 fix: Elbows are typically tucked against the torso during tackles,
    # so classify them as torso contact with a moderate coupling factor
    if nearest_body_idx in (L_ELBOW, R_ELBOW):
        return ("head_to_torso", 0.25)

    # Torso zone: hips are part of the trunk
    if nearest_body_idx in (L_HIP, R_HIP):
        return ("head_to_torso", 0.2)

    # Bug 2 fix: Wrists are sometimes extended (stiff-arm) but often near
    # the torso — higher coupling than knees/ankles
    if nearest_body_idx in (L_WRIST, R_WRIST):
        return ("head_to_limb", 0.15)

    # Knees, ankles: true limb extremities with minimal head coupling
    return ("head_to_limb", 0.08)


# ---------------------------------------------------------------------------
# 6. Closing velocity
# ---------------------------------------------------------------------------

def compute_closing_velocity(
    positions_a: list[np.ndarray],
    positions_b: list[np.ndarray],
    fps: float,
    meters_per_pixel: float,
    sg_window: int = 5,
    sg_polyorder: int = 2,
) -> float:
    """Compute the closing speed between two heads in metres per second.

    Uses the central-difference method to estimate each head's instantaneous
    velocity, then projects the relative velocity onto the approach direction
    (unit vector from A to B).  A positive closing speed means the two heads
    are moving toward each other.

    When enough history points are available, a Savitzky-Golay filter is
    applied to smooth the position arrays before differencing (reduces
    noise-induced velocity spikes).

    Parameters
    ----------
    positions_a : list of ndarray, each shape (2,)
        Recent head positions for person A, ordered oldest to newest.
        Requires at least 3 frames.
    positions_b : list of ndarray, each shape (2,)
        Recent head positions for person B, same length requirement.
    fps : float
        Frame rate in frames per second.
    meters_per_pixel : float
        Conversion factor from pixel distance to metres.
    sg_window : int
        Savitzky-Golay filter window length (must be odd, >= 3).
        Skipped silently when fewer points than the window.
    sg_polyorder : int
        Polynomial order for the SG filter (must be < sg_window).

    Returns
    -------
    float
        Closing speed in m/s.  Clamped to >= 0 (negative means separating,
        which we report as zero).
    """
    # Need at least 3 frames for central difference (prev, current, next)
    if len(positions_a) < 3 or len(positions_b) < 3:
        return 0.0

    if fps <= 0.0 or meters_per_pixel <= 0.0:
        return 0.0

    # Convert to arrays for potential filtering
    arr_a = np.array(positions_a)  # shape (N, 2)
    arr_b = np.array(positions_b)

    # Apply Savitzky-Golay filter when enough points exist
    n_pts = min(len(arr_a), len(arr_b))
    if sg_window >= 3 and sg_window % 2 == 1 and sg_window <= n_pts:
        arr_a = savgol_filter(arr_a, sg_window, sg_polyorder, axis=0)
        arr_b = savgol_filter(arr_b, sg_window, sg_polyorder, axis=0)

    # Time step between consecutive frames
    dt = 1.0 / fps

    # Central difference for velocity: v[i] = (pos[i+1] - pos[i-1]) / (2 * dt)
    # We use the latest triplet of positions for the most recent velocity estimate
    vel_a_px = (arr_a[-1] - arr_a[-3]) / (2.0 * dt)
    vel_b_px = (arr_b[-1] - arr_b[-3]) / (2.0 * dt)

    # Relative velocity of A with respect to B (positive toward B means closing)
    rel_vel_px = vel_a_px - vel_b_px

    # Approach direction: unit vector from A's latest position toward B's latest
    approach_vec = arr_b[-1] - arr_a[-1]
    approach_dist = float(np.linalg.norm(approach_vec))
    if approach_dist < 1e-6:
        # Heads are essentially at the same point — use velocity magnitude
        closing_px_per_s = float(np.linalg.norm(rel_vel_px))
    else:
        # Project relative velocity onto the approach direction
        approach_unit = approach_vec / approach_dist
        closing_px_per_s = float(np.dot(rel_vel_px, approach_unit))

    # Convert from px/s to m/s
    closing_speed_ms = closing_px_per_s * meters_per_pixel

    # Clamp: negative closing speed means heads are separating, report as zero
    return max(0.0, closing_speed_ms)


def compute_approach_angle(
    positions_a: list[np.ndarray],
    positions_b: list[np.ndarray],
    fps: float,
) -> float:
    """Compute the angle between relative velocity and the approach direction.

    Returns the angle in radians between the relative velocity vector
    (A w.r.t. B) and the line connecting A to B.  A head-on collision
    has angle ~0; a pure side-swipe has angle ~pi/2.

    Parameters
    ----------
    positions_a : list of ndarray, each shape (2,)
        Recent head positions for person A (oldest to newest, >= 3).
    positions_b : list of ndarray, each shape (2,)
        Recent head positions for person B (oldest to newest, >= 3).
    fps : float
        Frame rate in frames per second.

    Returns
    -------
    float
        Angle in radians in [0, pi].  Returns 0.0 if insufficient data.
    """
    if len(positions_a) < 3 or len(positions_b) < 3 or fps <= 0.0:
        return 0.0

    dt = 1.0 / fps
    vel_a = (positions_a[-1] - positions_a[-3]) / (2.0 * dt)
    vel_b = (positions_b[-1] - positions_b[-3]) / (2.0 * dt)
    rel_vel = vel_a - vel_b

    approach_vec = positions_b[-1] - positions_a[-1]
    rel_mag = float(np.linalg.norm(rel_vel))
    app_mag = float(np.linalg.norm(approach_vec))

    if rel_mag < 1e-6 or app_mag < 1e-6:
        return 0.0

    cos_theta = float(np.dot(rel_vel, approach_vec)) / (rel_mag * app_mag)
    # Clamp for numerical safety
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


# ---------------------------------------------------------------------------
# CollisionDetector class
# ---------------------------------------------------------------------------

class CollisionDetector:
    """Stateful collision detector for multi-person YOLO-pose tracking.

    Maintains a ring buffer of recent head positions per track ID so that
    closing velocity can be computed across frames.  The full collision check
    pipeline is:

    1. **Bounding-box IoU** (coarse filter) — reject pairs with no spatial
       overlap.
    2. **Head-to-body proximity** — measure distance from each person's head
       to the other's body keypoints in both directions.
    3. **Metric scale** — convert pixel distances to metres using shoulder
       width.
    4. **Contact zone** — classify the struck body region and look up the
       head-coupling factor.
    5. **Closing velocity** — compute approach speed from the head position
       history.

    Parameters
    ----------
    iou_threshold : float
        Minimum bounding-box IoU to pass the coarse filter.
    proximity_ratio : float
        Maximum head-to-body distance expressed as a fraction of the
        person's bounding-box diagonal.  Smaller values require the head to
        be closer before a collision is reported.
    history_frames : int
        Number of recent head positions to retain per track ID for velocity
        estimation.
    """

    def __init__(
        self,
        iou_threshold: float = 0.05,
        proximity_ratio: float = 0.5,
        history_frames: int = 5,
        max_closing_speed_ms: float = MAX_CLOSING_SPEED_MS,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.proximity_ratio = proximity_ratio
        self.history_frames = max(3, history_frames)  # need >= 3 for central diff
        self.max_closing_speed_ms = max_closing_speed_ms

        # Per-track-ID ring buffer of recent head positions (pixel coords)
        self._head_history: dict[int, deque[np.ndarray]] = {}

        # Latest bounding box per track ID, for IoU checks
        self._bboxes: dict[int, np.ndarray] = {}

    def update(
        self,
        track_id: int,
        head_px: np.ndarray | None,
        bbox: np.ndarray,
    ) -> None:
        """Store a new head position and bounding box for the given track.

        Call this once per frame per tracked person, *before* calling
        ``check_collision``.

        Parameters
        ----------
        track_id : int
            Unique tracker-assigned ID for this person.
        head_px : ndarray, shape (2,) or None
            Head centroid in pixel coordinates, or None if the head is not
            visible this frame.
        bbox : ndarray, shape (4,)
            Bounding box in (x1, y1, x2, y2) format.
        """
        # Always update the bounding box
        self._bboxes[track_id] = np.asarray(bbox, dtype=np.float64)

        if head_px is None:
            # No head visible — skip position history update
            return

        # Initialise the ring buffer for new tracks
        if track_id not in self._head_history:
            self._head_history[track_id] = deque(maxlen=self.history_frames)

        # Append the latest head position (as a copy to avoid aliasing)
        self._head_history[track_id].append(np.asarray(head_px, dtype=np.float64))

    def check_collision(
        self,
        tid_a: int,
        tid_b: int,
        kpts_a: np.ndarray,
        conf_a: np.ndarray,
        kpts_b: np.ndarray,
        conf_b: np.ndarray,
        fps: float,
        meters_per_pixel: float,
    ) -> dict[str, Any] | None:
        """Check whether persons A and B are in a head-contact collision.

        Parameters
        ----------
        tid_a : int
            Track ID of person A.
        tid_b : int
            Track ID of person B.
        kpts_a : ndarray, shape (17, 2)
            Keypoint positions for person A.
        conf_a : ndarray, shape (17,)
            Keypoint confidences for person A.
        kpts_b : ndarray, shape (17, 2)
            Keypoint positions for person B.
        conf_b : ndarray, shape (17,)
            Keypoint confidences for person B.
        fps : float
            Video frame rate (frames per second).
        meters_per_pixel : float
            Pre-computed metric scale (or use ``estimate_meters_per_pixel``).

        Returns
        -------
        dict or None
            None if no collision is detected.  Otherwise a dict with:
            - ``colliding`` (bool): always True
            - ``tid_a``, ``tid_b`` (int): the two track IDs
            - ``iou`` (float): bounding-box IoU
            - ``min_distance_px`` (float): closest head-to-body distance
            - ``contact_zone`` (str): e.g. "head_to_head", "head_to_shoulder"
            - ``head_coupling_factor`` (float): energy transfer coefficient
            - ``closing_speed_ms`` (float): approach speed in m/s
            - ``struck_tid`` (int): track ID of the person whose head is closer
              to the other's body (i.e. the person being struck)
        """
        # ------------------------------------------------------------------
        # Step 1: Bounding-box IoU coarse filter
        # ------------------------------------------------------------------
        bbox_a = self._bboxes.get(tid_a)
        bbox_b = self._bboxes.get(tid_b)
        if bbox_a is None or bbox_b is None:
            # One of the tracks has not been updated yet — cannot check
            return None

        iou = bbox_iou(bbox_a, bbox_b)
        if iou < self.iou_threshold:
            # Bounding boxes do not overlap enough — no collision
            return None

        # ------------------------------------------------------------------
        # Step 2: Head-to-body proximity in both directions
        # ------------------------------------------------------------------
        # A's head → B's body
        dist_a_to_b, nearest_on_b = head_to_body_min_distance(
            kpts_a, conf_a, kpts_b, conf_b,
        )
        # B's head → A's body
        dist_b_to_a, nearest_on_a = head_to_body_min_distance(
            kpts_b, conf_b, kpts_a, conf_a,
        )

        # Pick the smaller distance — that direction is the primary contact
        if dist_a_to_b <= dist_b_to_a:
            min_distance_px = dist_a_to_b
            nearest_body_idx = nearest_on_b
            # Person A's head is hitting person B's body → B is "struck"
            # but semantically the "struck" person is the one whose body
            # is receiving the head impact; however the person whose HEAD
            # is closer to the OTHER's body is the one leading with their
            # head.  We define struck_tid as the person whose head is
            # closer to the other's body (the one at head-injury risk).
            struck_tid = tid_a
        else:
            min_distance_px = dist_b_to_a
            nearest_body_idx = nearest_on_a
            struck_tid = tid_b

        # ------------------------------------------------------------------
        # Step 2b: Head-to-head proximity check (Bug 1 fix)
        # head_to_body_min_distance only searches BODY_INDICES (5-16), so
        # head-to-head collisions were never detected.  Compute the direct
        # head-centroid-to-head-centroid distance and use it if it is
        # smaller than both head-to-body distances.
        # ------------------------------------------------------------------
        h2h_dist = head_to_head_distance(kpts_a, conf_a, kpts_b, conf_b)
        # If heads are closer to each other than either head is to the
        # other's body, this is a head-to-head collision (most dangerous type)
        if h2h_dist < min_distance_px:
            min_distance_px = h2h_dist
            nearest_body_idx = NOSE  # Use nose index to trigger head_to_head zone
            # In head-to-head, both persons are at equal risk — pick arbitrary
            struck_tid = tid_a

        # If neither direction produced a valid distance, bail out
        if not np.isfinite(min_distance_px):
            return None

        # ------------------------------------------------------------------
        # Step 3: Proximity threshold based on bounding-box diagonal
        # ------------------------------------------------------------------
        # Use the average diagonal of both boxes as the reference scale
        diag_a = float(np.linalg.norm(bbox_a[2:] - bbox_a[:2]))
        diag_b = float(np.linalg.norm(bbox_b[2:] - bbox_b[:2]))
        avg_diag = (diag_a + diag_b) / 2.0

        if avg_diag <= 0.0:
            return None

        # The head must be within proximity_ratio * diagonal to count
        if min_distance_px > self.proximity_ratio * avg_diag:
            return None

        # ------------------------------------------------------------------
        # Step 4: Contact-zone classification
        # ------------------------------------------------------------------
        contact_zone, head_coupling_factor = determine_contact_zone(nearest_body_idx)

        # ------------------------------------------------------------------
        # Step 5: Closing velocity from head position history
        # ------------------------------------------------------------------
        history_a = self._head_history.get(tid_a)
        history_b = self._head_history.get(tid_b)

        approach_angle_rad = 0.0
        if history_a is not None and history_b is not None:
            closing_speed = compute_closing_velocity(
                list(history_a), list(history_b),
                fps, meters_per_pixel,
            )
            approach_angle_rad = compute_approach_angle(
                list(history_a), list(history_b), fps,
            )
        else:
            closing_speed = 0.0

        # Hard-clamp closing speed to a physiologically plausible maximum
        closing_speed_raw = closing_speed
        closing_speed_clamped = closing_speed > self.max_closing_speed_ms
        if closing_speed_clamped:
            print(
                f"[SPEED CLAMP] raw={closing_speed_raw:.1f} m/s "
                f"clamped to {self.max_closing_speed_ms:.1f} m/s"
            )
            closing_speed = self.max_closing_speed_ms

        # ------------------------------------------------------------------
        # Step 6: Minimum head confidence across both persons
        # ------------------------------------------------------------------
        head_confs = []
        for idx in HEAD_INDICES:
            if idx < len(conf_a) and conf_a[idx] >= CONF_THRESHOLD:
                head_confs.append(float(conf_a[idx]))
            if idx < len(conf_b) and conf_b[idx] >= CONF_THRESHOLD:
                head_confs.append(float(conf_b[idx]))
        min_head_confidence = min(head_confs) if head_confs else 0.0

        # ------------------------------------------------------------------
        # Build the collision report
        # ------------------------------------------------------------------
        return {
            "colliding": True,
            "tid_a": tid_a,
            "tid_b": tid_b,
            "iou": round(iou, 4),
            "min_distance_px": round(min_distance_px, 2),
            "contact_zone": contact_zone,
            "head_coupling_factor": head_coupling_factor,
            "closing_speed_ms": round(closing_speed, 4),
            "struck_tid": struck_tid,
            # New additive fields
            "closing_speed_raw_ms": round(closing_speed_raw, 4),
            "closing_speed_clamped": closing_speed_clamped,
            "approach_angle_rad": round(approach_angle_rad, 6),
            "calibration_confidence": "medium",  # placeholder; caller sets from estimate_meters_per_pixel
            "min_head_confidence": round(min_head_confidence, 4),
        }
