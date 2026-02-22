"""Standalone concussion monitoring via YOLOv11-Pose tracking with collision detection.

Tracks head keypoints (nose, ears) per person across frames, detects collisions
between skeleton pairs, computes head speed (m/s), linear acceleration (g-force),
and biomechanical concussion probability using the Rowson & Duma (2013) model.

Overlays results on each frame and logs impact/collision events to CSV.

Usage:
    python scripts/concussion_monitor.py path/to/video.mp4 [--output out.mp4] [--csv impacts.csv]

Dependencies: ultralytics, opencv-python, numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Import collision detection and biomechanics modules (same directory)
from _collision_detection import (
    CollisionDetector,
    compute_head_center,
    estimate_meters_per_pixel,
    compute_approach_angle,
)
from _biomechanics_model import (
    estimate_body_mass,
    score_collision,
    delta_v_to_peak_g,
    estimate_rotational_acceleration,
    concussion_probability_rowson_duma,
    IMPACT_DURATION_UNHELMETED_S,
)

# ── Constants ────────────────────────────────────────────────────────────────

# COCO keypoint indices for the head region
NOSE_IDX = 0
LEFT_EAR_IDX = 3
RIGHT_EAR_IDX = 4

# Shoulder indices for dynamic pixel-per-meter calibration
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

# Average adult biacromial (shoulder) width in meters
SHOULDER_WIDTH_METERS = 0.45

# Minimum keypoint confidence to consider a detection valid
CONF_THRESHOLD = 0.3

# Gravity constant for g-force conversion (m/s²)
GRAVITY = 9.81

# ── Risk thresholds (linear acceleration in g) ──────────────────────────────
# Based on biomechanical concussion literature:
#   - Sub-concussive impacts typically < 40g
#   - Concussion risk increases significantly 40-75g
#   - Mean peak concussive acceleration ≈ 98g (Pellman et al., 2003)
LOW_G_THRESHOLD = 40.0
MODERATE_G_THRESHOLD = 75.0

# Colors for risk levels and collision indicators (BGR for OpenCV)
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_RED = (0, 0, 255)
COLOR_MAGENTA = (255, 0, 200)       # collision connecting line
COLOR_COLLISION_BG = (40, 0, 60)    # collision label background
COLOR_SKELETON = (255, 180, 0)
COLOR_TEXT_BG = (30, 30, 30)

# COCO skeleton connections for drawing (pairs of keypoint indices)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

# Minimum closing speed (m/s) to report a collision as biomechanically relevant
MIN_CLOSING_SPEED_MS = 1.0

# Cumulative head delta-v threshold (m/s) — triggers alert for repeated impacts
CUMULATIVE_DELTA_V_THRESHOLD_MS = 40.0

# Cooldown frames between collision reports for the same pair
COLLISION_COOLDOWN_FRAMES = 15

# ── Ground-impact detection constants ────────────────────────────────────────
# Minimum downward head velocity (m/s) to consider a ground-impact event
MIN_DESCENT_SPEED_MS = 1.5

# Head Y position must be >= this fraction of the lowest visible keypoint Y
# (i.e. the head is near ground level). In image coords, Y increases downward.
HEAD_AT_GROUND_Y_RATIO = 0.85

# Deceleration ratio: current velocity must drop below this fraction of peak
# to register the sudden-stop condition
DECEL_RATIO_THRESHOLD = 0.30

# Cooldown frames between ground-impact reports for the same person
GROUND_IMPACT_COOLDOWN_FRAMES = 15

# Impact duration for ground strikes — ground is harder than a body (6 ms)
IMPACT_DURATION_GROUND_S = 0.006

# Colors for ground-impact overlays (BGR)
COLOR_GROUND_IMPACT = (0, 140, 255)       # orange
COLOR_GROUND_IMPACT_BG = (30, 50, 80)     # dark overlay background

# COCO ankle keypoint indices used for ground-level estimation
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16


@dataclass
class PersonState:
    """Tracks per-person kinematics across frames."""
    track_id: int
    # Previous frame's head position in pixels (x, y)
    prev_head_px: np.ndarray | None = None
    # Previous frame's speed in m/s (for acceleration calc)
    prev_speed_ms: float = 0.0
    # Previous frame's timestamp in seconds
    prev_time_sec: float = 0.0
    # Running peak g-force observed for this person (individual motion)
    peak_g: float = 0.0
    # Current risk level string (individual motion)
    risk_level: str = "LOW"
    # Peak concussion probability from collision events
    peak_collision_prob: float = 0.0
    # Worst collision risk level
    collision_risk_level: str = "LOW"
    # Number of collision events involving this person
    collision_count: int = 0
    # All impact events for CSV logging
    events: list[dict[str, Any]] = field(default_factory=list)
    # ── Ground-impact tracking ──
    # Recent head Y positions (pixels, image coords — Y increases downward)
    head_y_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Recent head vertical velocities (m/s, positive = downward)
    head_vy_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Cooldown counter to avoid duplicate ground-impact reports
    ground_impact_cooldown: int = 0
    # Peak concussion probability from ground-impact events
    peak_ground_impact_prob: float = 0.0
    # Worst ground-impact risk level
    ground_impact_risk_level: str = "LOW"
    # Number of ground-impact events for this person
    ground_impact_count: int = 0
    # ── Cumulative delta-v tracking ──
    cumulative_delta_v_head_ms: float = 0.0
    cumulative_delta_v_flagged: bool = False


@dataclass
class GroundImpactEvent:
    """Record of a single head-to-ground impact."""
    frame: int
    track_id: int
    head_speed_ms: float
    peak_linear_g: float
    peak_rotational_rads2: float
    concussion_prob: float
    risk_level: str
    head_y_ratio: float


@dataclass
class CollisionEvent:
    """Record of a single collision between two persons."""
    frame: int
    timestamp_sec: float
    tid_a: int
    tid_b: int
    struck_tid: int
    closing_speed_ms: float
    contact_zone: str
    head_coupling_factor: float
    peak_linear_g: float
    peak_rotational_rads2: float
    concussion_prob: float
    risk_level: str
    # New additive fields
    hic: float = 0.0
    model_applicability: str = "VALIDATED"


def _classify_risk(accel_g: float) -> str:
    """Classify concussion risk from peak linear acceleration in g.

    Thresholds:
        LOW      : < 40g  — sub-concussive, routine play
        MODERATE : 40–75g — elevated risk, monitor athlete
        HIGH     : ≥ 75g  — significant concussion risk, evaluate immediately
    """
    if accel_g >= MODERATE_G_THRESHOLD:
        return "HIGH"
    elif accel_g >= LOW_G_THRESHOLD:
        return "MODERATE"
    return "LOW"


def _risk_color(risk: str) -> tuple[int, int, int]:
    """Return BGR color for a given risk level."""
    if risk in ("HIGH", "CRITICAL"):
        return COLOR_RED
    elif risk == "MODERATE":
        return COLOR_YELLOW
    return COLOR_GREEN


def _collision_risk_color(risk: str) -> tuple[int, int, int]:
    """Return BGR color for collision risk level."""
    if risk == "CRITICAL":
        return (0, 0, 255)       # bright red
    elif risk == "HIGH":
        return (0, 50, 255)      # orange-red
    elif risk == "MODERATE":
        return (0, 180, 255)     # orange
    return (0, 200, 0)           # green


def _compute_head_center_local(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
) -> np.ndarray | None:
    """Compute head center from nose and ear keypoints.

    Uses the average position of all visible head keypoints (nose=0,
    left_ear=3, right_ear=4) weighted equally. Falls back to whichever
    subset is visible above CONF_THRESHOLD.
    """
    head_indices = [NOSE_IDX, LEFT_EAR_IDX, RIGHT_EAR_IDX]
    visible = [i for i in head_indices if keypoints_conf[i] >= CONF_THRESHOLD]
    if not visible:
        return None
    return np.mean(keypoints_xy[visible], axis=0)


def _compute_pixel_per_meter(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    fallback: float = 200.0,
) -> float:
    """Derive pixels-per-meter from shoulder width."""
    l_conf = keypoints_conf[LEFT_SHOULDER_IDX]
    r_conf = keypoints_conf[RIGHT_SHOULDER_IDX]
    if l_conf >= CONF_THRESHOLD and r_conf >= CONF_THRESHOLD:
        shoulder_px = np.linalg.norm(
            keypoints_xy[LEFT_SHOULDER_IDX] - keypoints_xy[RIGHT_SHOULDER_IDX]
        )
        if shoulder_px > 5.0:
            return float(shoulder_px / SHOULDER_WIDTH_METERS)
    return fallback


def _draw_skeleton(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
) -> None:
    """Draw COCO skeleton on the frame."""
    for i, j in COCO_SKELETON:
        if keypoints_conf[i] >= CONF_THRESHOLD and keypoints_conf[j] >= CONF_THRESHOLD:
            pt1 = tuple(keypoints_xy[i].astype(int))
            pt2 = tuple(keypoints_xy[j].astype(int))
            cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2, cv2.LINE_AA)

    for idx in range(keypoints_xy.shape[0]):
        if keypoints_conf[idx] >= CONF_THRESHOLD:
            pt = tuple(keypoints_xy[idx].astype(int))
            cv2.circle(frame, pt, 3, COLOR_SKELETON, -1, cv2.LINE_AA)


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    bg_color: tuple[int, int, int] = COLOR_TEXT_BG,
) -> int:
    """Draw a text label with a dark background rectangle. Returns the y advance."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + baseline + 2), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + baseline + 6


def _draw_collision_overlay(
    frame: np.ndarray,
    head_a: np.ndarray,
    head_b: np.ndarray,
    event: CollisionEvent,
) -> None:
    """Draw collision indicators between two persons.

    Draws a connecting line between heads, a pulsing collision badge at the
    midpoint, and the concussion probability + risk level.
    """
    pt_a = (int(head_a[0]), int(head_a[1]))
    pt_b = (int(head_b[0]), int(head_b[1]))
    color = _collision_risk_color(event.risk_level)

    # ── Dashed connecting line between the two heads ──
    cv2.line(frame, pt_a, pt_b, COLOR_MAGENTA, 2, cv2.LINE_AA)

    # ── Collision badge at the midpoint ──
    mid_x = (pt_a[0] + pt_b[0]) // 2
    mid_y = (pt_a[1] + pt_b[1]) // 2

    # Pulsing circle radius based on probability
    base_radius = 12
    pulse_radius = base_radius + int(event.concussion_prob * 20)
    cv2.circle(frame, (mid_x, mid_y), pulse_radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (mid_x, mid_y), base_radius, color, -1, cv2.LINE_AA)

    # ── Labels at the midpoint ──
    lx = mid_x + pulse_radius + 5
    ly = mid_y - 20

    # Concussion probability as percentage
    prob_pct = event.concussion_prob * 100
    dy = _draw_label(
        frame, f"P(conc): {prob_pct:.1f}%",
        (lx, ly), color, font_scale=0.6, thickness=2, bg_color=COLOR_COLLISION_BG,
    )
    ly += dy

    # Closing speed
    dy = _draw_label(
        frame, f"Close: {event.closing_speed_ms:.1f} m/s",
        (lx, ly), (200, 200, 200), bg_color=COLOR_COLLISION_BG,
    )
    ly += dy

    # Peak g-force from collision model
    dy = _draw_label(
        frame, f"Est G: {event.peak_linear_g:.0f}g",
        (lx, ly), color, bg_color=COLOR_COLLISION_BG,
    )
    ly += dy

    # Contact zone
    dy = _draw_label(
        frame, f"Zone: {event.contact_zone}",
        (lx, ly), (180, 180, 180), font_scale=0.45, bg_color=COLOR_COLLISION_BG,
    )
    ly += dy

    # Risk level
    _draw_label(
        frame, f"COLLISION {event.risk_level}",
        (lx, ly), color, font_scale=0.65, thickness=2, bg_color=COLOR_COLLISION_BG,
    )

    # ── Highlight the struck person's head with a warning ring ──
    struck_pt = pt_a if event.struck_tid == event.tid_a else pt_b
    cv2.circle(frame, struck_pt, 18, color, 3, cv2.LINE_AA)
    cv2.circle(frame, struck_pt, 22, (255, 255, 255), 1, cv2.LINE_AA)


def _compute_lowest_keypoint_y(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
) -> float | None:
    """Return the Y position (pixels) of the lowest visible keypoint.

    Prefers ankles (COCO 15, 16) but falls back to any visible keypoint.
    In image coordinates, Y increases downward, so "lowest" means largest Y.
    """
    # Try ankles first
    ankle_indices = [LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX]
    visible_ankles = [i for i in ankle_indices if keypoints_conf[i] >= CONF_THRESHOLD]
    if visible_ankles:
        return float(max(keypoints_xy[i][1] for i in visible_ankles))

    # Fallback: any visible keypoint with the largest Y
    visible = [i for i in range(keypoints_xy.shape[0]) if keypoints_conf[i] >= CONF_THRESHOLD]
    if not visible:
        return None
    return float(max(keypoints_xy[i][1] for i in visible))


def _score_ground_impact(head_speed_ms: float) -> dict[str, float | str]:
    """Score a head-to-ground impact using biomechanical models.

    Ground has effectively infinite mass, so the head's full pre-impact
    speed is the delta-v. Uses a shorter impact duration (6 ms) because
    the ground is harder than a human body.

    Returns dict with peak_linear_g, peak_rotational_rads2, concussion_prob,
    risk_level.
    """
    peak_g = delta_v_to_peak_g(head_speed_ms, IMPACT_DURATION_GROUND_S)
    rot_accel = estimate_rotational_acceleration(peak_g)
    prob = concussion_probability_rowson_duma(peak_g, rot_accel)

    if prob < 0.05:
        risk = "LOW"
    elif prob < 0.15:
        risk = "MODERATE"
    elif prob < 0.50:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "peak_linear_g": peak_g,
        "peak_rotational_rads2": rot_accel,
        "concussion_prob": prob,
        "risk_level": risk,
    }


def _draw_ground_impact_overlay(
    frame: np.ndarray,
    head_px: np.ndarray,
    event: GroundImpactEvent,
) -> None:
    """Draw ground-impact indicators at the head position.

    Draws a downward triangle, warning ring, and labels showing
    concussion probability and risk level.
    """
    hx, hy = int(head_px[0]), int(head_px[1])
    color = _collision_risk_color(event.risk_level)

    # ── Downward triangle above head ──
    tri_size = 14
    tri_pts = np.array([
        [hx - tri_size, hy - tri_size - 10],
        [hx + tri_size, hy - tri_size - 10],
        [hx, hy - 2],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [tri_pts], COLOR_GROUND_IMPACT)
    cv2.polylines(frame, [tri_pts], True, color, 2, cv2.LINE_AA)

    # ── Warning ring around head ──
    cv2.circle(frame, (hx, hy), 20, COLOR_GROUND_IMPACT, 3, cv2.LINE_AA)
    cv2.circle(frame, (hx, hy), 24, color, 1, cv2.LINE_AA)

    # ── Labels to the right ──
    lx = hx + 30
    ly = hy - 30

    prob_pct = event.concussion_prob * 100
    dy = _draw_label(
        frame, f"GROUND IMPACT",
        (lx, ly), COLOR_GROUND_IMPACT, font_scale=0.6, thickness=2,
        bg_color=COLOR_GROUND_IMPACT_BG,
    )
    ly += dy

    dy = _draw_label(
        frame, f"P(conc): {prob_pct:.1f}%",
        (lx, ly), color, bg_color=COLOR_GROUND_IMPACT_BG,
    )
    ly += dy

    dy = _draw_label(
        frame, f"Head spd: {event.head_speed_ms:.1f} m/s",
        (lx, ly), (200, 200, 200), bg_color=COLOR_GROUND_IMPACT_BG,
    )
    ly += dy

    dy = _draw_label(
        frame, f"Est G: {event.peak_linear_g:.0f}g",
        (lx, ly), color, bg_color=COLOR_GROUND_IMPACT_BG,
    )
    ly += dy

    _draw_label(
        frame, f"Risk: {event.risk_level}",
        (lx, ly), color, font_scale=0.65, thickness=2,
        bg_color=COLOR_GROUND_IMPACT_BG,
    )


class ConcussionMonitor:
    """Multi-person concussion risk monitor with collision detection.

    Tracks head keypoints per person across frames, detects when two
    skeletons collide, computes instantaneous head speed, linear
    acceleration, and biomechanical concussion probability from the
    collision kinematics. Annotates the video frame with all results.

    Args:
        model_name: Ultralytics model path (e.g. "yolo11x-pose.pt").
        csv_path: Optional path for impact event CSV log.
        conf: YOLO detection confidence threshold.
        imgsz: YOLO input image size.
        fps: Video frame rate (used for closing velocity calculation).
    """

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        csv_path: str | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
        fps: float = 30.0,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._conf = conf
        self._imgsz = imgsz
        self._fps = fps

        # Per-person state keyed by track_id
        self._persons: dict[int, PersonState] = {}

        # Frame counter for CSV logging
        self._frame_idx: int = 0

        # ── Collision detection subsystem ──
        self._collision_detector = CollisionDetector(
            iou_threshold=0.05,      # coarse bbox overlap filter
            proximity_ratio=0.5,     # head-to-body distance / bbox diagonal
            history_frames=5,        # frames of head history for velocity
        )

        # Collision cooldown tracker: (min_tid, max_tid) -> last_collision_frame
        self._collision_cooldowns: dict[tuple[int, int], int] = {}

        # All collision events for the report
        self._collision_events: list[CollisionEvent] = []

        # All ground-impact events for the report
        self._ground_impact_events: list[GroundImpactEvent] = []

        # ── CSV writer setup ──
        self._csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None
        if csv_path:
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "event_type", "track_id", "frame", "timestamp_sec",
                "speed_ms", "accel_g", "risk_level",
                # Collision-specific columns (empty for individual events)
                "other_track_id", "closing_speed_ms", "contact_zone",
                "peak_linear_g", "concussion_prob", "collision_risk",
            ])

        # Exponential moving average of pixel_per_meter for stability
        self._ppm_ema: float = 200.0
        self._ppm_alpha: float = 0.1

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one BGR frame: detect, track, compute kinematics, detect collisions, annotate.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Annotated BGR frame with skeleton, track IDs, speed, g-force,
            collision probability, and color-coded risk labels.
        """
        self._frame_idx += 1
        timestamp_sec = time.monotonic()

        # ── 1. Run YOLOv11-Pose with built-in tracker ───────────────────────
        results = self._model.track(
            frame,
            conf=self._conf,
            imgsz=self._imgsz,
            persist=True,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return frame

        result = results[0]
        annotated = frame.copy()

        boxes = result.boxes
        if boxes.id is None:
            return annotated

        track_ids = boxes.id.cpu().numpy().astype(int)
        all_xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) bounding boxes

        if result.keypoints is None or result.keypoints.data is None:
            return annotated
        all_kp = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

        # ── 2. Process each tracked person (individual kinematics) ───────────
        # Store per-frame data for collision detection pass
        frame_heads: dict[int, np.ndarray] = {}   # tid -> head_px
        frame_kp_xy: dict[int, np.ndarray] = {}   # tid -> (17, 2)
        frame_kp_conf: dict[int, np.ndarray] = {} # tid -> (17,)
        frame_bboxes: dict[int, np.ndarray] = {}  # tid -> (4,)

        for i, tid in enumerate(track_ids):
            tid = int(tid)
            kp = all_kp[i]
            kp_xy = kp[:, :2]
            kp_conf = kp[:, 2]
            bbox = all_xyxy[i]

            # ── 2a. Draw skeleton ──
            _draw_skeleton(annotated, kp_xy, kp_conf)

            # ── 2b. Head center ──
            head_px = _compute_head_center_local(kp_xy, kp_conf)
            if head_px is None:
                continue

            # ── 2c. Dynamic pixel-per-meter calibration ──
            ppm = _compute_pixel_per_meter(kp_xy, kp_conf, fallback=self._ppm_ema)
            self._ppm_ema += self._ppm_alpha * (ppm - self._ppm_ema)

            # ── 2d. Update collision detector state ──
            self._collision_detector.update(tid, head_px, bbox)

            # Store for collision pass
            frame_heads[tid] = head_px
            frame_kp_xy[tid] = kp_xy
            frame_kp_conf[tid] = kp_conf
            frame_bboxes[tid] = bbox

            # ── 2e. Individual kinematics ──
            if tid not in self._persons:
                self._persons[tid] = PersonState(
                    track_id=tid,
                    prev_head_px=head_px,
                    prev_time_sec=timestamp_sec,
                )
                speed_ms = 0.0
                accel_g = 0.0
            else:
                state = self._persons[tid]
                dt = timestamp_sec - state.prev_time_sec
                if dt < 1e-6:
                    dt = 1e-6

                # Speed from pixel displacement
                displacement_px = np.linalg.norm(head_px - state.prev_head_px)
                displacement_m = displacement_px / self._ppm_ema
                speed_ms = displacement_m / dt

                # Linear acceleration → g-force
                delta_v = speed_ms - state.prev_speed_ms
                accel_ms2 = abs(delta_v) / dt
                accel_g = accel_ms2 / GRAVITY

                # Update individual peak and risk
                state.peak_g = max(state.peak_g, accel_g)
                state.risk_level = _classify_risk(state.peak_g)

                # Update state
                state.prev_head_px = head_px
                state.prev_speed_ms = speed_ms
                state.prev_time_sec = timestamp_sec

                # Log individual events to CSV
                if self._csv_writer and accel_g > 1.0:
                    self._csv_writer.writerow([
                        "individual", tid, self._frame_idx,
                        f"{timestamp_sec:.4f}",
                        f"{speed_ms:.3f}", f"{accel_g:.2f}",
                        state.risk_level,
                        "", "", "", "", "", "",
                    ])

            # ── 2f. Ground-impact detection ──
            person = self._persons[tid]

            # Decrement cooldown
            if person.ground_impact_cooldown > 0:
                person.ground_impact_cooldown -= 1

            # Update head Y history and compute vertical velocity
            head_y = float(head_px[1])
            person.head_y_history.append(head_y)

            if len(person.head_y_history) >= 2:
                # Vertical velocity in pixels/frame (positive = downward in image coords)
                dy_px = person.head_y_history[-1] - person.head_y_history[-2]
                # Convert to m/s (positive = downward)
                vy_ms = (dy_px / self._ppm_ema) * self._fps
                person.head_vy_history.append(vy_ms)

                # Check 3 conditions for ground impact
                if (
                    person.ground_impact_cooldown == 0
                    and len(person.head_vy_history) >= 3
                ):
                    # Condition 1: Peak downward velocity in recent history > threshold
                    peak_descent = max(person.head_vy_history)
                    if peak_descent > MIN_DESCENT_SPEED_MS:
                        # Condition 2: Current velocity dropped to < 30% of peak (sudden stop)
                        current_vy = person.head_vy_history[-1]
                        if current_vy < peak_descent * DECEL_RATIO_THRESHOLD:
                            # Condition 3: Head at ground level
                            lowest_y = _compute_lowest_keypoint_y(kp_xy, kp_conf)
                            if lowest_y is not None and lowest_y > 0:
                                head_y_ratio = head_y / lowest_y
                                if head_y_ratio >= HEAD_AT_GROUND_Y_RATIO:
                                    # All 3 conditions met — score the impact
                                    score = _score_ground_impact(peak_descent)

                                    event = GroundImpactEvent(
                                        frame=self._frame_idx,
                                        track_id=tid,
                                        head_speed_ms=peak_descent,
                                        peak_linear_g=score["peak_linear_g"],
                                        peak_rotational_rads2=score["peak_rotational_rads2"],
                                        concussion_prob=score["concussion_prob"],
                                        risk_level=score["risk_level"],
                                        head_y_ratio=head_y_ratio,
                                    )

                                    self._ground_impact_events.append(event)
                                    person.ground_impact_cooldown = GROUND_IMPACT_COOLDOWN_FRAMES
                                    person.ground_impact_count += 1
                                    if score["concussion_prob"] > person.peak_ground_impact_prob:
                                        person.peak_ground_impact_prob = score["concussion_prob"]
                                        person.ground_impact_risk_level = score["risk_level"]

                                    # Draw ground-impact overlay
                                    _draw_ground_impact_overlay(annotated, head_px, event)

                                    # Log to CSV
                                    if self._csv_writer:
                                        self._csv_writer.writerow([
                                            "ground_impact", tid, self._frame_idx,
                                            f"{timestamp_sec:.4f}",
                                            f"{peak_descent:.3f}", f"{score['peak_linear_g']:.2f}",
                                            score["risk_level"],
                                            "", "", "",
                                            f"{score['peak_linear_g']:.1f}",
                                            f"{score['concussion_prob']:.6f}",
                                            score["risk_level"],
                                        ])

                                    # Console alert
                                    prob_pct = score["concussion_prob"] * 100
                                    print(
                                        f"  [GROUND IMPACT] Frame {self._frame_idx}: "
                                        f"ID:{tid} | "
                                        f"head_speed={peak_descent:.1f} m/s | "
                                        f"est_g={score['peak_linear_g']:.0f}g | "
                                        f"P(conc)={prob_pct:.1f}% | "
                                        f"risk={score['risk_level']} | "
                                        f"head_y_ratio={head_y_ratio:.2f}"
                                    )

            # ── 2g. Per-person annotation ──
            # Use the worst of individual, collision, and ground-impact risk for display
            risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
            display_risk = person.risk_level
            if risk_order.get(person.collision_risk_level, 0) > risk_order.get(display_risk, 0):
                display_risk = person.collision_risk_level
            if risk_order.get(person.ground_impact_risk_level, 0) > risk_order.get(display_risk, 0):
                display_risk = person.ground_impact_risk_level
            color = _risk_color(display_risk)

            lx = int(head_px[0]) + 10
            ly = int(head_px[1]) - 30

            dy = _draw_label(annotated, f"ID:{tid}", (lx, ly), (255, 255, 255))
            ly += dy
            dy = _draw_label(annotated, f"Speed: {speed_ms:.1f} m/s", (lx, ly), (200, 200, 200))
            ly += dy
            dy = _draw_label(annotated, f"Peak G: {person.peak_g:.1f}g", (lx, ly), color)
            ly += dy

            # Show collision probability if this person has been in a collision
            if person.peak_collision_prob > 0.001:
                coll_color = _collision_risk_color(person.collision_risk_level)
                dy = _draw_label(
                    annotated,
                    f"Coll P: {person.peak_collision_prob*100:.1f}%",
                    (lx, ly), coll_color,
                )
                ly += dy

            # Show ground-impact probability if this person has had a ground impact
            if person.peak_ground_impact_prob > 0.001:
                gi_color = _collision_risk_color(person.ground_impact_risk_level)
                dy = _draw_label(
                    annotated,
                    f"Ground P: {person.peak_ground_impact_prob*100:.1f}%",
                    (lx, ly), gi_color,
                )
                ly += dy

            _draw_label(annotated, f"Risk: {display_risk}", (lx, ly), color, font_scale=0.65, thickness=2)

            # Head indicator circle
            cv2.circle(annotated, (int(head_px[0]), int(head_px[1])), 8, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, (int(head_px[0]), int(head_px[1])), 8, (0, 0, 0), 1, cv2.LINE_AA)

        # ── 3. Collision detection between all pairs ─────────────────────────
        # meters_per_pixel for the collision detector (inverse of ppm_ema)
        meters_per_pixel = 1.0 / max(self._ppm_ema, 1.0)

        active_tids = list(frame_heads.keys())
        for tid_a, tid_b in combinations(active_tids, 2):
            # Check cooldown — don't re-report the same pair too quickly
            pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
            last_frame = self._collision_cooldowns.get(pair_key, -999)
            if self._frame_idx - last_frame < COLLISION_COOLDOWN_FRAMES:
                continue

            # ── 3a. Run collision detection pipeline ──
            collision = self._collision_detector.check_collision(
                tid_a, tid_b,
                frame_kp_xy[tid_a], frame_kp_conf[tid_a],
                frame_kp_xy[tid_b], frame_kp_conf[tid_b],
                fps=self._fps,
                meters_per_pixel=meters_per_pixel,
            )

            if collision is None:
                continue

            # ── 3b. Filter: require minimum closing speed ──
            closing_speed = collision["closing_speed_ms"]
            if closing_speed < MIN_CLOSING_SPEED_MS:
                continue

            # ── 3c. Estimate body masses from bounding boxes ──
            bbox_a = frame_bboxes[tid_a]
            bbox_b = frame_bboxes[tid_b]
            height_a_px = float(bbox_a[3] - bbox_a[1])
            height_b_px = float(bbox_b[3] - bbox_b[1])
            mass_a = estimate_body_mass(height_a_px, meters_per_pixel)
            mass_b = estimate_body_mass(height_b_px, meters_per_pixel)

            # ── 3d. Score collision via biomechanical model ──
            # Use unhelmeted impact duration for general sports
            score = score_collision(
                closing_speed_ms=closing_speed,
                mass_a_kg=mass_a,
                mass_b_kg=mass_b,
                head_coupling_factor=collision["head_coupling_factor"],
                impact_duration_s=IMPACT_DURATION_UNHELMETED_S,
                helmeted=False,
                approach_angle_rad=collision.get("approach_angle_rad", 0.0),
                min_pose_confidence=collision.get("min_head_confidence", 1.0),
            )

            # ── 3e. Create collision event ──
            event = CollisionEvent(
                frame=self._frame_idx,
                timestamp_sec=timestamp_sec,
                tid_a=tid_a,
                tid_b=tid_b,
                struck_tid=collision["struck_tid"],
                closing_speed_ms=closing_speed,
                contact_zone=collision["contact_zone"],
                head_coupling_factor=collision["head_coupling_factor"],
                peak_linear_g=score["peak_linear_g"],
                peak_rotational_rads2=score["peak_rotational_rads2"],
                concussion_prob=score["concussion_prob"],
                risk_level=score["risk_level"],
                hic=score.get("hic", 0.0),
                model_applicability=score.get("model_applicability", "VALIDATED"),
            )

            self._collision_events.append(event)
            self._collision_cooldowns[pair_key] = self._frame_idx

            # ── 3f. Update struck person's collision state ──
            struck_tid = collision["struck_tid"]
            if struck_tid in self._persons:
                p = self._persons[struck_tid]
                p.collision_count += 1
                if score["concussion_prob"] > p.peak_collision_prob:
                    p.peak_collision_prob = score["concussion_prob"]
                    p.collision_risk_level = score["risk_level"]
                # Accumulate cumulative delta-v on struck person
                p.cumulative_delta_v_head_ms += score["delta_v_head_ms"]
                if (
                    not p.cumulative_delta_v_flagged
                    and p.cumulative_delta_v_head_ms >= CUMULATIVE_DELTA_V_THRESHOLD_MS
                ):
                    p.cumulative_delta_v_flagged = True
                    print(
                        f"  [CUMULATIVE ALERT] ID:{struck_tid} "
                        f"cumulative_delta_v={p.cumulative_delta_v_head_ms:.1f} m/s "
                        f">= threshold {CUMULATIVE_DELTA_V_THRESHOLD_MS:.1f} m/s"
                    )

            # Also update the other person (they receive some impact too)
            other_tid = tid_b if struck_tid == tid_a else tid_a
            if other_tid in self._persons:
                p = self._persons[other_tid]
                p.collision_count += 1
                # The other person gets a lower probability (they're the striker)
                striker_prob = score["concussion_prob"] * 0.5
                if striker_prob > p.peak_collision_prob:
                    p.peak_collision_prob = striker_prob
                    # Re-classify at the lower probability
                    if striker_prob >= 0.50:
                        p.collision_risk_level = "CRITICAL"
                    elif striker_prob >= 0.15:
                        p.collision_risk_level = "HIGH"
                    elif striker_prob >= 0.05:
                        p.collision_risk_level = "MODERATE"
                    else:
                        p.collision_risk_level = "LOW"

            # ── 3g. Draw collision overlay ──
            _draw_collision_overlay(
                annotated,
                frame_heads[tid_a],
                frame_heads[tid_b],
                event,
            )

            # ── 3h. Log collision to CSV ──
            if self._csv_writer:
                self._csv_writer.writerow([
                    "collision", struck_tid, self._frame_idx,
                    f"{timestamp_sec:.4f}",
                    "", "",  # speed_ms and accel_g are individual metrics
                    score["risk_level"],
                    other_tid,
                    f"{closing_speed:.2f}",
                    collision["contact_zone"],
                    f"{score['peak_linear_g']:.1f}",
                    f"{score['concussion_prob']:.6f}",
                    score["risk_level"],
                ])

            # Print collision alert to console
            prob_pct = score["concussion_prob"] * 100
            print(
                f"  [COLLISION] Frame {self._frame_idx}: "
                f"ID:{tid_a} <-> ID:{tid_b} | "
                f"struck=ID:{struck_tid} | "
                f"close={closing_speed:.1f} m/s | "
                f"zone={collision['contact_zone']} | "
                f"est_g={score['peak_linear_g']:.0f}g | "
                f"P(conc)={prob_pct:.1f}% | "
                f"risk={score['risk_level']}"
            )

        return annotated

    def get_report(self) -> dict[str, Any]:
        """Return a summary dict of all tracked persons and collision events.

        Returns:
            {
                "total_persons": int,
                "persons": { <track_id>: { ... }, ... },
                "highest_risk": str,
                "max_peak_g": float,
                "total_collisions": int,
                "collision_events": [ { ... }, ... ],
                "max_concussion_prob": float,
                "worst_collision_risk": str,
                "total_ground_impacts": int,
                "ground_impact_events": [ { ... }, ... ],
                "max_ground_impact_prob": float,
                "worst_ground_impact_risk": str,
            }
        """
        persons_summary = {}
        max_g = 0.0
        worst_risk = "LOW"
        max_coll_prob = 0.0
        worst_coll_risk = "LOW"
        risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

        max_ground_prob = 0.0
        worst_ground_risk = "LOW"

        for tid, state in self._persons.items():
            persons_summary[tid] = {
                "peak_g": round(state.peak_g, 2),
                "risk_level": state.risk_level,
                "collision_count": state.collision_count,
                "peak_collision_prob": round(state.peak_collision_prob, 6),
                "collision_risk_level": state.collision_risk_level,
                "ground_impact_count": state.ground_impact_count,
                "peak_ground_impact_prob": round(state.peak_ground_impact_prob, 6),
                "ground_impact_risk_level": state.ground_impact_risk_level,
                "num_events": len(state.events),
                "cumulative_delta_v_head_ms": round(state.cumulative_delta_v_head_ms, 2),
                "cumulative_delta_v_flagged": state.cumulative_delta_v_flagged,
            }
            if state.peak_g > max_g:
                max_g = state.peak_g
                worst_risk = state.risk_level
            if state.peak_collision_prob > max_coll_prob:
                max_coll_prob = state.peak_collision_prob
            if risk_order.get(state.collision_risk_level, 0) > risk_order.get(worst_coll_risk, 0):
                worst_coll_risk = state.collision_risk_level
            if state.peak_ground_impact_prob > max_ground_prob:
                max_ground_prob = state.peak_ground_impact_prob
            if risk_order.get(state.ground_impact_risk_level, 0) > risk_order.get(worst_ground_risk, 0):
                worst_ground_risk = state.ground_impact_risk_level

        collision_records = [
            {
                "frame": e.frame,
                "tid_a": e.tid_a,
                "tid_b": e.tid_b,
                "struck_tid": e.struck_tid,
                "closing_speed_ms": round(e.closing_speed_ms, 2),
                "contact_zone": e.contact_zone,
                "peak_linear_g": round(e.peak_linear_g, 1),
                "concussion_prob": round(e.concussion_prob, 6),
                "risk_level": e.risk_level,
                "hic": round(e.hic, 1),
                "model_applicability": e.model_applicability,
            }
            for e in self._collision_events
        ]

        ground_impact_records = [
            {
                "frame": e.frame,
                "track_id": e.track_id,
                "head_speed_ms": round(e.head_speed_ms, 2),
                "peak_linear_g": round(e.peak_linear_g, 1),
                "peak_rotational_rads2": round(e.peak_rotational_rads2, 1),
                "concussion_prob": round(e.concussion_prob, 6),
                "risk_level": e.risk_level,
                "head_y_ratio": round(e.head_y_ratio, 3),
            }
            for e in self._ground_impact_events
        ]

        return {
            "total_persons": len(self._persons),
            "persons": persons_summary,
            "highest_risk": worst_risk,
            "max_peak_g": round(max_g, 2),
            "total_collisions": len(self._collision_events),
            "collision_events": collision_records,
            "max_concussion_prob": round(max_coll_prob, 6),
            "worst_collision_risk": worst_coll_risk,
            "total_ground_impacts": len(self._ground_impact_events),
            "ground_impact_events": ground_impact_records,
            "max_ground_impact_prob": round(max_ground_prob, 6),
            "worst_ground_impact_risk": worst_ground_risk,
        }

    def close(self) -> None:
        """Flush and close the CSV file if open."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


def run_video(
    video_path: str,
    output_path: str | None = None,
    csv_path: str | None = None,
    model_name: str = "yolo11x-pose.pt",
    show: bool = True,
) -> dict[str, Any]:
    """Process a video file end-to-end with concussion monitoring.

    Args:
        video_path: Path to input video.
        output_path: Optional path to save annotated output video.
        csv_path: Optional path for impact event CSV.
        model_name: YOLO model to use.
        show: Whether to display frames in an OpenCV window.

    Returns:
        Summary report dict from ConcussionMonitor.get_report().
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    monitor = ConcussionMonitor(
        model_name=model_name,
        csv_path=csv_path,
        fps=fps,
    )

    print(f"Processing: {video_path}")
    print(f"  Resolution: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    if output_path:
        print(f"  Output: {output_path}")
    if csv_path:
        print(f"  CSV log: {csv_path}")

    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated = monitor.process_frame(frame)

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Concussion Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_num % 100 == 0:
                pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_num}/{total_frames} ({pct:.0f}%)")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    report = monitor.get_report()
    monitor.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONCUSSION MONITORING REPORT")
    print(f"{'='*60}")
    print(f"Total persons tracked:   {report['total_persons']}")
    print(f"Highest individual risk: {report['highest_risk']}")
    print(f"Max individual peak g:   {report['max_peak_g']:.1f}g")
    print(f"{'─'*60}")
    print(f"Total collisions:        {report['total_collisions']}")
    print(f"Max concussion prob:     {report['max_concussion_prob']*100:.2f}%")
    print(f"Worst collision risk:    {report['worst_collision_risk']}")
    print(f"{'─'*60}")
    print(f"Total ground impacts:    {report['total_ground_impacts']}")
    print(f"Max ground impact prob:  {report['max_ground_impact_prob']*100:.2f}%")
    print(f"Worst ground risk:       {report['worst_ground_impact_risk']}")
    print(f"{'─'*60}")

    for tid, info in report["persons"].items():
        # Determine worst risk marker across collision and ground impact
        risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
        worst_event_risk = info["collision_risk_level"]
        if risk_order.get(info["ground_impact_risk_level"], 0) > risk_order.get(worst_event_risk, 0):
            worst_event_risk = info["ground_impact_risk_level"]
        risk_marker = {"LOW": "[OK]", "MODERATE": "[!!]", "HIGH": "[XX]", "CRITICAL": "[!!!]"}.get(
            worst_event_risk, ""
        )
        coll_str = ""
        if info["collision_count"] > 0:
            coll_str = (
                f" | collisions={info['collision_count']}"
                f" P(conc)={info['peak_collision_prob']*100:.1f}%"
                f" [{info['collision_risk_level']}]"
            )
        ground_str = ""
        if info["ground_impact_count"] > 0:
            ground_str = (
                f" | ground_impacts={info['ground_impact_count']}"
                f" P(conc)={info['peak_ground_impact_prob']*100:.1f}%"
                f" [{info['ground_impact_risk_level']}]"
            )
        print(f"  Person {tid}: peak={info['peak_g']:.1f}g  risk={info['risk_level']}{coll_str}{ground_str} {risk_marker}")

    if report["collision_events"]:
        print(f"{'─'*60}")
        print("  Collision Events:")
        for ev in report["collision_events"]:
            print(
                f"    Frame {ev['frame']}: ID:{ev['tid_a']} <-> ID:{ev['tid_b']}"
                f" | struck=ID:{ev['struck_tid']}"
                f" | {ev['closing_speed_ms']:.1f} m/s"
                f" | {ev['contact_zone']}"
                f" | {ev['peak_linear_g']:.0f}g"
                f" | P={ev['concussion_prob']*100:.1f}%"
                f" | {ev['risk_level']}"
            )

    if report["ground_impact_events"]:
        print(f"{'─'*60}")
        print("  Ground Impact Events:")
        for ev in report["ground_impact_events"]:
            print(
                f"    Frame {ev['frame']}: ID:{ev['track_id']}"
                f" | head_speed={ev['head_speed_ms']:.1f} m/s"
                f" | {ev['peak_linear_g']:.0f}g"
                f" | P={ev['concussion_prob']*100:.1f}%"
                f" | {ev['risk_level']}"
                f" | head_y_ratio={ev['head_y_ratio']:.2f}"
            )

    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concussion risk monitor with collision detection using YOLOv11-Pose tracking",
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Path to save annotated output video")
    parser.add_argument("--csv", "-c", default="impacts.csv", help="Path for impact event CSV (default: impacts.csv)")
    parser.add_argument("--model", "-m", default="yolo11x-pose.pt", help="YOLO model name (default: yolo11x-pose.pt)")
    parser.add_argument("--no-show", action="store_true", help="Disable OpenCV display window")

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_video(
        video_path=args.video,
        output_path=args.output,
        csv_path=args.csv,
        model_name=args.model,
        show=not args.no_show,
    )
