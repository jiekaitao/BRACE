"""Offline crash video analysis pipeline.

Processes uploaded video for collision detection and concussion risk scoring.
Uses the full physics pipeline:
  - CollisionDetector (coarse-to-fine keypoint collision detection)
  - score_collision (Rowson & Duma biomechanics model)
  - HeadImpactAnalyzer (head kinematics tracking)
  - ClosingSpeedTracker (pairwise approach speed)

Pattern mirrors basketball_processor.py.
"""

from __future__ import annotations

import uuid
from itertools import combinations
from typing import Any, Callable

import numpy as np

try:
    from collision_detection import (
        CollisionDetector,
        compute_head_center,
        estimate_meters_per_pixel,
    )
    from biomechanics_model import score_collision, DEFAULT_BODY_MASS_KG
except ImportError:
    from backend.collision_detection import (
        CollisionDetector,
        compute_head_center,
        estimate_meters_per_pixel,
    )
    from backend.biomechanics_model import score_collision, DEFAULT_BODY_MASS_KG

# MediaPipe-33 index for each COCO-17 keypoint (ordered by COCO index 0..16)
_MP_IDX_FOR_COCO = [
    0,   # COCO 0  nose        ← MP 0
    2,   # COCO 1  left_eye    ← MP 2
    5,   # COCO 2  right_eye   ← MP 5
    7,   # COCO 3  left_ear    ← MP 7
    8,   # COCO 4  right_ear   ← MP 8
    11,  # COCO 5  left_shoulder  ← MP 11
    12,  # COCO 6  right_shoulder ← MP 12
    13,  # COCO 7  left_elbow  ← MP 13
    14,  # COCO 8  right_elbow ← MP 14
    15,  # COCO 9  left_wrist  ← MP 15
    16,  # COCO 10 right_wrist ← MP 16
    23,  # COCO 11 left_hip    ← MP 23
    24,  # COCO 12 right_hip   ← MP 24
    25,  # COCO 13 left_knee   ← MP 25
    26,  # COCO 14 right_knee  ← MP 26
    27,  # COCO 15 left_ankle  ← MP 27
    28,  # COCO 16 right_ankle ← MP 28
]


def _landmarks_mp_to_coco17(
    landmarks_mp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract COCO-17 keypoints and confidences from (33, 4) MediaPipe landmarks.

    Returns
    -------
    kpts : ndarray, shape (17, 2)
        (x, y) pixel coordinates per COCO keypoint.
    conf : ndarray, shape (17,)
        Per-keypoint confidence / visibility score.
    """
    kpts = np.zeros((17, 2), dtype=np.float64)
    conf = np.zeros(17, dtype=np.float64)
    for coco_idx, mp_idx in enumerate(_MP_IDX_FOR_COCO):
        kpts[coco_idx, 0] = landmarks_mp[mp_idx, 0]  # x
        kpts[coco_idx, 1] = landmarks_mp[mp_idx, 1]  # y
        conf[coco_idx] = landmarks_mp[mp_idx, 3]      # visibility
    return kpts, conf


# Risk level ordering for worst-of comparisons
_RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

# Minimum frames between duplicate collision events for the same pair
_COLLISION_COOLDOWN_FRAMES = 15


def process_crash_video(
    pipeline,
    n_frames: int,
    fps: float,
    img_wh: tuple[int, int],
    progress_callback: Callable[[float, dict], None] | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    """Process pre-extracted pipeline results for crash/collision analysis.

    This is the synchronous core that works with a mock pipeline in tests
    or a real PoseBackend in production.

    Args:
        pipeline: Object with process_frame(rgb) -> list[PipelineResult].
        n_frames: Total number of frames to process.
        fps: Video frame rate.
        img_wh: (width, height) of the video frames.
        progress_callback: Optional callable(pct, data) for progress updates.
        analysis_id: Optional analysis ID (generated if not provided).

    Returns:
        CrashAnalysisResult dict.
    """
    if analysis_id is None:
        analysis_id = str(uuid.uuid4())

    w, h = img_wh
    safe_fps = max(fps, 1.0)  # avoid division by zero
    duration_sec = n_frames / safe_fps

    # Core detectors
    collision_detector = CollisionDetector(
        iou_threshold=0.05,
        proximity_ratio=0.5,
        history_frames=5,
    )

    # Track per-subject data
    subjects_seen: set[int] = set()
    collision_events: list[dict] = []

    # Cooldown tracking: (tid_a, tid_b) -> last collision frame
    pair_cooldown: dict[tuple[int, int], int] = {}

    # Synthetic frame for pipeline calls
    dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for frame_idx in range(n_frames):
        video_time = frame_idx / safe_fps

        # Run pipeline
        try:
            results = pipeline.process_frame(dummy_rgb)
        except Exception:
            results = []

        if not results:
            # Progress callback even on empty frames
            if progress_callback is not None and frame_idx % 30 == 0:
                pct = (frame_idx / n_frames * 100) if n_frames > 0 else 0
                progress_callback(pct, {
                    "frame_index": frame_idx,
                    "total_frames": n_frames,
                })
            continue

        # Extract per-subject data
        frame_subjects: dict[int, dict] = {}
        for pr in results:
            tid = pr.track_id
            subjects_seen.add(tid)

            kpts, conf = _landmarks_mp_to_coco17(pr.landmarks_mp)
            bbox = pr.bbox_pixel

            # Update collision detector state
            head_center = compute_head_center(kpts, conf)
            collision_detector.update(tid, head_center, bbox)

            # Estimate meters_per_pixel from this subject's skeleton
            m_per_px, _ = estimate_meters_per_pixel(kpts, conf)

            frame_subjects[tid] = {
                "kpts": kpts,
                "conf": conf,
                "bbox": bbox,
                "m_per_px": m_per_px,
            }

        # Check all pairs for collisions
        track_ids = list(frame_subjects.keys())
        for tid_a, tid_b in combinations(track_ids, 2):
            # Normalize pair key for cooldown
            pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))

            # Skip if in cooldown
            last_collision = pair_cooldown.get(pair_key, -_COLLISION_COOLDOWN_FRAMES - 1)
            if frame_idx - last_collision < _COLLISION_COOLDOWN_FRAMES:
                continue

            sa = frame_subjects[tid_a]
            sb = frame_subjects[tid_b]

            # Average meters_per_pixel from both subjects
            avg_mpp = (sa["m_per_px"] + sb["m_per_px"]) / 2.0

            collision = collision_detector.check_collision(
                tid_a, tid_b,
                sa["kpts"], sa["conf"],
                sb["kpts"], sb["conf"],
                fps=safe_fps,
                meters_per_pixel=avg_mpp,
            )

            if collision is not None:
                # Score the collision using the biomechanics model
                closing_speed = collision.get("closing_speed_ms", 0.0)
                head_coupling = collision.get("head_coupling_factor", 0.5)
                contact_zone = collision.get("contact_zone", "unknown")

                bio_result = score_collision(
                    closing_speed_ms=closing_speed,
                    mass_a_kg=DEFAULT_BODY_MASS_KG,
                    mass_b_kg=DEFAULT_BODY_MASS_KG,
                    head_coupling_factor=head_coupling,
                )

                event = {
                    "event_id": str(uuid.uuid4()),
                    "frame_index": frame_idx,
                    "video_time": round(video_time, 3),
                    "subject_a": tid_a,
                    "subject_b": tid_b,
                    "closing_speed_ms": round(closing_speed, 4),
                    "peak_linear_g": round(bio_result["peak_linear_g"], 2),
                    "peak_rotational_rads2": round(bio_result["peak_rotational_rads2"], 2),
                    "concussion_probability": round(bio_result["concussion_prob"], 4),
                    "risk_level": bio_result["risk_level"],
                    "recommendation": _risk_to_recommendation(bio_result["risk_level"]),
                    "contact_zone": contact_zone,
                    "head_coupling_factor": round(head_coupling, 3),
                    "hic": round(bio_result["hic"], 2),
                }

                collision_events.append(event)
                pair_cooldown[pair_key] = frame_idx

        # Progress callback
        if progress_callback is not None and frame_idx % 30 == 0:
            pct = (frame_idx / n_frames * 100) if n_frames > 0 else 0
            progress_callback(pct, {
                "frame_index": frame_idx,
                "total_frames": n_frames,
                "subjects_tracked": len(subjects_seen),
                "collision_count": len(collision_events),
            })

    # Final progress callback
    if progress_callback is not None:
        progress_callback(100.0, {
            "frame_index": n_frames,
            "total_frames": n_frames,
            "subjects_tracked": len(subjects_seen),
            "collision_count": len(collision_events),
        })

    # Sort events by frame
    collision_events.sort(key=lambda e: e["frame_index"])

    # Build per-subject summaries
    subject_summaries = _build_subject_summaries(subjects_seen, collision_events)

    # Overall risk = worst across all events
    if collision_events:
        overall_risk = max(
            collision_events,
            key=lambda e: _RISK_ORDER.get(e["risk_level"], 0),
        )["risk_level"]
    else:
        overall_risk = "LOW"

    return {
        "analysis_id": analysis_id,
        "status": "complete",
        "total_frames": n_frames,
        "duration_sec": round(duration_sec, 1),
        "fps": fps,
        "subjects_tracked": len(subjects_seen),
        "collision_events": collision_events,
        "subject_summaries": subject_summaries,
        "overall_risk": overall_risk,
        "overall_recommendation": _risk_to_recommendation(overall_risk),
    }


def _build_subject_summaries(
    subjects: set[int],
    collision_events: list[dict],
) -> dict[str, dict]:
    """Build per-subject crash summaries from collision events."""
    summaries: dict[str, dict] = {}

    for sid in subjects:
        # Find all events involving this subject
        events = [
            e for e in collision_events
            if e["subject_a"] == sid or e["subject_b"] == sid
        ]

        if events:
            max_prob = max(e["concussion_probability"] for e in events)
            worst_risk = max(
                events,
                key=lambda e: _RISK_ORDER.get(e["risk_level"], 0),
            )["risk_level"]
        else:
            max_prob = 0.0
            worst_risk = "NONE"

        summaries[str(sid)] = {
            "subject_id": sid,
            "collision_count": len(events),
            "max_concussion_probability": round(max_prob, 4),
            "worst_risk_level": worst_risk,
            "recommendation": _risk_to_recommendation(worst_risk),
        }

    return summaries


def _risk_to_recommendation(risk_level: str) -> str:
    """Convert risk level to a human-readable recommendation."""
    recommendations = {
        "CRITICAL": "EVALUATE NOW — remove from play immediately for concussion assessment",
        "HIGH": "EVALUATE — significant risk, consider immediate assessment",
        "MODERATE": "MONITOR — watch for concussion symptoms over next 24 hours",
        "LOW": "CLEARED — routine contact, no action needed",
        "NONE": "No collisions detected",
    }
    return recommendations.get(risk_level, "No assessment available")
