#!/usr/bin/env python3
"""Generate demo video showing the full BRACE pipeline:
   1. Calibration phase (blue bounding box) — building baseline from normal gait
   2. Normal phase (green bounding box) — normal gait passes
   3. Injury detection (red bounding box) — pathological gaits flagged
"""

import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from brace.data.kinect_loader import list_sequences, load_sequence, GAIT_TYPES
from brace.core.srp import normalize_to_body_frame_3d
from brace.core.features import extract_features_sequence, robust_std
from brace.core.gait_cycle import detect_heel_strikes, resample_cycle
from brace.data.joint_map import (
    ANKLE_LEFT, ANKLE_RIGHT, HIP_LEFT, HIP_RIGHT,
    SHOULDER_LEFT, SHOULDER_RIGHT, NUM_KINECT_JOINTS,
    FEATURE_LANDMARKS, FEATURE_JOINT_NAMES,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# ── Video parameters ──
WIDTH, HEIGHT = 1280, 720
FPS = 24
BG_COLOR = (18, 18, 24)  # Dark background

# ── Colors (BGR for OpenCV) ──
BLUE_CALIBRATE = (210, 160, 40)     # Warm blue
BLUE_CALIBRATE_FILL = (210, 160, 40)
GREEN_NORMAL = (80, 210, 60)         # Green
GREEN_NORMAL_DIM = (40, 100, 30)
RED_ANOMALY = (60, 60, 240)          # Red
ORANGE_WARNING = (40, 160, 240)      # Orange
WHITE = (240, 240, 240)
GRAY = (120, 120, 120)
DARK_GRAY = (60, 60, 60)
CYAN = (220, 200, 40)
SKELETON_COLOR_NORMAL = (200, 200, 200)

# Kinect skeleton bone connections (pairs of joint indices)
BONES = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7),
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11),
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),
    # Hands
    (7, 21), (7, 22), (11, 23), (11, 24),
]

# Gait type display info
GAIT_DISPLAY = {
    "normal": {"label": "NORMAL GAIT", "emoji": ""},
    "antalgic": {"label": "ANTALGIC (Limping)", "emoji": ""},
    "lurch": {"label": "LURCHING GAIT", "emoji": ""},
    "steppage": {"label": "STEPPAGE GAIT", "emoji": ""},
    "stiff_legged": {"label": "STIFF-LEGGED", "emoji": ""},
    "trendelenburg": {"label": "TRENDELENBURG", "emoji": ""},
}


def project_3d_to_2d(joints_3d: np.ndarray, target_cx: float = None, target_cy: float = None,
                     target_height: float = 500) -> np.ndarray:
    """Project 3D Kinect joints to 2D screen coordinates, auto-centered and auto-scaled.

    Kinect coordinate system: x=lateral, y=vertical (up), z=depth.
    We project onto the x-y plane (front view), auto-fitting to a target region.
    """
    xs = joints_3d[:, 0]
    ys = joints_3d[:, 1]
    x_range = max(xs.max() - xs.min(), 0.01)
    y_range = max(ys.max() - ys.min(), 0.01)

    # Scale to fit target_height while preserving aspect ratio
    scale = target_height / max(x_range, y_range)

    # Center of skeleton in world space
    x_center = (xs.max() + xs.min()) / 2.0
    y_center = (ys.max() + ys.min()) / 2.0

    if target_cx is None:
        target_cx = WIDTH // 2 - 100
    if target_cy is None:
        target_cy = HEIGHT // 2

    pts = np.zeros((joints_3d.shape[0], 2), dtype=np.float32)
    pts[:, 0] = target_cx + (xs - x_center) * scale
    pts[:, 1] = target_cy - (ys - y_center) * scale  # flip y
    return pts


def get_bbox(pts_2d: np.ndarray, padding: int = 30) -> tuple:
    """Get bounding box from 2D points with padding."""
    valid = pts_2d[~np.isnan(pts_2d).any(axis=1)]
    if len(valid) == 0:
        return (100, 100, 300, 500)
    x_min = int(np.min(valid[:, 0])) - padding
    y_min = int(np.min(valid[:, 1])) - padding
    x_max = int(np.max(valid[:, 0])) + padding
    y_max = int(np.max(valid[:, 1])) + padding
    return (x_min, y_min, x_max, y_max)


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=16):
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    # Straight edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_skeleton(img, pts_2d, color, joint_radius=5, bone_thickness=2):
    """Draw skeleton on image."""
    for a, b in BONES:
        pa = tuple(pts_2d[a].astype(int))
        pb = tuple(pts_2d[b].astype(int))
        if 0 < pa[0] < WIDTH and 0 < pa[1] < HEIGHT and 0 < pb[0] < WIDTH and 0 < pb[1] < HEIGHT:
            cv2.line(img, pa, pb, color, bone_thickness, cv2.LINE_AA)
    for i in range(NUM_KINECT_JOINTS):
        pt = tuple(pts_2d[i].astype(int))
        if 0 < pt[0] < WIDTH and 0 < pt[1] < HEIGHT:
            cv2.circle(img, pt, joint_radius, color, -1, cv2.LINE_AA)


def draw_deviation_arrows(img, pts_2d, deviation_indices, color, scale=20):
    """Draw arrows on joints that deviate most."""
    for idx, magnitude in deviation_indices:
        pt = tuple(pts_2d[idx].astype(int))
        if 0 < pt[0] < WIDTH and 0 < pt[1] < HEIGHT:
            # Pulsing circle around deviating joint
            r = int(8 + magnitude * 3)
            cv2.circle(img, pt, r, color, 2, cv2.LINE_AA)


def draw_score_bar(img, x, y, w, h, score, max_score=3.0, threshold=1.0):
    """Draw an anomaly score bar with gradient fill."""
    # Background
    cv2.rectangle(img, (x, y), (x + w, y + h), DARK_GRAY, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)

    # Threshold line
    thresh_x = x + int(w * (threshold / max_score))
    cv2.line(img, (thresh_x, y), (thresh_x, y + h), WHITE, 1)

    # Fill
    fill_w = int(w * min(score / max_score, 1.0))
    if score < threshold:
        fill_color = GREEN_NORMAL
    elif score < threshold * 1.8:
        fill_color = ORANGE_WARNING
    else:
        fill_color = RED_ANOMALY
    if fill_w > 0:
        cv2.rectangle(img, (x, y + 2), (x + fill_w, y + h - 2), fill_color, -1)


def draw_info_panel(img, phase, gait_label, anomaly_score, frame_num, total_frames,
                    calibration_progress=0, worst_joints=None):
    """Draw the HUD info panel on the right side."""
    panel_x = WIDTH - 380
    panel_y = 30
    panel_w = 350
    panel_h = 340

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (30, 30, 40), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  DARK_GRAY, 1)

    tx = panel_x + 20
    ty = panel_y + 30

    # Title
    cv2.putText(img, "BRACE", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)
    ty += 35

    # Phase indicator
    if phase == "calibrating":
        phase_color = BLUE_CALIBRATE
        phase_text = "CALIBRATING..."
    elif phase == "normal":
        phase_color = GREEN_NORMAL
        phase_text = "NORMAL"
    else:
        phase_color = RED_ANOMALY
        phase_text = "ANOMALY DETECTED"

    cv2.circle(img, (tx + 6, ty - 5), 6, phase_color, -1, cv2.LINE_AA)
    cv2.putText(img, phase_text, (tx + 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2, cv2.LINE_AA)
    ty += 30

    # Gait type label
    display = GAIT_DISPLAY.get(gait_label, {"label": gait_label})
    cv2.putText(img, display["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    ty += 30

    # Calibration progress bar or anomaly score
    if phase == "calibrating":
        cv2.putText(img, "Building baseline:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
        ty += 22
        bar_w = panel_w - 40
        bar_h = 16
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + bar_h), DARK_GRAY, -1)
        fill_w = int(bar_w * calibration_progress)
        cv2.rectangle(img, (tx, ty), (tx + fill_w, ty + bar_h), BLUE_CALIBRATE, -1)
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + bar_h), GRAY, 1)
        pct_text = f"{int(calibration_progress * 100)}%"
        cv2.putText(img, pct_text, (tx + bar_w + 8, ty + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        ty += 35
    else:
        cv2.putText(img, "Anomaly Score:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
        ty += 22
        draw_score_bar(img, tx, ty, panel_w - 80, 18, anomaly_score)
        score_text = f"{anomaly_score:.2f}"
        cv2.putText(img, score_text, (tx + panel_w - 70, ty + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                     GREEN_NORMAL if anomaly_score < 1.0 else (ORANGE_WARNING if anomaly_score < 1.8 else RED_ANOMALY),
                     2, cv2.LINE_AA)
        ty += 35

    # Worst joints (during anomaly phase)
    if worst_joints and phase == "anomaly":
        cv2.putText(img, "Flagged joints:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
        ty += 20
        for name, dev in worst_joints[:3]:
            bar_w_j = int(min(dev / 3.0, 1.0) * 120)
            cv2.rectangle(img, (tx + 110, ty - 10), (tx + 110 + bar_w_j, ty + 2), RED_ANOMALY, -1)
            cv2.putText(img, f"{name}", (tx + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
            cv2.putText(img, f"{dev:.1f}s", (tx + 240, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, RED_ANOMALY, 1, cv2.LINE_AA)
            ty += 20

    # Frame counter
    ty = panel_y + panel_h - 15
    cv2.putText(img, f"Frame {frame_num}/{total_frames}", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, DARK_GRAY, 1, cv2.LINE_AA)


def draw_phase_title(img, text, color, y=HEIGHT - 50):
    """Draw a big phase title at the bottom."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = (WIDTH - tw) // 2
    # Background pill
    cv2.rectangle(img, (tx - 20, y - th - 10), (tx + tw + 20, y + 10), (30, 30, 40), -1)
    cv2.rectangle(img, (tx - 20, y - th - 10), (tx + tw + 20, y + 10), color, 2)
    cv2.putText(img, text, (tx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def compute_live_anomaly(cycle_features, baseline_mean, baseline_std):
    """Compute anomaly score for a single cycle against baseline."""
    deviation = np.abs(cycle_features - baseline_mean) / baseline_std
    frame_rms = np.sqrt(np.mean(deviation ** 2, axis=1))
    overall = float(np.mean(frame_rms))

    # Per-joint scores
    n_joints = len(FEATURE_LANDMARKS)
    joint_scores = []
    for j_idx, joint_id in enumerate(FEATURE_LANDMARKS):
        feat_start = j_idx * 3
        feat_end = feat_start + 3
        dev = float(np.mean(deviation[:, feat_start:feat_end]))
        joint_scores.append((FEATURE_JOINT_NAMES[joint_id], dev))

    joint_scores.sort(key=lambda x: x[1], reverse=True)
    return overall, joint_scores, frame_rms


def main():
    print("Loading dataset...")
    all_seqs = list_sequences(DATA_ROOT)

    # Pick subject with best separation (human1 or human9)
    subject = "human1"
    print(f"Using subject: {subject}")

    # Load normal sequences for calibration
    normal_seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == "normal"]
    # Load pathological sequences
    patho_order = ["antalgic", "lurch", "stiff_legged", "trendelenburg"]
    patho_seqs = {}
    for gt in patho_order:
        seqs = [s for s in all_seqs if s["subject"] == subject and s["gait_type"] == gt]
        if seqs:
            patho_seqs[gt] = seqs

    # Load raw skeleton data
    print("Loading skeleton data...")
    calibration_raw = []
    for seq_info in normal_seqs[:6]:
        data = load_sequence(seq_info)
        if data.shape[0] > 0:
            calibration_raw.append(data)

    normal_test_raw = []
    for seq_info in normal_seqs[6:9]:
        data = load_sequence(seq_info)
        if data.shape[0] > 0:
            normal_test_raw.append(data)

    patho_raw = {}
    for gt, seqs in patho_seqs.items():
        loaded = []
        for seq_info in seqs[:3]:
            data = load_sequence(seq_info)
            if data.shape[0] > 0:
                loaded.append(data)
        if loaded:
            patho_raw[gt] = loaded

    # ── Build video frames ──
    video_path = OUTPUT_DIR / "brace_demo.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (WIDTH, HEIGHT))

    # ── PHASE 1: CALIBRATION (blue) ──
    print("Rendering Phase 1: Calibration...")
    baseline_features_all = []
    total_cal_frames = sum(d.shape[0] for d in calibration_raw[:4])
    cal_frame_count = 0

    for seq_idx, raw_seq in enumerate(calibration_raw[:4]):
        norm_seq, _ = normalize_to_body_frame_3d(raw_seq)
        feats = extract_features_sequence(norm_seq)

        # Subsample for video (every 2nd frame)
        for f_idx in range(0, raw_seq.shape[0], 2):
            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

            joints_3d = raw_seq[f_idx]
            pts_2d = project_3d_to_2d(joints_3d)
            bbox = get_bbox(pts_2d, padding=40)

            # Draw skeleton (blue tint)
            skel_color = tuple(int(c * 0.6) for c in BLUE_CALIBRATE)
            draw_skeleton(frame, pts_2d, skel_color, joint_radius=4, bone_thickness=2)

            # Draw bounding box (blue, rounded)
            draw_rounded_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              BLUE_CALIBRATE, thickness=3, radius=20)

            # "CALIBRATING" label above bbox
            label = "CALIBRATING"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_x = bbox[0]
            label_y = bbox[1] - 12
            cv2.rectangle(frame, (label_x - 4, label_y - lh - 6), (label_x + lw + 8, label_y + 4),
                          BLUE_CALIBRATE, -1)
            cv2.putText(frame, label, (label_x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 30), 2, cv2.LINE_AA)

            cal_frame_count += 2
            progress = min(1.0, cal_frame_count / total_cal_frames)

            total_video_frames = total_cal_frames // 2 + 200 + 150 * len(patho_raw)
            draw_info_panel(frame, "calibrating", "normal", 0, cal_frame_count // 2,
                            total_video_frames, calibration_progress=progress)

            draw_phase_title(frame, "PHASE 1: Building motion baseline from normal gait", BLUE_CALIBRATE)

            writer.write(frame)

        # Accumulate features for baseline
        baseline_features_all.append(feats)

    # Compute baseline
    all_feats = np.concatenate(baseline_features_all, axis=0)
    baseline_mean = np.mean(all_feats, axis=0)
    baseline_std = robust_std(all_feats)
    baseline_std = np.maximum(baseline_std, 1e-6)

    # ── Transition frame ──
    for _ in range(int(FPS * 1.5)):
        frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
        text = "BASELINE READY - Monitoring for anomalies..."
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(frame, text, ((WIDTH - tw) // 2, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN_NORMAL, 2, cv2.LINE_AA)
        writer.write(frame)

    # ── PHASE 2: NORMAL VERIFICATION (green) ──
    print("Rendering Phase 2: Normal gait (green)...")
    running_score = deque(maxlen=30)
    frame_counter = cal_frame_count // 2 + int(FPS * 1.5)

    for raw_seq in normal_test_raw[:2]:
        norm_seq, _ = normalize_to_body_frame_3d(raw_seq)
        feats = extract_features_sequence(norm_seq)

        # Compute rolling anomaly score using a sliding window
        window_size = 60
        for f_idx in range(0, raw_seq.shape[0], 2):
            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
            frame_counter += 1

            joints_3d = raw_seq[f_idx]
            pts_2d = project_3d_to_2d(joints_3d)
            bbox = get_bbox(pts_2d, padding=40)

            # Compute anomaly for recent window
            start = max(0, f_idx - window_size)
            window_feats = feats[start:f_idx + 1]
            if window_feats.shape[0] > 5:
                deviation = np.abs(window_feats - baseline_mean) / baseline_std
                frame_rms = np.sqrt(np.mean(deviation ** 2, axis=1))
                score = float(np.mean(frame_rms))
            else:
                score = 0.5
            running_score.append(score)
            avg_score = float(np.mean(running_score))

            # Draw skeleton (green)
            draw_skeleton(frame, pts_2d, GREEN_NORMAL, joint_radius=4, bone_thickness=2)

            # Green bbox
            draw_rounded_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              GREEN_NORMAL, thickness=3, radius=20)

            label = "NORMAL"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_x = bbox[0]
            label_y = bbox[1] - 12
            cv2.rectangle(frame, (label_x - 4, label_y - lh - 6), (label_x + lw + 8, label_y + 4),
                          GREEN_NORMAL, -1)
            cv2.putText(frame, label, (label_x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 30), 2, cv2.LINE_AA)

            draw_info_panel(frame, "normal", "normal", avg_score, frame_counter,
                            frame_counter + 300)

            draw_phase_title(frame, "PHASE 2: Normal gait verified - no anomalies", GREEN_NORMAL)

            writer.write(frame)

    # ── PHASE 3: PATHOLOGICAL GAITS (red) ──
    print("Rendering Phase 3: Pathological gaits (red)...")

    for gt in patho_order:
        if gt not in patho_raw:
            continue

        raw_seq = patho_raw[gt][0]
        norm_seq, _ = normalize_to_body_frame_3d(raw_seq)
        feats = extract_features_sequence(norm_seq)

        # Transition frame
        for _ in range(int(FPS * 0.8)):
            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
            text = f"Now testing: {GAIT_DISPLAY[gt]['label']}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, text, ((WIDTH - tw) // 2, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE_WARNING, 2, cv2.LINE_AA)
            writer.write(frame)
            frame_counter += 1

        running_score.clear()
        worst_joints_cache = None

        for f_idx in range(0, min(raw_seq.shape[0], 150), 2):
            frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)
            frame_counter += 1

            joints_3d = raw_seq[f_idx]
            pts_2d = project_3d_to_2d(joints_3d)
            bbox = get_bbox(pts_2d, padding=40)

            # Compute anomaly
            start = max(0, f_idx - 60)
            window_feats = feats[start:f_idx + 1]
            if window_feats.shape[0] > 5:
                score, joint_scores, _ = compute_live_anomaly(
                    window_feats, baseline_mean[:window_feats.shape[1]], baseline_std[:window_feats.shape[1]]
                )
                worst_joints_cache = joint_scores[:3]
            else:
                score = 1.0
            running_score.append(score)
            avg_score = float(np.mean(running_score))

            # Determine color based on score
            if avg_score < 1.0:
                bbox_color = GREEN_NORMAL
                skel_color = GREEN_NORMAL
                label = "NORMAL"
                phase = "normal"
            elif avg_score < 1.5:
                bbox_color = ORANGE_WARNING
                skel_color = ORANGE_WARNING
                label = "WARNING"
                phase = "anomaly"
            else:
                bbox_color = RED_ANOMALY
                skel_color = RED_ANOMALY
                label = "INJURY DETECTED"
                phase = "anomaly"

            # Draw skeleton
            draw_skeleton(frame, pts_2d, skel_color, joint_radius=4, bone_thickness=2)

            # Draw bounding box
            draw_rounded_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              bbox_color, thickness=3, radius=20)

            # Label above bbox
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            label_x = bbox[0]
            label_y = bbox[1] - 12
            cv2.rectangle(frame, (label_x - 4, label_y - lh - 6), (label_x + lw + 8, label_y + 4),
                          bbox_color, -1)
            cv2.putText(frame, label, (label_x + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 30), 2, cv2.LINE_AA)

            # Draw deviation highlights on worst joints
            if phase == "anomaly" and worst_joints_cache:
                for jname, jdev in worst_joints_cache[:3]:
                    # Find the joint index from name
                    for jid, jn in FEATURE_JOINT_NAMES.items():
                        if jn == jname:
                            pt = tuple(pts_2d[jid].astype(int))
                            if 0 < pt[0] < WIDTH and 0 < pt[1] < HEIGHT:
                                r = int(10 + jdev * 2)
                                cv2.circle(frame, pt, r, RED_ANOMALY, 2, cv2.LINE_AA)
                                cv2.circle(frame, pt, r + 4, RED_ANOMALY, 1, cv2.LINE_AA)
                            break

            draw_info_panel(frame, phase, gt, avg_score, frame_counter,
                            frame_counter + 200, worst_joints=worst_joints_cache)

            title = f"PHASE 3: Testing {GAIT_DISPLAY[gt]['label']}"
            draw_phase_title(frame, title, bbox_color)

            writer.write(frame)

    # ── End card ──
    print("Rendering end card...")
    for _ in range(int(FPS * 3)):
        frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

        cv2.putText(frame, "BRACE", (WIDTH // 2 - 180, HEIGHT // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, CYAN, 3, cv2.LINE_AA)

        cv2.putText(frame, "SRP-based gait anomaly detection", (WIDTH // 2 - 220, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1, cv2.LINE_AA)

        cv2.putText(frame, "Normal: 0.886  |  Pathological: 1.398  |  1.6x separation",
                    (WIDTH // 2 - 280, HEIGHT // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

        cv2.putText(frame, "10 subjects x 6 gait types x 1,200 sequences",
                    (WIDTH // 2 - 240, HEIGHT // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1, cv2.LINE_AA)

        writer.write(frame)

    writer.release()
    print(f"\nVideo saved: {video_path}")
    print(f"Duration: ~{frame_counter / FPS:.0f}s at {FPS} fps")


if __name__ == "__main__":
    main()
