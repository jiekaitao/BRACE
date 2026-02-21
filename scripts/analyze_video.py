#!/usr/bin/env python3
"""BRACE — Analyze any sports/exercise video.

Pipeline:
1. MediaPipe pose estimation on every frame
2. SRP normalization (position/scale/rotation invariant)
3. Segment into individual motions
4. Cluster repeated movements
5. Analyze consistency within clusters
6. Render annotated video:
   - Blue bbox: calibrating (first pass)
   - Green bbox: normal/consistent movement
   - Red/orange bbox: anomalous repetition (potential injury indicator)
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from brace.core.pose import (
    extract_poses_from_video,
    MP_BONES,
    FEATURE_INDICES,
    FEATURE_NAMES,
    LEFT_HIP, RIGHT_HIP,
)
from brace.core.motion_segments import (
    compute_feature_trajectory,
    segment_motions,
    cluster_segments,
    analyze_consistency,
    normalize_frame,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Colors (BGR)
BLUE = (210, 160, 40)
GREEN = (80, 210, 60)
RED = (60, 60, 240)
ORANGE = (40, 160, 240)
CYAN = (220, 200, 40)
WHITE = (240, 240, 240)
GRAY = (120, 120, 120)
DARK = (30, 30, 40)

CLUSTER_COLORS = [
    (255, 180, 50),   # blue
    (50, 220, 100),   # green
    (80, 120, 255),   # red-ish
    (200, 100, 255),  # pink
    (100, 255, 255),  # yellow
    (255, 100, 200),  # purple
]


def get_bbox_from_landmarks(lm: np.ndarray, padding: int = 30) -> tuple:
    """Get bounding box from landmarks (33, 4) using visible joints."""
    vis = lm[:, 3]
    xy = lm[:, :2]
    mask = vis > 0.3
    if mask.sum() < 4:
        return None
    visible = xy[mask]
    x1 = int(visible[:, 0].min()) - padding
    y1 = int(visible[:, 1].min()) - padding
    x2 = int(visible[:, 0].max()) + padding
    y2 = int(visible[:, 1].max()) + padding
    return (x1, y1, x2, y2)


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=14):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4, 30)
    if r < 2:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_skeleton(img, lm, color, thickness=2, joint_radius=4):
    """Draw MediaPipe skeleton on image."""
    vis = lm[:, 3]
    xy = lm[:, :2].astype(int)
    for a, b in MP_BONES:
        if vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(img, tuple(xy[a]), tuple(xy[b]), color, thickness, cv2.LINE_AA)
    for i in FEATURE_INDICES:
        if vis[i] > 0.3:
            cv2.circle(img, tuple(xy[i]), joint_radius, color, -1, cv2.LINE_AA)


def draw_label(img, text, x, y, bg_color, text_color=(20, 20, 30)):
    """Draw a label with background."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 6, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)


def draw_hud(img, width, height, phase, cluster_id, segment_idx, consistency_score,
             n_segments, n_clusters, cluster_analysis):
    """Draw the HUD panel."""
    px = width - 340
    py = 20
    pw = 320
    ph = 280

    overlay = img.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), DARK, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), GRAY, 1)

    tx = px + 16
    ty = py + 28

    cv2.putText(img, "BRACE", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2, cv2.LINE_AA)
    ty += 30

    # Phase
    if phase == "calibrating":
        cv2.circle(img, (tx + 5, ty - 4), 5, BLUE, -1)
        cv2.putText(img, "CALIBRATING", (tx + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2, cv2.LINE_AA)
    elif phase == "normal":
        cv2.circle(img, (tx + 5, ty - 4), 5, GREEN, -1)
        cv2.putText(img, "CONSISTENT", (tx + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2, cv2.LINE_AA)
    else:
        cv2.circle(img, (tx + 5, ty - 4), 5, RED, -1)
        cv2.putText(img, "INCONSISTENT", (tx + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
    ty += 28

    # Stats
    cv2.putText(img, f"Motions detected: {n_segments}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
    ty += 20
    cv2.putText(img, f"Motion clusters: {n_clusters}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
    ty += 20

    if cluster_id is not None:
        c_color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        cv2.putText(img, f"Current cluster: #{cluster_id}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_color, 1, cv2.LINE_AA)
        ty += 20

    if consistency_score is not None and phase != "calibrating":
        cv2.putText(img, "Consistency:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
        ty += 18
        bar_w = pw - 80
        bar_h = 14
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + bar_h), (50, 50, 60), -1)
        fill = min(consistency_score / 3.0, 1.0)
        fill_w = int(bar_w * fill)
        fill_color = GREEN if consistency_score < 1.2 else (ORANGE if consistency_score < 2.0 else RED)
        cv2.rectangle(img, (tx, ty), (tx + fill_w, ty + bar_h), fill_color, -1)
        cv2.rectangle(img, (tx, ty), (tx + bar_w, ty + bar_h), GRAY, 1)
        cv2.putText(img, f"{consistency_score:.2f}", (tx + bar_w + 6, ty + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, fill_color, 1, cv2.LINE_AA)
        ty += 28

    # Cluster summary
    if cluster_analysis:
        ty += 5
        cv2.putText(img, "Cluster summary:", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRAY, 1, cv2.LINE_AA)
        ty += 16
        for c_id in sorted(cluster_analysis.keys()):
            info = cluster_analysis[c_id]
            c_color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
            n_anom = len(info["anomaly_segments"])
            text = f"#{c_id}: {info['count']} reps"
            if n_anom > 0:
                text += f" ({n_anom} flagged)"
            cv2.putText(img, text, (tx + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c_color, 1, cv2.LINE_AA)
            ty += 15
            if ty > py + ph - 10:
                break


def main():
    parser = argparse.ArgumentParser(description="BRACE — analyze sports video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default="", help="Output video path (default: outputs/analyzed_<name>.mp4)")
    parser.add_argument("--model", default="", help="Path to pose_landmarker .task model")
    parser.add_argument("--cluster-threshold", type=float, default=1.5, help="Distance threshold for clustering")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    output_path = args.output or str(OUTPUT_DIR / f"analyzed_{video_path.stem}.mp4")
    model_path = args.model or None

    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print()

    # ── Step 1: Pose Estimation ──
    print("[1/5] Running MediaPipe pose estimation...")
    landmarks_list, vid_w, vid_h, fps = extract_poses_from_video(video_path, model_path)
    total_frames = len(landmarks_list)
    print(f"  Video: {vid_w}x{vid_h} @ {fps:.1f}fps, {total_frames} frames")

    # ── Step 2: SRP Normalize + Feature Extraction ──
    print("\n[2/5] SRP normalization + feature extraction...")
    features, valid_indices = compute_feature_trajectory(landmarks_list)
    print(f"  {features.shape[0]} valid frames, {features.shape[1]}D features")

    # ── Step 3: Segment Motions ──
    print("\n[3/5] Segmenting motions...")
    segments = segment_motions(features, valid_indices, fps)
    print(f"  {len(segments)} motion segments detected")

    # ── Step 4: Cluster ──
    print("\n[4/5] Clustering repeated motions...")
    segments = cluster_segments(segments, distance_threshold=args.cluster_threshold)
    n_clusters = len(set(s["cluster"] for s in segments))
    cluster_counts = defaultdict(int)
    for s in segments:
        cluster_counts[s["cluster"]] += 1
    print(f"  {n_clusters} clusters found:")
    for c_id in sorted(cluster_counts.keys()):
        print(f"    Cluster #{c_id}: {cluster_counts[c_id]} repetitions")

    # ── Step 5: Analyze Consistency ──
    print("\n[5/5] Analyzing consistency...")
    cluster_analysis = analyze_consistency(segments)
    for c_id, info in sorted(cluster_analysis.items()):
        n_anom = len(info["anomaly_segments"])
        print(f"  Cluster #{c_id}: {info['count']} reps, mean consistency={info['mean_score']:.2f}"
              + (f", {n_anom} flagged" if n_anom else ""))

    # ── Build frame-level lookup ──
    frame_to_segment = {}
    for seg_idx, seg in enumerate(segments):
        for f in range(seg["start_frame"], seg["end_frame"] + 1):
            frame_to_segment[f] = seg_idx

    # Figure out calibration window: first pass through each cluster's first rep
    # (first 20% of frames = calibration)
    calibration_end = int(total_frames * 0.2)

    # ── Render annotated video ──
    print(f"\nRendering annotated video...")
    cap = cv2.VideoCapture(str(video_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        lm = landmarks_list[frame_idx]
        seg_idx = frame_to_segment.get(frame_idx)

        if lm is not None:
            bbox = get_bbox_from_landmarks(lm)

            # Determine phase and colors
            if frame_idx < calibration_end:
                phase = "calibrating"
                bbox_color = BLUE
                skel_color = tuple(int(c * 0.7) for c in BLUE)
                label_text = "CALIBRATING"
            elif seg_idx is not None:
                seg = segments[seg_idx]
                is_anomaly = seg.get("is_anomaly", False)
                score = seg.get("consistency_score", 0)

                if is_anomaly:
                    phase = "anomaly"
                    bbox_color = RED if score > 2.0 else ORANGE
                    skel_color = bbox_color
                    label_text = "INCONSISTENT" if score > 2.0 else "WARNING"
                else:
                    phase = "normal"
                    bbox_color = GREEN
                    skel_color = GREEN
                    label_text = "CONSISTENT"
            else:
                phase = "normal"
                bbox_color = GREEN
                skel_color = tuple(int(c * 0.5) for c in GREEN)
                label_text = "TRACKING"

            # Draw skeleton
            draw_skeleton(frame, lm, skel_color, thickness=2, joint_radius=3)

            # Draw bbox
            if bbox:
                draw_rounded_rect(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  bbox_color, thickness=3)
                draw_label(frame, label_text, bbox[0], bbox[1] - 8, bbox_color)

                # Show cluster color indicator
                if seg_idx is not None:
                    c_id = segments[seg_idx].get("cluster", 0)
                    c_color = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
                    cv2.rectangle(frame, (bbox[2] - 30, bbox[1]), (bbox[2], bbox[1] + 20), c_color, -1)
                    cv2.putText(frame, f"#{c_id}", (bbox[2] - 27, bbox[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 30), 1, cv2.LINE_AA)

            # If anomaly, highlight deviating joints
            if phase == "anomaly" and lm is not None:
                norm = normalize_frame(lm)
                if norm is not None:
                    for ji in FEATURE_INDICES:
                        pt = tuple(lm[ji, :2].astype(int))
                        if lm[ji, 3] > 0.3:
                            cv2.circle(frame, pt, 10, RED, 2, cv2.LINE_AA)

            # HUD
            c_id = segments[seg_idx]["cluster"] if seg_idx is not None else None
            c_score = segments[seg_idx].get("consistency_score") if seg_idx is not None else None
            draw_hud(frame, vid_w, vid_h, phase, c_id, seg_idx, c_score,
                     len(segments), n_clusters, cluster_analysis)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"  Rendering: {frame_idx}/{total_frames}...")

    writer.release()
    cap.release()
    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
