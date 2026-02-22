"""Test: YOLO-pose tracking → closing speed → head acceleration → concussion probability."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import math
import cv2
import numpy as np
from ultralytics import YOLO
from movement_quality import HeadImpactAnalyzer, HeadKinematicState

# --- ClosingSpeedTracker (same as in main.py) ---
CLOSING_SPEED_THRESHOLD = 1.5

class ClosingSpeedTracker:
    def __init__(self):
        self.prev_centers = {}
        self.prev_time = -1.0

    def update(self, subjects_data, video_time, fps):
        current_centers = {}
        for sid_str, data in subjects_data.items():
            bbox = data.get("bbox")
            if bbox:
                cx = (bbox["x1"] + bbox["x2"]) / 2.0
                cy = (bbox["y1"] + bbox["y2"]) / 2.0
                current_centers[int(sid_str)] = (cx, cy)

        dt = 1.0 / max(fps, 1.0)
        if self.prev_time >= 0 and video_time > 0:
            actual_dt = video_time - self.prev_time
            if 0 < actual_dt < 1.0:
                dt = actual_dt

        pairs = []
        max_closing_speed = 0.0

        if self.prev_centers and len(current_centers) >= 2:
            sids = list(current_centers.keys())
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    a, b = sids[i], sids[j]
                    if a not in self.prev_centers or b not in self.prev_centers:
                        continue
                    ax, ay = current_centers[a]
                    bx, by = current_centers[b]
                    curr_dist = math.hypot(ax - bx, ay - by)
                    pax, pay = self.prev_centers[a]
                    pbx, pby = self.prev_centers[b]
                    prev_dist = math.hypot(pax - pbx, pay - pby)
                    closing_speed = (prev_dist - curr_dist) / dt
                    if closing_speed > 0.1:
                        pairs.append({"a": a, "b": b, "closing_speed": round(closing_speed, 3), "distance": round(curr_dist, 4)})
                        if closing_speed > max_closing_speed:
                            max_closing_speed = closing_speed

        self.prev_centers = current_centers
        self.prev_time = video_time
        pairs.sort(key=lambda p: p["closing_speed"], reverse=True)
        pairs = pairs[:5]
        return {
            "pairs": pairs,
            "max_closing_speed": round(max_closing_speed, 3),
            "collision_warning": max_closing_speed >= CLOSING_SPEED_THRESHOLD,
        }


# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "Sample-Data/Tackle4s_10s.mp4"
    print(f"Processing: {video_path}")
    print("=" * 80)

    model = YOLO("yolo11m-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fps:.0f} FPS, {total_frames} frames, {total_frames/fps:.1f}s")
    print("=" * 80)

    closing_tracker = ClosingSpeedTracker()
    impact_analyzer = HeadImpactAnalyzer(fps=fps)
    frame_idx = 0
    dt = 1.0 / fps

    collision_warnings = 0
    concussion_events = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        video_time = frame_idx / fps

        # Run YOLO-pose with tracking
        results = model.track(frame, persist=True, verbose=False, conf=0.25)
        result = results[0]

        subjects_data = {}
        keypoints_by_id = {}

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            kpts_xy = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
            kpts_conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else None

            for idx, track_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[idx]
                subjects_data[str(track_id)] = {
                    "bbox": {
                        "x1": round(x1 / w, 4),
                        "y1": round(y1 / h, 4),
                        "x2": round(x2 / w, 4),
                        "y2": round(y2 / h, 4),
                    }
                }
                if kpts_xy is not None and kpts_conf is not None:
                    keypoints_by_id[track_id] = {
                        "xy": kpts_xy[idx],      # (17, 2)
                        "conf": kpts_conf[idx],   # (17,)
                    }

        # 1) Closing speed
        proximity = closing_tracker.update(subjects_data, video_time, fps)

        # 2) Feed proximity to impact analyzer
        impact_analyzer.update_proximity(proximity["pairs"], frame_idx)

        # 3) Feed per-subject head data to impact analyzer
        for track_id, kp_data in keypoints_by_id.items():
            xy = kp_data["xy"]
            conf = kp_data["conf"]

            # Check head visibility (nose + ears)
            head_vis = [conf[NOSE], conf[L_EAR], conf[R_EAR]]
            if not all(v >= 0.3 for v in head_vis):
                continue

            # Head position: average of head keypoints
            head_px = (xy[NOSE] + xy[L_EAR] + xy[R_EAR]) / 3.0

            # Pixel→meter via shoulder width (~0.40m)
            sw_px = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
            px_to_m = 0.40 / max(sw_px, 1.0)
            head_m = head_px * px_to_m

            # Ear angle for rotational velocity
            ear_vec = xy[R_EAR] - xy[L_EAR]
            ear_angle = float(np.arctan2(ear_vec[1], ear_vec[0]))

            state = impact_analyzer.update_subject(
                track_id, head_m, ear_angle, dt, frame_idx, video_time,
            )

        # Decay scores
        impact_analyzer.decay_scores()

        # Print frame summary
        n_subjects = len(subjects_data)
        warning = ""
        if proximity["collision_warning"]:
            warning = " !! COLLISION WARNING !!"
            collision_warnings += 1

        # Check for new events
        events = impact_analyzer.get_recent_events()
        new_events = [e for e in events if e not in concussion_events]

        if n_subjects >= 2 and (proximity["max_closing_speed"] > 0.3 or new_events):
            # Get per-subject head data
            head_info = ""
            for tid in sorted(keypoints_by_id.keys()):
                tracker = impact_analyzer._trackers.get(tid)
                if tracker and tracker.latest_state:
                    s = tracker.latest_state
                    score = impact_analyzer.get_subject_score(tid)
                    if s.accel_magnitude_g > 0.5 or score > 0:
                        head_info += f"\n    S{tid}: accel={s.accel_magnitude_g:.2f}g dir_change={math.degrees(s.accel_direction_change):.1f}° rot={s.rotational_vel_rps:.1f}rad/s jerk={s.jerk_magnitude:.0f} score={score:.1f}"

            print(
                f"Frame {frame_idx:4d} | t={video_time:.2f}s | "
                f"{n_subjects} subj | "
                f"closing={proximity['max_closing_speed']:.2f}"
                f"{warning}{head_info}"
            )

        for evt in new_events:
            concussion_events.append(evt)
            print(f"\n{'='*60}")
            print(f"  CONCUSSION EVENT: {evt['event_id']}")
            print(f"  Subjects: {evt['subjects']}")
            print(f"  Frame: {evt['frame_index']} | t={evt['video_time']:.2f}s")
            print(f"  Closing speed: {evt['pre_impact_closing_speed']:.2f}")
            print(f"  Peak accel: {evt['peak_head_accel_g']:.2f}g")
            print(f"  Accel direction change: {evt['accel_direction_change_deg']:.1f}°")
            print(f"  Peak jerk: {evt['peak_jerk']:.0f}")
            print(f"  Rotational velocity: {evt['rotational_vel_peak']:.2f} rad/s")
            print(f"  CONCUSSION PROBABILITY: {evt['concussion_probability']:.1f}/100")
            print(f"{'='*60}\n")

        frame_idx += 1

    cap.release()

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Frames processed: {frame_idx}")
    print(f"Collision warnings: {collision_warnings}")
    print(f"Concussion events: {len(concussion_events)}")
    for evt in concussion_events:
        print(f"  [{evt['event_id']}] Subjects {evt['subjects']} | "
              f"accel={evt['peak_head_accel_g']:.2f}g | "
              f"dir_change={evt['accel_direction_change_deg']:.1f}° | "
              f"prob={evt['concussion_probability']:.1f}/100")


if __name__ == "__main__":
    main()
