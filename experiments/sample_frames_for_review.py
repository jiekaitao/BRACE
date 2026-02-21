#!/usr/bin/env python3
"""Sample a few frames from each video for visual review."""
import cv2
import os

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
OUTPUT_DIR = "/mnt/Data/GitHub/BRACE/experiments/review_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])

for vname in videos:
    vpath = os.path.join(VIDEO_DIR, vname)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        print(f"Skip {vname}")
        cap.release()
        continue

    duration = total / fps
    # Sample 5 evenly spaced frames
    times = [duration * i / 5 for i in range(5)]

    base = vname.replace('.mp4', '')
    for t in times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(OUTPUT_DIR, f"{base}_t{t:.1f}s.jpg")
            cv2.imwrite(out_path, frame)

    cap.release()
    print(f"Sampled {vname}: {len(times)} frames")

print(f"\nFrames saved to {OUTPUT_DIR}")
