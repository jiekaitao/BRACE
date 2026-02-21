#!/usr/bin/env python3
"""Test clustering on all demo videos at various thresholds."""
import sys
sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import os
import cv2
import numpy as np
from ultralytics import YOLO
from brace.core.motion_segments import (
    normalize_frame, segment_motions, cluster_segments, _segment_distance,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

VIDEOS_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
model = YOLO("yolo11n-pose.pt")

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    features = []
    valid_indices = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            xy = kp.xy.cpu().numpy()[0]
            conf = kp.conf.cpu().numpy()[0]
            kp_with_conf = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_with_conf, img_w, img_h)
            feat = normalize_frame(mp33)
            if feat is not None:
                feat_vec = feat[FEATURE_INDICES, :2].flatten()
                features.append(feat_vec)
                valid_indices.append(frame_idx)
        frame_idx += 1
    cap.release()
    return np.stack(features) if features else None, valid_indices, fps

thresholds = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5]

for fname in sorted(os.listdir(VIDEOS_DIR)):
    if not fname.endswith('.mp4'):
        continue
    path = os.path.join(VIDEOS_DIR, fname)
    print(f"\n{'='*60}")
    print(f"Video: {fname}")
    feats, vi, fps = extract_features(path)
    if feats is None or feats.shape[0] < 5:
        print("  Not enough features")
        continue
    print(f"  Frames with features: {feats.shape[0]}, dim: {feats.shape[1]}, FPS: {fps:.0f}")
    
    # Show segments
    segs = segment_motions(feats, vi, fps)
    print(f"  Segments: {len(segs)}")
    if len(segs) >= 2:
        print(f"  Pairwise distances:")
        for i in range(len(segs)):
            for j in range(i+1, len(segs)):
                d = _segment_distance(segs[i], segs[j])
                print(f"    d({i},{j}) = {d:.3f}")
    
    print(f"\n  {'Threshold':>10} | {'Clusters':>8} | Labels")
    for t in thresholds:
        segs_t = segment_motions(feats, vi, fps)
        segs_t = cluster_segments(segs_t, distance_threshold=t)
        nc = len(set(s.get("cluster", 0) for s in segs_t))
        labs = [s.get("cluster", -1) for s in segs_t]
        marker = " <-- TARGET" if t == 2.0 else ""
        print(f"    {t:>8.1f} | {nc:>8} | {labs}{marker}")
