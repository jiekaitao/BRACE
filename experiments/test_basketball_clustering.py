#!/usr/bin/env python3
"""Test clustering on the real basketball_solo video with various thresholds.

Processes the video with MediaPipe to get actual pose landmarks,
then runs SRP normalization + segmentation + clustering.
Shows exact pairwise distances to determine optimal threshold.
"""

import sys
sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import cv2
import numpy as np
from ultralytics import YOLO
from brace.core.motion_segments import (
    normalize_frame,
    normalize_frame_3d_real,
    segment_motions,
    cluster_segments,
    _segment_distance,
    detect_motion_boundaries,
    analyze_consistency,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

VIDEO_PATH = "/mnt/Data/GitHub/BRACE/data/sports_videos/basketball_solo.mp4"


def extract_landmarks(video_path: str) -> tuple[list[np.ndarray], float]:
    """Extract YOLO-pose landmarks from video (COCO 17 -> MediaPipe 33)."""
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]  # first person
            xy = kp.xy.cpu().numpy()[0]  # (17, 2) pixel coords
            conf = kp.conf.cpu().numpy()[0]  # (17,)
            # Build (17, 3) [x_pixel, y_pixel, confidence]
            kp_with_conf = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_with_conf, img_w, img_h)
            landmarks_list.append(mp33)
        else:
            landmarks_list.append(None)

    cap.release()
    return landmarks_list, fps


def main():
    print(f"Processing {VIDEO_PATH}...")
    landmarks_list, fps = extract_landmarks(VIDEO_PATH)
    total_frames = len(landmarks_list)
    valid_count = sum(1 for l in landmarks_list if l is not None)
    print(f"Total frames: {total_frames}, Valid: {valid_count}, FPS: {fps:.1f}")

    # Normalize features
    features = []
    valid_indices = []
    for i, lm in enumerate(landmarks_list):
        if lm is None:
            continue
        feat = normalize_frame(lm)
        if feat is None:
            continue
        feat_vec = feat[FEATURE_INDICES, :2].flatten()
        features.append(feat_vec)
        valid_indices.append(i)

    features_arr = np.stack(features)
    print(f"Feature frames: {features_arr.shape[0]}, dim: {features_arr.shape[1]}")

    # Detect motion boundaries
    boundaries = detect_motion_boundaries(features_arr, fps, min_segment_sec=1.0)
    print(f"\nMotion boundaries at frames: {boundaries}")
    print(f"Number of segments: {len(boundaries)}")

    # Create segments
    segments = segment_motions(features_arr, valid_indices, fps)
    n_seg = len(segments)
    print(f"Segments created: {n_seg}")
    for i, seg in enumerate(segments):
        dur = (seg['end_frame'] - seg['start_frame']) / fps
        print(f"  Segment {i}: frames {seg['start_frame']}-{seg['end_frame']} ({dur:.1f}s), "
              f"features: {seg['features'].shape[0]}")

    if n_seg < 2:
        print("\nOnly 1 segment - nothing to cluster. Try lower min_segment_sec.")
        # Try with lower min_segment_sec
        for ms in [0.5, 0.3]:
            segs2 = segment_motions(features_arr, valid_indices, fps, min_segment_sec=ms)
            print(f"  With min_segment_sec={ms}: {len(segs2)} segments")
        return

    # Compute pairwise distances
    print(f"\nPairwise segment distances:")
    for i in range(n_seg):
        for j in range(i + 1, n_seg):
            d = _segment_distance(segments[i], segments[j])
            print(f"  d({i},{j}) = {d:.4f}")

    # Test various thresholds
    thresholds = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5]
    print(f"\n{'Threshold':>10} | {'Clusters':>8} | Cluster labels")
    print("-" * 60)
    for t in thresholds:
        # Re-segment each time since cluster_segments modifies in-place
        segs = segment_motions(features_arr, valid_indices, fps)
        segs = cluster_segments(segs, distance_threshold=t)
        n_clusters = len(set(s.get("cluster", 0) for s in segs))
        labels = [s.get("cluster", -1) for s in segs]
        print(f"  {t:>8.1f} | {n_clusters:>8} | {labels}")

    # Also test with min_segment_sec=0.5 for more segments
    print(f"\n--- With min_segment_sec=0.5 ---")
    segs05 = segment_motions(features_arr, valid_indices, fps, min_segment_sec=0.5)
    n05 = len(segs05)
    print(f"Segments: {n05}")
    if n05 >= 2:
        for i in range(n05):
            for j in range(i + 1, n05):
                d = _segment_distance(segs05[i], segs05[j])
                print(f"  d({i},{j}) = {d:.4f}")
        for t in thresholds:
            segs_t = segment_motions(features_arr, valid_indices, fps, min_segment_sec=0.5)
            segs_t = cluster_segments(segs_t, distance_threshold=t)
            n_clusters = len(set(s.get("cluster", 0) for s in segs_t))
            labels = [s.get("cluster", -1) for s in segs_t]
            print(f"  {t:>8.1f} | {n_clusters:>8} | {labels}")


if __name__ == "__main__":
    main()
