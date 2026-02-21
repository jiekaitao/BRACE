#!/usr/bin/env python3
"""Validate clustering pipeline on all demo videos using CLIP labels.

For each video:
1. Extract pose features via YOLO-pose + SRP normalization
2. Run velocity segmentation + agglomerative clustering at threshold 2.0
3. For each cluster, get representative frames and classify with CLIP
4. Check: do different clusters get different labels?

OVER-SEGMENTED: 2+ clusters share same CLIP label (threshold too low)
UNDER-SEGMENTED: 1 cluster contains different activities (threshold too high)
"""

import sys
import os
import json
import copy
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import open_clip
from ultralytics import YOLO
from brace.core.motion_segments import normalize_frame, segment_motions, cluster_segments
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
OUTPUT_PATH = "/mnt/Data/GitHub/BRACE/experiments/clustering_validation.json"

# Action labels for CLIP classification (same as analyze_videos_clip.py)
ACTION_LABELS = [
    "a person dribbling a basketball",
    "a person shooting a basketball",
    "a person doing a layup",
    "a person dunking a basketball",
    "a person running",
    "a person jogging",
    "a person walking",
    "a person standing still",
    "a person resting",
    "a person doing pushups",
    "a person doing squats",
    "a person doing pullups on a bar",
    "a person doing lunges",
    "a person doing burpees",
    "a person doing jumping jacks",
    "a person doing a plank",
    "a person stretching",
    "a person doing yoga",
    "a person lifting weights",
    "a person doing bench press",
    "a person doing deadlift",
    "a person doing barbell curls",
    "a person doing overhead press",
    "a person doing rows",
    "a person doing kettlebell swings",
    "a person doing clean and jerk",
    "a person doing snatch",
    "a person jumping rope",
    "a person doing box jumps",
    "a person doing wall balls",
    "a person doing muscle ups",
    "a person doing handstand pushups",
    "a person doing rope climbs",
    "a person doing double unders",
    "a person dribbling a soccer ball",
    "a person kicking a soccer ball",
    "a person heading a soccer ball",
    "a person doing soccer tricks",
    "multiple people playing soccer",
    "a person boxing",
    "a person kicking",
    "a person punching a bag",
    "a person doing a backflip",
    "a person doing a cartwheel",
    "a person swimming",
    "a person serving in tennis",
    "a person hitting a tennis ball",
    "a person doing mountain climbers",
    "a person doing high knees",
]

LABEL_SIMPLIFY = {
    "a person dribbling a basketball": "dribbling basketball",
    "a person shooting a basketball": "shooting basketball",
    "a person doing a layup": "shooting basketball",
    "a person dunking a basketball": "dunking basketball",
    "a person running": "running",
    "a person jogging": "running",
    "a person walking": "walking",
    "a person standing still": "standing/resting",
    "a person resting": "standing/resting",
    "a person doing pushups": "pushups",
    "a person doing squats": "squats",
    "a person doing pullups on a bar": "pull-ups",
    "a person doing lunges": "lunges",
    "a person doing burpees": "burpees",
    "a person doing jumping jacks": "jumping jacks",
    "a person doing a plank": "plank",
    "a person stretching": "stretching",
    "a person doing yoga": "yoga/stretching",
    "a person lifting weights": "weightlifting",
    "a person doing bench press": "bench press",
    "a person doing deadlift": "deadlift",
    "a person doing barbell curls": "barbell curls",
    "a person doing overhead press": "overhead press",
    "a person doing rows": "rows",
    "a person doing kettlebell swings": "kettlebell swings",
    "a person doing clean and jerk": "clean and jerk",
    "a person doing snatch": "snatch",
    "a person jumping rope": "jump rope",
    "a person doing box jumps": "box jumps",
    "a person doing wall balls": "wall balls",
    "a person doing muscle ups": "muscle ups",
    "a person doing handstand pushups": "handstand pushups",
    "a person doing rope climbs": "rope climbs",
    "a person doing double unders": "double unders/jump rope",
    "a person dribbling a soccer ball": "dribbling soccer ball",
    "a person kicking a soccer ball": "kicking soccer ball",
    "a person heading a soccer ball": "heading soccer ball",
    "a person doing soccer tricks": "soccer tricks",
    "multiple people playing soccer": "soccer match play",
    "a person boxing": "boxing/striking",
    "a person kicking": "kicking",
    "a person punching a bag": "punching bag",
    "a person doing a backflip": "acrobatics",
    "a person doing a cartwheel": "acrobatics",
    "a person swimming": "swimming",
    "a person serving in tennis": "tennis serve",
    "a person hitting a tennis ball": "tennis hitting",
    "a person doing mountain climbers": "mountain climbers",
    "a person doing high knees": "high knees",
}


def load_clip_model():
    """Load CLIP model for zero-shot classification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


def extract_features_and_frames(video_path, yolo_model):
    """Extract SRP features AND store raw frames for CLIP classification.

    Returns: (features, valid_indices, fps, raw_frames_dict)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    features, valid_indices, raw_frames = [], [], {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Store raw frame (RGB) for CLIP classification later
        raw_frames[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = yolo_model(frame, verbose=False)
        if results and len(results[0].keypoints) > 0:
            kp = results[0].keypoints[0]
            xy = kp.xy.cpu().numpy()[0]      # (17, 2)
            conf = kp.conf.cpu().numpy()[0]  # (17,)
            kp_wc = np.column_stack([xy, conf])  # (17, 3)
            mp33 = coco_keypoints_to_landmarks(kp_wc, img_w, img_h)
            feat = normalize_frame(mp33)
            if feat is not None:
                features.append(feat[FEATURE_INDICES, :2].flatten())
                valid_indices.append(frame_idx)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    Frame {frame_idx}/{total}...")

    cap.release()

    if not features:
        return None, [], fps, raw_frames

    return np.stack(features), valid_indices, fps, raw_frames


def classify_cluster_frames(frames_list, clip_model, preprocess, text_features, device):
    """Classify a set of frames using CLIP. Return top label and confidence."""
    if not frames_list:
        return "unknown", 0.0, {}

    # Sample up to 8 frames evenly spaced
    n = len(frames_list)
    indices = np.linspace(0, n - 1, min(8, n), dtype=int)
    sampled = [frames_list[i] for i in indices]

    # Encode all frames
    images = torch.stack([preprocess(Image.fromarray(f)) for f in sampled]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Average frame features, then compare to text
    avg_feat = image_features.mean(dim=0, keepdim=True)
    avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
    similarity = (avg_feat @ text_features.T).squeeze()

    # Get top-3 for debugging
    probs = (similarity * 100.0).softmax(dim=-1)
    top3_idx = probs.topk(3).indices.tolist()
    top3 = {LABEL_SIMPLIFY.get(ACTION_LABELS[i], ACTION_LABELS[i]): round(probs[i].item(), 3) for i in top3_idx}

    best_idx = probs.argmax().item()
    raw_label = ACTION_LABELS[best_idx]
    label = LABEL_SIMPLIFY.get(raw_label, raw_label)
    conf = float(probs[best_idx])

    return label, conf, top3


def main():
    # Load ground truth
    gt = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH) as f:
            gt = json.load(f)
        print(f"Loaded ground truth for {len(gt)} videos")

    # Load models
    print("Loading YOLO-pose model...")
    yolo_model = YOLO("yolo11n-pose.pt")

    print("Loading CLIP model...")
    clip_model, preprocess, tokenizer, device = load_clip_model()

    # Pre-encode text labels
    text_tokens = tokenizer(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Models loaded on {device}\n")

    results = {}
    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    print(f"Found {len(videos)} videos\n")

    for fname in videos:
        path = os.path.join(VIDEO_DIR, fname)
        print(f"{'='*60}")
        print(f"Processing: {fname}")

        # 1. Extract features + raw frames
        feats, vi, fps, raw_frames = extract_features_and_frames(path, yolo_model)
        if feats is None or len(feats) < 10:
            print(f"  Not enough features ({0 if feats is None else len(feats)}). Skipping.")
            results[fname] = {"error": "insufficient features", "n_features": 0}
            continue

        print(f"  Features: {feats.shape[0]}, FPS: {fps:.1f}")

        # 2. Segment and cluster
        segments = segment_motions(feats, vi, fps, min_segment_sec=1.0)
        if len(segments) < 2:
            segments = segment_motions(feats, vi, fps, min_segment_sec=0.5)

        print(f"  Segments before clustering: {len(segments)}")

        if len(segments) >= 2:
            clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0

        n_clusters = len(set(s["cluster"] for s in clustered))
        print(f"  Clusters: {n_clusters}")

        # 3. Get frames per cluster
        cluster_frames_map = {}
        for seg in clustered:
            cid = seg["cluster"]
            if cid not in cluster_frames_map:
                cluster_frames_map[cid] = []
            for fi in range(seg["start_frame"], seg["end_frame"] + 1):
                if fi in raw_frames:
                    cluster_frames_map[cid].append(raw_frames[fi])

        # 4. Classify each cluster with CLIP
        cluster_labels = {}
        for cid in sorted(cluster_frames_map.keys()):
            frames = cluster_frames_map[cid]
            label, conf, top3 = classify_cluster_frames(
                frames, clip_model, preprocess, text_features, device
            )
            cluster_labels[cid] = {
                "label": label,
                "confidence": round(conf, 3),
                "n_frames": len(frames),
                "top3": top3,
            }
            print(f"  Cluster {cid}: '{label}' (conf={conf:.3f}, {len(frames)} frames)")

        # 5. Validate: check for duplicate labels across clusters
        labels_only = [v["label"] for v in cluster_labels.values()]
        unique_labels = set(labels_only)
        duplicate_labels = len(labels_only) != len(unique_labels)

        failure_reason = None
        if duplicate_labels:
            from collections import Counter
            label_counts = Counter(labels_only)
            dups = [l for l, c in label_counts.items() if c > 1]
            dup_clusters = []
            for l in dups:
                cids = [str(cid) for cid, v in cluster_labels.items() if v["label"] == l]
                dup_clusters.append(f"clusters {','.join(cids)} share label '{l}'")
            failure_reason = "OVER-SEGMENTED: " + "; ".join(dup_clusters)
            print(f"  *** FAIL: {failure_reason}")
        else:
            print(f"  PASS: all {n_clusters} clusters have distinct labels")

        # 6. Compare to ground truth
        gt_info = gt.get(fname, {})
        gt_expected = gt_info.get("expected_clusters", None)
        gt_activities = gt_info.get("distinct_activities", [])
        gt_match = None
        if gt_expected is not None:
            gt_match = n_clusters == gt_expected
            if gt_match:
                print(f"  GT MATCH: {n_clusters} clusters == expected {gt_expected}")
            else:
                print(f"  GT MISMATCH: {n_clusters} clusters != expected {gt_expected}")

        # Segment detail
        seg_info = []
        for i, seg in enumerate(clustered):
            dur = (seg["end_frame"] - seg["start_frame"]) / max(fps, 1)
            seg_info.append({
                "segment": i,
                "cluster": seg["cluster"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "duration_sec": round(dur, 2),
            })

        results[fname] = {
            "n_features": int(feats.shape[0]),
            "n_segments": len(clustered),
            "n_clusters": n_clusters,
            "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
            "all_labels_distinct": not duplicate_labels,
            "failure_reason": failure_reason,
            "gt_expected_clusters": gt_expected,
            "gt_cluster_match": gt_match,
            "gt_activities": gt_activities,
            "segments": seg_info,
        }

        # Free frames memory
        del raw_frames

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = sum(1 for v in results.values() if "error" not in v)
    passed = sum(1 for v in results.values() if v.get("all_labels_distinct"))
    failed = total - passed
    gt_matched = sum(1 for v in results.values() if v.get("gt_cluster_match") is True)
    gt_total = sum(1 for v in results.values() if v.get("gt_cluster_match") is not None)

    print(f"  Total videos processed: {total}")
    print(f"  Distinct-labels PASS: {passed}")
    print(f"  Distinct-labels FAIL: {failed}")
    print(f"  Ground-truth cluster count match: {gt_matched}/{gt_total}")

    print(f"\n  Per-video results:")
    for fname in sorted(results.keys()):
        r = results[fname]
        if "error" in r:
            print(f"    SKIP  {fname}: {r['error']}")
            continue
        status = "PASS" if r["all_labels_distinct"] else "FAIL"
        gt_str = ""
        if r.get("gt_cluster_match") is not None:
            gt_str = f", GT={'MATCH' if r['gt_cluster_match'] else 'MISMATCH'}"
            gt_str += f" (got {r['n_clusters']} vs expected {r['gt_expected_clusters']})"
        labels = [v["label"] for v in r["cluster_labels"].values()]
        print(f"    {status}  {fname}: {r['n_clusters']} clusters, labels={labels}{gt_str}")
        if r.get("failure_reason"):
            print(f"          Reason: {r['failure_reason']}")


if __name__ == "__main__":
    main()
