#!/usr/bin/env python3
"""Sweep clustering thresholds across all demo videos and validate with CLIP.

Memory-efficient: caches only pose features (small), reads frames on-demand for CLIP.
Also tests CLIP post-merge: cluster at low threshold, then merge clusters sharing a label.
"""
import sys, os, json, copy, time
import cv2
import numpy as np
import torch

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from ultralytics import YOLO
from brace.core.motion_segments import (
    normalize_frame, segment_motions, cluster_segments, _segment_distance,
)
from brace.core.pose import FEATURE_INDICES, coco_keypoints_to_landmarks

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ACTION_LABELS = [
    "a person running", "a person jogging", "a person walking",
    "a person dribbling a basketball", "a person shooting a basketball",
    "a person doing a layup", "a person dunking a basketball",
    "a person doing pushups", "a person doing squats", "a person doing pullups",
    "a person lifting weights", "a person doing bench press",
    "a person jumping rope", "a person stretching", "a person doing yoga",
    "a person boxing", "a person punching a bag", "a person shadowboxing",
    "a person swimming", "a person serving in tennis", "a person hitting a tennis ball",
    "a person dribbling a soccer ball", "a person kicking a soccer ball",
    "a person standing still", "a person resting",
    "a person doing burpees", "a person doing jumping jacks",
    "a person doing lunges", "a person doing kettlebell swings",
    "a person doing overhead press", "a person doing rows",
    "a person doing clean and jerk", "a person doing crunches",
    "a person doing planks", "a person doing deadlifts",
    "a person doing barbell curls", "a person doing tricep extensions",
    "a person doing lateral raises", "a person doing mountain climbers",
    "a person doing high knees", "a person doing box jumps",
]

THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]


def extract_and_cache_features(video_path: str) -> tuple:
    """Extract pose features, cache to disk. Returns (features, valid_indices, fps)."""
    fname = os.path.basename(video_path)
    cache_path = os.path.join(CACHE_DIR, f"{fname}.feats.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["features"], data["valid_indices"].tolist(), float(data["fps"])

    print(f"  Extracting features from {fname}...", flush=True)
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    features, valid_indices = [], []
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
            kp_wc = np.column_stack([xy, conf])
            mp33 = coco_keypoints_to_landmarks(kp_wc, img_w, img_h)
            feat = normalize_frame(mp33)
            if feat is not None:
                features.append(feat[FEATURE_INDICES, :2].flatten())
                valid_indices.append(frame_idx)
        frame_idx += 1
    cap.release()

    if not features:
        return None, [], fps

    feats = np.stack(features)
    np.savez_compressed(cache_path, features=feats,
                        valid_indices=np.array(valid_indices), fps=np.array(fps))
    print(f"  Cached {feats.shape[0]} features ({feats.shape[1]}D)", flush=True)
    return feats, valid_indices, fps


def classify_cluster_with_clip(video_path, cluster_frame_ranges, clip_model, preprocess,
                                text_features, device):
    """Read specific frame ranges from video and classify with CLIP.
    Samples ~8 frames total across all ranges. Memory-efficient: no caching."""
    from PIL import Image

    total_frames = sum(e - s + 1 for s, e in cluster_frame_ranges)
    n_sample = min(8, total_frames)

    all_frame_idxs = []
    for s, e in cluster_frame_ranges:
        all_frame_idxs.extend(range(s, e + 1))

    if len(all_frame_idxs) == 0:
        return "unknown", 0.0

    sample_idxs = sorted(set(
        all_frame_idxs[i]
        for i in np.linspace(0, len(all_frame_idxs) - 1, n_sample, dtype=int)
    ))

    cap = cv2.VideoCapture(video_path)
    sampled_frames = []
    frame_idx = 0
    sample_set = set(sample_idxs)
    max_idx = max(sample_idxs)

    while cap.isOpened() and frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(rgb)
        frame_idx += 1
    cap.release()

    if not sampled_frames:
        return "unknown", 0.0

    images = torch.stack([preprocess(Image.fromarray(f)) for f in sampled_frames]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    avg_feat = image_features.mean(dim=0, keepdim=True)
    avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
    similarity = (avg_feat @ text_features.T).squeeze()

    best_idx = similarity.argmax().item()
    return ACTION_LABELS[best_idx], float(similarity[best_idx])


def clip_post_merge(clustered_segments, cluster_labels):
    """Merge clusters that share the same CLIP label."""
    label_to_cids = {}
    for cid, info in cluster_labels.items():
        label = info["label"]
        label_to_cids.setdefault(label, []).append(cid)

    merge_map = {}
    for label, cids in label_to_cids.items():
        if len(cids) > 1:
            target = min(cids)
            for cid in cids:
                merge_map[cid] = target

    if not merge_map:
        return clustered_segments, cluster_labels

    merged_segments = copy.deepcopy(clustered_segments)
    for seg in merged_segments:
        old_cid = seg["cluster"]
        if old_cid in merge_map:
            seg["cluster"] = merge_map[old_cid]

    new_labels = {}
    for cid, info in cluster_labels.items():
        target = merge_map.get(cid, cid)
        if target not in new_labels:
            new_labels[target] = info

    return merged_segments, new_labels


def validate_video(video_path, feats, valid_indices, fps, threshold,
                   clip_model, preprocess, text_features, device):
    """Run full pipeline: segment -> cluster -> CLIP classify -> validate."""
    segments = segment_motions(feats, valid_indices, fps, min_segment_sec=1.0)
    if len(segments) < 2:
        segments = segment_motions(feats, valid_indices, fps, min_segment_sec=0.5)

    if len(segments) >= 2:
        clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=threshold)
    else:
        clustered = segments
        for s in clustered:
            s["cluster"] = 0

    n_clusters = len(set(s["cluster"] for s in clustered))

    cluster_ranges = {}
    for seg in clustered:
        cid = seg["cluster"]
        cluster_ranges.setdefault(cid, []).append(
            (seg["start_frame"], seg["end_frame"])
        )

    cluster_labels = {}
    for cid, ranges in sorted(cluster_ranges.items()):
        label, conf = classify_cluster_with_clip(
            video_path, ranges, clip_model, preprocess, text_features, device
        )
        n_frames = sum(e - s + 1 for s, e in ranges)
        cluster_labels[cid] = {"label": label, "confidence": round(conf, 3), "n_frames": n_frames}

    labels_only = [v["label"] for v in cluster_labels.values()]
    unique_labels = set(labels_only)
    all_distinct = len(labels_only) == len(unique_labels)

    merged_segs, merged_labels = clip_post_merge(clustered, cluster_labels)
    n_clusters_after_merge = len(set(s["cluster"] for s in merged_segs))
    merged_labels_only = [v["label"] for v in merged_labels.values()]
    merged_distinct = len(merged_labels_only) == len(set(merged_labels_only))

    return {
        "n_segments": len(clustered),
        "n_clusters": n_clusters,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "all_distinct": all_distinct,
        "n_clusters_after_clip_merge": n_clusters_after_merge,
        "merged_labels": {str(k): v for k, v in merged_labels.items()},
        "merged_distinct": merged_distinct,
    }


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    print(f"Videos: {len(videos)}", flush=True)

    # Phase 1: Extract all features (expensive, cached)
    video_data = {}
    for fname in videos:
        path = os.path.join(VIDEO_DIR, fname)
        feats, vi, fps = extract_and_cache_features(path)
        if feats is not None and len(feats) >= 10:
            video_data[fname] = (path, feats, vi, fps)
            print(f"  {fname}: {feats.shape[0]} features", flush=True)
        else:
            print(f"  {fname}: SKIPPED (insufficient features)", flush=True)

    # Phase 2: Sweep thresholds
    all_results = {}
    for threshold in THRESHOLDS:
        print(f"\n{'='*70}", flush=True)
        print(f"THRESHOLD = {threshold}", flush=True)
        print(f"{'='*70}", flush=True)

        threshold_results = {}
        pass_count = 0
        merge_pass_count = 0

        for fname, (path, feats, vi, fps) in sorted(video_data.items()):
            result = validate_video(
                path, feats, vi, fps, threshold,
                clip_model, preprocess, text_features, device
            )
            threshold_results[fname] = result

            nc = result["n_clusters"]
            labels = [v["label"] for v in result["cluster_labels"].values()]
            distinct = result["all_distinct"]
            nc_merged = result["n_clusters_after_clip_merge"]
            m_distinct = result["merged_distinct"]

            status = "PASS" if distinct else "FAIL"
            m_status = "PASS" if m_distinct else "FAIL"
            if distinct:
                pass_count += 1
            if m_distinct:
                merge_pass_count += 1

            print(f"  {status:4s} {fname:30s} clusters={nc:2d} labels={labels}", flush=True)
            if not distinct:
                print(f"       -> CLIP-merge: {m_status} clusters={nc_merged} labels={[v['label'] for v in result['merged_labels'].values()]}", flush=True)

        total = len(video_data)
        print(f"\n  THRESHOLD {threshold}: {pass_count}/{total} pass, {merge_pass_count}/{total} pass after CLIP-merge", flush=True)

        all_results[str(threshold)] = {
            "pass_count": pass_count,
            "merge_pass_count": merge_pass_count,
            "total": total,
            "videos": threshold_results,
        }

    # Save results
    out_path = "/mnt/Data/GitHub/BRACE/experiments/threshold_sweep_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Threshold':>10} {'Raw Pass':>10} {'CLIP-Merge Pass':>16} {'Total':>6}", flush=True)
    for t in THRESHOLDS:
        r = all_results[str(t)]
        print(f"  {t:>10.1f} {r['pass_count']:>10}/{r['total']} {r['merge_pass_count']:>13}/{r['total']} {r['total']:>6}", flush=True)


if __name__ == "__main__":
    main()
