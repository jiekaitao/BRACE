#!/usr/bin/env python3
"""End-to-end validation combining all 3 agents' best findings.

Fixes applied:
1. Segmentation: 3x smoothing kernel (fps*0.9), prominence 1.0, min_segment_sec 2.0
2. Clustering: average linkage + small cluster post-merge (<2 segs or <3s)
3. Labels: CLIP ViT-L/14 + expanded 61-label sport-specific vocabulary
"""
import sys, os, json, copy
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from brace.core.motion_segments import (
    _segment_distance, _resample_segment, _merge_adjacent_clusters,
    segment_motions, cluster_segments,
)

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"

# Expanded 61-label vocabulary (from label-researcher findings)
EXPANDED_LABELS = [
    # Running/walking
    "a person running at full speed", "a person jogging at a moderate pace",
    "a person walking", "a person sprinting",
    # Basketball
    "a person dribbling a basketball while stationary",
    "a person dribbling a basketball while moving",
    "a person shooting a basketball jump shot",
    "a person doing a basketball layup", "a person dunking a basketball",
    "a person passing a basketball", "a person defending in basketball",
    # General fitness
    "a person doing pushups", "a person doing squats",
    "a person doing pullups", "a person doing burpees",
    "a person doing jumping jacks", "a person doing lunges",
    "a person doing mountain climbers", "a person doing high knees",
    "a person doing box jumps", "a person doing planks",
    "a person doing crunches", "a person doing stretching",
    # Weight training
    "a person lifting weights overhead",
    "a person doing bench press", "a person doing deadlifts",
    "a person doing barbell curls", "a person doing rows",
    "a person doing kettlebell swings", "a person doing clean and jerk",
    "a person doing lateral raises", "a person doing tricep extensions",
    # Boxing/martial arts
    "a person shadowboxing", "a person punching a heavy bag",
    "a person doing boxing footwork", "a person in a fighting stance",
    # Jump rope
    "a person jumping rope with both feet",
    "a person doing double unders with a jump rope",
    # Swimming
    "a person swimming freestyle", "a person swimming backstroke",
    "a person swimming breaststroke", "a person doing a flip turn",
    # Tennis
    "a person hitting a tennis forehand", "a person hitting a tennis backhand",
    "a person serving in tennis", "a person doing a tennis volley",
    "a person doing a tennis overhead smash",
    "a person doing a tennis split step ready position",
    # Soccer
    "a person dribbling a soccer ball",
    "a person kicking a soccer ball",
    "a person doing soccer juggling tricks",
    "a person doing soccer footwork drills",
    # Yoga
    "a person doing a yoga standing pose",
    "a person doing a yoga balance pose",
    "a person doing a yoga seated pose",
    "a person doing a yoga inversion",
    "a person transitioning between yoga poses",
    # Calisthenics
    "a person doing bodyweight dips",
    "a person doing handstand practice",
    "a person doing muscle ups",
    # General
    "a person standing still", "a person resting between exercises",
    "a person warming up",
]


def detect_motion_boundaries_improved(
    features: np.ndarray,
    fps: float = 24.0,
    min_segment_sec: float = 2.0,
) -> list[int]:
    """Improved boundary detection: wider smoothing + higher prominence threshold."""
    n = features.shape[0]
    if n < 5:
        return [0]

    velocity = np.zeros(n)
    for i in range(1, n):
        velocity[i] = float(np.linalg.norm(features[i] - features[i - 1]))

    # FIX 1: 3x wider smoothing kernel (fps*0.9 instead of fps*0.3)
    kernel_size = max(5, int(fps * 0.9))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(velocity, kernel, mode="same")

    # FIX 2: Higher min_segment_sec (2.0 instead of 1.0)
    min_frames = max(int(fps * min_segment_sec), 5)
    positive_vals = smoothed[smoothed > 0]
    if len(positive_vals) == 0:
        return [0]

    median_vel = float(np.median(positive_vals))
    # FIX 3: Higher prominence threshold (1.0 instead of 0.5)
    min_prominence = median_vel * 1.0

    inv_smooth = -smoothed
    peaks, _ = find_peaks(inv_smooth, distance=min_frames, prominence=min_prominence)

    boundaries = [0]
    for p in peaks:
        if p - boundaries[-1] >= min_frames:
            boundaries.append(int(p))

    return boundaries


def segment_motions_improved(features, valid_indices, fps=24.0, min_segment_sec=2.0):
    """Segment with improved boundary detection."""
    boundaries = detect_motion_boundaries_improved(features, fps, min_segment_sec)

    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else features.shape[0]
        if end - start < 3:
            continue
        seg_features = features[start:end]
        segments.append({
            "start_valid": start,
            "end_valid": end,
            "start_frame": valid_indices[start] if start < len(valid_indices) else 0,
            "end_frame": valid_indices[end - 1] if end - 1 < len(valid_indices) else valid_indices[-1],
            "features": seg_features,
            "mean_feature": seg_features.mean(axis=0),
        })
    return segments


def cluster_segments_improved(segments, distance_threshold=2.0):
    """Cluster with average linkage + small cluster post-merge."""
    if not segments:
        return segments
    n = len(segments)
    if n == 1:
        segments[0]["cluster"] = 0
        return segments

    # Build pairwise distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _segment_distance(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # FIX: Use average linkage instead of single
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Merge adjacent same-cluster segments
    segments = _merge_adjacent_clusters(segments)

    # FIX: Small cluster post-merge
    segments = _merge_small_clusters(segments, dist_matrix=None, min_segments=2, min_seconds=3.0, fps=30.0)

    return segments


def _merge_small_clusters(segments, dist_matrix=None, min_segments=2, min_seconds=3.0, fps=30.0):
    """Merge small clusters (<min_segments or <min_seconds) into nearest neighbor."""
    if len(segments) <= 1:
        return segments

    # Count segments and duration per cluster
    cluster_info = {}
    for seg in segments:
        cid = seg["cluster"]
        if cid not in cluster_info:
            cluster_info[cid] = {"count": 0, "total_frames": 0, "segments": []}
        cluster_info[cid]["count"] += 1
        cluster_info[cid]["total_frames"] += seg["end_frame"] - seg["start_frame"]
        cluster_info[cid]["segments"].append(seg)

    # Identify small clusters
    small_cids = set()
    for cid, info in cluster_info.items():
        duration_sec = info["total_frames"] / max(fps, 1)
        if info["count"] < min_segments and duration_sec < min_seconds:
            small_cids.add(cid)

    if not small_cids or len(small_cids) >= len(cluster_info):
        return segments  # Don't merge if ALL clusters are small

    # For each small cluster, find nearest large cluster by centroid distance
    large_cids = [cid for cid in cluster_info if cid not in small_cids]
    large_centroids = {}
    for cid in large_cids:
        feats = [s["mean_feature"] for s in cluster_info[cid]["segments"]]
        large_centroids[cid] = np.mean(feats, axis=0)

    merge_map = {}
    for small_cid in small_cids:
        small_centroid = np.mean(
            [s["mean_feature"] for s in cluster_info[small_cid]["segments"]], axis=0
        )
        best_cid = None
        best_dist = float("inf")
        for large_cid, centroid in large_centroids.items():
            d = float(np.linalg.norm(small_centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_cid = large_cid
        if best_cid is not None:
            merge_map[small_cid] = best_cid

    # Apply merge
    for seg in segments:
        if seg["cluster"] in merge_map:
            seg["cluster"] = merge_map[seg["cluster"]]

    # Re-merge adjacent same-cluster segments after small cluster merge
    segments = _merge_adjacent_clusters(segments)

    # Relabel to 0-indexed contiguous
    unique = sorted(set(s["cluster"] for s in segments))
    remap = {old: new for new, old in enumerate(unique)}
    for seg in segments:
        seg["cluster"] = remap[seg["cluster"]]

    return segments


def classify_cluster_with_clip(video_path, cluster_frame_ranges, clip_model, preprocess,
                                text_features, device, labels):
    """Classify cluster frames with CLIP. Memory-efficient: reads on-demand."""
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
    return labels[best_idx], float(similarity[best_idx])


def run_pipeline(video_path, feats, valid_indices, fps, mode, clip_model, preprocess,
                 text_features, device, labels):
    """Run segmentation + clustering + CLIP validation for a given mode."""
    if mode == "baseline":
        segments = segment_motions(feats, valid_indices, fps, min_segment_sec=1.0)
        if len(segments) >= 2:
            clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0
    elif mode == "improved":
        segments = segment_motions_improved(feats, valid_indices, fps, min_segment_sec=2.0)
        if len(segments) >= 2:
            clustered = cluster_segments_improved(copy.deepcopy(segments), distance_threshold=2.0)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0
    elif mode == "improved_3s":
        segments = segment_motions_improved(feats, valid_indices, fps, min_segment_sec=3.0)
        if len(segments) >= 2:
            clustered = cluster_segments_improved(copy.deepcopy(segments), distance_threshold=2.0)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_clusters = len(set(s["cluster"] for s in clustered))

    # CLIP classify each cluster
    cluster_ranges = {}
    for seg in clustered:
        cid = seg["cluster"]
        cluster_ranges.setdefault(cid, []).append(
            (seg["start_frame"], seg["end_frame"])
        )

    cluster_labels = {}
    for cid, ranges in sorted(cluster_ranges.items()):
        label, conf = classify_cluster_with_clip(
            video_path, ranges, clip_model, preprocess, text_features, device, labels
        )
        n_frames = sum(e - s + 1 for s, e in ranges)
        cluster_labels[cid] = {"label": label, "confidence": round(conf, 3), "n_frames": n_frames}

    labels_only = [v["label"] for v in cluster_labels.values()]
    unique_labels = set(labels_only)
    all_distinct = len(labels_only) == len(unique_labels)

    # CLIP post-merge: merge clusters with same label
    if not all_distinct and n_clusters > 1:
        label_to_cids = {}
        for cid, info in cluster_labels.items():
            label_to_cids.setdefault(info["label"], []).append(cid)
        merge_map = {}
        for label, cids in label_to_cids.items():
            if len(cids) > 1:
                target = min(cids)
                for cid in cids:
                    merge_map[cid] = target
        merged_segs = copy.deepcopy(clustered)
        for seg in merged_segs:
            old_cid = seg["cluster"]
            if old_cid in merge_map:
                seg["cluster"] = merge_map[old_cid]
        merged_segs = _merge_adjacent_clusters(merged_segs)
        n_after_merge = len(set(s["cluster"] for s in merged_segs))
    else:
        n_after_merge = n_clusters

    return {
        "n_segments": len(clustered),
        "n_clusters": n_clusters,
        "n_clusters_after_clip_merge": n_after_merge,
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "all_distinct": all_distinct,
    }


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load CLIP ViT-L/14 (better model from label-researcher findings)
    print("Loading CLIP ViT-L/14...", flush=True)
    clip_model_l14, preprocess_l14 = clip.load("ViT-L/14", device=device)
    text_tokens_expanded = clip.tokenize(EXPANDED_LABELS).to(device)
    with torch.no_grad():
        text_features_expanded = clip_model_l14.encode_text(text_tokens_expanded)
        text_features_expanded = text_features_expanded / text_features_expanded.norm(dim=-1, keepdim=True)

    # Also load ViT-B/32 for baseline comparison
    print("Loading CLIP ViT-B/32...", flush=True)
    clip_model_b32, preprocess_b32 = clip.load("ViT-B/32", device=device)
    from experiments.threshold_sweep_v2 import ACTION_LABELS as BASELINE_LABELS
    text_tokens_baseline = clip.tokenize(BASELINE_LABELS).to(device)
    with torch.no_grad():
        text_features_baseline = clip_model_b32.encode_text(text_tokens_baseline)
        text_features_baseline = text_features_baseline / text_features_baseline.norm(dim=-1, keepdim=True)

    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    print(f"Videos: {len(videos)}\n", flush=True)

    # Load cached features
    video_data = {}
    for fname in videos:
        cache_path = os.path.join(CACHE_DIR, f"{fname}.feats.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            feats = data["features"]
            vi = data["valid_indices"].tolist()
            fps = float(data["fps"])
            if len(feats) >= 10:
                path = os.path.join(VIDEO_DIR, fname)
                video_data[fname] = (path, feats, vi, fps)

    print(f"Loaded features for {len(video_data)} videos\n", flush=True)

    # Run 4 configurations:
    # A: baseline segmentation + baseline clustering + baseline CLIP (ViT-B/32 + 41 labels)
    # B: improved segmentation + baseline clustering + baseline CLIP
    # C: baseline segmentation + improved clustering + baseline CLIP
    # D: improved segmentation + improved clustering + improved CLIP (ViT-L/14 + 61 labels)
    configs = {
        "D_all_improved": {
            "mode": "improved",
            "clip_model": clip_model_l14, "preprocess": preprocess_l14,
            "text_features": text_features_expanded, "labels": EXPANDED_LABELS,
        },
        "E_improved_3s": {
            "mode": "improved_3s",
            "clip_model": clip_model_l14, "preprocess": preprocess_l14,
            "text_features": text_features_expanded, "labels": EXPANDED_LABELS,
        },
    }

    all_results = {}
    for config_name, cfg in configs.items():
        print(f"{'='*70}", flush=True)
        print(f"CONFIG: {config_name}", flush=True)
        print(f"{'='*70}", flush=True)

        config_results = {}
        pass_count = 0

        for fname, (path, feats, vi, fps) in sorted(video_data.items()):
            result = run_pipeline(
                path, feats, vi, fps, cfg["mode"],
                cfg["clip_model"], cfg["preprocess"], cfg["text_features"],
                device, cfg["labels"],
            )
            config_results[fname] = result

            nc = result["n_clusters"]
            labels = [v["label"] for v in result["cluster_labels"].values()]
            distinct = result["all_distinct"]
            status = "PASS" if distinct else "FAIL"
            if distinct:
                pass_count += 1

            # Truncate labels for display
            short_labels = [l.replace("a person ", "").replace("a person doing ", "")[:30] for l in labels]
            print(f"  {status:4s} {fname:35s} segs={result['n_segments']:2d} clust={nc:2d} {short_labels}", flush=True)

        total = len(video_data)
        print(f"\n  {config_name}: {pass_count}/{total} pass ({100*pass_count/total:.0f}%)\n", flush=True)
        all_results[config_name] = {
            "pass_count": pass_count,
            "total": total,
            "videos": config_results,
        }

    # Save results
    out_path = "/mnt/Data/GitHub/BRACE/experiments/final_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("FINAL COMPARISON", flush=True)
    print(f"{'='*70}", flush=True)
    for name, r in all_results.items():
        print(f"  {name:30s}: {r['pass_count']}/{r['total']} pass", flush=True)

        # Per-video cluster counts
        for fname, vr in sorted(r["videos"].items()):
            labels = [v["label"] for v in vr["cluster_labels"].values()]
            print(f"    {fname:35s} {vr['n_clusters']:2d} clusters, distinct={vr['all_distinct']}", flush=True)


if __name__ == "__main__":
    main()
