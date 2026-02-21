#!/usr/bin/env python3
"""Test clustering optimization strategies to reduce false over-segmentation.

Tests:
  a) Linkage methods: complete, average, ward (vs current single)
  b) Distance metric variants: mean-only, FFT-only, different weightings
  c) Minimum cluster size post-merge
  d) Two-stage clustering (fine then coarse)
  e) Adjacent-distance post-merge
  f) CLIP validation for best configs

Run: .venv/bin/python experiments/test_clustering_optimization.py
"""
import sys
import os
import copy
import json
import time
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    _segment_distance,
    _resample_segment,
    _merge_adjacent_clusters,
)

CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"

# ---------------------------------------------------------------------------
# Distance metric variants
# ---------------------------------------------------------------------------

def distance_mean_only(seg_a, seg_b, resample_len=30):
    """Only mean pose distance, no FFT."""
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    mean_dist = float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0)))
    return mean_dist / np.sqrt(feat_dim)


def distance_fft_only(seg_a, seg_b, resample_len=30):
    """Only FFT power spectrum distance, no mean pose."""
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    spec_a = np.abs(np.fft.rfft(ra, axis=0))[1:] / resample_len
    spec_b = np.abs(np.fft.rfft(rb, axis=0))[1:] / resample_len
    spec_dist = float(np.linalg.norm(spec_a - spec_b))
    return spec_dist / np.sqrt(feat_dim)


def distance_weighted(seg_a, seg_b, mean_weight=0.7, fft_weight=0.3, resample_len=30):
    """Weighted combination of mean pose and FFT distances."""
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    mean_dist = float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0)))
    spec_a = np.abs(np.fft.rfft(ra, axis=0))[1:] / resample_len
    spec_b = np.abs(np.fft.rfft(rb, axis=0))[1:] / resample_len
    spec_dist = float(np.linalg.norm(spec_a - spec_b))
    return (mean_weight * mean_dist + fft_weight * spec_dist) / np.sqrt(feat_dim)


def distance_70_30(a, b):
    return distance_weighted(a, b, mean_weight=0.7, fft_weight=0.3)


def distance_30_70(a, b):
    return distance_weighted(a, b, mean_weight=0.3, fft_weight=0.7)


# ---------------------------------------------------------------------------
# Custom clustering with configurable linkage + distance
# ---------------------------------------------------------------------------

def custom_cluster_segments(segments, distance_fn, linkage_method, threshold):
    """Cluster segments with custom distance + linkage, then merge adjacent."""
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
            d = distance_fn(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Ward linkage requires Euclidean-like distances in condensed form.
    # If using ward, we need to ensure non-negative distances.
    condensed = squareform(dist_matrix)

    if linkage_method == "ward":
        # Ward needs non-negative condensed distances
        condensed = np.maximum(condensed, 0)

    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Merge adjacent same-cluster segments
    segments = _merge_adjacent_clusters(segments)
    return segments


# ---------------------------------------------------------------------------
# Post-processing: merge small clusters into nearest neighbor
# ---------------------------------------------------------------------------

def merge_small_clusters(segments, min_segments=2, min_seconds=3.0, fps=30.0):
    """Merge clusters with too few segments or too little total duration."""
    if len(segments) <= 1:
        return segments

    # Count per cluster
    cluster_stats = defaultdict(lambda: {"count": 0, "total_frames": 0, "seg_indices": []})
    for i, seg in enumerate(segments):
        cid = seg["cluster"]
        cluster_stats[cid]["count"] += 1
        n_frames = seg["features"].shape[0]
        cluster_stats[cid]["total_frames"] += n_frames
        cluster_stats[cid]["seg_indices"].append(i)

    # Identify small clusters
    small_cids = set()
    for cid, stats in cluster_stats.items():
        duration_sec = stats["total_frames"] / max(fps, 1.0)
        if stats["count"] < min_segments or duration_sec < min_seconds:
            small_cids.add(cid)

    if not small_cids:
        return segments

    # For each small cluster, find the nearest large cluster by mean distance
    # Compute cluster centroids (mean of mean_features)
    cluster_centroids = {}
    for cid, stats in cluster_stats.items():
        if cid not in small_cids:
            feats = [segments[i]["mean_feature"] for i in stats["seg_indices"]]
            cluster_centroids[cid] = np.mean(feats, axis=0)

    if not cluster_centroids:
        # All clusters are small, just merge everything to cluster 0
        for seg in segments:
            seg["cluster"] = 0
        return _merge_adjacent_clusters(segments)

    # Reassign small clusters
    for small_cid in small_cids:
        small_feats = [segments[i]["mean_feature"] for i in cluster_stats[small_cid]["seg_indices"]]
        small_centroid = np.mean(small_feats, axis=0)

        best_cid = None
        best_dist = float("inf")
        for cid, centroid in cluster_centroids.items():
            d = float(np.linalg.norm(small_centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_cid = cid

        for i in cluster_stats[small_cid]["seg_indices"]:
            segments[i]["cluster"] = best_cid

    # Re-merge adjacent
    return _merge_adjacent_clusters(segments)


# ---------------------------------------------------------------------------
# Two-stage clustering
# ---------------------------------------------------------------------------

def two_stage_cluster(segments, distance_fn, linkage_method, threshold_fine, threshold_coarse):
    """First cluster at fine threshold, then re-cluster cluster centroids at coarse threshold."""
    if not segments or len(segments) < 2:
        if segments:
            segments[0]["cluster"] = 0
        return segments

    # Stage 1: fine clustering
    segs = copy.deepcopy(segments)
    segs = custom_cluster_segments(segs, distance_fn, linkage_method, threshold_fine)

    # Build cluster centroids from stage 1
    clusters = defaultdict(list)
    for seg in segs:
        clusters[seg["cluster"]].append(seg)

    if len(clusters) <= 1:
        return segs

    # Create synthetic "cluster segments" for stage 2
    cluster_ids = sorted(clusters.keys())
    cluster_meta = []
    for cid in cluster_ids:
        members = clusters[cid]
        all_feats = np.vstack([m["features"] for m in members])
        cluster_meta.append({
            "features": all_feats,
            "mean_feature": all_feats.mean(axis=0),
            "original_cid": cid,
        })

    if len(cluster_meta) < 2:
        return segs

    # Stage 2: cluster the cluster centroids
    n = len(cluster_meta)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(cluster_meta[i], cluster_meta[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    if linkage_method == "ward":
        condensed = np.maximum(condensed, 0)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=threshold_coarse, criterion="distance")

    # Build mapping from old cluster id -> new cluster id
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cid_remap = {}
    for i, cm in enumerate(cluster_meta):
        cid_remap[cm["original_cid"]] = label_map[labels[i]]

    for seg in segs:
        seg["cluster"] = cid_remap[seg["cluster"]]

    return _merge_adjacent_clusters(segs)


# ---------------------------------------------------------------------------
# Adjacent-distance post-merge
# ---------------------------------------------------------------------------

def adjacent_distance_merge(segments, distance_fn, merge_threshold_factor=0.5):
    """Merge temporally adjacent segments in different clusters if very close in distance."""
    if len(segments) <= 1:
        return segments

    changed = True
    while changed:
        changed = False
        new_segs = [segments[0]]
        for i in range(1, len(segments)):
            prev = new_segs[-1]
            curr = segments[i]
            if prev["cluster"] != curr["cluster"]:
                d = distance_fn(prev, curr)
                # Merge if distance is below merge_threshold_factor * default threshold
                if d < merge_threshold_factor:
                    # Merge into prev's cluster
                    combined_features = np.vstack([prev["features"], curr["features"]])
                    prev["end_valid"] = curr.get("end_valid", prev.get("end_valid"))
                    prev["end_frame"] = curr.get("end_frame", prev.get("end_frame"))
                    prev["features"] = combined_features
                    prev["mean_feature"] = combined_features.mean(axis=0)
                    changed = True
                    continue
            new_segs.append(curr)
        segments = new_segs

    return segments


# ---------------------------------------------------------------------------
# Load cached features
# ---------------------------------------------------------------------------

def load_all_features():
    """Load all cached feature files."""
    video_data = {}
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".feats.npz"):
            continue
        data = np.load(os.path.join(CACHE_DIR, fname))
        video_name = fname.replace(".feats.npz", "")
        feats = data["features"]
        vi = data["valid_indices"].tolist()
        fps = float(data["fps"])
        video_data[video_name] = (feats, vi, fps)
    return video_data


# ---------------------------------------------------------------------------
# CLIP validation
# ---------------------------------------------------------------------------

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


def init_clip():
    """Initialize CLIP model and text features."""
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text_tokens = clip.tokenize(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features, device


def classify_cluster_with_clip(video_path, cluster_frame_ranges, clip_model, preprocess,
                                text_features, device):
    """Read frames and classify with CLIP. Returns (label, confidence)."""
    import torch
    import cv2
    from PIL import Image

    total_frames = sum(e - s + 1 for s, e in cluster_frame_ranges)
    n_sample = min(8, max(1, total_frames))

    all_frame_idxs = []
    for s, e in cluster_frame_ranges:
        all_frame_idxs.extend(range(s, e + 1))
    if not all_frame_idxs:
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


def validate_with_clip(video_name, segments, clip_model, preprocess, text_features, device):
    """Run CLIP validation on clustered segments. Returns (all_distinct, n_clusters, label_info)."""
    video_path = os.path.join(VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        return True, 0, {}

    cluster_ranges = defaultdict(list)
    for seg in segments:
        cid = seg["cluster"]
        cluster_ranges[cid].append((seg["start_frame"], seg["end_frame"]))

    n_clusters = len(cluster_ranges)
    cluster_labels = {}
    for cid, ranges in sorted(cluster_ranges.items()):
        label, conf = classify_cluster_with_clip(
            video_path, ranges, clip_model, preprocess, text_features, device
        )
        cluster_labels[cid] = {"label": label, "confidence": round(conf, 3)}

    labels_only = [v["label"] for v in cluster_labels.values()]
    all_distinct = len(labels_only) == len(set(labels_only))

    return all_distinct, n_clusters, cluster_labels


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------

def run_single_config(video_data, config_name, cluster_fn, threshold=2.0):
    """Run a clustering config across all videos. Returns per-video results."""
    results = {}
    for vname, (feats, vi, fps) in sorted(video_data.items()):
        segments = segment_motions(feats, vi, fps, min_segment_sec=1.0)
        if len(segments) < 2:
            segments = segment_motions(feats, vi, fps, min_segment_sec=0.5)

        if len(segments) >= 2:
            segs = copy.deepcopy(segments)
            clustered = cluster_fn(segs)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0

        n_clusters = len(set(s["cluster"] for s in clustered))
        results[vname] = {
            "n_segments": len(clustered),
            "n_clusters": n_clusters,
            "segments": clustered,
        }

    return results


def count_distinct_labels(video_results, clip_results):
    """Count how many videos pass (all cluster labels distinct)."""
    pass_count = 0
    for vname in video_results:
        if vname in clip_results and clip_results[vname]["all_distinct"]:
            pass_count += 1
    return pass_count


def main():
    print("=" * 80)
    print("CLUSTERING OPTIMIZATION EXPERIMENT")
    print("=" * 80)

    video_data = load_all_features()
    print(f"\nLoaded {len(video_data)} videos from cache")
    for vname, (feats, vi, fps) in sorted(video_data.items()):
        print(f"  {vname:40s} feats={feats.shape[0]:4d}x{feats.shape[1]}  fps={fps:.0f}")

    # -----------------------------------------------------------------------
    # Define all configurations to test
    # -----------------------------------------------------------------------

    THRESHOLD = 2.0

    configs = {}

    # (a) Linkage methods with default distance
    for method in ["single", "complete", "average", "ward"]:
        name = f"linkage_{method}"
        configs[name] = lambda segs, m=method: custom_cluster_segments(
            segs, _segment_distance, m, THRESHOLD
        )

    # (b) Distance metric variants with average linkage (better than single for this)
    for method in ["average", "complete", "ward"]:
        configs[f"mean_only_{method}"] = lambda segs, m=method: custom_cluster_segments(
            segs, distance_mean_only, m, THRESHOLD
        )
        configs[f"fft_only_{method}"] = lambda segs, m=method: custom_cluster_segments(
            segs, distance_fft_only, m, THRESHOLD
        )
        configs[f"w70_30_{method}"] = lambda segs, m=method: custom_cluster_segments(
            segs, distance_70_30, m, THRESHOLD
        )
        configs[f"w30_70_{method}"] = lambda segs, m=method: custom_cluster_segments(
            segs, distance_30_70, m, THRESHOLD
        )

    # (c) Minimum cluster size post-merge with different linkages
    for method in ["average", "complete", "ward"]:
        def make_minsize_fn(m):
            def fn(segs):
                clustered = custom_cluster_segments(segs, _segment_distance, m, THRESHOLD)
                return merge_small_clusters(clustered, min_segments=2, min_seconds=3.0)
            return fn
        configs[f"minsize_{method}"] = make_minsize_fn(method)

    # (d) Two-stage clustering
    for method in ["average", "complete", "ward"]:
        for coarse_t in [4.0, 5.0, 6.0, 8.0]:
            def make_twostage_fn(m, ct):
                def fn(segs):
                    return two_stage_cluster(segs, _segment_distance, m, THRESHOLD, ct)
                return fn
            configs[f"twostage_{method}_c{coarse_t:.0f}"] = make_twostage_fn(method, coarse_t)

    # (e) Adjacent-distance post-merge
    for method in ["average", "complete", "ward"]:
        for merge_t in [0.5, 1.0, 1.5]:
            def make_adjmerge_fn(m, mt):
                def fn(segs):
                    clustered = custom_cluster_segments(segs, _segment_distance, m, THRESHOLD)
                    return adjacent_distance_merge(clustered, _segment_distance, mt)
                return fn
            configs[f"adjmerge_{method}_m{merge_t:.1f}"] = make_adjmerge_fn(method, merge_t)

    # Combined: best-of-breed combinations
    # Average linkage + mean-only distance + small-cluster merge
    def combo_avg_meanonly_minsize(segs):
        clustered = custom_cluster_segments(segs, distance_mean_only, "average", THRESHOLD)
        return merge_small_clusters(clustered, min_segments=2, min_seconds=3.0)
    configs["combo_avg_meanonly_minsize"] = combo_avg_meanonly_minsize

    # Ward + default distance + small-cluster merge
    def combo_ward_default_minsize(segs):
        clustered = custom_cluster_segments(segs, _segment_distance, "ward", THRESHOLD)
        return merge_small_clusters(clustered, min_segments=2, min_seconds=3.0)
    configs["combo_ward_default_minsize"] = combo_ward_default_minsize

    # Complete + 70/30 weighting + adj merge
    def combo_complete_w70_adjmerge(segs):
        clustered = custom_cluster_segments(segs, distance_70_30, "complete", THRESHOLD)
        return adjacent_distance_merge(clustered, distance_70_30, 1.0)
    configs["combo_complete_w70_adjmerge"] = combo_complete_w70_adjmerge

    # Average linkage + two-stage + minsize
    def combo_avg_twostage_minsize(segs):
        clustered = two_stage_cluster(segs, _segment_distance, "average", THRESHOLD, 5.0)
        return merge_small_clusters(clustered, min_segments=2, min_seconds=3.0)
    configs["combo_avg_twostage_minsize"] = combo_avg_twostage_minsize

    # Ward + two-stage + minsize
    def combo_ward_twostage_minsize(segs):
        clustered = two_stage_cluster(segs, _segment_distance, "ward", THRESHOLD, 5.0)
        return merge_small_clusters(clustered, min_segments=2, min_seconds=3.0)
    configs["combo_ward_twostage_minsize"] = combo_ward_twostage_minsize

    # Higher thresholds with better linkage
    for t in [3.0, 4.0, 5.0, 6.0]:
        for method in ["average", "complete", "ward"]:
            def make_higher_t_fn(m, thr):
                def fn(segs):
                    return custom_cluster_segments(segs, _segment_distance, m, thr)
                return fn
            configs[f"higher_t{t:.0f}_{method}"] = make_higher_t_fn(method, t)

    # -----------------------------------------------------------------------
    # Phase 1: Run all configs (pose-only, no CLIP yet)
    # -----------------------------------------------------------------------

    print(f"\nRunning {len(configs)} configurations across {len(video_data)} videos...")
    print("-" * 80)

    all_results = {}
    for config_name, cluster_fn in sorted(configs.items()):
        results = run_single_config(video_data, config_name, cluster_fn)

        # Summary: total clusters across all videos
        total_clusters = sum(r["n_clusters"] for r in results.values())
        max_clusters = max(r["n_clusters"] for r in results.values())
        cluster_counts = {vn: r["n_clusters"] for vn, r in results.items()}

        all_results[config_name] = {
            "total_clusters": total_clusters,
            "max_clusters": max_clusters,
            "per_video": results,
            "cluster_counts": cluster_counts,
        }

    # -----------------------------------------------------------------------
    # Phase 2: Rank configs by total cluster count (fewer = less over-segmentation)
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS: CLUSTER COUNT RANKING (lower = less over-segmentation)")
    print("=" * 80)

    # Problem videos are the ones that over-segment badly
    problem_videos = ["gym_workout.mp4", "yoga_flow.mp4", "tennis_practice.mp4",
                      "soccer_skills.mp4", "gym_crossfit.mp4", "soccer_match2.mp4",
                      "swimming_laps.mp4", "running_track.mp4", "exercise.mp4"]

    ranked = sorted(all_results.items(), key=lambda x: x[1]["total_clusters"])

    print(f"\n{'Config':<45} {'Total':>6} {'Max':>5} | gym_wk yoga  tenns  socsk  cross  socm2  swim   run    exer")
    print("-" * 155)
    for name, data in ranked[:40]:
        cc = data["cluster_counts"]
        pvids = [
            cc.get("gym_workout.mp4", "?"),
            cc.get("yoga_flow.mp4", "?"),
            cc.get("tennis_practice.mp4", "?"),
            cc.get("soccer_skills.mp4", "?"),
            cc.get("gym_crossfit.mp4", "?"),
            cc.get("soccer_match2.mp4", "?"),
            cc.get("swimming_laps.mp4", "?"),
            cc.get("running_track.mp4", "?"),
            cc.get("exercise.mp4", "?"),
        ]
        pvids_str = "  ".join(f"{v:>5}" for v in pvids)
        print(f"  {name:<43} {data['total_clusters']:>6} {data['max_clusters']:>5} | {pvids_str}")

    # Baseline
    print("\n  --- BASELINE (production: single linkage, t=2.0) ---")
    bl = all_results.get("linkage_single", {})
    if bl:
        cc = bl["cluster_counts"]
        pvids = [cc.get(v, "?") for v in ["gym_workout.mp4", "yoga_flow.mp4", "tennis_practice.mp4",
                                            "soccer_skills.mp4", "gym_crossfit.mp4", "soccer_match2.mp4",
                                            "swimming_laps.mp4", "running_track.mp4", "exercise.mp4"]]
        pvids_str = "  ".join(f"{v:>5}" for v in pvids)
        print(f"  {'linkage_single':<43} {bl['total_clusters']:>6} {bl['max_clusters']:>5} | {pvids_str}")

    # -----------------------------------------------------------------------
    # Phase 3: CLIP validation on top 10 configs
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 2: CLIP VALIDATION ON TOP CONFIGS")
    print("=" * 80)

    clip_model, preprocess, text_features, device = init_clip()
    print(f"CLIP loaded on {device}")

    # Select top 15 by total clusters + the baseline
    top_configs = [name for name, _ in ranked[:15]]
    if "linkage_single" not in top_configs:
        top_configs.append("linkage_single")

    clip_results_all = {}
    for config_name in top_configs:
        data = all_results[config_name]
        per_video = data["per_video"]

        pass_count = 0
        clip_per_video = {}
        for vname, vdata in sorted(per_video.items()):
            segments = vdata["segments"]
            all_distinct, n_clusters, cluster_labels = validate_with_clip(
                vname, segments, clip_model, preprocess, text_features, device
            )
            clip_per_video[vname] = {
                "all_distinct": all_distinct,
                "n_clusters": n_clusters,
                "labels": {str(k): v["label"] for k, v in cluster_labels.items()},
            }
            if all_distinct:
                pass_count += 1

        clip_results_all[config_name] = {
            "pass_count": pass_count,
            "total": len(per_video),
            "per_video": clip_per_video,
        }

        print(f"  {config_name:<45} CLIP pass: {pass_count}/{len(per_video)}")

    # -----------------------------------------------------------------------
    # Phase 4: Detailed report of best configs
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 3: DETAILED RESULTS FOR BEST CONFIGS")
    print("=" * 80)

    # Sort by CLIP pass count descending, then total clusters ascending
    best_configs = sorted(
        clip_results_all.items(),
        key=lambda x: (-x[1]["pass_count"], all_results[x[0]]["total_clusters"])
    )

    for config_name, clip_data in best_configs[:5]:
        print(f"\n--- {config_name} (CLIP pass: {clip_data['pass_count']}/{clip_data['total']}, "
              f"total clusters: {all_results[config_name]['total_clusters']}) ---")

        for vname, vclip in sorted(clip_data["per_video"].items()):
            status = "PASS" if vclip["all_distinct"] else "FAIL"
            labels = list(vclip["labels"].values())
            n_c = vclip["n_clusters"]
            print(f"  {status:4s} {vname:35s} clusters={n_c:2d} labels={labels}")

    # -----------------------------------------------------------------------
    # Also test CLIP post-merge on baseline vs best
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 4: CLIP POST-MERGE COMPARISON")
    print("=" * 80)

    for config_name in [best_configs[0][0], "linkage_single"]:
        clip_data = clip_results_all.get(config_name, {})
        if not clip_data:
            continue

        # Count how many would pass after CLIP-merge
        merge_pass = 0
        for vname, vclip in clip_data["per_video"].items():
            labels = list(vclip["labels"].values())
            if len(labels) == len(set(labels)):
                merge_pass += 1
            else:
                # Simulate merge
                unique = len(set(labels))
                merge_pass += 1  # After merge, all distinct by definition

        print(f"  {config_name:<45} Raw CLIP pass: {clip_data['pass_count']}/{clip_data['total']}, "
              f"After CLIP-merge: {merge_pass}/{clip_data['total']}")

    # -----------------------------------------------------------------------
    # Save full results
    # -----------------------------------------------------------------------

    # Build serializable summary
    summary = {
        "experiment": "clustering_optimization",
        "threshold": THRESHOLD,
        "n_configs": len(configs),
        "n_videos": len(video_data),
        "ranking": [],
    }

    for config_name, clip_data in best_configs:
        entry = {
            "config": config_name,
            "clip_pass": clip_data["pass_count"],
            "total_videos": clip_data["total"],
            "total_clusters": all_results[config_name]["total_clusters"],
            "max_clusters": all_results[config_name]["max_clusters"],
            "cluster_counts": all_results[config_name]["cluster_counts"],
            "per_video_labels": {
                vn: vd["labels"] for vn, vd in clip_data["per_video"].items()
            },
            "per_video_pass": {
                vn: vd["all_distinct"] for vn, vd in clip_data["per_video"].items()
            },
        }
        summary["ranking"].append(entry)

    # Also add the full ranking (without CLIP) for all configs
    summary["full_cluster_ranking"] = [
        {
            "config": name,
            "total_clusters": data["total_clusters"],
            "max_clusters": data["max_clusters"],
            "cluster_counts": data["cluster_counts"],
        }
        for name, data in ranked
    ]

    out_path = "/mnt/Data/GitHub/BRACE/experiments/clustering_optimization_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # -----------------------------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_name = best_configs[0][0]
    best_clip = best_configs[0][1]
    best_clusters = all_results[best_name]

    print(f"\n  Best config:       {best_name}")
    print(f"  CLIP pass rate:    {best_clip['pass_count']}/{best_clip['total']}")
    print(f"  Total clusters:    {best_clusters['total_clusters']} (baseline: {all_results.get('linkage_single', {}).get('total_clusters', '?')})")
    print(f"  Max clusters:      {best_clusters['max_clusters']} (baseline: {all_results.get('linkage_single', {}).get('max_clusters', '?')})")

    baseline_clip = clip_results_all.get("linkage_single", {})
    if baseline_clip:
        print(f"\n  Baseline CLIP pass: {baseline_clip['pass_count']}/{baseline_clip['total']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
