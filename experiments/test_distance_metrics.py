#!/usr/bin/env python3
"""Experiment: Compare segment distance metrics for motion clustering.

Tests multiple distance metrics against the current spectral distance
(mean pose + FFT power spectrum) used in production. For each metric,
computes pairwise distance matrices on all 14 demo videos, feeds into
agglomerative clustering (average linkage) at thresholds 1.5, 2.0, 3.0,
and validates with CLIP ViT-L/14 zero-shot classification.

Metrics tested:
1. Spectral (current production) - mean pose + FFT power spectrum
2. DTW (Dynamic Time Warping) via dtaidistance
3. Soft-DTW via tslearn
4. SBD (Shape-Based Distance) via tslearn
5. CID (Complexity-Invariant Distance) - custom implementation
6. Wasserstein/EMD between segment feature distributions
7. Matrix Profile via stumpy (motif-based similarity)
8. PCA-reduced Euclidean distance
9. Mean-only L2 (ablation: mean pose without FFT)

Usage:
    .venv/bin/python experiments/test_distance_metrics.py
"""

import sys
import os
import copy
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    _resample_segment,
    _segment_distance,
    _merge_adjacent_clusters,
    _merge_small_clusters,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
RESAMPLE_LEN = 30
THRESHOLDS = [1.5, 2.0, 3.0]

# ACTION_LABELS for CLIP validation (61 expanded labels from run_clustering_validation.py)
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
    # Extended labels for finer discrimination
    "a person doing forehand tennis stroke",
    "a person doing backhand tennis stroke",
    "a person doing seated yoga pose",
    "a person doing standing yoga pose",
    "a person doing cobra yoga pose",
    "a person doing child's pose",
    "a person doing a forward fold",
    "a person doing side lunges",
    "a person doing front kicks",
    "a person doing shadowboxing combinations",
    "a person doing uppercuts",
    "a person doing jab cross combinations",
    "a person talking to camera",
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
    "a person doing yoga": "yoga",
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
    "a person doing forehand tennis stroke": "tennis forehand",
    "a person doing backhand tennis stroke": "tennis backhand",
    "a person doing seated yoga pose": "seated yoga",
    "a person doing standing yoga pose": "standing yoga",
    "a person doing cobra yoga pose": "cobra pose",
    "a person doing child's pose": "child's pose",
    "a person doing a forward fold": "forward fold",
    "a person doing side lunges": "side lunges",
    "a person doing front kicks": "front kicks",
    "a person doing shadowboxing combinations": "shadowboxing",
    "a person doing uppercuts": "uppercuts",
    "a person doing jab cross combinations": "jab-cross",
    "a person talking to camera": "talking/standing",
}


# ---------------------------------------------------------------------------
# Distance metric implementations
# ---------------------------------------------------------------------------

def dist_spectral(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Current production metric: mean pose + FFT power spectrum."""
    return _segment_distance(seg_a, seg_b, resample_len=resample_len)


def dist_mean_only(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Ablation: only mean pose distance, no FFT."""
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    mean_dist = float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0)))
    return mean_dist / np.sqrt(feat_dim)


def dist_dtw(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Dynamic Time Warping via dtaidistance (C-accelerated)."""
    from dtaidistance import dtw_ndim
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    d = dtw_ndim.distance(ra.astype(np.double), rb.astype(np.double))
    return d / (resample_len * np.sqrt(feat_dim))


def dist_soft_dtw(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Soft-DTW via tslearn (differentiable DTW variant)."""
    from tslearn.metrics import soft_dtw
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]
    # gamma controls softness; 1.0 is a good default
    d = soft_dtw(ra, rb, gamma=1.0)
    # soft-dtw can return negative values; shift to ensure non-negative
    d = max(d, 0.0)
    return np.sqrt(d) / (resample_len * np.sqrt(feat_dim))


def dist_sbd(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Shape-Based Distance via cross-correlation (used in k-Shape).

    SBD is naturally in [0, 2] range. Phase-invariant via shift optimization.
    """
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)

    # SBD computed per-dimension and averaged
    sbd_total = 0.0
    for d in range(ra.shape[1]):
        x = ra[:, d] - ra[:, d].mean()
        y = rb[:, d] - rb[:, d].mean()
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        if norm_x < 1e-8 or norm_y < 1e-8:
            sbd_total += 1.0  # max distance for flat signals
            continue
        # Normalized cross-correlation
        ncc = np.correlate(x, y, mode="full") / (norm_x * norm_y)
        sbd_total += 1.0 - ncc.max()

    return sbd_total / ra.shape[1]


def dist_cid(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Complexity-Invariant Distance.

    CID = ED * CF, where CF is the complexity correction factor.
    Complexity = sum of first-difference norms (captures how "wiggly" a signal is).
    """
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]

    # Euclidean distance
    ed = float(np.linalg.norm(ra - rb))

    # Complexity of each segment
    def complexity(x):
        diffs = np.diff(x, axis=0)
        return float(np.sqrt(np.sum(diffs ** 2)))

    ce_a = complexity(ra)
    ce_b = complexity(rb)

    # Complexity correction factor
    if ce_a < 1e-8 and ce_b < 1e-8:
        cf = 1.0
    else:
        cf = max(ce_a, ce_b) / (min(ce_a, ce_b) + 1e-8)

    return (ed * cf) / (resample_len * np.sqrt(feat_dim))


def dist_wasserstein(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Wasserstein/EMD distance between segment feature distributions.

    Treats each segment as an empirical distribution of feature vectors
    and computes 1D Wasserstein per dimension, then averages.
    """
    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]

    wd = 0.0
    for d in range(feat_dim):
        wd += wasserstein_distance(ra[:, d], rb[:, d])

    return wd / np.sqrt(feat_dim)


def dist_matrix_profile(seg_a, seg_b, resample_len=RESAMPLE_LEN):
    """Matrix Profile distance via stumpy.

    Uses the matrix profile between two segments to measure subsequence
    similarity. Returns the mean of the cross-matrix-profile as distance.
    """
    import stumpy

    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]

    # Matrix profile operates on 1D; compute per-dimension and average
    m = min(5, resample_len // 3)  # subsequence length
    if m < 3:
        m = 3
    if resample_len < m + 1:
        # Too short for matrix profile
        return float(np.linalg.norm(ra.mean(axis=0) - rb.mean(axis=0))) / np.sqrt(feat_dim)

    mp_dists = []
    for d in range(feat_dim):
        ts_a = ra[:, d].astype(np.float64)
        ts_b = rb[:, d].astype(np.float64)
        # Check for near-constant signals (stumpy fails on these)
        if np.std(ts_a) < 1e-8 or np.std(ts_b) < 1e-8:
            mp_dists.append(abs(ts_a.mean() - ts_b.mean()))
            continue
        try:
            mp = stumpy.stump(ts_a, m=m, T_B=ts_b, ignore_trivial=False)
            mp_dists.append(float(np.mean(mp[:, 0])))
        except Exception:
            mp_dists.append(abs(ts_a.mean() - ts_b.mean()))

    return np.mean(mp_dists) / np.sqrt(feat_dim)


def dist_pca_euclidean(seg_a, seg_b, resample_len=RESAMPLE_LEN, n_components=8):
    """PCA-reduced Euclidean distance.

    Projects resampled segments into PCA space (fit on their concatenation)
    and computes L2 distance on the flattened PCA representation.
    """
    from sklearn.decomposition import PCA

    ra = _resample_segment(seg_a["features"], resample_len)
    rb = _resample_segment(seg_b["features"], resample_len)
    feat_dim = ra.shape[1]

    combined = np.vstack([ra, rb])
    n_comp = min(n_components, combined.shape[1], combined.shape[0])
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(combined)

    pa = transformed[:resample_len].flatten()
    pb = transformed[resample_len:].flatten()

    return float(np.linalg.norm(pa - pb)) / (resample_len * np.sqrt(n_comp))


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRICS = {
    "spectral": dist_spectral,
    "mean_only": dist_mean_only,
    "dtw": dist_dtw,
    "soft_dtw": dist_soft_dtw,
    "sbd": dist_sbd,
    "cid": dist_cid,
    "wasserstein": dist_wasserstein,
    # "matrix_profile": dist_matrix_profile,  # ~15s/pair, impractical for pairwise
    "pca_euclidean": dist_pca_euclidean,
}


# ---------------------------------------------------------------------------
# Clustering with custom distance
# ---------------------------------------------------------------------------

def cluster_with_metric(segments, metric_fn, threshold, fps=30.0):
    """Cluster segments using a custom distance metric + agglomerative clustering.

    Applies the same post-processing as production: merge adjacent same-cluster
    segments, absorb small clusters.
    """
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
            d = metric_fn(segments[i], segments[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Agglomerative clustering with average linkage
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for i, seg in enumerate(segments):
        seg["cluster"] = label_map[labels[i]]

    # Same post-processing as production
    segments = _merge_adjacent_clusters(segments)
    segments = _merge_small_clusters(segments, fps=fps)

    return segments


# ---------------------------------------------------------------------------
# CLIP validation
# ---------------------------------------------------------------------------

def load_clip_model():
    """Load CLIP ViT-L/14 for zero-shot classification."""
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer, device


def encode_text_labels(clip_model, tokenizer, device):
    """Pre-encode all text labels."""
    import torch
    text_tokens = tokenizer(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def classify_cluster_frames(frames_list, clip_model, preprocess, text_features, device):
    """Classify frames with CLIP. Returns simplified label and confidence."""
    import torch
    from PIL import Image

    if not frames_list:
        return "unknown", 0.0

    n = len(frames_list)
    indices = np.linspace(0, n - 1, min(8, n), dtype=int)
    sampled = [frames_list[i] for i in indices]

    images = torch.stack([preprocess(Image.fromarray(f)) for f in sampled]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    avg_feat = image_features.mean(dim=0, keepdim=True)
    avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
    similarity = (avg_feat @ text_features.T).squeeze()
    probs = (similarity * 100.0).softmax(dim=-1)
    best_idx = probs.argmax().item()
    raw_label = ACTION_LABELS[best_idx]
    label = LABEL_SIMPLIFY.get(raw_label, raw_label)
    conf = float(probs[best_idx])
    return label, conf


def validate_with_clip(clustered_segments, video_path, clip_model, preprocess,
                       text_features, device):
    """Validate clustering: do different clusters get different CLIP labels?

    Returns (passed, n_clusters, labels_dict).
    """
    import cv2

    n_clusters = len(set(s["cluster"] for s in clustered_segments))
    if n_clusters <= 1:
        return True, n_clusters, {0: "single_cluster"}

    # Extract representative frames per cluster
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gather frame indices per cluster
    cluster_frame_indices = {}
    for seg in clustered_segments:
        cid = seg["cluster"]
        if cid not in cluster_frame_indices:
            cluster_frame_indices[cid] = []
        for fi in range(seg["start_frame"], min(seg["end_frame"] + 1, total)):
            cluster_frame_indices[cid].append(fi)

    # Sample up to 8 frames per cluster
    cluster_sample_indices = {}
    for cid, indices in cluster_frame_indices.items():
        n = len(indices)
        if n == 0:
            continue
        sample = np.linspace(0, n - 1, min(8, n), dtype=int)
        cluster_sample_indices[cid] = [indices[i] for i in sample]

    # Read the needed frames
    all_needed = set()
    for indices in cluster_sample_indices.values():
        all_needed.update(indices)

    frames_map = {}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in all_needed:
            frames_map[frame_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_idx += 1
        if frame_idx > max(all_needed):
            break
    cap.release()

    # Classify each cluster
    cluster_labels = {}
    for cid in sorted(cluster_sample_indices.keys()):
        frames = [frames_map[fi] for fi in cluster_sample_indices[cid] if fi in frames_map]
        if not frames:
            cluster_labels[cid] = "unknown"
            continue
        label, conf = classify_cluster_frames(
            frames, clip_model, preprocess, text_features, device
        )
        cluster_labels[cid] = label

    # Check: all clusters have distinct labels
    labels_list = list(cluster_labels.values())
    unique_labels = set(labels_list)
    passed = len(labels_list) == len(unique_labels)

    return passed, n_clusters, cluster_labels


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def load_cached_features():
    """Load all cached feature files."""
    videos = {}
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".feats.npz"):
            continue
        video_name = fname.replace(".feats.npz", "")
        data = np.load(os.path.join(CACHE_DIR, fname))
        videos[video_name] = {
            "features": data["features"],
            "valid_indices": data["valid_indices"].tolist(),
            "fps": float(data["fps"]),
        }
    return videos


def run_experiment(use_clip=True):
    """Run all distance metrics on all videos and compare."""
    print("Loading cached features...")
    videos = load_cached_features()
    print(f"Loaded {len(videos)} videos\n")

    # Load ground truth
    with open(GT_PATH) as f:
        gt = json.load(f)

    # Load CLIP if requested
    clip_model = preprocess = text_features = device = None
    if use_clip:
        print("Loading CLIP ViT-L/14 model...")
        clip_model, preprocess, tokenizer, device = load_clip_model()
        text_features = encode_text_labels(clip_model, tokenizer, device)
        print(f"CLIP loaded on {device}\n")

    # Results storage
    results = {}
    timing = {}

    for metric_name, metric_fn in METRICS.items():
        print(f"\n{'='*70}")
        print(f"  METRIC: {metric_name}")
        print(f"{'='*70}")

        results[metric_name] = {}
        t0 = time.time()

        for threshold in THRESHOLDS:
            pass_count = 0
            total_count = 0
            video_results = {}

            for video_name, vdata in videos.items():
                feats = vdata["features"]
                vi = vdata["valid_indices"]
                fps = vdata["fps"]

                if len(feats) < 10:
                    continue

                # Segment
                segments = segment_motions(feats, vi, fps, min_segment_sec=2.0)
                if len(segments) < 2:
                    segments = segment_motions(feats, vi, fps, min_segment_sec=0.5)

                if len(segments) < 2:
                    # Only 1 segment = 1 cluster = auto-pass
                    for s in segments:
                        s["cluster"] = 0
                    clustered = segments
                else:
                    # Cluster with the test metric
                    clustered = cluster_with_metric(
                        copy.deepcopy(segments), metric_fn, threshold, fps
                    )

                n_clusters = len(set(s["cluster"] for s in clustered))

                # Ground truth check
                gt_info = gt.get(video_name, {})
                expected_range = gt_info.get("expected_clusters_range", None)

                # CLIP validation
                passed = True
                labels = {}
                if use_clip and n_clusters > 1:
                    video_path = f"/mnt/Data/GitHub/BRACE/data/sports_videos/{video_name}"
                    if os.path.exists(video_path):
                        passed, _, labels = validate_with_clip(
                            clustered, video_path, clip_model, preprocess,
                            text_features, device
                        )

                total_count += 1
                if passed:
                    pass_count += 1

                video_results[video_name] = {
                    "n_clusters": n_clusters,
                    "n_segments": len(clustered),
                    "passed": passed,
                    "labels": labels,
                    "expected_range": expected_range,
                }

            results[metric_name][threshold] = {
                "pass_count": pass_count,
                "total_count": total_count,
                "pass_rate": f"{pass_count}/{total_count}",
                "videos": video_results,
            }

            print(f"  t={threshold:.1f}: {pass_count}/{total_count} pass"
                  f"  ({pass_count/max(total_count,1)*100:.0f}%)")

        elapsed = time.time() - t0
        timing[metric_name] = round(elapsed, 1)
        print(f"  Time: {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("  SUMMARY: Pass rates by metric and threshold")
    print(f"{'='*70}")

    header = f"{'Metric':<20}"
    for t in THRESHOLDS:
        header += f"  {'t='+str(t):<12}"
    header += f"  {'Time(s)':<10}"
    print(header)
    print("-" * len(header))

    for metric_name in METRICS:
        row = f"{metric_name:<20}"
        for t in THRESHOLDS:
            r = results[metric_name][t]
            row += f"  {r['pass_rate']:<12}"
        row += f"  {timing.get(metric_name, '?'):<10}"
        print(row)

    # Per-video breakdown at t=2.0
    print(f"\n\n{'='*70}")
    print("  PER-VIDEO BREAKDOWN at t=2.0")
    print(f"{'='*70}")

    all_videos = sorted(list(videos.keys()))
    header = f"{'Video':<35}"
    for mn in METRICS:
        header += f" {mn[:7]:>7}"
    print(header)
    print("-" * len(header))

    for vname in all_videos:
        row = f"{vname:<35}"
        for mn in METRICS:
            vr = results[mn].get(2.0, {}).get("videos", {}).get(vname, {})
            if not vr:
                row += f" {'---':>7}"
            else:
                status = "P" if vr["passed"] else "F"
                nc = vr["n_clusters"]
                row += f" {status}({nc}){'':<2}"[:8].rjust(7)
        print(row)

    # Pairwise distance statistics for each metric at t=2.0
    print(f"\n\n{'='*70}")
    print("  DISTANCE STATISTICS (mean pairwise distance per video)")
    print(f"{'='*70}")

    for metric_name, metric_fn in METRICS.items():
        mean_dists = []
        for video_name, vdata in videos.items():
            feats = vdata["features"]
            vi = vdata["valid_indices"]
            fps = vdata["fps"]
            if len(feats) < 10:
                continue
            segments = segment_motions(feats, vi, fps, min_segment_sec=2.0)
            if len(segments) < 2:
                continue
            dists = []
            for i in range(len(segments)):
                for j in range(i + 1, len(segments)):
                    d = metric_fn(segments[i], segments[j])
                    dists.append(d)
            if dists:
                mean_dists.append(np.mean(dists))
        if mean_dists:
            print(f"  {metric_name:<20}: mean={np.mean(mean_dists):.3f}, "
                  f"std={np.std(mean_dists):.3f}, "
                  f"min={np.min(mean_dists):.3f}, max={np.max(mean_dists):.3f}")

    # Save results
    output_path = "/mnt/Data/GitHub/BRACE/experiments/distance_metrics_results.json"
    # Convert labels dicts (with int keys) to string keys for JSON
    json_results = {}
    for mn, thresh_data in results.items():
        json_results[mn] = {}
        for t, tdata in thresh_data.items():
            json_results[mn][str(t)] = {
                "pass_count": tdata["pass_count"],
                "total_count": tdata["total_count"],
                "pass_rate": tdata["pass_rate"],
                "videos": {}
            }
            for vn, vd in tdata.get("videos", {}).items():
                json_results[mn][str(t)]["videos"][vn] = {
                    "n_clusters": vd["n_clusters"],
                    "n_segments": vd["n_segments"],
                    "passed": vd["passed"],
                    "labels": {str(k): v for k, v in vd.get("labels", {}).items()},
                    "expected_range": vd.get("expected_range"),
                }

    with open(output_path, "w") as f:
        json.dump({"results": json_results, "timing": timing}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results, timing


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clip", action="store_true",
                        help="Skip CLIP validation (faster, distance-only analysis)")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Comma-separated list of metrics to test (default: all)")
    args = parser.parse_args()

    if args.metrics:
        selected = args.metrics.split(",")
        METRICS = {k: v for k, v in METRICS.items() if k in selected}
        print(f"Running metrics: {list(METRICS.keys())}")

    run_experiment(use_clip=not args.no_clip)
