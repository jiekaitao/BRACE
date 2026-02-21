#!/usr/bin/env python3
"""Test change point detection methods for motion segmentation.

Compares multiple CPD algorithms from the `ruptures` library against the
production velocity-based boundary detector.  Each method is evaluated on
all 14 demo videos using cached SRP features, then segments are clustered
with agglomerative clustering (average linkage, t=2.0) and classified with
CLIP ViT-L/14 zero-shot labels.  A method "passes" a video when every
resulting cluster receives a distinct CLIP label.

Usage:
    .venv/bin/python experiments/test_changepoint_methods.py
"""

import sys
import os
import json
import copy
import time
from collections import Counter

import numpy as np

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

import ruptures as rpt
from brace.core.motion_segments import (
    segment_motions,
    cluster_segments,
    _segment_distance,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"
OUTPUT_PATH = "/mnt/Data/GitHub/BRACE/experiments/changepoint_results.json"

# ---------------------------------------------------------------------------
# CLIP labels (same 61-label vocab used in clustering validation)
# ---------------------------------------------------------------------------
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
    # Expanded labels for better discrimination
    "a person doing a forehand tennis stroke",
    "a person doing a backhand tennis stroke",
    "a person doing a seated yoga pose",
    "a person doing a standing yoga pose",
    "a person doing a prone yoga pose",
    "a person doing a kneeling yoga pose",
    "a person doing a side kick",
    "a person doing a roundhouse kick",
    "a person doing a jab punch",
    "a person doing an uppercut",
    "a person doing a cross punch",
    "a person doing crunches",
    "a person doing sit ups",
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
    "a person doing a forehand tennis stroke": "tennis forehand",
    "a person doing a backhand tennis stroke": "tennis backhand",
    "a person doing a seated yoga pose": "seated yoga",
    "a person doing a standing yoga pose": "standing yoga",
    "a person doing a prone yoga pose": "prone yoga",
    "a person doing a kneeling yoga pose": "kneeling yoga",
    "a person doing a side kick": "side kick",
    "a person doing a roundhouse kick": "roundhouse kick",
    "a person doing a jab punch": "jab",
    "a person doing an uppercut": "uppercut",
    "a person doing a cross punch": "cross punch",
    "a person doing crunches": "crunches",
    "a person doing sit ups": "sit ups",
}


# ======================================================================
# Change Point Detection Methods
# ======================================================================

def cpd_production(features, valid_indices, fps, min_segment_sec=2.0):
    """Production method: velocity-based boundary detection."""
    segments = segment_motions(features, valid_indices, fps, min_segment_sec)
    return segments


def _boundaries_to_segments(bkps, features, valid_indices):
    """Convert ruptures breakpoints to segment dicts compatible with cluster_segments.

    ruptures returns breakpoints as sorted list ending with n_samples.
    E.g. [50, 120, 300] means segments [0:50], [50:120], [120:300].
    """
    segments = []
    prev = 0
    for bp in bkps:
        if bp <= prev:
            continue
        end = min(bp, features.shape[0])
        if end - prev < 3:
            prev = end
            continue
        seg_features = features[prev:end]
        start_frame = valid_indices[prev] if prev < len(valid_indices) else 0
        end_frame = valid_indices[end - 1] if end - 1 < len(valid_indices) else valid_indices[-1]
        segments.append({
            "start_valid": prev,
            "end_valid": end,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "features": seg_features,
            "mean_feature": seg_features.mean(axis=0),
        })
        prev = end
    return segments


def _auto_penalty(features, fps, min_segment_sec):
    """Compute a reasonable penalty for PELT/Binseg given feature dims.

    Uses a modified BIC-like penalty: pen = C * dim * log(n), where C is
    tuned so that we get ~1 breakpoint per min_segment_sec seconds.
    """
    n, dim = features.shape
    # BIC-style: dim * log(n) tends to work well.
    # We scale by a factor to target the right segment density.
    return dim * np.log(n) * 1.5


def cpd_pelt_l2(features, valid_indices, fps, min_segment_sec=2.0):
    """PELT with L2 cost (Gaussian model)."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_pelt_rbf(features, valid_indices, fps, min_segment_sec=2.0):
    """PELT with RBF kernel (non-parametric, distribution changes)."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_binseg_l2(features, valid_indices, fps, min_segment_sec=2.0):
    """Binary segmentation with L2 cost."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_binseg_rbf(features, valid_indices, fps, min_segment_sec=2.0):
    """Binary segmentation with RBF kernel."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Binseg(model="rbf", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_bottomup_l2(features, valid_indices, fps, min_segment_sec=2.0):
    """Bottom-up segmentation with L2 cost."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.BottomUp(model="l2", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_bottomup_rbf(features, valid_indices, fps, min_segment_sec=2.0):
    """Bottom-up segmentation with RBF kernel."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.BottomUp(model="rbf", min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_window_l2(features, valid_indices, fps, min_segment_sec=2.0):
    """Window-based CPD with L2 cost."""
    min_size = max(int(fps * min_segment_sec), 5)
    width = max(int(fps * min_segment_sec * 2), 20)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Window(model="l2", width=width, min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_window_rbf(features, valid_indices, fps, min_segment_sec=2.0):
    """Window-based CPD with RBF kernel."""
    min_size = max(int(fps * min_segment_sec), 5)
    width = max(int(fps * min_segment_sec * 2), 20)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.Window(model="rbf", width=width, min_size=min_size, jump=1).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_kernelcpd_rbf(features, valid_indices, fps, min_segment_sec=2.0):
    """KernelCPD with RBF kernel (C-optimized, fastest kernel method)."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_kernelcpd_linear(features, valid_indices, fps, min_segment_sec=2.0):
    """KernelCPD with linear kernel."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.KernelCPD(kernel="linear", min_size=min_size).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_kernelcpd_cosine(features, valid_indices, fps, min_segment_sec=2.0):
    """KernelCPD with cosine kernel."""
    min_size = max(int(fps * min_segment_sec), 5)
    pen = _auto_penalty(features, fps, min_segment_sec)
    algo = rpt.KernelCPD(kernel="cosine", min_size=min_size).fit(features)
    bkps = algo.predict(pen=pen)
    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_bottomup_l2_elbow(features, valid_indices, fps, min_segment_sec=2.0):
    """BottomUp L2 on standardized features with elbow-based k selection.

    Standardizes features first (zero mean, unit variance), runs BottomUp L2,
    then picks k via elbow method (marginal cost reduction < 2% of total).
    """
    from sklearn.preprocessing import StandardScaler
    n, dim = features.shape
    min_size = max(int(fps * min_segment_sec), 5)

    scaler = StandardScaler()
    feats_std = scaler.fit_transform(features)

    algo = rpt.BottomUp(model="l2", min_size=min_size, jump=5).fit(feats_std)

    max_k = min(10, n // min_size - 1)
    costs = []
    for k in range(0, max_k + 1):
        if k == 0:
            bkps = [n]
        else:
            try:
                bkps = algo.predict(n_bkps=k)
            except Exception:
                costs.append(costs[-1] if costs else 0)
                continue
        cost = 0.0
        prev = 0
        for bp in bkps:
            seg = feats_std[prev:bp]
            if len(seg) > 1:
                cost += float(np.sum(np.var(seg, axis=0)) * len(seg))
            prev = bp
        costs.append(cost)

    threshold = costs[0] * 0.02
    elbow_k = 0
    for k in range(1, len(costs)):
        reduction = costs[k - 1] - costs[k]
        if reduction > threshold:
            elbow_k = k
        else:
            break

    if elbow_k == 0:
        bkps = [n]
    else:
        bkps = algo.predict(n_bkps=elbow_k)

    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_pelt_l2_std(features, valid_indices, fps, min_segment_sec=2.0):
    """PELT L2 on standardized features with lower penalty (scale=0.5).

    Standardizing features helps PELT work better on high-dimensional data
    by making the L2 cost function more balanced across dimensions.
    """
    from sklearn.preprocessing import StandardScaler
    n, dim = features.shape
    min_size = max(int(fps * min_segment_sec), 5)

    scaler = StandardScaler()
    feats_std = scaler.fit_transform(features)
    pen = dim * np.log(n) * 0.5

    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(feats_std)
    bkps = algo.predict(pen=pen)

    return _boundaries_to_segments(bkps, features, valid_indices)


def cpd_dynp_l2(features, valid_indices, fps, min_segment_sec=2.0, n_bkps_range=(1, 8)):
    """Dynamic programming with L2 (exact, but needs n_bkps).

    Tries multiple n_bkps and picks the one with best BIC.
    """
    min_size = max(int(fps * min_segment_sec), 5)
    n = features.shape[0]
    dim = features.shape[1]

    algo = rpt.Dynp(model="l2", min_size=min_size, jump=max(1, min_size // 4)).fit(features)

    best_bkps = [n]
    best_bic = float("inf")

    max_bkps = min(n_bkps_range[1], n // min_size - 1)
    for k in range(n_bkps_range[0], max(max_bkps + 1, 2)):
        try:
            bkps = algo.predict(n_bkps=k)
        except Exception:
            continue
        # Compute BIC: sum of within-segment variance + k * dim * log(n)
        cost = 0.0
        prev = 0
        for bp in bkps:
            seg = features[prev:bp]
            if len(seg) > 0:
                cost += float(np.sum(np.var(seg, axis=0)) * len(seg))
            prev = bp
        bic = cost + k * dim * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_bkps = bkps

    return _boundaries_to_segments(best_bkps, features, valid_indices)


# Penalty sweep versions: try multiple penalty values and pick best
def cpd_pelt_l2_sweep(features, valid_indices, fps, min_segment_sec=2.0):
    """PELT L2 with penalty sweep (tries 5 penalty values, picks best BIC)."""
    min_size = max(int(fps * min_segment_sec), 5)
    n, dim = features.shape
    base_pen = dim * np.log(n)

    best_bkps = [n]
    best_bic = float("inf")

    for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
        pen = base_pen * scale
        try:
            algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(features)
            bkps = algo.predict(pen=pen)
        except Exception:
            continue
        k = len(bkps) - 1  # number of changepoints (last bkp is n)
        cost = 0.0
        prev = 0
        for bp in bkps:
            seg = features[prev:bp]
            if len(seg) > 0:
                cost += float(np.sum(np.var(seg, axis=0)) * len(seg))
            prev = bp
        bic = cost + k * dim * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_bkps = bkps

    return _boundaries_to_segments(best_bkps, features, valid_indices)


# ======================================================================
# CLIP Classification
# ======================================================================

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


def precompute_text_features(clip_model, tokenizer, device):
    """Pre-encode CLIP text features for all action labels."""
    import torch
    text_tokens = tokenizer(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def classify_cluster_clip(frames_list, clip_model, preprocess, text_features, device):
    """Classify a list of frames via CLIP. Returns (simplified_label, confidence)."""
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


def load_video_frames(video_path):
    """Load all frames from a video as RGB numpy arrays.

    Returns dict mapping frame_index -> RGB numpy array.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = {}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return frames


def get_cluster_frames(segments, raw_frames, valid_indices):
    """Group raw video frames by cluster assignment.

    Returns dict: cluster_id -> list of RGB frames.
    """
    cluster_frames = {}
    for seg in segments:
        cid = seg.get("cluster", 0)
        if cid not in cluster_frames:
            cluster_frames[cid] = []
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        # Sample up to 30 frames per segment
        frame_range = range(start_frame, min(end_frame + 1, max(raw_frames.keys()) + 1))
        step = max(1, len(frame_range) // 30)
        for fi in range(start_frame, end_frame + 1, step):
            if fi in raw_frames:
                cluster_frames[cid].append(raw_frames[fi])
    return cluster_frames


# ======================================================================
# Validation
# ======================================================================

def validate_clusters(segments, raw_frames, valid_indices, clip_model,
                      preprocess, text_features, device):
    """Classify each cluster with CLIP and check for distinct labels.

    Returns (passed, n_clusters, labels_dict, failure_reason).
    """
    if not segments:
        return False, 0, {}, "no segments"

    # Run clustering
    if len(segments) >= 2:
        clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
    else:
        clustered = copy.deepcopy(segments)
        for s in clustered:
            s["cluster"] = 0

    n_clusters = len(set(s["cluster"] for s in clustered))
    cluster_frames = get_cluster_frames(clustered, raw_frames, valid_indices)

    labels_dict = {}
    for cid in sorted(cluster_frames.keys()):
        label, conf = classify_cluster_clip(
            cluster_frames[cid], clip_model, preprocess, text_features, device
        )
        labels_dict[cid] = {"label": label, "confidence": round(conf, 3)}

    labels_list = [v["label"] for v in labels_dict.values()]
    unique = set(labels_list)
    if len(labels_list) != len(unique):
        counts = Counter(labels_list)
        dups = [l for l, c in counts.items() if c > 1]
        reason = "duplicate labels: " + ", ".join(dups)
        return False, n_clusters, labels_dict, reason

    return True, n_clusters, labels_dict, None


# ======================================================================
# Main
# ======================================================================

# All methods to test
METHODS = {
    "production_velocity": cpd_production,
    "pelt_l2": cpd_pelt_l2,
    "pelt_rbf": cpd_pelt_rbf,
    "pelt_l2_sweep": cpd_pelt_l2_sweep,
    "pelt_l2_std": cpd_pelt_l2_std,
    "binseg_l2": cpd_binseg_l2,
    "binseg_rbf": cpd_binseg_rbf,
    "bottomup_l2": cpd_bottomup_l2,
    "bottomup_rbf": cpd_bottomup_rbf,
    "bottomup_l2_elbow": cpd_bottomup_l2_elbow,
    "window_l2": cpd_window_l2,
    "window_rbf": cpd_window_rbf,
    "kernelcpd_rbf": cpd_kernelcpd_rbf,
    "kernelcpd_linear": cpd_kernelcpd_linear,
    "kernelcpd_cosine": cpd_kernelcpd_cosine,
    "dynp_l2": cpd_dynp_l2,
}


def main():
    # Load ground truth
    gt = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH) as f:
            gt = json.load(f)
        print(f"Loaded ground truth for {len(gt)} videos")

    # Load CLIP model
    print("Loading CLIP ViT-L/14...")
    clip_model, preprocess, tokenizer, device = load_clip_model()
    text_features = precompute_text_features(clip_model, tokenizer, device)
    print(f"CLIP loaded on {device}")

    # Discover cached feature files
    feat_files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".feats.npz"))
    print(f"\nFound {len(feat_files)} cached feature files")

    # Results: method -> video -> result_dict
    all_results = {method: {} for method in METHODS}
    method_timings = {method: [] for method in METHODS}

    for feat_file in feat_files:
        video_name = feat_file.replace(".feats.npz", "")
        video_path = os.path.join(VIDEO_DIR, video_name)
        feat_path = os.path.join(CACHE_DIR, feat_file)

        print(f"\n{'='*70}")
        print(f"Video: {video_name}")

        # Load cached features
        data = np.load(feat_path)
        features = data["features"]
        valid_indices = list(data["valid_indices"])
        fps = float(data["fps"])
        print(f"  Features: {features.shape}, FPS: {fps:.1f}")

        # Load video frames for CLIP (do once per video)
        if os.path.exists(video_path):
            raw_frames = load_video_frames(video_path)
            print(f"  Loaded {len(raw_frames)} video frames")
        else:
            print(f"  WARNING: video not found at {video_path}, skipping CLIP")
            raw_frames = {}

        gt_info = gt.get(video_name, {})
        gt_expected = gt_info.get("expected_clusters", None)

        # Test each method
        for method_name, method_fn in METHODS.items():
            try:
                t0 = time.time()
                segments = method_fn(features, valid_indices, fps)
                elapsed = time.time() - t0
                method_timings[method_name].append(elapsed)
            except Exception as e:
                print(f"  {method_name}: ERROR - {e}")
                all_results[method_name][video_name] = {
                    "error": str(e),
                    "n_segments": 0,
                    "n_clusters": 0,
                    "passed": False,
                }
                continue

            n_segments = len(segments)

            if raw_frames:
                passed, n_clusters, labels, reason = validate_clusters(
                    segments, raw_frames, valid_indices,
                    clip_model, preprocess, text_features, device
                )
            else:
                # No video -> can only report segment counts
                if n_segments >= 2:
                    clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0)
                else:
                    clustered = copy.deepcopy(segments)
                    for s in clustered:
                        s["cluster"] = 0
                n_clusters = len(set(s["cluster"] for s in clustered))
                passed = None
                labels = {}
                reason = "no video for CLIP"

            status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
            label_strs = [v["label"] for v in labels.values()] if labels else []
            gt_str = f" (expected {gt_expected})" if gt_expected is not None else ""
            print(f"  {method_name:25s}: {status} | {n_segments} segs -> {n_clusters} clusters{gt_str} | {label_strs} | {elapsed*1000:.0f}ms")

            all_results[method_name][video_name] = {
                "n_segments": n_segments,
                "n_clusters": n_clusters,
                "passed": passed,
                "labels": {str(k): v for k, v in labels.items()},
                "failure_reason": reason,
                "gt_expected": gt_expected,
                "gt_match": (n_clusters == gt_expected) if gt_expected is not None else None,
                "time_ms": round(elapsed * 1000, 1),
            }

        # Free video frames
        del raw_frames

    # ---- Summary ----
    print(f"\n\n{'='*70}")
    print("SUMMARY: Change Point Detection Methods Comparison")
    print(f"{'='*70}")
    print(f"\n{'Method':30s} {'Pass':>5s} {'Fail':>5s} {'Total':>6s} {'Rate':>7s} {'AvgSeg':>7s} {'AvgClust':>9s} {'AvgTime':>9s}")
    print("-" * 90)

    summary = {}
    for method_name in METHODS:
        results = all_results[method_name]
        passed = sum(1 for v in results.values() if v.get("passed") is True)
        failed = sum(1 for v in results.values() if v.get("passed") is False)
        total = passed + failed
        rate = passed / total * 100 if total > 0 else 0

        avg_seg = np.mean([v["n_segments"] for v in results.values() if "error" not in v])
        avg_clust = np.mean([v["n_clusters"] for v in results.values() if "error" not in v])
        timings = method_timings[method_name]
        avg_time = np.mean(timings) * 1000 if timings else 0

        print(f"{method_name:30s} {passed:5d} {failed:5d} {total:6d} {rate:6.1f}% {avg_seg:7.1f} {avg_clust:9.1f} {avg_time:7.0f}ms")

        summary[method_name] = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": round(rate, 1),
            "avg_segments": round(float(avg_seg), 1),
            "avg_clusters": round(float(avg_clust), 1),
            "avg_time_ms": round(avg_time, 1),
        }

    # GT match rates
    print(f"\n{'Method':30s} {'GT Match':>9s} {'GT Total':>9s} {'GT Rate':>8s}")
    print("-" * 60)
    for method_name in METHODS:
        results = all_results[method_name]
        gt_matched = sum(1 for v in results.values() if v.get("gt_match") is True)
        gt_total = sum(1 for v in results.values() if v.get("gt_match") is not None)
        gt_rate = gt_matched / gt_total * 100 if gt_total > 0 else 0
        print(f"{method_name:30s} {gt_matched:9d} {gt_total:9d} {gt_rate:7.1f}%")
        summary[method_name]["gt_matched"] = gt_matched
        summary[method_name]["gt_total"] = gt_total
        summary[method_name]["gt_rate"] = round(gt_rate, 1)

    # Per-video pass matrix
    print(f"\n\nPer-Video Pass Matrix:")
    print(f"{'Video':35s}", end="")
    short_names = {m: m[:12] for m in METHODS}
    for m in METHODS:
        print(f" {short_names[m]:>12s}", end="")
    print()
    print("-" * (35 + 13 * len(METHODS)))

    for feat_file in feat_files:
        video_name = feat_file.replace(".feats.npz", "")
        print(f"{video_name:35s}", end="")
        for method_name in METHODS:
            r = all_results[method_name].get(video_name, {})
            if r.get("passed") is True:
                s = "PASS"
            elif r.get("passed") is False:
                s = "FAIL"
            elif "error" in r:
                s = "ERR"
            else:
                s = "SKIP"
            n_c = r.get("n_clusters", "?")
            print(f" {s}({n_c})", end="")
            # pad to 12
            printed = len(f"{s}({n_c})")
            print(" " * max(0, 12 - printed), end="")
        print()

    # Save detailed results
    output = {
        "summary": summary,
        "per_video": all_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
