#!/usr/bin/env python3
"""
Test temporal action segmentation approaches for BRACE.

Compares:
1. SlowFast / R3D video features + clustering
2. CLIP frame features + zero-shot text classification
3. VideoMAE pretrained features + clustering
4. X-CLIP zero-shot video classification
5. 1D-CNN on pose features with pseudo-labels

Each method is tested on basketball_solo.mp4 and reports:
- Detected actions/segments
- Runtime per frame
- GPU memory usage
- Comparison to current velocity+clustering approach
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path("/mnt/Data/GitHub/BRACE")
VIDEO_DIR = PROJECT / "data" / "sports_videos"
VIDEOS = {
    "basketball": VIDEO_DIR / "basketball_solo.mp4",
    "exercise": VIDEO_DIR / "exercise.mp4",
    "crossfit": VIDEO_DIR / "gym_crossfit.mp4",
    "mma": VIDEO_DIR / "mma_spar.mp4",
    "soccer": VIDEO_DIR / "soccer_match2.mp4",
}


@dataclass
class SegmentResult:
    method: str
    video: str
    segments: list  # list of (start_sec, end_sec, label, confidence)
    runtime_total_s: float
    runtime_per_frame_ms: float
    gpu_mem_peak_mb: float
    n_frames: int
    notes: str = ""


def load_video_frames(path: str, max_frames: int = 0, resize: tuple = None):
    """Load video frames as numpy arrays (RGB)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames), fps, total


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0


def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def cluster_features(features: np.ndarray, fps: float, method_name: str, n_clusters_hint: int = 0):
    """Simple temporal segmentation via feature similarity."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    # Compute cosine distances between consecutive frames
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed = features / norms

    # Compute pairwise distances
    dists = pdist(normed, metric="cosine")
    Z = linkage(dists, method="ward")

    # Try different thresholds to find reasonable segmentation
    best_labels = None
    for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        labels = fcluster(Z, t=t, criterion="distance")
        n_unique = len(np.unique(labels))
        if 2 <= n_unique <= 10:
            best_labels = labels
            break
    if best_labels is None:
        best_labels = fcluster(Z, t=2, criterion="maxclust")

    # Convert frame labels to segments
    segments = []
    current_label = best_labels[0]
    start_frame = 0
    for i, label in enumerate(best_labels):
        if label != current_label:
            segments.append((
                start_frame / fps,
                i / fps,
                f"cluster_{current_label}",
                1.0,
            ))
            current_label = label
            start_frame = i
    segments.append((
        start_frame / fps,
        len(best_labels) / fps,
        f"cluster_{current_label}",
        1.0,
    ))
    return segments, best_labels


# ═══════════════════════════════════════════════════════════════════════
# METHOD 1: R3D-18 (3D ResNet) Video Features
# ═══════════════════════════════════════════════════════════════════════

def test_r3d_features(video_path: str) -> SegmentResult:
    """Extract per-clip features using R3D-18 and cluster them."""
    print("\n" + "=" * 70)
    print("METHOD 1: R3D-18 (3D ResNet) Video Features")
    print("=" * 70)

    reset_gpu()
    from torchvision.models.video import r3d_18, R3D_18_Weights

    frames, fps, total = load_video_frames(video_path, resize=(224, 224))
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    # Load pretrained R3D-18
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights).cuda().eval()

    # Remove classification head - get features
    model.fc = torch.nn.Identity()

    # R3D expects (B, C, T, H, W) with ImageNet normalization
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).cuda()
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).cuda()

    # Process in sliding windows of 16 frames
    clip_len = 16
    stride = 4  # stride for overlapping clips
    features = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_frames - clip_len + 1, stride):
            clip = frames[start : start + clip_len]  # (16, 224, 224, 3)
            # Convert to (1, 3, 16, 224, 224) - batch, channels, time, h, w
            clip_t = torch.from_numpy(clip).float().permute(0, 3, 1, 2) / 255.0  # (16, 3, 224, 224)
            clip_t = clip_t.permute(1, 0, 2, 3).unsqueeze(0).cuda()  # (1, 3, 16, 224, 224)
            clip_t = (clip_t - mean) / std
            feat = model(clip_t)  # (1, 512)
            features.append(feat.cpu().numpy()[0])

    elapsed = time.perf_counter() - t0
    features = np.array(features)
    print(f"  Extracted {len(features)} clip features, shape: {features.shape}")
    print(f"  Time: {elapsed:.2f}s ({elapsed / n_frames * 1000:.1f} ms/frame)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    # Cluster features
    segments, labels = cluster_features(features, fps / stride, "R3D-18")
    n_clusters = len(np.unique(labels))
    print(f"  Found {n_clusters} clusters -> {len(segments)} segments")
    for s in segments:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]} (conf={s[3]:.2f})")

    return SegmentResult(
        method="R3D-18",
        video=str(video_path),
        segments=segments,
        runtime_total_s=elapsed,
        runtime_per_frame_ms=elapsed / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"{n_clusters} clusters from {len(features)} clip features (512D)",
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 2: CLIP Frame Features + Zero-Shot Text Classification
# ═══════════════════════════════════════════════════════════════════════

def test_clip_zeroshot(video_path: str) -> SegmentResult:
    """CLIP per-frame features + zero-shot text classification."""
    print("\n" + "=" * 70)
    print("METHOD 2: CLIP Zero-Shot Action Classification")
    print("=" * 70)

    reset_gpu()
    import open_clip

    frames, fps, total = load_video_frames(video_path, resize=(224, 224))
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.cuda().eval()

    # Define action labels for zero-shot classification
    action_labels = [
        "a person dribbling a basketball",
        "a person shooting a basketball",
        "a person dunking a basketball",
        "a person standing still",
        "a person running",
        "a person jumping",
        "a person doing push-ups",
        "a person doing squats",
        "a person stretching",
        "a person punching",
        "a person kicking",
        "a person wrestling",
        "a person playing soccer",
        "a person lifting weights",
        "a person doing burpees",
        "a person resting",
    ]

    # Encode text labels
    text_tokens = tokenizer(action_labels).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    # Encode frames
    t0 = time.perf_counter()
    frame_labels = []
    frame_confs = []
    all_features = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i : i + batch_size]
            # Manual preprocessing: normalize to CLIP expected range
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            # CLIP normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            batch_t = (batch_t - mean) / std
            batch_t = batch_t.cuda()

            img_features = model.encode_image(batch_t)
            img_features = F.normalize(img_features, dim=-1)

            # Cosine similarity with text labels
            sims = (img_features @ text_features.T) * 100  # scale
            probs = F.softmax(sims, dim=-1)
            confs, idxs = probs.max(dim=-1)

            for j in range(len(batch)):
                frame_labels.append(action_labels[idxs[j].item()])
                frame_confs.append(confs[j].item())
            all_features.append(img_features.cpu().numpy())

    elapsed = time.perf_counter() - t0
    all_features = np.concatenate(all_features)
    print(f"  Extracted {len(all_features)} frame features, shape: {all_features.shape}")
    print(f"  Time: {elapsed:.2f}s ({elapsed / n_frames * 1000:.1f} ms/frame)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    # Build segments from consecutive same-label frames
    segments = []
    current_label = frame_labels[0]
    current_confs = [frame_confs[0]]
    start_frame = 0

    for i in range(1, len(frame_labels)):
        if frame_labels[i] != current_label:
            mean_conf = np.mean(current_confs)
            segments.append((
                start_frame / fps,
                i / fps,
                current_label,
                mean_conf,
            ))
            current_label = frame_labels[i]
            current_confs = [frame_confs[i]]
            start_frame = i
        else:
            current_confs.append(frame_confs[i])

    segments.append((
        start_frame / fps,
        len(frame_labels) / fps,
        current_label,
        np.mean(current_confs),
    ))

    # Merge very short segments (< 0.5s) into neighbors
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < 0.5:
            # Merge into previous
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2], prev[3])
        else:
            merged.append(seg)
    segments = merged

    print(f"  Found {len(segments)} segments (after merging short ones)")
    # Show unique labels and their total durations
    from collections import Counter
    label_durations = Counter()
    for s in segments:
        label_durations[s[2]] += s[1] - s[0]
    print(f"  Action distribution:")
    for label, dur in label_durations.most_common():
        print(f"    {label}: {dur:.1f}s")

    for s in segments[:20]:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]} (conf={s[3]:.2f})")
    if len(segments) > 20:
        print(f"    ... and {len(segments) - 20} more segments")

    return SegmentResult(
        method="CLIP-ZeroShot",
        video=str(video_path),
        segments=segments,
        runtime_total_s=elapsed,
        runtime_per_frame_ms=elapsed / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"ViT-B/32, {len(action_labels)} action labels, {len(segments)} segments after merge",
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 3: VideoMAE Features + Clustering
# ═══════════════════════════════════════════════════════════════════════

def test_videomae(video_path: str) -> SegmentResult:
    """VideoMAE pretrained features + temporal clustering."""
    print("\n" + "=" * 70)
    print("METHOD 3: VideoMAE Pretrained Features")
    print("=" * 70)

    reset_gpu()
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    frames, fps, total = load_video_frames(video_path, resize=(224, 224))
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    # Load VideoMAE - use the Kinetics-400 finetuned model
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    print(f"  Loading {model_name}...")
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name).cuda().eval()

    # VideoMAE expects 16 frames per clip
    clip_len = 16
    stride = 8

    t0 = time.perf_counter()
    clip_predictions = []
    clip_features = []

    with torch.no_grad():
        for start in range(0, n_frames - clip_len + 1, stride):
            clip = list(frames[start : start + clip_len])
            inputs = processor(clip, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1)
            top5_vals, top5_ids = probs.topk(5)

            # Get top prediction
            top_label = model.config.id2label[top5_ids[0].item()]
            top_conf = top5_vals[0].item()
            clip_predictions.append({
                "start": start / fps,
                "end": (start + clip_len) / fps,
                "label": top_label,
                "confidence": top_conf,
                "top5": [
                    (model.config.id2label[top5_ids[i].item()], top5_vals[i].item())
                    for i in range(5)
                ],
            })
            clip_features.append(logits.cpu().numpy())

    elapsed = time.perf_counter() - t0
    print(f"  Processed {len(clip_predictions)} clips")
    print(f"  Time: {elapsed:.2f}s ({elapsed / n_frames * 1000:.1f} ms/frame)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    # Build segments from clip predictions
    segments = []
    if clip_predictions:
        current_label = clip_predictions[0]["label"]
        current_confs = [clip_predictions[0]["confidence"]]
        start_time = clip_predictions[0]["start"]

        for pred in clip_predictions[1:]:
            if pred["label"] != current_label:
                segments.append((
                    start_time,
                    pred["start"],
                    current_label,
                    np.mean(current_confs),
                ))
                current_label = pred["label"]
                current_confs = [pred["confidence"]]
                start_time = pred["start"]
            else:
                current_confs.append(pred["confidence"])

        segments.append((
            start_time,
            clip_predictions[-1]["end"],
            current_label,
            np.mean(current_confs),
        ))

    # Show results
    print(f"\n  Kinetics-400 predictions (top-5 per clip):")
    for pred in clip_predictions[:5]:
        print(f"    {pred['start']:5.1f}s - {pred['end']:5.1f}s:")
        for label, conf in pred["top5"]:
            print(f"      {conf:.3f} {label}")

    print(f"\n  Merged segments: {len(segments)}")
    for s in segments:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]} (conf={s[3]:.2f})")

    return SegmentResult(
        method="VideoMAE",
        video=str(video_path),
        segments=segments,
        runtime_total_s=elapsed,
        runtime_per_frame_ms=elapsed / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"videomae-base-finetuned-kinetics, {len(clip_predictions)} clips, Kinetics-400 labels",
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 4: X-CLIP Zero-Shot Video Classification
# ═══════════════════════════════════════════════════════════════════════

def test_xclip(video_path: str) -> SegmentResult:
    """X-CLIP zero-shot temporal action classification."""
    print("\n" + "=" * 70)
    print("METHOD 4: X-CLIP Zero-Shot Video Classification")
    print("=" * 70)

    reset_gpu()
    from transformers import XCLIPProcessor, XCLIPModel

    frames, fps, total = load_video_frames(video_path, resize=(224, 224))
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    # Load X-CLIP
    model_name = "microsoft/xclip-base-patch32"
    print(f"  Loading {model_name}...")
    from transformers import VideoMAEImageProcessor, AutoTokenizer
    processor = XCLIPProcessor.from_pretrained(model_name)
    img_processor = VideoMAEImageProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name).cuda().eval()

    # Zero-shot labels
    action_labels = [
        "dribbling a basketball",
        "shooting a basketball",
        "dunking a basketball",
        "doing push-ups",
        "doing squats",
        "doing jumping jacks",
        "stretching",
        "punching",
        "kicking",
        "wrestling",
        "running",
        "standing still",
        "lifting weights",
        "doing burpees",
        "playing soccer",
        "resting",
    ]

    # X-CLIP takes 8 frames per clip
    clip_len = 8
    stride = 4

    t0 = time.perf_counter()
    clip_predictions = []

    # Tokenize text labels once
    text_inputs = tokenizer(action_labels, padding=True, return_tensors="pt")
    text_inputs = {k: v.cuda() for k, v in text_inputs.items()}

    from PIL import Image

    with torch.no_grad():
        for start in range(0, n_frames - clip_len + 1, stride):
            clip = [Image.fromarray(f) for f in frames[start : start + clip_len]]
            # Process video frames separately
            vid_inputs = img_processor(clip, return_tensors="pt")
            inputs = {**text_inputs, "pixel_values": vid_inputs["pixel_values"].cuda()}

            outputs = model(**inputs)
            # outputs.logits_per_video shape: (1, n_labels)
            probs = F.softmax(outputs.logits_per_video, dim=-1)[0]
            top_conf, top_idx = probs.max(dim=-1)

            clip_predictions.append({
                "start": start / fps,
                "end": (start + clip_len) / fps,
                "label": action_labels[top_idx.item()],
                "confidence": top_conf.item(),
                "all_probs": {
                    action_labels[i]: probs[i].item()
                    for i in range(len(action_labels))
                },
            })

    elapsed = time.perf_counter() - t0
    print(f"  Processed {len(clip_predictions)} clips")
    print(f"  Time: {elapsed:.2f}s ({elapsed / n_frames * 1000:.1f} ms/frame)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    # Build segments
    segments = []
    if clip_predictions:
        current_label = clip_predictions[0]["label"]
        current_confs = [clip_predictions[0]["confidence"]]
        start_time = clip_predictions[0]["start"]

        for pred in clip_predictions[1:]:
            if pred["label"] != current_label:
                segments.append((
                    start_time,
                    pred["start"],
                    current_label,
                    np.mean(current_confs),
                ))
                current_label = pred["label"]
                current_confs = [pred["confidence"]]
                start_time = pred["start"]
            else:
                current_confs.append(pred["confidence"])

        segments.append((
            start_time,
            clip_predictions[-1]["end"],
            current_label,
            np.mean(current_confs),
        ))

    # Merge short segments
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < 0.3:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2], prev[3])
        else:
            merged.append(seg)
    segments = merged

    print(f"\n  X-CLIP predictions sample:")
    for pred in clip_predictions[:5]:
        sorted_probs = sorted(pred["all_probs"].items(), key=lambda x: -x[1])[:3]
        print(f"    {pred['start']:5.1f}s - {pred['end']:5.1f}s: {pred['label']} ({pred['confidence']:.3f})")
        for label, p in sorted_probs:
            print(f"      {p:.3f} {label}")

    print(f"\n  Merged segments: {len(segments)}")
    for s in segments:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]} (conf={s[3]:.2f})")

    return SegmentResult(
        method="X-CLIP",
        video=str(video_path),
        segments=segments,
        runtime_total_s=elapsed,
        runtime_per_frame_ms=elapsed / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"xclip-base-patch32, {len(action_labels)} labels, {len(segments)} segments",
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 5: 1D-CNN on Pose Features (Self-Supervised)
# ═══════════════════════════════════════════════════════════════════════

def test_pose_1dcnn(video_path: str) -> SegmentResult:
    """
    1D-CNN on pose features. Uses velocity-based boundaries as pseudo-labels
    and trains a small temporal convolution network to classify segments.
    This tests whether learnable temporal patterns beat handcrafted features.
    """
    print("\n" + "=" * 70)
    print("METHOD 5: 1D-CNN on Pose Features (Self-Supervised)")
    print("=" * 70)

    reset_gpu()

    # Extract pose features using the project's own pipeline
    sys.path.insert(0, str(PROJECT))
    from brace.core.motion_segments import (
        normalize_frame,
        detect_motion_boundaries,
        segment_motions,
        cluster_segments,
    )

    # Use YOLO11-pose for quick feature extraction
    from ultralytics import YOLO

    yolo = YOLO("yolo11n-pose.pt")
    frames_rgb, fps, total = load_video_frames(video_path)
    n_frames = len(frames_rgb)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    # Extract pose features
    t0 = time.perf_counter()
    COCO_TO_MP = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
    FEATURE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    raw_features = []
    valid_indices = []
    for fi, frame in enumerate(frames_rgb):
        results = yolo(frame, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) x,y,conf
            # Map COCO -> MediaPipe subset
            landmarks = np.zeros((33, 2))
            for coco_idx, mp_idx in COCO_TO_MP.items():
                if coco_idx < len(kp) and kp[coco_idx][2] > 0.3:
                    landmarks[mp_idx] = kp[coco_idx][:2]

            normed = normalize_frame(landmarks)
            if normed is not None:
                feat = normed[FEATURE_INDICES].flatten()
                raw_features.append(feat)
                valid_indices.append(fi)

    pose_extract_time = time.perf_counter() - t0
    features = np.array(raw_features)
    print(f"  Extracted {features.shape[0]} pose features, dim={features.shape[1]}")
    print(f"  Pose extraction: {pose_extract_time:.2f}s ({pose_extract_time / n_frames * 1000:.1f} ms/frame)")

    # Current method: velocity-based segmentation + agglomerative clustering
    segments_list = segment_motions(features, valid_indices, fps=fps, min_segment_sec=1.0)
    clustered = cluster_segments(segments_list, distance_threshold=3.5)
    n_seg = len(clustered)
    n_clust = len(set(s["cluster"] for s in clustered))
    print(f"  Current method: {n_seg} segments, {n_clust} clusters")

    # Now build a 1D-CNN that learns temporal patterns
    # Use the cluster labels as pseudo-labels for supervised training
    t1 = time.perf_counter()

    # Create frame-level labels from segments
    n_valid = len(features)
    frame_labels = np.zeros(n_valid, dtype=np.int64)
    for seg in clustered:
        s = seg["start_valid"]
        e = seg["end_valid"]
        frame_labels[s:e] = seg["cluster"]

    feat_dim = features.shape[1]
    n_classes = max(n_clust, 2)  # at least 2 classes for the CNN

    class TemporalCNN(torch.nn.Module):
        def __init__(self, in_dim, n_classes, hidden=64, kernel=15):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(in_dim, hidden, kernel, padding=kernel // 2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(hidden, n_classes, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = TemporalCNN(feat_dim, n_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    X = torch.from_numpy(features.T).float().unsqueeze(0).cuda()  # (1, feat_dim, T)
    Y = torch.from_numpy(frame_labels).long().cuda()  # (T,)

    for epoch in range(200):
        logits = model(X)[0]  # (n_classes, T)
        loss = loss_fn(logits.T, Y)  # need (T, n_classes) vs (T,)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(X)[0]  # (n_classes, T)
        pred_labels = logits.argmax(dim=0).cpu().numpy()

    cnn_time = time.perf_counter() - t1

    # Convert to segments
    segments = []
    current_label = pred_labels[0]
    start_frame = 0
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != current_label:
            segments.append((
                start_frame / fps,
                i / fps,
                f"action_{current_label}",
                1.0,
            ))
            current_label = pred_labels[i]
            start_frame = i
    segments.append((
        start_frame / fps,
        len(pred_labels) / fps,
        f"action_{current_label}",
        1.0,
    ))

    # Merge very short
    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < 0.3:
            prev = merged[-1]
            merged[-1] = (prev[0], seg[1], prev[2], prev[3])
        else:
            merged.append(seg)
    segments = merged

    total_time = pose_extract_time + cnn_time

    # Accuracy vs pseudo-labels
    acc = (pred_labels == frame_labels).mean()
    print(f"  1D-CNN training + inference: {cnn_time:.2f}s")
    print(f"  Total pipeline: {total_time:.2f}s ({total_time / n_frames * 1000:.1f} ms/frame)")
    print(f"  Accuracy vs pseudo-labels: {acc:.1%}")
    print(f"  CNN segments: {len(segments)} (vs {n_seg} original)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    for s in segments:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]}")

    return SegmentResult(
        method="1D-CNN-Pose",
        video=str(video_path),
        segments=segments,
        runtime_total_s=total_time,
        runtime_per_frame_ms=total_time / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"3-layer 1D-CNN, {feat_dim}D input, {n_classes} classes, acc={acc:.1%}, pose+CNN time",
    )


# ═══════════════════════════════════════════════════════════════════════
# METHOD 6 (Bonus): CLIP Feature Clustering (no text, just visual similarity)
# ═══════════════════════════════════════════════════════════════════════

def test_clip_clustering(video_path: str) -> SegmentResult:
    """CLIP visual features + temporal clustering (no text labels)."""
    print("\n" + "=" * 70)
    print("METHOD 6: CLIP Visual Feature Clustering (Unsupervised)")
    print("=" * 70)

    reset_gpu()
    import open_clip

    frames, fps, total = load_video_frames(video_path, resize=(224, 224))
    n_frames = len(frames)
    print(f"  Loaded {n_frames} frames at {fps:.1f} fps")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.cuda().eval()

    # Extract per-frame features
    t0 = time.perf_counter()
    all_features = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = frames[i : i + batch_size]
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            batch_t = (batch_t - mean) / std
            batch_t = batch_t.cuda()
            feats = model.encode_image(batch_t)
            all_features.append(feats.cpu().numpy())

    elapsed = time.perf_counter() - t0
    features = np.concatenate(all_features)
    print(f"  Extracted {features.shape[0]} features, dim={features.shape[1]}")
    print(f"  Time: {elapsed:.2f}s ({elapsed / n_frames * 1000:.1f} ms/frame)")
    print(f"  GPU peak: {gpu_mem_mb():.0f} MB")

    # Temporal clustering
    segments, labels = cluster_features(features, fps, "CLIP-Cluster")
    n_clusters = len(np.unique(labels))
    print(f"  Found {n_clusters} clusters -> {len(segments)} segments")
    for s in segments:
        print(f"    {s[0]:5.1f}s - {s[1]:5.1f}s : {s[2]} (conf={s[3]:.2f})")

    return SegmentResult(
        method="CLIP-Clustering",
        video=str(video_path),
        segments=segments,
        runtime_total_s=elapsed,
        runtime_per_frame_ms=elapsed / n_frames * 1000,
        gpu_mem_peak_mb=gpu_mem_mb(),
        n_frames=n_frames,
        notes=f"ViT-B/32 512D features, ward clustering, {n_clusters} clusters",
    )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def print_comparison(results: list[SegmentResult]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)

    print(f"\n{'Method':<20} {'Segments':>8} {'ms/frame':>10} {'GPU MB':>10} {'Notes'}")
    print("-" * 90)
    for r in results:
        print(f"{r.method:<20} {len(r.segments):>8} {r.runtime_per_frame_ms:>10.1f} {r.gpu_mem_peak_mb:>10.0f} {r.notes[:45]}")

    print("\n" + "-" * 90)
    print("Real-time feasibility (target: <100ms/frame):")
    for r in results:
        feasible = "YES" if r.runtime_per_frame_ms < 100 else "NO"
        print(f"  {r.method:<20} {r.runtime_per_frame_ms:>8.1f} ms/frame  [{feasible}]")


def main():
    video = str(VIDEOS["basketball"])
    print(f"Testing on: {video}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []

    # Run each method
    try:
        results.append(test_r3d_features(video))
    except Exception as e:
        print(f"  R3D-18 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_clip_zeroshot(video))
    except Exception as e:
        print(f"  CLIP ZeroShot FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_videomae(video))
    except Exception as e:
        print(f"  VideoMAE FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_xclip(video))
    except Exception as e:
        print(f"  X-CLIP FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_pose_1dcnn(video))
    except Exception as e:
        print(f"  1D-CNN Pose FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        results.append(test_clip_clustering(video))
    except Exception as e:
        print(f"  CLIP Clustering FAILED: {e}")
        import traceback; traceback.print_exc()

    # Comparison
    if results:
        print_comparison(results)

    # Save results
    out_path = PROJECT / "experiments" / "action_segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "method": r.method,
                    "video": r.video,
                    "segments": r.segments,
                    "runtime_total_s": r.runtime_total_s,
                    "runtime_per_frame_ms": r.runtime_per_frame_ms,
                    "gpu_mem_peak_mb": r.gpu_mem_peak_mb,
                    "n_frames": r.n_frames,
                    "notes": r.notes,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
