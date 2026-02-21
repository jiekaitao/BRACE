#!/usr/bin/env python3
"""Validate the production code changes against all 14 demo videos.

Uses the actual production functions (not experimental overrides) to verify
the clustering improvements are working correctly.
"""
import sys, os, json, copy
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from brace.core.motion_segments import segment_motions, cluster_segments

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"

# Expanded labels from label-researcher
EXPANDED_LABELS = [
    "a person running at full speed", "a person jogging at a moderate pace",
    "a person walking", "a person sprinting",
    "a person dribbling a basketball while stationary",
    "a person dribbling a basketball while moving",
    "a person shooting a basketball jump shot",
    "a person doing a basketball layup", "a person dunking a basketball",
    "a person passing a basketball", "a person defending in basketball",
    "a person doing pushups", "a person doing squats",
    "a person doing pullups", "a person doing burpees",
    "a person doing jumping jacks", "a person doing lunges",
    "a person doing mountain climbers", "a person doing high knees",
    "a person doing box jumps", "a person doing planks",
    "a person doing crunches", "a person doing stretching",
    "a person lifting weights overhead",
    "a person doing bench press", "a person doing deadlifts",
    "a person doing barbell curls", "a person doing rows",
    "a person doing kettlebell swings", "a person doing clean and jerk",
    "a person doing lateral raises", "a person doing tricep extensions",
    "a person shadowboxing", "a person punching a heavy bag",
    "a person doing boxing footwork", "a person in a fighting stance",
    "a person jumping rope with both feet",
    "a person doing double unders with a jump rope",
    "a person swimming freestyle", "a person swimming backstroke",
    "a person swimming breaststroke", "a person doing a flip turn",
    "a person hitting a tennis forehand", "a person hitting a tennis backhand",
    "a person serving in tennis", "a person doing a tennis volley",
    "a person doing a tennis overhead smash",
    "a person doing a tennis split step ready position",
    "a person dribbling a soccer ball",
    "a person kicking a soccer ball",
    "a person doing soccer juggling tricks",
    "a person doing soccer footwork drills",
    "a person doing a yoga standing pose",
    "a person doing a yoga balance pose",
    "a person doing a yoga seated pose",
    "a person doing a yoga inversion",
    "a person transitioning between yoga poses",
    "a person doing bodyweight dips",
    "a person doing handstand practice",
    "a person doing muscle ups",
    "a person standing still", "a person resting between exercises",
    "a person warming up",
]


def classify_cluster(video_path, frame_ranges, clip_model, preprocess, text_features, device):
    total = sum(e - s + 1 for s, e in frame_ranges)
    n_sample = min(8, total)
    all_idxs = []
    for s, e in frame_ranges:
        all_idxs.extend(range(s, e + 1))
    if not all_idxs:
        return "unknown", 0.0
    sample_idxs = sorted(set(
        all_idxs[i] for i in np.linspace(0, len(all_idxs) - 1, n_sample, dtype=int)
    ))
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    sample_set = set(sample_idxs)
    max_idx = max(sample_idxs)
    while cap.isOpened() and idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    if not frames:
        return "unknown", 0.0
    images = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    avg = img_feat.mean(dim=0, keepdim=True)
    avg = avg / avg.norm(dim=-1, keepdim=True)
    sim = (avg @ text_features.T).squeeze()
    best = sim.argmax().item()
    return EXPANDED_LABELS[best], float(sim[best])


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    tokens = clip.tokenize(EXPANDED_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    print(f"Videos: {len(videos)}\n")

    pass_count = 0
    total = 0
    results = {}

    for fname in videos:
        cache_path = os.path.join(CACHE_DIR, f"{fname}.feats.npz")
        if not os.path.exists(cache_path):
            print(f"  SKIP {fname} (no cache)")
            continue
        data = np.load(cache_path)
        feats = data["features"]
        vi = data["valid_indices"].tolist()
        fps = float(data["fps"])
        if len(feats) < 10:
            print(f"  SKIP {fname} (too few features)")
            continue

        total += 1
        path = os.path.join(VIDEO_DIR, fname)

        # Use production functions directly
        segments = segment_motions(feats, vi, fps)
        if len(segments) >= 2:
            clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0, fps=fps)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0

        n_clusters = len(set(s["cluster"] for s in clustered))

        # CLIP classify
        cluster_ranges = {}
        for seg in clustered:
            cid = seg["cluster"]
            cluster_ranges.setdefault(cid, []).append((seg["start_frame"], seg["end_frame"]))

        cluster_labels = {}
        for cid, ranges in sorted(cluster_ranges.items()):
            label, conf = classify_cluster(path, ranges, clip_model, preprocess, text_features, device)
            cluster_labels[cid] = {"label": label, "confidence": round(conf, 3)}

        labels_only = [v["label"] for v in cluster_labels.values()]
        all_distinct = len(labels_only) == len(set(labels_only))
        status = "PASS" if all_distinct else "FAIL"
        if all_distinct:
            pass_count += 1

        short = [l.replace("a person ", "")[:35] for l in labels_only]
        print(f"  {status:4s} {fname:35s} segs={len(clustered):2d} clust={n_clusters:2d} {short}")

        results[fname] = {
            "n_segments": len(clustered),
            "n_clusters": n_clusters,
            "labels": labels_only,
            "all_distinct": all_distinct,
        }

    print(f"\nFINAL: {pass_count}/{total} pass ({100*pass_count/total:.0f}%)")

    out = "/mnt/Data/GitHub/BRACE/experiments/production_validation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
