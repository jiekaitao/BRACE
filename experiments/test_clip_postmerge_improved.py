#!/usr/bin/env python3
"""Test CLIP post-merge with the improved pipeline (average linkage, wider smoothing, min_seg=2.0).

For each threshold in [1.5, 2.0, 2.5, 3.0, 4.0]:
  1. Run production segment_motions() + cluster_segments()
  2. Classify each cluster with CLIP ViT-L/14 + 61 expanded labels
  3. Count pass/fail (all distinct labels)
  4. Apply CLIP post-merge (merge clusters sharing same label)
  5. Report pass rates: raw vs post-merge

Special attention to basketball_solo: with ViT-L/14 it should already have
distinct labels (dribble vs dunk), so post-merge should NOT collapse it.
"""
import sys, os, copy, time
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from brace.core.motion_segments import (
    segment_motions, cluster_segments, _merge_adjacent_clusters,
)

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"

THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 4.0]

# Expanded 61-label vocabulary (from final_validation.py)
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


def classify_cluster_with_clip(video_path, cluster_frame_ranges, clip_model, preprocess,
                                text_features, device):
    """Classify cluster frames with CLIP. Samples ~8 frames from the cluster."""
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
    return EXPANDED_LABELS[best_idx], float(similarity[best_idx])


def clip_post_merge(clustered_segments, cluster_labels):
    """Merge clusters that share the same CLIP label. Returns merged segments + new label map."""
    label_to_cids = {}
    for cid, info in cluster_labels.items():
        label_to_cids.setdefault(info["label"], []).append(cid)

    merge_map = {}
    for label, cids in label_to_cids.items():
        if len(cids) > 1:
            target = min(cids)
            for cid in cids:
                merge_map[cid] = target

    if not merge_map:
        return clustered_segments, cluster_labels, len(cluster_labels)

    merged_segments = copy.deepcopy(clustered_segments)
    for seg in merged_segments:
        old_cid = seg["cluster"]
        if old_cid in merge_map:
            seg["cluster"] = merge_map[old_cid]

    merged_segments = _merge_adjacent_clusters(merged_segments)

    new_labels = {}
    for cid, info in cluster_labels.items():
        target = merge_map.get(cid, cid)
        if target not in new_labels:
            new_labels[target] = info

    n_after = len(set(s["cluster"] for s in merged_segments))
    return merged_segments, new_labels, n_after


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading CLIP ViT-L/14...", flush=True)
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    text_tokens = clip.tokenize(EXPANDED_LABELS).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load cached features
    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    video_data = {}
    for fname in videos:
        cache_path = os.path.join(CACHE_DIR, f"{fname}.feats.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            feats = data["features"]
            vi = data["valid_indices"].tolist()
            fps = float(data["fps"])
            if len(feats) >= 10:
                video_data[fname] = (os.path.join(VIDEO_DIR, fname), feats, vi, fps)

    print(f"Loaded features for {len(video_data)} videos\n", flush=True)

    # Track results per threshold
    summary_rows = []

    for threshold in THRESHOLDS:
        print(f"\n{'='*80}", flush=True)
        print(f"  THRESHOLD = {threshold}", flush=True)
        print(f"{'='*80}", flush=True)

        raw_pass = 0
        merge_pass = 0
        flipped = []  # videos that go FAIL->PASS with post-merge
        basketball_detail = None

        for fname, (path, feats, vi, fps) in sorted(video_data.items()):
            # Run production pipeline
            segments = segment_motions(feats, vi, fps, min_segment_sec=2.0)
            if len(segments) >= 2:
                clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=threshold, fps=fps)
            else:
                clustered = segments
                for s in clustered:
                    s["cluster"] = 0

            n_clusters = len(set(s["cluster"] for s in clustered))

            # Build cluster frame ranges
            cluster_ranges = {}
            for seg in clustered:
                cid = seg["cluster"]
                cluster_ranges.setdefault(cid, []).append(
                    (seg["start_frame"], seg["end_frame"])
                )

            # CLIP classify each cluster
            cluster_labels = {}
            for cid, ranges in sorted(cluster_ranges.items()):
                label, conf = classify_cluster_with_clip(
                    path, ranges, clip_model, preprocess, text_features, device
                )
                cluster_labels[cid] = {"label": label, "confidence": round(conf, 3)}

            # Check raw distinctness
            labels_only = [v["label"] for v in cluster_labels.values()]
            raw_distinct = len(labels_only) == len(set(labels_only))
            if raw_distinct:
                raw_pass += 1

            # CLIP post-merge
            merged_segs, merged_labels, n_after_merge = clip_post_merge(
                clustered, cluster_labels
            )
            merged_labels_only = [v["label"] for v in merged_labels.values()]
            merge_distinct = len(merged_labels_only) == len(set(merged_labels_only))
            if merge_distinct:
                merge_pass += 1

            if not raw_distinct and merge_distinct:
                flipped.append(fname)

            raw_status = "PASS" if raw_distinct else "FAIL"
            merge_status = "PASS" if merge_distinct else "FAIL"
            short_labels = [l.replace("a person ", "").replace("doing ", "")[:35] for l in labels_only]

            flip_marker = " ** FLIP **" if (not raw_distinct and merge_distinct) else ""
            print(f"  {raw_status:4s}->{merge_status:4s}  {fname:35s}  "
                  f"clust={n_clusters:2d}->{n_after_merge:2d}  {short_labels}{flip_marker}",
                  flush=True)

            # Basketball solo special check
            if fname == "basketball_solo.mp4":
                basketball_detail = {
                    "n_clusters_raw": n_clusters,
                    "n_clusters_merged": n_after_merge,
                    "raw_labels": labels_only,
                    "merged_labels": merged_labels_only,
                    "raw_distinct": raw_distinct,
                    "merge_distinct": merge_distinct,
                }

        total = len(video_data)
        print(f"\n  THRESHOLD {threshold}: raw={raw_pass}/{total}, post-merge={merge_pass}/{total}",
              flush=True)
        if flipped:
            print(f"  Flipped FAIL->PASS: {flipped}", flush=True)

        summary_rows.append({
            "threshold": threshold,
            "raw_pass": raw_pass,
            "merge_pass": merge_pass,
            "total": total,
            "flipped": flipped,
        })

        if basketball_detail:
            print(f"\n  ** basketball_solo detail at t={threshold}:", flush=True)
            print(f"     raw: {basketball_detail['n_clusters_raw']} clusters, "
                  f"labels={basketball_detail['raw_labels']}, "
                  f"distinct={basketball_detail['raw_distinct']}", flush=True)
            print(f"     merged: {basketball_detail['n_clusters_merged']} clusters, "
                  f"labels={basketball_detail['merged_labels']}, "
                  f"distinct={basketball_detail['merge_distinct']}", flush=True)

    # Final summary table
    print(f"\n\n{'='*80}", flush=True)
    print("FINAL SUMMARY: CLIP Post-Merge with Improved Pipeline", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Threshold':>10}  {'Raw Pass':>10}  {'Post-Merge':>12}  {'Flipped':>10}", flush=True)
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}", flush=True)
    for row in summary_rows:
        t = row["threshold"]
        rp = row["raw_pass"]
        mp = row["merge_pass"]
        total = row["total"]
        nf = len(row["flipped"])
        print(f"  {t:>10.1f}  {rp:>4}/{total:<4}  {mp:>6}/{total:<4}  {nf:>10}", flush=True)

    print(f"\nFlipped videos (FAIL->PASS with post-merge) per threshold:", flush=True)
    for row in summary_rows:
        if row["flipped"]:
            print(f"  t={row['threshold']}: {row['flipped']}", flush=True)


if __name__ == "__main__":
    main()
