#!/usr/bin/env python3
"""Test label quality improvements for CLIP-based cluster classification.

Tests four approaches:
  A) ViT-L/14 (larger model) vs ViT-B/32 baseline
  B) Better label vocabulary — sport-specific sub-labels
  C) Prompt engineering — different prompt templates
  D) Multi-frame voting — per-frame classification + majority vote

Focuses on problem videos: basketball_solo, gym_workout, tennis_practice.
"""
import sys, os, copy, time
import cv2
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from brace.core.motion_segments import segment_motions, cluster_segments

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"

# Focus on problem videos
PROBLEM_VIDEOS = ["basketball_solo.mp4", "gym_workout.mp4", "tennis_practice.mp4"]

# ---------- LABEL VOCABULARIES ----------

# Baseline: same 41 labels from threshold_sweep_v2.py
LABELS_BASELINE = [
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

# Expanded vocabulary with sport-specific sub-labels
LABELS_EXPANDED = [
    # Basketball — fine-grained
    "a person dribbling a basketball in place",
    "a person dribbling a basketball while moving",
    "a person crossing over with a basketball",
    "a person shooting a basketball jump shot",
    "a person shooting a basketball free throw",
    "a person doing a basketball layup drive",
    "a person dunking a basketball",
    "a person passing a basketball",
    "a person defending in basketball with arms out",
    "a person rebounding a basketball",
    # Tennis — fine-grained
    "a person serving a tennis ball overhead",
    "a person hitting a tennis forehand",
    "a person hitting a tennis backhand",
    "a person hitting a tennis volley at the net",
    "a person doing a tennis overhead smash",
    "a person in a ready stance waiting for a tennis ball",
    "a person running on a tennis court to return a ball",
    "a person doing a tennis split step",
    # Gym / Workout — fine-grained
    "a person doing pushups on the floor",
    "a person doing bodyweight squats",
    "a person doing barbell squats with a rack",
    "a person doing pullups on a bar",
    "a person doing dumbbell curls",
    "a person doing bench press with a barbell",
    "a person doing overhead press with dumbbells",
    "a person doing bent over rows",
    "a person doing deadlifts with a barbell",
    "a person doing kettlebell swings",
    "a person doing burpees",
    "a person doing jumping jacks",
    "a person doing lunges forward",
    "a person doing walking lunges",
    "a person doing lateral raises with dumbbells",
    "a person doing tricep dips",
    "a person doing mountain climbers on the floor",
    "a person doing plank hold",
    "a person doing crunches on the floor",
    "a person doing box jumps onto a platform",
    "a person doing high knees running in place",
    "a person doing jump rope",
    "a person doing cable rows",
    "a person doing lat pulldowns on a machine",
    "a person doing leg press on a machine",
    "a person doing leg curls on a machine",
    "a person doing chest flys with dumbbells",
    "a person doing battle ropes exercise",
    "a person stretching on the floor",
    "a person doing yoga pose",
    "a person resting between exercises",
    "a person standing still",
    # General movement
    "a person running fast",
    "a person jogging slowly",
    "a person walking",
    # Boxing
    "a person shadowboxing throwing punches",
    "a person boxing hitting a heavy bag",
    "a person doing boxing footwork",
    # Soccer
    "a person dribbling a soccer ball with feet",
    "a person kicking a soccer ball",
    "a person juggling a soccer ball",
    # Swimming
    "a person swimming freestyle",
    "a person swimming backstroke",
]

# ---------- PROMPT TEMPLATES ----------

PROMPT_TEMPLATES = {
    "baseline": "a person {action}",
    "video_frame": "a video frame showing a person {action}",
    "athlete": "an athlete {action}",
    "sports_clip": "a sports video clip of a person {action}",
    "photo_of": "a photo of a person {action}",
}

# Action stems (no "a person" prefix) for prompt template testing
ACTION_STEMS_BASELINE = [
    "running", "jogging", "walking",
    "dribbling a basketball", "shooting a basketball",
    "doing a layup", "dunking a basketball",
    "doing pushups", "doing squats", "doing pullups",
    "lifting weights", "doing bench press",
    "jumping rope", "stretching", "doing yoga",
    "boxing", "punching a bag", "shadowboxing",
    "swimming", "serving in tennis", "hitting a tennis ball",
    "dribbling a soccer ball", "kicking a soccer ball",
    "standing still", "resting",
    "doing burpees", "doing jumping jacks",
    "doing lunges", "doing kettlebell swings",
    "doing overhead press", "doing rows",
    "doing clean and jerk", "doing crunches",
    "doing planks", "doing deadlifts",
    "doing barbell curls", "doing tricep extensions",
    "doing lateral raises", "doing mountain climbers",
    "doing high knees", "doing box jumps",
]

ACTION_STEMS_EXPANDED = [
    # Basketball
    "dribbling a basketball in place",
    "dribbling a basketball while moving",
    "crossing over with a basketball",
    "shooting a basketball jump shot",
    "shooting a basketball free throw",
    "doing a basketball layup drive",
    "dunking a basketball",
    "passing a basketball",
    "defending in basketball with arms out",
    "rebounding a basketball",
    # Tennis
    "serving a tennis ball overhead",
    "hitting a tennis forehand",
    "hitting a tennis backhand",
    "hitting a tennis volley at the net",
    "doing a tennis overhead smash",
    "in a ready stance waiting for a tennis ball",
    "running on a tennis court to return a ball",
    "doing a tennis split step",
    # Gym
    "doing pushups on the floor",
    "doing bodyweight squats",
    "doing barbell squats with a rack",
    "doing pullups on a bar",
    "doing dumbbell curls",
    "doing bench press with a barbell",
    "doing overhead press with dumbbells",
    "doing bent over rows",
    "doing deadlifts with a barbell",
    "doing kettlebell swings",
    "doing burpees",
    "doing jumping jacks",
    "doing lunges forward",
    "doing walking lunges",
    "doing lateral raises with dumbbells",
    "doing tricep dips",
    "doing mountain climbers on the floor",
    "doing plank hold",
    "doing crunches on the floor",
    "doing box jumps onto a platform",
    "doing high knees running in place",
    "doing jump rope",
    "doing cable rows",
    "doing lat pulldowns on a machine",
    "doing leg press on a machine",
    "doing leg curls on a machine",
    "doing chest flys with dumbbells",
    "doing battle ropes exercise",
    "stretching on the floor",
    "doing yoga pose",
    "resting between exercises",
    "standing still",
    # General
    "running fast", "jogging slowly", "walking",
    # Boxing
    "shadowboxing throwing punches",
    "boxing hitting a heavy bag",
    "doing boxing footwork",
    # Soccer
    "dribbling a soccer ball with feet",
    "kicking a soccer ball",
    "juggling a soccer ball",
    # Swimming
    "swimming freestyle", "swimming backstroke",
]


def load_cached_features(video_name):
    """Load pre-computed pose features from cache."""
    cache_path = os.path.join(CACHE_DIR, f"{video_name}.feats.npz")
    if not os.path.exists(cache_path):
        print(f"  WARNING: No cached features for {video_name}")
        return None, [], 30.0
    data = np.load(cache_path)
    return data["features"], data["valid_indices"].tolist(), float(data["fps"])


def get_cluster_frame_ranges(feats, valid_indices, fps, threshold=2.0):
    """Segment + cluster, return cluster_id -> list of (start_frame, end_frame)."""
    segments = segment_motions(feats, valid_indices, fps, min_segment_sec=1.0)
    if len(segments) < 2:
        segments = segment_motions(feats, valid_indices, fps, min_segment_sec=0.5)

    if len(segments) >= 2:
        clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=threshold)
    else:
        clustered = segments
        for s in clustered:
            s["cluster"] = 0

    cluster_ranges = {}
    for seg in clustered:
        cid = seg["cluster"]
        cluster_ranges.setdefault(cid, []).append(
            (seg["start_frame"], seg["end_frame"])
        )
    n_clusters = len(cluster_ranges)
    return cluster_ranges, n_clusters


def sample_frames_from_ranges(video_path, frame_ranges, n_sample=8):
    """Read sampled frames from a video for given frame ranges."""
    from PIL import Image

    all_frame_idxs = []
    for s, e in frame_ranges:
        all_frame_idxs.extend(range(s, e + 1))

    if not all_frame_idxs:
        return []

    sample_idxs = sorted(set(
        all_frame_idxs[i]
        for i in np.linspace(0, len(all_frame_idxs) - 1, min(n_sample, len(all_frame_idxs)), dtype=int)
    ))

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    sample_set = set(sample_idxs)
    max_idx = max(sample_idxs)

    while cap.isOpened() and frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        frame_idx += 1
    cap.release()
    return frames


def classify_averaged(frames, clip_model, preprocess, text_features, labels, device):
    """Average image features across frames, then match to labels. Returns (label, confidence)."""
    from PIL import Image

    if not frames:
        return "unknown", 0.0

    images = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
    with torch.no_grad():
        img_feats = clip_model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    avg_feat = img_feats.mean(dim=0, keepdim=True)
    avg_feat = avg_feat / avg_feat.norm(dim=-1, keepdim=True)
    sim = (avg_feat @ text_features.T).squeeze()

    best_idx = sim.argmax().item()
    return labels[best_idx], float(sim[best_idx])


def classify_voting(frames, clip_model, preprocess, text_features, labels, device):
    """Classify each frame independently, return majority vote. Returns (label, vote_fraction)."""
    from PIL import Image

    if not frames:
        return "unknown", 0.0

    images = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
    with torch.no_grad():
        img_feats = clip_model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    sims = img_feats @ text_features.T  # (N, num_labels)
    per_frame_labels = [labels[idx.item()] for idx in sims.argmax(dim=-1)]

    counts = Counter(per_frame_labels)
    majority_label, majority_count = counts.most_common(1)[0]
    vote_fraction = majority_count / len(per_frame_labels)

    return majority_label, vote_fraction


def encode_labels(clip_model, labels, device):
    """Encode a list of text labels into normalized CLIP features."""
    import clip
    tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


def run_experiment(name, clip_model, preprocess, text_features, labels, device,
                   video_data, classify_fn=classify_averaged, threshold=2.0):
    """Run classification for all problem videos and print results."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  Labels: {len(labels)}, Threshold: {threshold}")
    print(f"{'='*70}")

    results = {}
    for vname in PROBLEM_VIDEOS:
        if vname not in video_data:
            continue
        path, feats, vi, fps = video_data[vname]
        cluster_ranges, n_clusters = get_cluster_frame_ranges(feats, vi, fps, threshold)

        cluster_labels = {}
        for cid, ranges in sorted(cluster_ranges.items()):
            frames = sample_frames_from_ranges(path, ranges, n_sample=8)
            label, conf = classify_fn(frames, clip_model, preprocess, text_features, labels, device)
            n_frames = sum(e - s + 1 for s, e in ranges)
            cluster_labels[cid] = {"label": label, "confidence": round(conf, 3), "n_frames": n_frames}

        labels_only = [v["label"] for v in cluster_labels.values()]
        unique = set(labels_only)
        all_distinct = len(labels_only) == len(unique)
        n_unique = len(unique)

        status = "PASS" if all_distinct else "FAIL"
        print(f"\n  {status} {vname} — {n_clusters} clusters, {n_unique} unique labels")
        for cid, info in sorted(cluster_labels.items()):
            print(f"    cluster {cid:2d}: {info['label']} (conf={info['confidence']:.3f}, frames={info['n_frames']})")

        results[vname] = {
            "n_clusters": n_clusters,
            "n_unique_labels": n_unique,
            "all_distinct": all_distinct,
            "cluster_labels": cluster_labels,
        }

    return results


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load features for problem videos
    video_data = {}
    for vname in PROBLEM_VIDEOS:
        path = os.path.join(VIDEO_DIR, vname)
        feats, vi, fps = load_cached_features(vname)
        if feats is not None and len(feats) >= 10:
            video_data[vname] = (path, feats, vi, fps)
            print(f"  {vname}: {feats.shape[0]} features ({feats.shape[1]}D)")
        else:
            print(f"  {vname}: SKIPPED")

    if not video_data:
        print("No video data loaded, exiting.")
        return

    all_results = {}

    # ====== EXPERIMENT A: ViT-B/32 baseline vs ViT-L/14 ======
    print("\n" + "#" * 70)
    print("# SECTION A: Model Comparison (ViT-B/32 vs ViT-L/14 vs ViT-L/14@336px)")
    print("#" * 70)

    for model_name in ["ViT-B/32", "ViT-L/14", "ViT-L/14@336px"]:
        print(f"\nLoading {model_name}...")
        t0 = time.time()
        m, p = clip.load(model_name, device=device)
        print(f"  Loaded in {time.time()-t0:.1f}s")
        tf = encode_labels(m, LABELS_BASELINE, device)

        key = f"A_{model_name}_baseline_labels"
        all_results[key] = run_experiment(
            f"{model_name} + Baseline Labels (41)",
            m, p, tf, LABELS_BASELINE, device, video_data,
        )

        # Free VRAM between models
        if model_name != "ViT-L/14@336px":
            del m, p, tf
            torch.cuda.empty_cache()

    # Keep ViT-L/14@336px loaded for further experiments (best model)
    # If it's too slow, fall back to ViT-L/14
    print(f"\nUsing ViT-L/14@336px for remaining experiments")
    best_model, best_preprocess = clip.load("ViT-L/14@336px", device=device)

    # ====== EXPERIMENT B: Expanded label vocabulary ======
    print("\n" + "#" * 70)
    print("# SECTION B: Expanded Label Vocabulary")
    print("#" * 70)

    tf_expanded = encode_labels(best_model, LABELS_EXPANDED, device)
    all_results["B_expanded_labels"] = run_experiment(
        f"ViT-L/14@336px + Expanded Labels ({len(LABELS_EXPANDED)})",
        best_model, best_preprocess, tf_expanded, LABELS_EXPANDED, device, video_data,
    )

    # ====== EXPERIMENT C: Prompt Templates ======
    print("\n" + "#" * 70)
    print("# SECTION C: Prompt Engineering")
    print("#" * 70)

    for tmpl_name, tmpl in PROMPT_TEMPLATES.items():
        # Build labels by applying template to expanded action stems
        prompted_labels = [tmpl.replace("a person {action}", "").strip() if False
                           else tmpl.format(action=stem) for stem in ACTION_STEMS_EXPANDED]
        tf_prompted = encode_labels(best_model, prompted_labels, device)

        key = f"C_prompt_{tmpl_name}"
        all_results[key] = run_experiment(
            f"ViT-L/14@336px + Prompt: \"{tmpl}\" ({len(prompted_labels)} labels)",
            best_model, best_preprocess, tf_prompted, prompted_labels, device, video_data,
        )

    # ====== EXPERIMENT D: Multi-frame Voting ======
    print("\n" + "#" * 70)
    print("# SECTION D: Multi-frame Voting vs Averaging")
    print("#" * 70)

    # Use expanded labels with best model
    tf_expanded = encode_labels(best_model, LABELS_EXPANDED, device)

    all_results["D_averaging"] = run_experiment(
        "ViT-L/14@336px + Expanded + Averaging (baseline)",
        best_model, best_preprocess, tf_expanded, LABELS_EXPANDED, device, video_data,
        classify_fn=classify_averaged,
    )

    all_results["D_voting"] = run_experiment(
        "ViT-L/14@336px + Expanded + Majority Voting",
        best_model, best_preprocess, tf_expanded, LABELS_EXPANDED, device, video_data,
        classify_fn=classify_voting,
    )

    # Also test voting with more frames (16 instead of 8)
    def classify_voting_16(frames_unused, clip_model, preprocess, text_features, labels, device):
        # This wrapper is a hack — we need to re-sample more frames
        # Instead, we'll run this separately below
        pass

    # Re-run voting with 16 frames by modifying sample count
    print(f"\n{'='*70}")
    print("EXPERIMENT: ViT-L/14@336px + Expanded + Majority Voting (16 frames)")
    print(f"{'='*70}")

    voting_16_results = {}
    for vname in PROBLEM_VIDEOS:
        if vname not in video_data:
            continue
        path, feats, vi, fps = video_data[vname]
        cluster_ranges, n_clusters = get_cluster_frame_ranges(feats, vi, fps, threshold=2.0)

        cluster_labels = {}
        for cid, ranges in sorted(cluster_ranges.items()):
            frames = sample_frames_from_ranges(path, ranges, n_sample=16)
            label, conf = classify_voting(
                frames, best_model, best_preprocess, tf_expanded, LABELS_EXPANDED, device
            )
            n_frames_total = sum(e - s + 1 for s, e in ranges)
            cluster_labels[cid] = {"label": label, "confidence": round(conf, 3), "n_frames": n_frames_total}

        labels_only = [v["label"] for v in cluster_labels.values()]
        unique = set(labels_only)
        all_distinct = len(labels_only) == len(unique)
        n_unique = len(unique)

        status = "PASS" if all_distinct else "FAIL"
        print(f"\n  {status} {vname} — {n_clusters} clusters, {n_unique} unique labels")
        for cid, info in sorted(cluster_labels.items()):
            print(f"    cluster {cid:2d}: {info['label']} (conf={info['confidence']:.3f}, frames={info['n_frames']})")

        voting_16_results[vname] = {
            "n_clusters": n_clusters,
            "n_unique_labels": n_unique,
            "all_distinct": all_distinct,
            "cluster_labels": cluster_labels,
        }

    all_results["D_voting_16frames"] = voting_16_results

    # ====== FINAL SUMMARY ======
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print(f"\n{'Experiment':<55} {'basketball':<14} {'gym':<14} {'tennis':<14} {'Score':<6}")
    print("-" * 103)

    for key, result in all_results.items():
        scores = []
        cells = []
        for vname in PROBLEM_VIDEOS:
            if vname in result:
                r = result[vname]
                nc = r["n_clusters"]
                nu = r["n_unique_labels"]
                distinct = r["all_distinct"]
                cell = f"{nu}/{nc}"
                if distinct:
                    cell += " PASS"
                    scores.append(1)
                else:
                    cell += " FAIL"
                    scores.append(0)
                cells.append(cell)
            else:
                cells.append("N/A")
                scores.append(0)

        total = f"{sum(scores)}/3"
        # Truncate experiment name
        short_key = key[:53]
        print(f"  {short_key:<53} {cells[0]:<14} {cells[1]:<14} {cells[2]:<14} {total}")

    # Save all results
    import json
    out_path = "/mnt/Data/GitHub/BRACE/experiments/label_quality_results.json"
    # Convert cluster_labels keys from int to str for JSON
    serializable = {}
    for exp_key, exp_result in all_results.items():
        serializable[exp_key] = {}
        for vname, vresult in exp_result.items():
            sr = dict(vresult)
            if "cluster_labels" in sr:
                sr["cluster_labels"] = {str(k): v for k, v in sr["cluster_labels"].items()}
            serializable[exp_key][vname] = sr
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
