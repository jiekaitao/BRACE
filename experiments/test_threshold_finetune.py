#!/usr/bin/env python3
"""Fine-tune clustering threshold and deeply analyze the 5 failing videos.

Part A: Threshold sweep with production pipeline + CLIP ViT-L/14 + expanded labels.
Part B: Deep failure analysis with very specific CLIP prompts per failing video.
"""
import sys, os, json, copy, time
import cv2
import numpy as np
import torch
from PIL import Image
from collections import Counter

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")
from brace.core.motion_segments import segment_motions, cluster_segments

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"
GT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"

# Same expanded labels used in validate_production.py (ViT-L/14 validated)
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

THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# Videos known to fail at t=2.0 with CLIP ViT-L/14
FAILING_VIDEOS = [
    "gym_workout.mp4",
    "soccer_match2.mp4",
    "soccer_skills.mp4",
    "tennis_practice.mp4",
    "yoga_flow.mp4",
]

# Very specific CLIP prompts for failure analysis
FAILURE_SPECIFIC_LABELS = {
    "gym_workout.mp4": [
        "a person doing jumping jacks with arms raised overhead",
        "a person doing jumping jacks with arms at sides",
        "a person doing side leg raises while standing",
        "a person doing standing side lunges",
        "a person doing high knee marching",
        "a person doing butt kicks while standing",
        "a person doing squat jumps",
        "a person doing standing arm circles",
        "a person doing standing calf raises",
        "a person doing standing hip circles",
        "a person resting between exercises",
        "a person transitioning between exercises",
        "a person doing a standing warm up stretch",
        "a person doing lateral shuffles",
        "a person doing standing knee raises",
        "a person doing standing oblique crunches",
    ],
    "tennis_practice.mp4": [
        "a person hitting a tennis forehand with topspin",
        "a person hitting a tennis forehand slice",
        "a person hitting a flat tennis forehand",
        "a person hitting a tennis backhand with topspin",
        "a person hitting a tennis backhand slice",
        "a person hitting a flat tennis backhand",
        "a person hitting a two-handed tennis backhand",
        "a person serving a tennis ball",
        "a person doing a tennis volley at the net",
        "a person doing a tennis overhead smash",
        "a person in tennis ready position waiting for the ball",
        "a person doing a tennis split step",
        "a person running to hit a tennis ball",
        "a person recovering after hitting a tennis shot",
        "a person doing a tennis approach shot",
    ],
    "yoga_flow.mp4": [
        "a person doing cobra pose (lying face down, chest lifted)",
        "a person doing upward facing dog",
        "a person doing downward facing dog",
        "a person doing child's pose (kneeling, forehead on floor)",
        "a person doing a seated twist yoga pose",
        "a person doing a seated forward fold",
        "a person doing warrior one pose",
        "a person doing warrior two pose",
        "a person doing tree pose (standing on one leg)",
        "a person doing cat cow stretch on all fours",
        "a person doing pigeon pose",
        "a person doing bridge pose (lying on back, hips up)",
        "a person doing a standing yoga forward bend",
        "a person in savasana lying flat on back",
        "a person doing a kneeling yoga lunge",
        "a person transitioning between yoga poses",
    ],
    "soccer_match2.mp4": [
        "a person doing a short pass with foot",
        "a person doing a long soccer kick",
        "a person dribbling a soccer ball closely",
        "a person running with a soccer ball",
        "a person running without the ball",
        "a person doing a sliding tackle",
        "a person trapping a soccer ball with foot",
        "a person heading a soccer ball",
        "a person standing with hands on hips on soccer field",
        "a person celebrating after scoring a goal",
        "a person doing a throw in from sideline",
        "a person defending in soccer with arms out",
    ],
    "soccer_skills.mp4": [
        "a person talking to camera while standing still",
        "a person doing soccer ball juggling with feet",
        "a person dribbling a soccer ball through cones",
        "a person doing soccer passing drills",
        "a person doing soccer shooting practice",
        "a person doing soccer footwork ladder drills",
        "a person running with a soccer ball",
        "a person standing still watching others play",
        "a person doing soccer ball control exercises",
        "a person demonstrating a soccer technique to camera",
    ],
}


def load_cached_features(fname):
    """Load features from cache."""
    cache_path = os.path.join(CACHE_DIR, f"{fname}.feats.npz")
    if not os.path.exists(cache_path):
        return None, [], 0.0
    data = np.load(cache_path)
    return data["features"], data["valid_indices"].tolist(), float(data["fps"])


def classify_cluster_clip(video_path, frame_ranges, clip_model, preprocess,
                          text_features, labels_list, device, n_sample=8):
    """Classify cluster frames with CLIP. Returns (label, confidence, top3_dict)."""
    total = sum(e - s + 1 for s, e in frame_ranges)
    n = min(n_sample, total)
    all_idxs = []
    for s, e in frame_ranges:
        all_idxs.extend(range(s, e + 1))
    if not all_idxs:
        return "unknown", 0.0, {}
    sample_idxs = sorted(set(
        all_idxs[i] for i in np.linspace(0, len(all_idxs) - 1, n, dtype=int)
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
        return "unknown", 0.0, {}
    images = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    avg = img_feat.mean(dim=0, keepdim=True)
    avg = avg / avg.norm(dim=-1, keepdim=True)
    sim = (avg @ text_features.T).squeeze()
    probs = (sim * 100.0).softmax(dim=-1)
    top3_idx = probs.topk(3).indices.tolist()
    top3 = {}
    for i in top3_idx:
        short = labels_list[i].replace("a person ", "").replace("doing ", "")[:50]
        top3[short] = round(probs[i].item(), 4)
    best = probs.argmax().item()
    label = labels_list[best]
    conf = float(probs[best])
    return label, conf, top3


def run_pipeline(feats, vi, fps, threshold=2.0, min_segment_sec=2.0):
    """Run production segmentation + clustering."""
    segments = segment_motions(feats, vi, fps, min_segment_sec=min_segment_sec)
    if len(segments) >= 2:
        clustered = cluster_segments(
            copy.deepcopy(segments),
            distance_threshold=threshold,
            fps=fps,
        )
    else:
        clustered = segments
        for s in clustered:
            s["cluster"] = 0
    return clustered


def encode_text_features(clip_model, labels, device):
    """Encode text labels with CLIP."""
    import clip
    tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def part_a_threshold_sweep(clip_model, preprocess, device):
    """Part A: Sweep thresholds, report per-video pass/fail."""
    print("=" * 70)
    print("PART A: THRESHOLD SWEEP")
    print("=" * 70)

    text_features = encode_text_features(clip_model, EXPANDED_LABELS, device)

    # Load ground truth
    gt = {}
    if os.path.exists(GT_PATH):
        with open(GT_PATH) as f:
            gt = json.load(f)

    videos = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    video_data = {}
    for fname in videos:
        feats, vi, fps = load_cached_features(fname)
        if feats is not None and len(feats) >= 10:
            video_data[fname] = (feats, vi, fps)
        else:
            print(f"  SKIP {fname} (no cache or too few features)")

    print(f"Loaded {len(video_data)} videos from cache\n")

    sweep_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'='*60}")
        print(f"  THRESHOLD = {threshold}")
        print(f"{'='*60}")

        pass_count = 0
        total = 0
        threshold_detail = {}

        for fname in sorted(video_data.keys()):
            feats, vi, fps = video_data[fname]
            path = os.path.join(VIDEO_DIR, fname)
            total += 1

            clustered = run_pipeline(feats, vi, fps, threshold=threshold)
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
                label, conf, top3 = classify_cluster_clip(
                    path, ranges, clip_model, preprocess,
                    text_features, EXPANDED_LABELS, device
                )
                cluster_labels[cid] = {"label": label, "conf": round(conf, 3), "top3": top3}

            labels_only = [v["label"] for v in cluster_labels.values()]
            all_distinct = len(labels_only) == len(set(labels_only))
            if all_distinct:
                pass_count += 1

            # Check ground truth range
            gt_info = gt.get(fname, {})
            gt_range = gt_info.get("expected_clusters_range", None)
            in_gt_range = None
            if gt_range:
                in_gt_range = gt_range[0] <= n_clusters <= gt_range[1]

            status = "PASS" if all_distinct else "FAIL"
            short_labels = [l.replace("a person ", "")[:30] for l in labels_only]
            gt_str = ""
            if in_gt_range is not None:
                gt_str = f" GT={'OK' if in_gt_range else 'X'}({gt_range[0]}-{gt_range[1]})"
            print(f"  {status:4s} {fname:35s} clust={n_clusters:2d} {short_labels}{gt_str}")

            threshold_detail[fname] = {
                "n_clusters": n_clusters,
                "n_segments": len(clustered),
                "labels": labels_only,
                "all_distinct": all_distinct,
                "in_gt_range": in_gt_range,
                "cluster_detail": {str(k): v for k, v in cluster_labels.items()},
            }

        pct = 100 * pass_count / total if total > 0 else 0
        print(f"\n  RESULT: {pass_count}/{total} pass ({pct:.0f}%)")
        sweep_results[str(threshold)] = {
            "pass_count": pass_count,
            "total": total,
            "pass_rate": round(pct, 1),
            "videos": threshold_detail,
        }

    # Summary table
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Threshold':>10} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    for t in THRESHOLDS:
        r = sweep_results[str(t)]
        print(f"  {t:>10.1f} {r['pass_count']:>6}/{r['total']} {r['pass_rate']:>7.0f}%")

    return sweep_results


def part_b_failure_analysis(clip_model, preprocess, device):
    """Part B: Deep analysis of failing videos with specific CLIP prompts."""
    print(f"\n\n{'='*70}")
    print("PART B: FAILURE ANALYSIS (at threshold=2.0)")
    print(f"{'='*70}")

    analysis_results = {}

    for fname in FAILING_VIDEOS:
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {fname}")
        print(f"{'='*60}")

        feats, vi, fps = load_cached_features(fname)
        if feats is None or len(feats) < 10:
            print(f"  SKIP (no features)")
            continue

        path = os.path.join(VIDEO_DIR, fname)
        clustered = run_pipeline(feats, vi, fps, threshold=2.0)
        n_clusters = len(set(s["cluster"] for s in clustered))

        # Get cluster ranges and frame info
        cluster_info = {}
        for seg in clustered:
            cid = seg["cluster"]
            if cid not in cluster_info:
                cluster_info[cid] = {
                    "segments": [],
                    "frame_ranges": [],
                    "total_frames": 0,
                    "duration_sec": 0.0,
                }
            s_frame = seg["start_frame"]
            e_frame = seg["end_frame"]
            dur = (e_frame - s_frame) / max(fps, 1)
            cluster_info[cid]["segments"].append({
                "start_frame": s_frame,
                "end_frame": e_frame,
                "duration_sec": round(dur, 2),
                "n_features": seg["features"].shape[0],
            })
            cluster_info[cid]["frame_ranges"].append((s_frame, e_frame))
            cluster_info[cid]["total_frames"] += (e_frame - s_frame)
            cluster_info[cid]["duration_sec"] += dur

        print(f"  {n_clusters} clusters, {len(clustered)} segments")
        for cid in sorted(cluster_info.keys()):
            info = cluster_info[cid]
            ranges_str = ", ".join(f"{s}-{e}" for s, e in info["frame_ranges"])
            print(f"  Cluster {cid}: {info['duration_sec']:.1f}s, "
                  f"{len(info['segments'])} seg(s), frames=[{ranges_str}]")

        # 1. Classify with expanded labels (same as Part A)
        text_features_expanded = encode_text_features(clip_model, EXPANDED_LABELS, device)
        print(f"\n  --- EXPANDED LABELS ({len(EXPANDED_LABELS)} labels) ---")
        expanded_results = {}
        for cid in sorted(cluster_info.keys()):
            ranges = cluster_info[cid]["frame_ranges"]
            label, conf, top3 = classify_cluster_clip(
                path, ranges, clip_model, preprocess,
                text_features_expanded, EXPANDED_LABELS, device, n_sample=8
            )
            expanded_results[cid] = {"label": label, "conf": conf, "top3": top3}
            short = label.replace("a person ", "")[:40]
            print(f"  Cluster {cid}: '{short}' (conf={conf:.4f})")
            for k, v in top3.items():
                print(f"            {k}: {v:.4f}")

        # 2. Classify with video-specific labels
        specific_labels = FAILURE_SPECIFIC_LABELS.get(fname, [])
        if specific_labels:
            text_features_specific = encode_text_features(clip_model, specific_labels, device)
            print(f"\n  --- SPECIFIC LABELS ({len(specific_labels)} labels) ---")
            specific_results = {}
            for cid in sorted(cluster_info.keys()):
                ranges = cluster_info[cid]["frame_ranges"]
                label, conf, top3 = classify_cluster_clip(
                    path, ranges, clip_model, preprocess,
                    text_features_specific, specific_labels, device, n_sample=8
                )
                specific_results[cid] = {"label": label, "conf": conf, "top3": top3}
                short = label.replace("a person ", "")[:45]
                print(f"  Cluster {cid}: '{short}' (conf={conf:.4f})")
                for k, v in top3.items():
                    print(f"            {k}: {v:.4f}")

            # 3. Determine if clustering is correct or wrong
            expanded_labels_list = [expanded_results[c]["label"] for c in sorted(expanded_results)]
            specific_labels_list = [specific_results[c]["label"] for c in sorted(specific_results)]
            expanded_distinct = len(set(expanded_labels_list)) == len(expanded_labels_list)
            specific_distinct = len(set(specific_labels_list)) == len(specific_labels_list)

            print(f"\n  --- DIAGNOSIS ---")
            print(f"  Expanded labels distinct? {expanded_distinct}")
            print(f"  Specific labels distinct? {specific_distinct}")

            if specific_distinct:
                print(f"  --> CLUSTERING CORRECT: Specific prompts distinguish the clusters.")
                print(f"      The standard labels are just too coarse.")
                diagnosis = "CLUSTERING_CORRECT_LABELS_TOO_COARSE"
            else:
                # Check if pose features actually differ
                cluster_mean_feats = {}
                for cid_idx, seg in enumerate(clustered):
                    cid = seg["cluster"]
                    if cid not in cluster_mean_feats:
                        cluster_mean_feats[cid] = []
                    cluster_mean_feats[cid].append(seg["mean_feature"])

                centroids = {}
                for cid, feats_list in cluster_mean_feats.items():
                    centroids[cid] = np.mean(feats_list, axis=0)

                if len(centroids) >= 2:
                    cids = sorted(centroids.keys())
                    dists = []
                    for i in range(len(cids)):
                        for j in range(i+1, len(cids)):
                            d = float(np.linalg.norm(centroids[cids[i]] - centroids[cids[j]]))
                            dists.append((cids[i], cids[j], d))
                    for c1, c2, d in dists:
                        print(f"  Pose centroid distance (cluster {c1} vs {c2}): {d:.3f}")

                    avg_dist = np.mean([d for _, _, d in dists])
                    if avg_dist > 1.0:
                        print(f"  --> CLUSTERING LIKELY CORRECT: pose centroids differ significantly ({avg_dist:.3f})")
                        print(f"      Vision model cannot tell the difference, but pose space can.")
                        diagnosis = "CLUSTERING_CORRECT_VISION_LIMITED"
                    else:
                        print(f"  --> POSSIBLE OVER-SEGMENTATION: pose centroids are close ({avg_dist:.3f})")
                        diagnosis = "POSSIBLE_OVER_SEGMENTATION"
                else:
                    diagnosis = "SINGLE_CLUSTER"
        else:
            specific_results = {}
            diagnosis = "NO_SPECIFIC_LABELS"

        analysis_results[fname] = {
            "n_clusters": n_clusters,
            "n_segments": len(clustered),
            "cluster_info": {str(k): {
                "duration_sec": round(v["duration_sec"], 2),
                "n_segments": len(v["segments"]),
                "frame_ranges": v["frame_ranges"],
            } for k, v in cluster_info.items()},
            "expanded_labels": {str(k): v for k, v in expanded_results.items()},
            "specific_labels": {str(k): v for k, v in specific_results.items()},
            "diagnosis": diagnosis,
        }

    return analysis_results


def main():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading CLIP ViT-L/14...")
    t0 = time.time()
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    print(f"CLIP loaded in {time.time() - t0:.1f}s\n")

    # Part A
    sweep_results = part_a_threshold_sweep(clip_model, preprocess, device)

    # Part B
    failure_results = part_b_failure_analysis(clip_model, preprocess, device)

    # Save all results
    out_path = "/mnt/Data/GitHub/BRACE/experiments/threshold_finetune_results.json"
    combined = {
        "sweep": sweep_results,
        "failure_analysis": failure_results,
        "config": {
            "clip_model": "ViT-L/14",
            "n_expanded_labels": len(EXPANDED_LABELS),
            "thresholds": THRESHOLDS,
            "min_segment_sec": 2.0,
            "linkage": "average",
        },
    }
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
