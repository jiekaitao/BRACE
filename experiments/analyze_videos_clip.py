#!/usr/bin/env python3
"""
Analyze sports videos using CLIP zero-shot classification to determine
ground truth activities and expected cluster counts.
"""

import json
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

import open_clip

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
OUTPUT_PATH = "/mnt/Data/GitHub/BRACE/experiments/video_ground_truth.json"

# Action labels for zero-shot classification
ACTION_LABELS = [
    # Basketball
    "a person dribbling a basketball",
    "a person shooting a basketball",
    "a person doing a layup",
    "a person dunking a basketball",
    # General movement
    "a person running",
    "a person jogging",
    "a person walking",
    "a person standing still",
    "a person resting",
    # Exercises
    "a person doing pushups",
    "a person doing squats",
    "a person doing pullups on a bar",
    "a person doing lunges",
    "a person doing burpees",
    "a person doing jumping jacks",
    "a person doing a plank",
    "a person stretching",
    "a person doing yoga",
    # Weightlifting
    "a person lifting weights",
    "a person doing bench press",
    "a person doing deadlift",
    "a person doing barbell curls",
    "a person doing overhead press",
    "a person doing rows",
    "a person doing kettlebell swings",
    "a person doing clean and jerk",
    "a person doing snatch",
    # Crossfit
    "a person jumping rope",
    "a person doing box jumps",
    "a person doing wall balls",
    "a person doing muscle ups",
    "a person doing handstand pushups",
    "a person doing rope climbs",
    "a person doing double unders",
    # Soccer
    "a person dribbling a soccer ball",
    "a person kicking a soccer ball",
    "a person heading a soccer ball",
    "a person doing soccer tricks",
    "multiple people playing soccer",
    # Combat
    "a person boxing",
    "a person kicking",
    "a person punching a bag",
    # Other
    "a person doing a backflip",
    "a person doing a cartwheel",
]

# Simplified label mapping - group similar actions together
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
}


def load_clip_model():
    """Load CLIP model for zero-shot classification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


def classify_frames(video_path, model, preprocess, tokenizer, device, sample_fps=2):
    """Classify video frames using CLIP zero-shot classification."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        cap.release()
        print(f"  WARNING: Cannot read {os.path.basename(video_path)} (fps={fps}, frames={total_frames})")
        return [], 0.0, 0.0
    duration = total_frames / fps
    sample_interval = max(1, int(fps / sample_fps))

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    print(f"  Sampling every {sample_interval} frames ({sample_fps} fps)")

    # Pre-encode text labels
    text_tokens = tokenizer(ACTION_LABELS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Convert BGR to RGB and preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_features = model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                similarity = (img_features @ text_features.T).squeeze(0)
                # Scale by temperature (100 is CLIP's default logit_scale)
                probs = (similarity * 100.0).softmax(dim=-1)

            top_idx = probs.argmax().item()
            top_label = ACTION_LABELS[top_idx]
            top_conf = probs[top_idx].item()

            # Get top-3 for debugging
            top3_indices = probs.topk(3).indices.tolist()
            top3 = [(ACTION_LABELS[i], probs[i].item()) for i in top3_indices]

            time_sec = frame_idx / fps
            simplified = LABEL_SIMPLIFY.get(top_label, top_label)

            frame_results.append({
                "time_sec": round(time_sec, 2),
                "frame_idx": frame_idx,
                "label": simplified,
                "raw_label": top_label,
                "confidence": round(top_conf, 4),
                "top3": top3,
            })

        frame_idx += 1

    cap.release()

    return frame_results, duration, fps


def segment_activities(frame_results, min_segment_sec=1.0):
    """Group consecutive frames with same label into segments."""
    if not frame_results:
        return []

    segments = []
    current_label = frame_results[0]["label"]
    current_start = frame_results[0]["time_sec"]
    current_confs = [frame_results[0]["confidence"]]

    for i in range(1, len(frame_results)):
        fr = frame_results[i]
        if fr["label"] != current_label:
            seg_duration = frame_results[i - 1]["time_sec"] - current_start
            segments.append({
                "start_sec": round(current_start, 2),
                "end_sec": round(frame_results[i - 1]["time_sec"], 2),
                "label": current_label,
                "confidence": round(np.mean(current_confs), 4),
                "duration": round(seg_duration, 2),
            })
            current_label = fr["label"]
            current_start = fr["time_sec"]
            current_confs = [fr["confidence"]]
        else:
            current_confs.append(fr["confidence"])

    # Last segment
    segments.append({
        "start_sec": round(current_start, 2),
        "end_sec": round(frame_results[-1]["time_sec"], 2),
        "label": current_label,
        "confidence": round(np.mean(current_confs), 4),
        "duration": round(frame_results[-1]["time_sec"] - current_start, 2),
    })

    # Merge very short segments with neighbors
    merged = merge_short_segments(segments, min_segment_sec)
    return merged


def merge_short_segments(segments, min_sec=1.0):
    """Merge segments shorter than min_sec into adjacent segments."""
    if len(segments) <= 1:
        return segments

    # First pass: merge tiny segments into their longest neighbor
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            seg = merged[i]
            if seg["duration"] < min_sec and len(merged) > 1:
                # Merge into the longer neighbor
                if i == 0:
                    # Merge into next
                    merged[i + 1]["start_sec"] = seg["start_sec"]
                    merged[i + 1]["duration"] = round(
                        merged[i + 1]["end_sec"] - merged[i + 1]["start_sec"], 2
                    )
                    changed = True
                    i += 1
                    continue
                elif i == len(merged) - 1:
                    # Merge into previous
                    new_merged[-1]["end_sec"] = seg["end_sec"]
                    new_merged[-1]["duration"] = round(
                        new_merged[-1]["end_sec"] - new_merged[-1]["start_sec"], 2
                    )
                    changed = True
                    i += 1
                    continue
                else:
                    # Merge into longer neighbor
                    prev_dur = new_merged[-1]["duration"] if new_merged else 0
                    next_dur = merged[i + 1]["duration"]
                    if prev_dur >= next_dur and new_merged:
                        new_merged[-1]["end_sec"] = seg["end_sec"]
                        new_merged[-1]["duration"] = round(
                            new_merged[-1]["end_sec"] - new_merged[-1]["start_sec"], 2
                        )
                    else:
                        merged[i + 1]["start_sec"] = seg["start_sec"]
                        merged[i + 1]["duration"] = round(
                            merged[i + 1]["end_sec"] - merged[i + 1]["start_sec"], 2
                        )
                    changed = True
                    i += 1
                    continue
            new_merged.append(seg)
            i += 1
        merged = new_merged

    # Second pass: merge adjacent segments with same label
    final = [merged[0]]
    for seg in merged[1:]:
        if seg["label"] == final[-1]["label"]:
            final[-1]["end_sec"] = seg["end_sec"]
            final[-1]["duration"] = round(
                final[-1]["end_sec"] - final[-1]["start_sec"], 2
            )
            final[-1]["confidence"] = round(
                (final[-1]["confidence"] + seg["confidence"]) / 2, 4
            )
        else:
            final.append(seg)

    return final


def analyze_video(video_path, model, preprocess, tokenizer, device):
    """Analyze a single video and return ground truth data."""
    frame_results, duration, fps = classify_frames(
        video_path, model, preprocess, tokenizer, device, sample_fps=2
    )

    if not frame_results:
        print("  SKIPPED: no valid frames")
        return {
            "duration_sec": 0,
            "fps": 0,
            "activities": [],
            "distinct_activities": [],
            "expected_clusters": 0,
            "reasoning": "Video could not be read",
            "frame_predictions": [],
        }

    # Print per-frame results for debugging
    print(f"\n  Frame-by-frame CLIP predictions:")
    for fr in frame_results:
        print(f"    t={fr['time_sec']:6.2f}s  {fr['label']:<30s}  conf={fr['confidence']:.3f}")

    # Segment activities
    segments = segment_activities(frame_results, min_segment_sec=1.0)

    # Get distinct activities
    distinct = sorted(set(seg["label"] for seg in segments))

    # Filter out brief "standing/resting" unless it's substantial
    total_time = sum(s["duration"] for s in segments)
    activity_segments = []
    for seg in segments:
        if seg["label"] == "standing/resting" and seg["duration"] < 2.0:
            continue
        activity_segments.append(seg)

    distinct_activities = sorted(set(seg["label"] for seg in activity_segments))
    # standing/resting doesn't count as a distinct activity for clustering purposes
    cluster_activities = [a for a in distinct_activities if a != "standing/resting"]
    expected_clusters = max(1, len(cluster_activities))

    print(f"\n  Segments:")
    for seg in segments:
        print(f"    [{seg['start_sec']:6.2f} - {seg['end_sec']:6.2f}]  "
              f"{seg['label']:<30s}  dur={seg['duration']:.1f}s  conf={seg['confidence']:.3f}")
    print(f"\n  Distinct activities (excl. rest): {cluster_activities}")
    print(f"  Expected clusters: {expected_clusters}")

    return {
        "duration_sec": round(duration, 1),
        "fps": round(fps, 1),
        "activities": [
            {
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "label": seg["label"],
                "confidence": round(seg["confidence"], 3),
            }
            for seg in segments
        ],
        "distinct_activities": distinct_activities,
        "expected_clusters": expected_clusters,
        "reasoning": "",  # Will be filled in after review
        "frame_predictions": [
            {
                "time_sec": fr["time_sec"],
                "label": fr["label"],
                "confidence": round(fr["confidence"], 3),
            }
            for fr in frame_results
        ],
    }


def main():
    print("Loading CLIP model...")
    model, preprocess, tokenizer, device = load_clip_model()
    print(f"CLIP model loaded on {device}\n")

    results = {}

    # Find all videos
    video_dir = Path(VIDEO_DIR)
    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {VIDEO_DIR}\n")

    for vf in video_files:
        print(f"\n{'='*70}")
        print(f"Analyzing: {vf.name}")
        print(f"{'='*70}")
        results[vf.name] = analyze_video(str(vf), model, preprocess, tokenizer, device)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {OUTPUT_PATH}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, data in results.items():
        print(f"  {name}: {data['duration_sec']}s, "
              f"{len(data['activities'])} segments, "
              f"{data['expected_clusters']} expected clusters, "
              f"activities: {data['distinct_activities']}")


if __name__ == "__main__":
    main()
