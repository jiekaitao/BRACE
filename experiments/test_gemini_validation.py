#!/usr/bin/env python3
"""Test Gemini 2.0 Flash as a cluster validator (replacing CLIP zero-shot).

For each video:
1. Load cached SRP features from .feature_cache/
2. Run production segment_motions() + cluster_segments(threshold=2.0)
3. For each cluster, extract ~4 representative frames from the video
4. Send frames to Gemini 2.0 Flash with a specific prompt
5. Compare labels for distinctiveness across clusters

Tests all 14 videos, with special focus on the 5 that fail with CLIP.
"""

import sys
import os
import copy
import time
import io

import cv2
import numpy as np

sys.path.insert(0, "/mnt/Data/GitHub/BRACE")

from brace.core.motion_segments import segment_motions, cluster_segments

VIDEO_DIR = "/mnt/Data/GitHub/BRACE/data/sports_videos"
CACHE_DIR = "/mnt/Data/GitHub/BRACE/experiments/.feature_cache"

# The 5 videos that fail with CLIP (duplicate labels)
CLIP_FAILURES = {
    "gym_workout.mp4",
    "soccer_match2.mp4",
    "soccer_skills.mp4",
    "tennis_practice.mp4",
    "yoga_flow.mp4",
}

GEMINI_PROMPT = (
    "What specific exercise or movement is this person performing in these frames? "
    "Be as specific as possible about the type, body position, and technique. "
    "Answer in 5 words or fewer."
)

# Minimum seconds between Gemini API calls
RATE_LIMIT_SEC = 2.0


def get_gemini_client():
    """Initialize Gemini client using google-genai SDK."""
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        # Fallback to hardcoded key from gemini_classifier.py
        api_key = "AIzaSyDrfeCSYbecPiJ1aLWUlY9MdGDBA96__W8"

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Quick test
        return client
    except Exception as e:
        print(f"ERROR: Could not initialize Gemini client: {e}")
        return None


def encode_frame_jpeg(frame_rgb: np.ndarray, max_dim: int = 512) -> bytes:
    """Resize and JPEG-encode a frame for the API."""
    h, w = frame_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def get_representative_frames(video_path: str, frame_indices: list[int], count: int = 4) -> list[np.ndarray]:
    """Extract evenly-spaced representative frames from the video file."""
    if not frame_indices:
        return []

    n = len(frame_indices)
    if n <= count:
        selected_indices = frame_indices
    else:
        step = n / count
        selected_indices = [frame_indices[int(i * step)] for i in range(count)]

    cap = cv2.VideoCapture(video_path)
    frames = []
    for fi in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def classify_with_gemini(client, frames: list[np.ndarray], prompt: str) -> str:
    """Send frames to Gemini 2.0 Flash and get an activity label."""
    from PIL import Image

    contents = []
    for frame in frames:
        jpeg_bytes = encode_frame_jpeg(frame)
        img = Image.open(io.BytesIO(jpeg_bytes))
        contents.append(img)
    contents.append(prompt)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
        )
        label = response.text.strip().lower().rstrip(".")
        # Take first line, first 5 words
        label = label.split("\n")[0].strip()
        words = label.split()
        if len(words) > 5:
            label = " ".join(words[:5])
        return label if label else "unknown"
    except Exception as e:
        print(f"    Gemini API error: {e}")
        return "unknown"


def main():
    # Initialize Gemini
    client = get_gemini_client()
    if client is None:
        print("SKIP: Gemini client not available. Set GOOGLE_GEMINI_API_KEY or install google-genai.")
        return

    # Verify with a quick test call
    print("Testing Gemini API connection...")
    try:
        test_resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Say 'ok' and nothing else."],
        )
        print(f"  API test response: {test_resp.text.strip()}")
    except Exception as e:
        print(f"SKIP: Gemini API test failed: {e}")
        return

    print()

    # Find all cached feature files
    cache_files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".feats.npz"))
    print(f"Found {len(cache_files)} cached feature files\n")

    results = {}
    api_calls = 0
    last_call_time = 0.0

    for cache_file in cache_files:
        video_name = cache_file.replace(".feats.npz", "")
        video_path = os.path.join(VIDEO_DIR, video_name)
        cache_path = os.path.join(CACHE_DIR, cache_file)

        if not os.path.exists(video_path):
            print(f"SKIP {video_name}: video file not found")
            continue

        is_clip_failure = video_name in CLIP_FAILURES
        marker = " *** CLIP-FAILURE ***" if is_clip_failure else ""
        print(f"{'='*60}")
        print(f"Processing: {video_name}{marker}")

        # Load cached features
        data = np.load(cache_path)
        features = data["features"]
        valid_indices = data["valid_indices"].tolist()
        fps = float(data["fps"])

        print(f"  Features: {features.shape[0]}, FPS: {fps:.1f}")

        if features.shape[0] < 10:
            print(f"  Not enough features. Skipping.")
            results[video_name] = {"error": "insufficient features"}
            continue

        # Segment and cluster (production settings)
        segments = segment_motions(features, valid_indices, fps, min_segment_sec=2.0)
        if len(segments) < 2:
            segments = segment_motions(features, valid_indices, fps, min_segment_sec=1.0)

        print(f"  Segments before clustering: {len(segments)}")

        if len(segments) >= 2:
            clustered = cluster_segments(copy.deepcopy(segments), distance_threshold=2.0, fps=fps)
        else:
            clustered = segments
            for s in clustered:
                s["cluster"] = 0

        n_clusters = len(set(s["cluster"] for s in clustered))
        print(f"  Clusters: {n_clusters}")

        # For each cluster, gather frame indices and classify with Gemini
        cluster_frame_ranges = {}
        for seg in clustered:
            cid = seg["cluster"]
            if cid not in cluster_frame_ranges:
                cluster_frame_ranges[cid] = []
            # Collect all original frame indices for this segment
            start_vi = seg["start_valid"]
            end_vi = seg["end_valid"]
            for vi_idx in range(start_vi, min(end_vi, len(valid_indices))):
                cluster_frame_ranges[cid].append(valid_indices[vi_idx])

        cluster_labels = {}
        for cid in sorted(cluster_frame_ranges.keys()):
            frame_indices = cluster_frame_ranges[cid]
            n_frames = len(frame_indices)

            # Get representative frames from video
            frames = get_representative_frames(video_path, frame_indices, count=4)
            if not frames:
                cluster_labels[cid] = {"label": "unknown", "n_frames": n_frames}
                print(f"  Cluster {cid}: 'unknown' (no frames extracted)")
                continue

            # Rate limit
            now = time.monotonic()
            wait = RATE_LIMIT_SEC - (now - last_call_time)
            if wait > 0:
                time.sleep(wait)

            label = classify_with_gemini(client, frames, GEMINI_PROMPT)
            last_call_time = time.monotonic()
            api_calls += 1

            cluster_labels[cid] = {"label": label, "n_frames": n_frames}
            print(f"  Cluster {cid}: '{label}' ({n_frames} frames)")

        # Validate: check for duplicate labels
        labels_only = [v["label"] for v in cluster_labels.values()]
        unique_labels = set(labels_only)
        all_distinct = len(labels_only) == len(unique_labels)

        failure_reason = None
        if not all_distinct:
            from collections import Counter
            label_counts = Counter(labels_only)
            dups = [l for l, c in label_counts.items() if c > 1]
            dup_info = []
            for l in dups:
                cids = [str(cid) for cid, v in cluster_labels.items() if v["label"] == l]
                dup_info.append(f"clusters {','.join(cids)} share '{l}'")
            failure_reason = "DUPLICATE: " + "; ".join(dup_info)
            print(f"  *** FAIL: {failure_reason}")
        else:
            print(f"  PASS: all {n_clusters} clusters have distinct labels")

        # Segment detail
        seg_info = []
        for i, seg in enumerate(clustered):
            dur = (seg["end_frame"] - seg["start_frame"]) / max(fps, 1)
            seg_info.append({
                "segment": i,
                "cluster": seg["cluster"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
                "duration_sec": round(dur, 2),
            })

        results[video_name] = {
            "n_features": int(features.shape[0]),
            "n_segments": len(clustered),
            "n_clusters": n_clusters,
            "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
            "all_labels_distinct": all_distinct,
            "failure_reason": failure_reason,
            "is_clip_failure": is_clip_failure,
            "segments": seg_info,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total = sum(1 for v in results.values() if "error" not in v)
    passed = sum(1 for v in results.values() if v.get("all_labels_distinct"))
    failed = total - passed
    clip_fail_passed = sum(
        1 for v in results.values()
        if v.get("is_clip_failure") and v.get("all_labels_distinct")
    )
    clip_fail_total = sum(1 for v in results.values() if v.get("is_clip_failure"))

    print(f"  Total videos processed: {total}")
    print(f"  Gemini distinct-labels PASS: {passed}/{total}")
    print(f"  Gemini distinct-labels FAIL: {failed}/{total}")
    print(f"  CLIP-failure videos fixed by Gemini: {clip_fail_passed}/{clip_fail_total}")
    print(f"  Total Gemini API calls: {api_calls}")
    print(f"  Estimated cost: ${api_calls * 0.00011:.5f}")

    # Detailed per-video
    print(f"\n  Per-video results:")
    # Show CLIP failures first
    for video_name in sorted(results.keys(), key=lambda v: (v not in CLIP_FAILURES, v)):
        r = results[video_name]
        if "error" in r:
            print(f"    SKIP  {video_name}: {r['error']}")
            continue
        status = "PASS" if r["all_labels_distinct"] else "FAIL"
        clip_tag = " [CLIP-FAIL]" if r.get("is_clip_failure") else ""
        labels = [v["label"] for v in r["cluster_labels"].values()]
        print(f"    {status}  {video_name}{clip_tag}: {r['n_clusters']} clusters, labels={labels}")
        if r.get("failure_reason"):
            print(f"          Reason: {r['failure_reason']}")


if __name__ == "__main__":
    main()
