"""Precompute full analysis for demo videos at maximum quality.

Processes every frame through the pipeline and stores per-frame results
in memory so the frontend can replay with perfect skeleton-video sync.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline_interface import PoseBackend, PipelineResult
from subject_manager import SubjectManager
from identity_resolver import IdentityResolver
from embedding_extractor import EmbeddingExtractor
from scene_detector import InlineSceneDetector

try:
    from gemini_classifier import GeminiActivityClassifier
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# Only send joints used by frontend skeleton renderer
_SEND_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]


# In-memory cache: filename -> precomputed data
_precompute_cache: dict[str, dict[str, Any]] = {}

# Active jobs: job_id -> status dict
_precompute_jobs: dict[str, dict[str, Any]] = {}

# Disk cache directory (inside container at /app/precompute_cache/)
_CACHE_DIR = Path("/app/precompute_cache")
_CACHE_DIR.mkdir(exist_ok=True)


def _disk_cache_path(filename: str) -> Path:
    """Return the gzipped JSON cache path for a video filename."""
    return _CACHE_DIR / f"{filename}.json.gz"


def _save_to_disk(filename: str, data: dict[str, Any]) -> None:
    """Persist precomputed data to disk (gzipped JSON)."""
    try:
        path = _disk_cache_path(filename)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as exc:
        print(f"[precompute] Failed to save cache for {filename}: {exc}")


def _load_from_disk(filename: str) -> dict[str, Any] | None:
    """Load precomputed data from disk cache, or None."""
    path = _disk_cache_path(filename)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[precompute] Failed to load cache for {filename}: {exc}")
        return None


def get_cached(filename: str) -> dict[str, Any] | None:
    """Return cached precomputed data for a video, or None.

    Checks in-memory cache first, then falls back to disk.
    """
    cached = _precompute_cache.get(filename)
    if cached is not None:
        return cached
    # Try disk
    cached = _load_from_disk(filename)
    if cached is not None:
        _precompute_cache[filename] = cached
    return cached


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return job status dict, or None."""
    return _precompute_jobs.get(job_id)


def _build_subject_response(
    track_id: int,
    label: str,
    response: dict[str, Any],
    srp_joints: list[list[float]] | None,
    embedding_update: dict[str, Any] | None,
    identity_status: str = "confirmed",
    identity_confidence: float = 1.0,
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build per-subject data dict matching the WebSocket format."""
    out: dict[str, Any] = {
        "label": label,
        "phase": response["phase"],
        "n_segments": response["n_segments"],
        "n_clusters": response["n_clusters"],
        "landmarks": response["landmarks"],
        "bbox": response["bbox"],
        "cluster_id": response["cluster_id"],
        "consistency_score": response["consistency_score"],
        "is_anomaly": response["is_anomaly"],
        "cluster_summary": response["cluster_summary"],
        "srp_joints": srp_joints,
        "identity_status": identity_status,
        "identity_confidence": round(identity_confidence, 3),
    }
    if embedding_update is not None:
        out["embedding_update"] = embedding_update
    if response.get("velocity") is not None:
        out["velocity"] = response["velocity"]
        out["rolling_velocity"] = response.get("rolling_velocity", 0)
        out["fatigue_index"] = response.get("fatigue_index", 0)
        out["peak_velocity"] = response.get("peak_velocity", 0)
    if quality:
        out["quality"] = quality
    return out


def precompute_video_sync(
    video_path: str | Path,
    backend: PoseBackend,
    job_id: str,
    cluster_threshold: float = 2.0,
    reid_extractor: EmbeddingExtractor | None = None,
    cross_cut_extractor: EmbeddingExtractor | None = None,
) -> dict[str, Any] | None:
    """Process an entire video synchronously (runs in thread pool).

    Returns the full precomputed data dict, or None on error.
    Updates _precompute_jobs[job_id] with progress.
    """
    job = _precompute_jobs[job_id]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        job["status"] = "error"
        job["error"] = f"Cannot open video: {video_path}"
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    job["total_frames"] = total_frames
    job["fps"] = fps
    job["status"] = "processing"

    manager = SubjectManager(fps=fps, cluster_threshold=cluster_threshold)
    resolver = (
        IdentityResolver(reid_extractor, cross_cut_extractor=cross_cut_extractor)
        if reid_extractor is not None
        else None
    )
    scene_detector = InlineSceneDetector()

    # Gemini activity classification
    gemini: GeminiActivityClassifier | None = None
    if _GEMINI_AVAILABLE:
        gemini = GeminiActivityClassifier()

    # Ring buffer for Gemini classification
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 600

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster(analyzer, cluster_id: int, gem: GeminiActivityClassifier):
        indices = analyzer.get_cluster_frame_indices(cluster_id)
        if not indices:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        bbox = analyzer.get_cluster_bbox(cluster_id)
        if bbox is None:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        frames = gem.get_representative_frames(indices, _get_buffered_frame, count=4)
        if not frames:
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        label = gem.classify_activity(frames, bbox)
        analyzer.set_activity_label(cluster_id, label)

    backend.reset()
    frames_data: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            # Check for cancellation
            if job.get("cancelled"):
                job["status"] = "cancelled"
                return None

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store for Gemini
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > _FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            # Scene cut detection
            if scene_detector.process_frame(rgb):
                backend.on_scene_cut()
                if resolver is not None:
                    resolver.on_scene_cut()

            # Pipeline inference (full quality - direct from cv2, no JPEG)
            results = backend.process_frame(rgb)

            # Identity resolution
            if resolver is not None and results:
                resolved = resolver.resolve_pipeline_results(results, rgb, width, height)
            else:
                resolved = None

            # Process through analyzers
            subjects_data: dict[str, dict[str, Any]] = {}

            items = resolved if resolved is not None else results

            for item in items:
                if resolved is not None:
                    rp = item
                    pr = rp.pipeline_result
                    subject_id = rp.subject_id
                    label = rp.label
                    identity_status = rp.identity_status
                    identity_confidence = rp.identity_confidence
                else:
                    pr = item
                    subject_id = pr.track_id
                    label = manager.get_label(pr.track_id)
                    identity_status = "confirmed"
                    identity_confidence = 1.0

                landmarks_xyzv = pr.landmarks_mp
                analyzer = manager.get_or_create_analyzer(subject_id)
                analyzer.last_seen_frame = frame_idx

                response = analyzer.process_frame(landmarks_xyzv, img_wh=(width, height))

                # Normalize landmarks
                response["landmarks"] = [
                    {
                        "i": i,
                        "x": round(float(landmarks_xyzv[i, 0] / width), 4),
                        "y": round(float(landmarks_xyzv[i, 1] / height), 4),
                        "v": round(float(landmarks_xyzv[i, 3]), 3),
                    }
                    for i in _SEND_INDICES
                ]

                nb = pr.bbox_normalized
                response["bbox"] = {
                    "x1": round(nb[0], 4),
                    "y1": round(nb[1], 4),
                    "x2": round(nb[2], 4),
                    "y2": round(nb[3], 4),
                }

                # Re-analysis
                if analyzer.needs_reanalysis():
                    analyzer.run_analysis()
                    if gemini is not None and gemini.available:
                        for cid in analyzer.get_clusters_needing_classification():
                            analyzer.mark_classification_pending(cid)
                            _classify_cluster(analyzer, cid, gemini)

                # UMAP embedding
                embedding_update = None
                if analyzer.needs_umap_refit():
                    embedding_update = analyzer.run_umap_fit()
                elif len(analyzer.features_list) > 0 and analyzer._umap_mapper is not None:
                    feat = analyzer.features_list[-1]
                    embedding_update = analyzer.run_umap_transform(feat)

                srp_joints = analyzer.get_srp_joints()
                if srp_joints is not None:
                    srp_joints = [[round(v, 4) for v in jt] for jt in srp_joints]

                # Quality metrics
                quality = None
                if hasattr(analyzer, '_quality_tracker') and analyzer._quality_tracker:
                    qt = analyzer._quality_tracker
                    quality = {}
                    if hasattr(qt, 'get_frame_quality'):
                        quality = qt.get_frame_quality()

                subjects_data[str(subject_id)] = _build_subject_response(
                    subject_id, label, response, srp_joints, embedding_update,
                    identity_status, identity_confidence, quality,
                )

            # Cleanup stale
            manager.cleanup_stale(frame_idx)
            if resolver is not None:
                active_track_ids = {pr.track_id for pr in results}
                resolver.cleanup_stale_tracks(active_track_ids)

            active_ids = manager.get_active_track_ids()

            video_time = frame_idx / fps

            frames_data.append({
                "frame_index": frame_idx,
                "video_time": round(video_time, 4),
                "subjects": subjects_data,
                "active_track_ids": active_ids,
            })

            frame_idx += 1
            job["processed_frames"] = frame_idx
            job["progress"] = round(frame_idx / max(total_frames, 1), 3)

    finally:
        cap.release()

    # Final analysis pass
    for subject_id, analyzer in manager.analyzers.items():
        if analyzer.needs_reanalysis() or len(analyzer._segments) == 0:
            analyzer.run_analysis()

    result = {
        "video_filename": Path(video_path).name,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "frames": frames_data,
    }

    # Persist to disk so it survives restarts
    _save_to_disk(Path(video_path).name, result)

    job["status"] = "complete"
    job["progress"] = 1.0
    job["processed_frames"] = total_frames

    return result
