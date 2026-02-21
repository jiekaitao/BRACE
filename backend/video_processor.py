"""Process uploaded MP4 files frame-by-frame through the pose analysis pipeline."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import WebSocket

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
_SEND_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]


def _build_subject_response(
    track_id: int,
    label: str,
    response: dict[str, Any],
    srp_joints: list[list[float]] | None,
    embedding_update: dict[str, Any] | None,
    identity_status: str = "confirmed",
    identity_confidence: float = 1.0,
    smpl_params: dict | None = None,
    uv_texture_b64: str | None = None,
) -> dict[str, Any]:
    """Build per-subject data dict for the multi-subject response."""
    out = {
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
    if smpl_params is not None:
        out["smpl_params"] = smpl_params
    if uv_texture_b64 is not None:
        out["uv_texture"] = uv_texture_b64
    return out


async def process_video(
    video_path: str | Path,
    websocket: WebSocket,
    backend: PoseBackend,
    cluster_threshold: float = 2.0,
    reid_extractor: EmbeddingExtractor | None = None,
    cross_cut_extractor: EmbeddingExtractor | None = None,
) -> None:
    """Process an uploaded video and stream multi-person results over WebSocket."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"Cannot open video: {video_path}"})
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manager = SubjectManager(fps=fps, cluster_threshold=cluster_threshold)
    resolver = (
        IdentityResolver(reid_extractor, cross_cut_extractor=cross_cut_extractor)
        if reid_extractor is not None
        else None
    )
    scene_detector = InlineSceneDetector()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=8)

    # Gemini activity classification
    gemini: GeminiActivityClassifier | None = None
    if _GEMINI_AVAILABLE:
        gemini = GeminiActivityClassifier()

    # Ring buffer of recent RGB frames for Gemini classification (indexed by frame_idx)
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 600  # keep last ~20 seconds at 30fps

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster_bg(analyzer, cluster_id: int, gem: GeminiActivityClassifier):
        """Background thread: classify a single cluster."""
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

    # Reset backend tracker for this video
    backend.reset()

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store frame in ring buffer for Gemini classification
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > _FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            # Step 0.5: Detect scene cuts (camera changes)
            if scene_detector.process_frame(rgb):
                backend.on_scene_cut()
                if resolver is not None:
                    resolver.on_scene_cut()

            # Step 1: Pipeline processes frame -> PipelineResults
            results = await loop.run_in_executor(executor, backend.process_frame, rgb)

            # Step 1.5: Resolve identities
            if resolver is not None and results:
                resolved = await loop.run_in_executor(
                    executor, resolver.resolve_pipeline_results, results, rgb, width, height
                )
            else:
                resolved = None

            # Step 2: Process through analyzers
            subjects_data: dict[str, dict[str, Any]] = {}

            if resolved is not None:
                items = resolved
            else:
                items = results

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

                # Process through analyzer (uses pixel coordinates for SRP)
                response = analyzer.process_frame(landmarks_xyzv, img_wh=(width, height))

                # Normalize landmarks to [0, 1] -- sparse, body joints only
                # Round to 4 decimal places to reduce JSON payload size
                response["landmarks"] = [
                    {
                        "i": i,
                        "x": round(float(landmarks_xyzv[i, 0] / width), 4),
                        "y": round(float(landmarks_xyzv[i, 1] / height), 4),
                        "v": round(float(landmarks_xyzv[i, 3]), 3),
                    }
                    for i in _SEND_INDICES
                ]
                # Use bbox from pipeline result (rounded)
                nb = pr.bbox_normalized
                response["bbox"] = {
                    "x1": round(nb[0], 4),
                    "y1": round(nb[1], 4),
                    "x2": round(nb[2], 4),
                    "y2": round(nb[3], 4),
                }

                # Run re-analysis if needed
                if analyzer.needs_reanalysis():
                    await loop.run_in_executor(executor, analyzer.run_analysis)

                    # Trigger Gemini classification for stable clusters (background)
                    if gemini is not None and gemini.available:
                        for cid in analyzer.get_clusters_needing_classification():
                            analyzer.mark_classification_pending(cid)
                            loop.run_in_executor(
                                executor, _classify_cluster_bg, analyzer, cid, gemini
                            )

                # Handle UMAP embedding
                embedding_update = None
                if analyzer.needs_umap_refit():
                    embedding_update = await loop.run_in_executor(
                        executor, analyzer.run_umap_fit
                    )
                elif len(analyzer.features_list) > 0 and analyzer._umap_mapper is not None:
                    feat = analyzer.features_list[-1]
                    embedding_update = analyzer.run_umap_transform(feat)

                srp_joints = analyzer.get_srp_joints()
                if srp_joints is not None:
                    srp_joints = [[round(v, 4) for v in jt] for jt in srp_joints]

                # Encode UV texture if available
                uv_texture_b64 = None
                if pr.smpl_texture_uv is not None:
                    import base64
                    _, buf = cv2.imencode(".jpg", pr.smpl_texture_uv,
                                         [cv2.IMWRITE_JPEG_QUALITY, 60])
                    uv_texture_b64 = base64.b64encode(buf.tobytes()).decode()

                subjects_data[str(subject_id)] = _build_subject_response(
                    subject_id, label, response, srp_joints, embedding_update,
                    identity_status, identity_confidence,
                    pr.smpl_params, uv_texture_b64,
                )

            # Step 3: Cleanup stale subjects and resolver tracks
            manager.cleanup_stale(frame_idx)
            if resolver is not None:
                active_track_ids = {pr.track_id for pr in results}
                resolver.cleanup_stale_tracks(active_track_ids)

            # Step 4: Send combined multi-subject response
            active_ids = manager.get_active_track_ids()
            await websocket.send_json({
                "frame_index": frame_idx,
                "subjects": subjects_data,
                "active_track_ids": active_ids,
            })

            # Send progress every 10 frames
            if frame_idx % 10 == 0:
                progress = frame_idx / max(total_frames, 1)
                await websocket.send_json({
                    "type": "video_progress",
                    "progress": round(progress, 3),
                    "frame": frame_idx,
                    "total": total_frames,
                })

            frame_idx += 1

    finally:
        cap.release()
        executor.shutdown(wait=False)

    # Final analysis pass for all subjects
    for subject_id, analyzer in manager.analyzers.items():
        if analyzer.needs_reanalysis() or len(analyzer._segments) == 0:
            analyzer.run_analysis()

    # Build final summary per subject
    final_summaries = {}
    for subject_id, analyzer in manager.analyzers.items():
        label = manager.get_label(subject_id)
        summary = analyzer.get_final_summary()
        summary["label"] = label
        final_summaries[str(subject_id)] = summary

    await websocket.send_json({
        "type": "video_complete",
        "final_summary": final_summaries,
    })
