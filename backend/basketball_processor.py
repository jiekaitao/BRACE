"""Async basketball game analysis pipeline.

Processes uploaded basketball video: runs pose detection, identity resolution,
jersey detection, activity classification, per-player risk tracking,
and stores results in MongoDB.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


async def process_basketball_game(
    video_path: Path,
    game_id: str,
    pipeline,
    resolver,
    manager,
    jersey_detector=None,
    gemini_classifier=None,
    vector_classifier=None,
    db_collection=None,
    progress_callback=None,
    executor: ThreadPoolExecutor | None = None,
    fps_override: float | None = None,
) -> dict[str, Any]:
    """Process a basketball video end-to-end.

    Args:
        video_path: Path to the video file.
        game_id: Unique game ID for database storage.
        pipeline: PoseBackend instance.
        resolver: IdentityResolver instance.
        manager: SubjectManager instance.
        jersey_detector: Optional JerseyDetector for jersey recognition.
        gemini_classifier: Optional GeminiActivityClassifier.
        vector_classifier: Optional VectorActivityClassifier.
        db_collection: Optional async MongoDB collection for game updates.
        progress_callback: Optional async callable(progress_pct: float, data: dict).
        executor: ThreadPoolExecutor for blocking operations.
        fps_override: Override video FPS (for testing).

    Returns:
        Summary dict with player results and statistics.
    """
    loop = asyncio.get_event_loop()
    _executor = executor or ThreadPoolExecutor(max_workers=4)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if fps > 0 else 0

    # Import risk engine lazily to avoid circular imports
    try:
        from player_risk_engine import PlayerRiskEngine
    except ImportError:
        PlayerRiskEngine = None

    # Per-subject risk engines
    risk_engines: dict[int, Any] = {}

    # Frame buffer for Gemini classification
    frame_buffer: dict[int, np.ndarray] = {}
    FRAME_BUFFER_MAX = int(fps * 60)  # 60 seconds

    # Jersey detection scheduling
    jersey_detection_done = False
    JERSEY_DETECT_FRAME = int(fps * 3)  # detect jerseys at ~3 seconds in
    JERSEY_DETECT_INTERVAL = int(fps * 30)  # re-detect every 30 seconds

    frame_idx = 0
    t_start = time.monotonic()

    # Update game status to processing
    if db_collection is not None:
        try:
            await db_collection.update_one(
                {"_id": game_id},
                {"$set": {"status": "processing", "total_frames": total_frames}},
            )
        except Exception:
            pass

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            video_time = frame_idx / fps

            # Store frame for Gemini
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            # Run pipeline
            try:
                results = await loop.run_in_executor(_executor, pipeline.process_frame, rgb)
            except Exception:
                results = []
                frame_idx += 1
                continue

            # Resolve identities
            if resolver is not None and results:
                try:
                    resolved = await loop.run_in_executor(
                        _executor, resolver.resolve_pipeline_results, results, rgb, w, h
                    )
                except Exception:
                    resolved = None
            else:
                resolved = None

            # Process each detected person
            items = resolved if resolved is not None else results
            for item in items:
                if resolved is not None:
                    rp = item
                    pr = rp.pipeline_result
                    subject_id = rp.subject_id
                else:
                    pr = item
                    subject_id = pr.track_id

                landmarks = pr.landmarks_mp
                analyzer = manager.get_or_create_analyzer(subject_id)
                analyzer.last_seen_frame = frame_idx
                response = analyzer.process_frame(landmarks, img_wh=(w, h))

                # Re-analysis if needed (segmentation/clustering)
                if analyzer.needs_reanalysis():
                    analyzer.run_analysis()

                # Activity classification for unlabeled clusters
                for cid in analyzer.get_clusters_needing_classification():
                    label = "unknown"
                    # Fast path: VectorAI similarity search
                    if vector_classifier is not None and vector_classifier.available:
                        features = analyzer.features_list
                        if features:
                            label = vector_classifier.classify(features[-1])
                    # Slow path: Gemini vision classification
                    if label == "unknown" and gemini_classifier is not None and gemini_classifier.available:
                        indices = analyzer.get_cluster_frame_indices(cid)
                        bbox = analyzer.get_cluster_bbox(cid)
                        if indices and bbox:
                            frames = gemini_classifier.get_representative_frames(
                                indices, lambda idx: frame_buffer.get(idx), count=4
                            )
                            if frames:
                                label = gemini_classifier.classify_activity(frames, bbox)
                    analyzer.set_activity_label(cid, label)

                # Risk engine per player
                if PlayerRiskEngine is not None:
                    if subject_id not in risk_engines:
                        risk_engines[subject_id] = PlayerRiskEngine(fps=fps)
                    quality = response.get("quality")
                    velocity = response.get("velocity", 0.0)
                    risk_engines[subject_id].process_frame(
                        quality, frame_idx, video_time, velocity
                    )

            # Jersey detection at scheduled frames
            if (
                jersey_detector is not None
                and not jersey_detection_done
                and frame_idx == JERSEY_DETECT_FRAME
                and resolved is not None
            ):
                crops = {}
                for item in resolved:
                    pr = item.pipeline_result
                    bbox = pr.bbox_pixel
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crops[item.subject_id] = rgb[y1:y2, x1:x2]

                if crops:
                    try:
                        jerseys = await loop.run_in_executor(
                            _executor, jersey_detector.detect_batch, crops
                        )
                        for sid, info in jerseys.items():
                            resolver.set_jersey(sid, number=info.number, color=info.color)
                        # Try merging duplicate subjects by jersey
                        merged = resolver.merge_by_jersey()
                        for from_id, to_id in merged:
                            manager.merge_subject(from_id, to_id)
                        jersey_detection_done = True
                    except Exception as e:
                        print(f"[basketball] jersey detection failed: {e}", flush=True)

            # Re-detect jerseys periodically for new players
            if (
                jersey_detector is not None
                and jersey_detection_done
                and frame_idx > 0
                and frame_idx % JERSEY_DETECT_INTERVAL == 0
                and resolved is not None
            ):
                # Only detect for subjects without jersey info
                crops = {}
                confirmed = resolver.get_confirmed_subjects()
                for item in resolved:
                    sid = item.subject_id
                    if sid in confirmed and confirmed[sid].get("jersey_number") is not None:
                        continue
                    pr = item.pipeline_result
                    bbox = pr.bbox_pixel
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crops[sid] = rgb[y1:y2, x1:x2]

                if crops:
                    try:
                        jerseys = await loop.run_in_executor(
                            _executor, jersey_detector.detect_batch, crops
                        )
                        for sid, info in jerseys.items():
                            resolver.set_jersey(sid, number=info.number, color=info.color)
                        merged = resolver.merge_by_jersey()
                        for from_id, to_id in merged:
                            manager.merge_subject(from_id, to_id)
                    except Exception:
                        pass

            # Progress reporting
            if progress_callback is not None and frame_idx % 30 == 0:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                try:
                    await progress_callback(progress, {
                        "frame_index": frame_idx,
                        "total_frames": total_frames,
                        "player_count": len(manager.analyzers),
                        "video_time": round(video_time, 1),
                    })
                except Exception:
                    pass

            # DB progress update every 5 seconds
            if db_collection is not None and frame_idx % (int(fps) * 5) == 0:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                try:
                    await db_collection.update_one(
                        {"_id": game_id},
                        {"$set": {
                            "progress": round(progress, 1),
                            "player_count": len(manager.analyzers),
                        }},
                    )
                except Exception:
                    pass

            frame_idx += 1

    finally:
        cap.release()

    elapsed = time.monotonic() - t_start

    # Build player summaries
    players: dict[str, dict] = {}
    confirmed = resolver.get_confirmed_subjects(min_frames=10) if resolver else {}

    for sid, analyzer in manager.analyzers.items():
        subject_info = confirmed.get(sid, {})
        risk_summary = None
        if sid in risk_engines:
            risk_summary = risk_engines[sid].get_player_summary()

        players[str(sid)] = {
            "subject_id": sid,
            "label": subject_info.get("label", f"S{sid}"),
            "jersey_number": subject_info.get("jersey_number"),
            "jersey_color": subject_info.get("jersey_color"),
            "total_frames": subject_info.get("total_frames", 0),
            "risk": risk_summary,
            "n_clusters": len(analyzer._cluster_labels) if hasattr(analyzer, '_cluster_labels') else 0,
            "n_segments": getattr(analyzer, '_segment_count', 0),
        }

    summary = {
        "game_id": game_id,
        "status": "complete",
        "total_frames": frame_idx,
        "duration_sec": round(duration_sec, 1),
        "processing_time_sec": round(elapsed, 1),
        "player_count": len(players),
        "players": players,
        "fps": fps,
    }

    # Update DB with final status
    if db_collection is not None:
        try:
            await db_collection.update_one(
                {"_id": game_id},
                {"$set": {
                    "status": "complete",
                    "progress": 100.0,
                    "player_count": len(players),
                    "total_frames": frame_idx,
                    "duration_sec": round(duration_sec, 1),
                    "processing_time_sec": round(elapsed, 1),
                }},
            )
        except Exception:
            pass

    return summary
