"""Process basketball game videos asynchronously with per-player analysis.

Reuses existing pipeline components (SubjectManager, IdentityResolver,
StreamingAnalyzer) and adds jersey detection, team clustering, and
MongoDB persistence.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pipeline_interface import PoseBackend, PipelineResult
    from subject_manager import SubjectManager
    from identity_resolver import IdentityResolver
    from embedding_extractor import EmbeddingExtractor
    from scene_detector import InlineSceneDetector
except ImportError:
    from backend.pipeline_interface import PoseBackend, PipelineResult
    from backend.subject_manager import SubjectManager
    from backend.identity_resolver import IdentityResolver
    from backend.embedding_extractor import EmbeddingExtractor
    from backend.scene_detector import InlineSceneDetector

try:
    from gemini_classifier import GeminiActivityClassifier
    _GEMINI_AVAILABLE = True
except ImportError:
    try:
        from backend.gemini_classifier import GeminiActivityClassifier
        _GEMINI_AVAILABLE = True
    except ImportError:
        _GEMINI_AVAILABLE = False

try:
    from jersey_detector import JerseyDetector, cluster_teams
except ImportError:
    try:
        from backend.jersey_detector import JerseyDetector, cluster_teams
    except ImportError:
        JerseyDetector = None
        cluster_teams = None

try:
    from db import get_collection, make_game_doc, make_game_player_doc, make_player_frame_doc
except ImportError:
    from backend.db import get_collection, make_game_doc, make_game_player_doc, make_player_frame_doc


# Frames to wait before triggering jersey detection for a player (~3s at 30fps)
_JERSEY_DETECT_FRAME_THRESHOLD = 90

# Interval (in frames) between player frame snapshots stored to MongoDB
_SNAPSHOT_INTERVAL = 300

# Basketball-specific activity classification prompt
_BASKETBALL_CLASSIFY_PROMPT = (
    "What basketball activity is this player performing? "
    "Respond with ONLY a single word or short phrase. "
    "Prefer these specific labels when applicable: "
    "dunk, dunking, rebound, rebounding, block, blocking, alley-oop, slam dunk, "
    "crossover, euro step, cutting, direction change, "
    "shooting, jump shot, free throw, layup, three pointer, floater, hook shot, "
    "dribbling, ball handling, "
    "defensive slide, guarding, closeout, defensive shuffle, "
    "running, sprinting, jogging, walking, "
    "jump, landing. "
    "Other valid labels: passing, screening, boxing out, fast break, "
    "catching, standing, stretching."
)


async def process_basketball_game(
    game_id: str,
    video_path: str | Path,
    backend: PoseBackend,
    reid_extractor: EmbeddingExtractor | None = None,
    cross_cut_extractor: EmbeddingExtractor | None = None,
    cluster_threshold: float = 2.0,
    progress_callback: Callable[[dict], Any] | None = None,
) -> None:
    """Process a basketball game video and store results in MongoDB.

    Args:
        game_id: MongoDB game document _id (as string).
        video_path: Path to the MP4 file.
        backend: Initialized PoseBackend instance.
        reid_extractor: Optional re-ID extractor for identity resolution.
        cross_cut_extractor: Optional CLIP-ReID for cross-cut matching.
        cluster_threshold: Clustering distance threshold.
        progress_callback: Optional async/sync callable receiving progress dicts.
    """
    games_col = get_collection("games")
    players_col = get_collection("game_players")
    frames_col = get_collection("player_frames")

    # Update game status to processing
    await games_col.update_one(
        {"_id": _to_object_id(game_id)},
        {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc)}},
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await _fail_game(games_col, game_id, f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    await games_col.update_one(
        {"_id": _to_object_id(game_id)},
        {"$set": {"total_frames": total_frames, "duration_sec": round(duration_sec, 1)}},
    )

    manager = SubjectManager(fps=fps, cluster_threshold=cluster_threshold)
    resolver = (
        IdentityResolver(reid_extractor, cross_cut_extractor=cross_cut_extractor)
        if reid_extractor is not None
        else None
    )
    scene_detector = InlineSceneDetector()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    # Gemini activity classification
    gemini: GeminiActivityClassifier | None = None
    if _GEMINI_AVAILABLE:
        gemini = GeminiActivityClassifier()

    # Jersey detection
    jersey_detector: JerseyDetector | None = None
    if JerseyDetector is not None:
        jersey_detector = JerseyDetector()

    # Ring buffer for Gemini classification
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 600

    # Track per-subject frame counts for jersey detection timing
    subject_frame_counts: dict[int, int] = {}
    # Track which subjects we've started jersey detection for
    jersey_detection_started: set[int] = set()

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster_bg(analyzer, cluster_id: int, gem: GeminiActivityClassifier):
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
        label = gem.classify_activity(frames, bbox, prompt=_BASKETBALL_CLASSIFY_PROMPT)
        analyzer.set_activity_label(cluster_id, label)

    def _detect_jersey_bg(detector: JerseyDetector, subject_id: int, crop: np.ndarray):
        if detector.has_result(subject_id):
            return
        result = detector.detect(crop)
        detector.store_result(subject_id, result)

    backend.reset()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Store frame in ring buffer
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > _FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            # Detect scene cuts
            if scene_detector.process_frame(rgb):
                backend.on_scene_cut()
                if resolver is not None:
                    resolver.on_scene_cut()

            # Pipeline processes frame
            results = await loop.run_in_executor(executor, backend.process_frame, rgb)

            # Resolve identities
            if resolver is not None and results:
                resolved = await loop.run_in_executor(
                    executor, resolver.resolve_pipeline_results, results, rgb, width, height
                )
            else:
                resolved = None

            # Process through analyzers
            if resolved is not None:
                items = resolved
            else:
                items = results

            for item in items:
                if resolved is not None:
                    rp = item
                    pr = rp.pipeline_result
                    subject_id = rp.subject_id
                else:
                    pr = item
                    subject_id = pr.track_id

                landmarks_xyzv = pr.landmarks_mp
                analyzer = manager.get_or_create_analyzer(subject_id)
                analyzer.last_seen_frame = frame_idx

                response = analyzer.process_frame(landmarks_xyzv, img_wh=(width, height))

                # Run re-analysis if needed
                if analyzer.needs_reanalysis():
                    await loop.run_in_executor(executor, analyzer.run_analysis)

                    # Trigger basketball-specific Gemini classification
                    if gemini is not None and gemini.available:
                        for cid in analyzer.get_clusters_needing_classification():
                            analyzer.mark_classification_pending(cid)
                            loop.run_in_executor(
                                executor, _classify_cluster_bg, analyzer, cid, gemini
                            )

                # Track subject frame counts for jersey detection
                subject_frame_counts[subject_id] = subject_frame_counts.get(subject_id, 0) + 1

                # Trigger jersey detection after enough frames
                if (
                    jersey_detector is not None
                    and jersey_detector.available
                    and subject_id not in jersey_detection_started
                    and subject_frame_counts.get(subject_id, 0) >= _JERSEY_DETECT_FRAME_THRESHOLD
                    and pr.crop_rgb is not None
                    and pr.crop_rgb.size > 0
                ):
                    jersey_detection_started.add(subject_id)
                    loop.run_in_executor(
                        executor, _detect_jersey_bg, jersey_detector, subject_id, pr.crop_rgb
                    )

                # Store periodic snapshots to MongoDB
                if frame_idx > 0 and frame_idx % _SNAPSHOT_INTERVAL == 0:
                    quality = response.get("quality")
                    if quality:
                        doc = make_player_frame_doc(
                            game_id=game_id,
                            subject_id=subject_id,
                            frame_idx=frame_idx,
                            quality=quality,
                            biomechanics=quality.get("biomechanics"),
                        )
                        # Fire-and-forget insert
                        asyncio.ensure_future(frames_col.insert_one(doc))

            # Cleanup stale subjects
            manager.cleanup_stale(frame_idx)
            if resolver is not None:
                active_track_ids = {pr.track_id for pr in results}
                resolver.cleanup_stale_tracks(active_track_ids)

            # Update progress every 30 frames
            if frame_idx % 30 == 0:
                progress = frame_idx / max(total_frames, 1)
                await games_col.update_one(
                    {"_id": _to_object_id(game_id)},
                    {"$set": {
                        "progress": round(progress, 3),
                        "frame_idx": frame_idx,
                        "player_count": len(manager.analyzers),
                        "updated_at": datetime.now(timezone.utc),
                    }},
                )

                if progress_callback is not None:
                    msg = {
                        "type": "game_progress",
                        "game_id": game_id,
                        "progress": round(progress, 3),
                        "frame": frame_idx,
                        "total": total_frames,
                        "player_count": len(manager.analyzers),
                    }
                    try:
                        result = progress_callback(msg)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass

            frame_idx += 1

    except Exception as e:
        cap.release()
        executor.shutdown(wait=False)
        await _fail_game(games_col, game_id, str(e))
        return

    cap.release()
    executor.shutdown(wait=False)

    # --- Post-processing ---

    # Final analysis pass
    for subject_id, analyzer in manager.analyzers.items():
        if analyzer.needs_reanalysis() or len(analyzer._segments) == 0:
            analyzer.run_analysis()

    # Team clustering from jersey detection
    team_assignments: dict[int, int] = {}
    team_colors_summary: list[dict] = []
    if jersey_detector is not None:
        all_jersey = jersey_detector.get_all_results()
        if len(all_jersey) >= 2 and cluster_teams is not None:
            team_assignments = cluster_teams(all_jersey)

            # Build team color summary
            from collections import defaultdict
            team_color_names: dict[int, list[str]] = defaultdict(list)
            team_color_rgbs: dict[int, list[list[int]]] = defaultdict(list)
            for sid, tid in team_assignments.items():
                jr = all_jersey.get(sid, {})
                team_color_names[tid].append(jr.get("jersey_color_name", "unknown"))
                team_color_rgbs[tid].append(jr.get("dominant_color", [128, 128, 128]))

            for tid in sorted(team_color_names.keys()):
                # Most common color name
                names = team_color_names[tid]
                most_common = max(set(names), key=names.count)
                # Average RGB
                rgbs = team_color_rgbs[tid]
                avg_rgb = [int(np.mean([c[i] for c in rgbs])) for i in range(3)]
                team_colors_summary.append({
                    "team_id": tid,
                    "color_name": most_common,
                    "rgb": avg_rgb,
                })

    # Store per-player documents
    for subject_id, analyzer in manager.analyzers.items():
        label = manager.get_label(subject_id)
        summary = analyzer.get_final_summary()
        summary["label"] = label

        jersey_info = jersey_detector.get_result(subject_id) if jersey_detector else None
        jersey_number = jersey_info["jersey_number"] if jersey_info else None
        jersey_color = jersey_info["jersey_color_name"] if jersey_info else None
        team_id = team_assignments.get(subject_id)

        # Collect injury events from quality tracker
        injury_events = []
        quality_data = None
        if hasattr(analyzer, '_quality_tracker') and analyzer._quality_tracker is not None:
            qt = analyzer._quality_tracker
            if hasattr(qt, 'injury_events'):
                injury_events = qt.injury_events
            if hasattr(qt, 'get_quality'):
                quality_data = qt.get_quality()

        doc = make_game_player_doc(
            game_id=game_id,
            subject_id=subject_id,
            jersey_number=jersey_number,
            team_id=team_id,
            jersey_color=jersey_color,
        )
        doc["injury_events"] = injury_events
        doc["final_quality"] = quality_data
        doc["analysis_summary"] = summary

        # Upsert player document
        await players_col.update_one(
            {"game_id": game_id, "subject_id": subject_id},
            {"$set": doc},
            upsert=True,
        )

    # Update game as complete
    await games_col.update_one(
        {"_id": _to_object_id(game_id)},
        {"$set": {
            "status": "complete",
            "progress": 1.0,
            "frame_idx": frame_idx,
            "player_count": len(manager.analyzers),
            "team_colors": team_colors_summary,
            "updated_at": datetime.now(timezone.utc),
        }},
    )

    if progress_callback is not None:
        msg = {
            "type": "game_complete",
            "game_id": game_id,
            "player_count": len(manager.analyzers),
            "team_colors": team_colors_summary,
        }
        try:
            result = progress_callback(msg)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass


async def _fail_game(games_col, game_id: str, error: str) -> None:
    """Mark a game as failed in MongoDB."""
    print(f"[game] {game_id} failed: {error}", flush=True)
    await games_col.update_one(
        {"_id": _to_object_id(game_id)},
        {"$set": {
            "status": "failed",
            "error": error,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


def _to_object_id(id_str: str):
    """Convert string to ObjectId, handling both ObjectId strings and raw strings."""
    try:
        from bson import ObjectId
        return ObjectId(id_str)
    except Exception:
        return id_str
