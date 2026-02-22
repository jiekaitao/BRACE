"""FastAPI WebSocket server for real-time multi-person BRACE analysis.

Receives JPEG frames (webcam) or processes uploaded videos,
runs GPU inference for detection + pose estimation + motion
consistency analysis, and streams multi-subject results back to the frontend.
"""

import asyncio
import base64
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

import struct

try:
    from concussion_pipeline import router as concussion_router
except ImportError:
    concussion_router = None

try:
    import webrtc_api
    _WEBRTC_AVAILABLE = True
except ImportError:
    _WEBRTC_AVAILABLE = False
# GPU JPEG decode via nvJPEG (torchvision + CUDA)
_GPU_JPEG_AVAILABLE = False
try:
    if torch.cuda.is_available():
        from torchvision.io import decode_jpeg as _tv_decode_jpeg
        _GPU_JPEG_AVAILABLE = True
except ImportError:
    pass

from pipeline_interface import PoseBackend, PipelineResult
from subject_manager import SubjectManager
from video_processor import process_video
from identity_resolver import IdentityResolver
from embedding_extractor import EmbeddingExtractor, DummyExtractor
from tensorrt_utils import ensure_tensorrt_engine
from device_utils import get_best_device

try:
    from auth import router as auth_router
except ImportError:
    auth_router = None

try:
    from chat_agent import router as chat_router
except ImportError:
    chat_router = None

try:
    from gemini_classifier import GeminiActivityClassifier
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

try:
    from dashboard_api import router as dashboard_router
except ImportError:
    dashboard_router = None

try:
    from jersey_detector import JerseyDetector, cluster_teams_visual, TeamClustering
    _JERSEY_DETECTOR_AVAILABLE = True
except ImportError:
    _JERSEY_DETECTOR_AVAILABLE = False

# VectorAI integration (required)
_vectorai_store = None
_movement_search = None
_vector_classifier = None
from vectorai_store import VectorAIStore
from vector_movement_search import MovementSearchEngine
from vector_activity_classifier import VectorActivityClassifier

try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False

SCRIPT_DIR = Path(__file__).parent
_default_upload = "/app/uploads" if Path("/app").exists() else str(SCRIPT_DIR.parent / "data" / "uploads")
_default_demo = "/app/data/sports_videos" if Path("/app/data").exists() else str(SCRIPT_DIR.parent / "data" / "sports_videos")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", _default_upload))
DEMO_VIDEOS_DIR = Path(os.environ.get("DEMO_VIDEOS_DIR", _default_demo))

app = FastAPI(title="BRACE API")

if auth_router is not None:
    app.include_router(auth_router)

if chat_router is not None:
    app.include_router(chat_router)

if concussion_router is not None:
    app.include_router(concussion_router)

if dashboard_router is not None:
    app.include_router(dashboard_router)

if _WEBRTC_AVAILABLE:
    app.include_router(webrtc_api.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=8)

# Send head points (0,1,2,3,4) + torso/limbs for frontend skeleton renderer
_SEND_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

pipeline: PoseBackend | None = None
_reid_extractor: EmbeddingExtractor | None = None
_clip_reid_extractor: EmbeddingExtractor | None = None

# Store uploaded video paths by session ID
_uploads: dict[str, Path] = {}

# Active WebSocket stream registry for /api/streams
_active_streams: dict[str, dict[str, Any]] = {}

DISABLE_REID = os.environ.get("DISABLE_REID", "0") == "1"
PIPELINE_BACKEND = os.environ.get("PIPELINE_BACKEND", "legacy")


@app.on_event("startup")
def load_models():
    global pipeline, _reid_extractor, _clip_reid_extractor

    device = get_best_device()
    print(f"[startup] Detected compute device: {device}", flush=True)

    # Pre-convert YOLO model to TensorRT FP16 engine (one-time, ~2-5 min)
    yolo_model = ensure_tensorrt_engine(
        os.environ.get("YOLO_MODEL", "yolo11m-pose.pt")
    )
    print(f"[startup] YOLO model path: {yolo_model}", flush=True)

    if PIPELINE_BACKEND == "advanced":
        try:
            from advanced_backend import AdvancedPoseBackend
            pipeline = AdvancedPoseBackend(model_name=yolo_model)
            print(f"[startup] Advanced pipeline loaded", flush=True)
        except Exception as e:
            print(f"[startup] Advanced backend failed ({e}), falling back to legacy", flush=True)
            _load_legacy_pipeline(yolo_model)
    else:
        _load_legacy_pipeline(yolo_model)

    # Load re-identification model (OSNet) unless disabled
    if not DISABLE_REID:
        try:
            from embedding_extractor import OSNetExtractor
            _reid_extractor = OSNetExtractor()
            print("[startup] OSNet re-ID extractor loaded", flush=True)
        except Exception as e:
            print(f"[startup] OSNet unavailable ({e}), using DummyExtractor", flush=True)
            _reid_extractor = DummyExtractor()

        # Load CLIP-ReID extractor for cross-cut matching (optional)
        try:
            from clip_reid_extractor import CLIPReIDExtractor
            clip_ext = CLIPReIDExtractor()
            if clip_ext.available:
                _clip_reid_extractor = clip_ext
                print("[startup] CLIP-ReID extractor loaded for cross-cut matching", flush=True)
            else:
                print("[startup] CLIP-ReID not available, cross-cut matching disabled", flush=True)
        except Exception as e:
            print(f"[startup] CLIP-ReID unavailable ({e}), cross-cut matching disabled", flush=True)
    else:
        print("[startup] Re-ID disabled (DISABLE_REID=1)", flush=True)

    if _WEBRTC_AVAILABLE:
        webrtc_api.inject_globals({
            "pipeline": pipeline,
            "executor": _executor,
            "SubjectManager": SubjectManager,
            "IdentityResolver": IdentityResolver,
            "DummyExtractor": DummyExtractor,
            "reid_extractor": _reid_extractor,
            "DISABLE_REID": DISABLE_REID,
            "GEMINI_AVAILABLE": _GEMINI_AVAILABLE,
            "GeminiActivityClassifier": GeminiActivityClassifier if _GEMINI_AVAILABLE else None,
            "SEND_INDICES": _SEND_INDICES,
        })

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[startup] Upload dir: {UPLOAD_DIR}", flush=True)

    # Initialize VectorAI store (required — raises on failure)
    global _vectorai_store, _movement_search, _vector_classifier
    _vectorai_store = VectorAIStore()
    if not _vectorai_store.health_check():
        raise RuntimeError("[startup] VectorAI health check failed after connection")
    _movement_search = MovementSearchEngine(_vectorai_store)
    _vector_classifier = VectorActivityClassifier(_vectorai_store)
    print("[startup] VectorAI store and search engine initialized", flush=True)

    # Ensure MongoDB indexes for auth/chat/workout collections
    try:
        from db import ensure_indexes

        async def _ensure_indexes():
            try:
                await ensure_indexes()
                print("[startup] MongoDB indexes ensured", flush=True)
            except Exception as exc:
                print(f"[startup] MongoDB indexes failed ({exc}), auth may not work", flush=True)

        asyncio.ensure_future(_ensure_indexes())
    except Exception as e:
        print(f"[startup] MongoDB indexes import failed ({e}), auth may not work", flush=True)


def _load_legacy_pipeline(model_name: str = "yolo11x-pose.pt"):
    global pipeline
    from legacy_backend import LegacyPoseBackend
    pipeline = LegacyPoseBackend(model_name=model_name, conf_threshold=0.3)
    print(f"[startup] YOLO-pose device: {pipeline.model.device}", flush=True)
    pipeline.warmup()
    print("[startup] Legacy pipeline loaded and warmed up", flush=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tracker_loaded": pipeline is not None,
    }


@app.get("/api/streams")
def get_active_streams():
    """Return metadata for all active WebSocket analysis streams."""
    now = time.time()
    streams = []
    for sid, info in list(_active_streams.items()):
        streams.append({
            "stream_id": sid,
            "mode": info.get("mode", "webcam"),
            "client_type": info.get("client_type", "web"),
            "connected_at": info.get("connected_at", 0),
            "uptime_sec": round(now - info.get("connected_at", now), 1),
            "frame_count": info.get("frame_count", 0),
            "subject_count": info.get("subject_count", 0),
            "fps": info.get("fps", 0),
            "last_thumbnail": info.get("last_thumbnail"),
            "resolution": info.get("resolution"),
        })
    return {"count": len(streams), "streams": streams}


@app.get("/api/streams/{stream_id}/frame")
def get_stream_frame(
    stream_id: str,
    w: int = Query(default=640, ge=80, le=1920),
    overlay: bool = Query(default=False),
):
    """Return the latest frame from a stream as a JPEG image.

    If overlay=true, draw bounding boxes and labels on the frame.
    """
    info = _active_streams.get(stream_id)
    if info is None:
        return Response(status_code=404, content="Stream not found")
    rgb = info.get("_last_rgb")
    if rgb is None:
        return Response(status_code=204, content="No frame yet")
    h_orig, w_orig = rgb.shape[:2]

    # Work on a copy so we don't mutate the shared frame
    frame = rgb.copy()

    # Draw bounding boxes if requested
    if overlay:
        subjects = info.get("_last_subjects") or {}
        for sid_str, sdata in subjects.items():
            bbox = sdata.get("bbox")
            if bbox is None:
                continue
            x1 = int(bbox["x1"] * w_orig)
            y1 = int(bbox["y1"] * h_orig)
            x2 = int(bbox["x2"] * w_orig)
            y2 = int(bbox["y2"] * h_orig)
            # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label background
            label = sdata.get("label", f"S{sid_str}")
            phase = sdata.get("phase", "")
            txt = f"{label} [{phase}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if w < w_orig:
        scale = w / w_orig
        new_h = int(h_orig * scale)
        frame = cv2.resize(frame, (w, new_h))
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/streams/{stream_id}/data")
def get_stream_data(stream_id: str):
    """Return the latest per-subject analysis data for a stream (JSON)."""
    info = _active_streams.get(stream_id)
    if info is None:
        return Response(status_code=404, content="Stream not found")
    response = info.get("_last_response") or {}
    subjects = info.get("_last_subjects") or {}
    # Build a stripped-down per-subject summary (skip heavy fields like
    # embedding_update, cluster_representatives, srp_joints, etc.)
    summary: dict[str, Any] = {}
    for sid_str, sdata in subjects.items():
        entry: dict[str, Any] = {
            "label": sdata.get("label"),
            "phase": sdata.get("phase"),
            "bbox": sdata.get("bbox"),
            "cluster_id": sdata.get("cluster_id"),
            "consistency_score": sdata.get("consistency_score"),
            "is_anomaly": sdata.get("is_anomaly"),
            "n_segments": sdata.get("n_segments"),
            "n_clusters": sdata.get("n_clusters"),
            "identity_status": sdata.get("identity_status"),
            "identity_confidence": sdata.get("identity_confidence"),
            "velocity": sdata.get("velocity"),
            "rolling_velocity": sdata.get("rolling_velocity"),
            "fatigue_index": sdata.get("fatigue_index"),
            "jersey_number": sdata.get("jersey_number"),
            "jersey_color": sdata.get("jersey_color"),
        }
        quality = sdata.get("quality")
        if quality:
            entry["quality"] = quality
        summary[sid_str] = entry
    return {**response, "subjects": summary}


@app.get("/api/gpu-stats")
def gpu_stats():
    """Return GPU utilization, VRAM, temperature, and power draw."""
    if not _nvml_available:
        return {"available": False}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "available": True,
            "name": name,
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "vram_used_gb": round(mem.used / 1e9, 1),
            "vram_total_gb": round(mem.total / 1e9, 1),
            "temp_c": temp,
            "power_w": round(power, 0),
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept an MP4 upload, save to temp dir, return session_id."""
    session_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{session_id}.mp4"
    content = await file.read()
    dest.write_bytes(content)
    _uploads[session_id] = dest
    return {"session_id": session_id, "filename": file.filename}


@app.get("/api/demo-videos")
def list_demo_videos():
    """List available demo videos from the sports_videos directory."""
    videos = []
    if DEMO_VIDEOS_DIR.is_dir():
        for f in sorted(DEMO_VIDEOS_DIR.iterdir()):
            if f.suffix == ".mp4":
                videos.append({
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                })
    return {"videos": videos}


@app.get("/api/demo-videos/{filename}")
def serve_demo_video(filename: str):
    """Serve a demo video file for streaming playback."""
    path = DEMO_VIDEOS_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        return {"error": "Not found"}
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/demo-videos/{filename}/thumbnail")
def demo_video_thumbnail(filename: str):
    """Extract a thumbnail frame from a demo video."""
    path = DEMO_VIDEOS_DIR / filename
    if not path.exists() or path.suffix != ".mp4":
        return Response(status_code=404)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Seek to ~1 second in for a more representative frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return Response(status_code=404)
    # Resize to 320px wide, maintain aspect ratio
    h, w = frame.shape[:2]
    new_w = 320
    new_h = int(h * new_w / w)
    thumb = cv2.resize(frame, (new_w, new_h))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(content=buf.tobytes(), media_type="image/jpeg")



# ---------------------------------------------------------------------------
# VectorAI REST API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/vectorai/health")
def vectorai_health():
    """Return VectorAI connection status."""
    if _vectorai_store.health_check():
        return {"status": "ok", "host": _vectorai_store._host, "port": _vectorai_store._port}
    return {"status": "unhealthy", "host": _vectorai_store._host, "port": _vectorai_store._port}


@app.get("/api/movements/similar")
def movements_similar(
    session_id: str = "",
    person_id: str = "",
    activity_label: str = "",
    top_k: int = 5,
):
    """Find similar movements from past sessions.

    Query params:
        session_id: (optional) filter to a specific session
        person_id: (optional) filter to a specific person
        activity_label: (optional) filter to a specific activity
        top_k: number of results (default 5)
    """
    # VectorAI is always available — no guard needed

    # Build filters
    filters = {}
    if person_id:
        filters["person_id"] = person_id
    if activity_label:
        filters["activity_label"] = activity_label
    if session_id:
        filters["session_id"] = session_id

    # For a general query without a specific vector, we return recent segments
    # A proper query requires a feature vector — this endpoint is best used
    # with a segment_idx param that references stored features
    return {"error": "Provide a feature vector via POST", "results": []}


@app.post("/api/movements/similar")
async def movements_similar_post(request: dict):
    """Find similar movements by posting a feature vector.

    Body: {"features": [float, ...], "top_k": int, "filters": {...}}
    """
    try:
        features = np.array(request.get("features", []), dtype=np.float32)
        if features.shape[0] == 0:
            return {"error": "Empty features vector", "results": []}
        top_k = request.get("top_k", 5)
        filters = request.get("filters")
        results = _movement_search.find_similar(features, top_k=top_k, filters=filters)
        return {"results": results}
    except Exception as e:
        return {"error": str(e), "results": []}


@app.get("/api/person/{person_id}/history")
def person_history(person_id: str):
    """Return cross-session history for a person."""
    # Query motion_segments filtered by person_id
    # We use a zero vector as a neutral query to get all segments for this person
    try:
        dummy_query = np.zeros(42, dtype=np.float32)
        results = _vectorai_store.find_similar_movements(
            dummy_query, top_k=50, filters={"person_id": person_id}
        )
        return {
            "person_id": person_id,
            "history": [
                {
                    "activity_label": r["metadata"].get("activity_label", "unknown"),
                    "session_id": r["metadata"].get("session_id", ""),
                    "risk_score": r["metadata"].get("risk_score", 0.0),
                    "timestamp": r["metadata"].get("timestamp", 0.0),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"error": str(e), "history": []}


# ---------------------------------------------------------------------------
# Game Analysis API Endpoints
# ---------------------------------------------------------------------------

# In-memory game processing tasks
_game_tasks: dict[str, asyncio.Task] = {}
_game_results: dict[str, dict] = {}
_game_progress_sockets: dict[str, list[WebSocket]] = {}


@app.post("/api/games")
async def create_game(
    session_id: str = Query(..., description="Upload session ID"),
    sport: str = Query("basketball", description="Sport type"),
    user_id: str = Query("", description="Optional user ID"),
):
    """Submit a video for game analysis.

    The session_id must reference a previously uploaded video via POST /api/upload.
    """
    video_path = _uploads.get(session_id)
    if not video_path or not video_path.exists():
        return {"error": "Video not found. Upload first via POST /api/upload."}

    game_id = str(uuid.uuid4())

    # Store game doc in MongoDB
    try:
        from db import get_collection, make_game_doc
        games_col = get_collection("games")
        doc = make_game_doc(
            session_id=session_id,
            video_name=video_path.name,
            sport=sport,
            user_id=user_id or None,
        )
        doc["_id"] = game_id
        await games_col.insert_one(doc)
    except Exception as e:
        print(f"[game] DB insert failed: {e}", flush=True)

    # Start async processing task
    task = asyncio.create_task(_process_game_bg(game_id, video_path, sport, user_id))
    _game_tasks[game_id] = task
    task.add_done_callback(lambda t: _game_tasks.pop(game_id, None))

    return {"game_id": game_id, "status": "processing"}


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """Get game analysis status and results."""
    # Check in-memory results first
    if game_id in _game_results:
        return _game_results[game_id]

    # Check if still processing
    if game_id in _game_tasks:
        return {"game_id": game_id, "status": "processing"}

    # Check MongoDB
    try:
        from db import get_collection
        games_col = get_collection("games")
        doc = await games_col.find_one({"_id": game_id})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
    except Exception:
        pass

    return {"error": "Game not found"}


@app.get("/api/games/{game_id}/players")
async def get_game_players(game_id: str):
    """List all players detected in a game."""
    if game_id in _game_results:
        result = _game_results[game_id]
        return {"players": list(result.get("players", {}).values())}

    try:
        from db import get_collection
        players_col = get_collection("game_players")
        cursor = players_col.find({"game_id": game_id}).sort("subject_id", 1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return {"players": results}
    except Exception as e:
        return {"error": str(e), "players": []}


@app.get("/api/games/{game_id}/players/{subject_id}")
async def get_game_player_detail(game_id: str, subject_id: int):
    """Get detailed analysis for a specific player."""
    if game_id in _game_results:
        result = _game_results[game_id]
        player = result.get("players", {}).get(str(subject_id))
        if player:
            return player
        return {"error": "Player not found"}

    try:
        from db import get_collection
        players_col = get_collection("game_players")
        doc = await players_col.find_one(
            {"game_id": game_id, "subject_id": subject_id}
        )
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return {"error": "Player not found"}
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/games/{game_id}")
async def ws_game_progress(websocket: WebSocket, game_id: str):
    """WebSocket for real-time game processing progress."""
    await websocket.accept()

    if game_id not in _game_progress_sockets:
        _game_progress_sockets[game_id] = []
    _game_progress_sockets[game_id].append(websocket)

    try:
        # If already complete, send result immediately
        if game_id in _game_results:
            await websocket.send_json({
                "type": "complete",
                "data": _game_results[game_id],
            })
            return

        # Keep connection alive while processing
        while game_id in _game_tasks:
            try:
                # Wait for client messages (keepalive)
                msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break

        # Send final result if available
        if game_id in _game_results:
            await websocket.send_json({
                "type": "complete",
                "data": _game_results[game_id],
            })
    except WebSocketDisconnect:
        pass
    finally:
        socks = _game_progress_sockets.get(game_id, [])
        if websocket in socks:
            socks.remove(websocket)


@app.post("/api/activity-templates")
async def seed_activity_templates(body: dict):
    """Seed VectorAI with activity classification templates.

    Body: {"templates": {"label": [feature_vector], ...}}
    """
    templates = body.get("templates", {})
    seeded = {}
    for label, features in templates.items():
        try:
            arr = np.array(features, dtype=np.float32)
            _vector_classifier.seed_templates({label: arr})
            seeded[label] = True
        except Exception as e:
            seeded[label] = str(e)

    return {"seeded": seeded}


async def _process_game_bg(
    game_id: str, video_path: Path, sport: str, user_id: str
) -> None:
    """Background task to process a game video."""
    try:
        from basketball_processor import process_basketball_game

        # Build resolver
        use_reid = (
            not DISABLE_REID
            and _reid_extractor is not None
            and not isinstance(_reid_extractor, DummyExtractor)
        )
        resolver = IdentityResolver(
            _reid_extractor,
            cross_cut_extractor=_clip_reid_extractor,
            vectorai_store=_vectorai_store,
        ) if use_reid else None

        manager = SubjectManager(fps=30.0, cluster_threshold=2.0, vectorai_store=_vectorai_store)

        # Jersey detector
        jersey_det = None
        if _JERSEY_DETECTOR_AVAILABLE and _GEMINI_AVAILABLE:
            jersey_det = JerseyDetector()

        # Gemini classifier
        gemini = GeminiActivityClassifier() if _GEMINI_AVAILABLE else None

        # DB collection
        games_col = None
        try:
            from db import get_collection
            games_col = get_collection("games")
        except Exception:
            pass

        async def _progress_cb(pct: float, data: dict):
            # Broadcast to WebSocket subscribers
            sockets = _game_progress_sockets.get(game_id, [])
            for ws in list(sockets):
                try:
                    await ws.send_json({
                        "type": "progress",
                        "progress": round(pct, 1),
                        "data": data,
                    })
                except Exception:
                    sockets.remove(ws)

        result = await process_basketball_game(
            video_path=video_path,
            game_id=game_id,
            pipeline=pipeline,
            resolver=resolver,
            manager=manager,
            jersey_detector=jersey_det,
            gemini_classifier=gemini,
            vector_classifier=_vector_classifier,
            db_collection=games_col,
            progress_callback=_progress_cb,
            executor=_executor,
        )

        _game_results[game_id] = result

        # Store player documents in MongoDB
        try:
            from db import get_collection, make_game_player_doc
            players_col = get_collection("game_players")
            for sid_str, pdata in result.get("players", {}).items():
                risk = pdata.get("risk") or {}
                doc = make_game_player_doc(
                    game_id=game_id,
                    subject_id=pdata["subject_id"],
                    label=pdata["label"],
                    jersey_number=pdata.get("jersey_number"),
                    jersey_color=pdata.get("jersey_color"),
                    risk_status=risk.get("status", "GREEN"),
                    total_frames=pdata.get("total_frames", 0),
                    injury_events=risk.get("events", []),
                    workload=risk.get("workload", {}),
                )
                await players_col.insert_one(doc)
        except Exception as e:
            print(f"[game] player DB insert failed: {e}", flush=True)

        # Notify WebSocket subscribers of completion
        sockets = _game_progress_sockets.get(game_id, [])
        for ws in list(sockets):
            try:
                await ws.send_json({"type": "complete", "data": result})
            except Exception:
                pass

    except Exception as e:
        print(f"[game] processing failed: {e}", flush=True)
        _game_results[game_id] = {
            "game_id": game_id,
            "status": "error",
            "error": str(e),
        }
        # Update DB
        try:
            from db import get_collection
            games_col = get_collection("games")
            await games_col.update_one(
                {"_id": game_id},
                {"$set": {"status": "error", "error": str(e)}},
            )
        except Exception:
            pass


def _gpu_decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes on GPU via nvJPEG, return RGB numpy array."""
    tensor = torch.frombuffer(jpeg_bytes, dtype=torch.uint8)
    # decode_jpeg with device="cuda" uses nvJPEG hardware decoder
    rgb_tensor = _tv_decode_jpeg(tensor, device="cuda")  # (3, H, W) uint8
    # Permute to (H, W, 3) and move to CPU as contiguous numpy
    return rgb_tensor.permute(1, 2, 0).cpu().numpy()


def decode_frame(data: str) -> np.ndarray:
    """Decode a base64-encoded JPEG into an RGB numpy array."""
    raw = base64.b64decode(data)
    if _GPU_JPEG_AVAILABLE:
        return _gpu_decode_jpeg(raw)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_frame_bytes(data: bytes) -> np.ndarray:
    """Decode raw JPEG bytes into an RGB numpy array."""
    if _GPU_JPEG_AVAILABLE:
        return _gpu_decode_jpeg(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_frame_with_timestamp(data: bytes) -> tuple[np.ndarray, float]:
    """Extract video_time (8-byte float64 prefix) + JPEG from binary message."""
    if len(data) > 8:
        video_time = struct.unpack("<d", data[:8])[0]
        jpeg_bytes = data[8:]
    else:
        video_time = 0.0
        jpeg_bytes = data
    rgb = decode_frame_bytes(jpeg_bytes)
    return rgb, video_time


@app.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket):
    """Primary analysis WebSocket endpoint.

    Query params:
        mode: "webcam" or "video"
        session_id: required for video mode (from POST /api/upload)
        cluster_threshold: optional (default 2.0)
    """
    mode = websocket.query_params.get("mode", "webcam")
    session_id = websocket.query_params.get("session_id", "")
    cluster_threshold = float(websocket.query_params.get("cluster_threshold", "2.0"))
    client_type = websocket.query_params.get("client", "web")  # "web" or "vr"
    fps_param = websocket.query_params.get("fps")
    target_fps = float(fps_param) if fps_param else 30.0
    user_id_param = websocket.query_params.get("user_id", "")

    await websocket.accept()

    # Register in active streams
    stream_id = str(uuid.uuid4())
    _active_streams[stream_id] = {
        "mode": mode,
        "client_type": client_type,
        "connected_at": time.time(),
        "frame_count": 0,
        "subject_count": 0,
        "fps": target_fps,
        "last_thumbnail": None,
        "resolution": None,
    }

    if pipeline is None:
        await websocket.send_json({"type": "error", "message": "Models not loaded"})
        await websocket.close()
        _active_streams.pop(stream_id, None)
        return

    # Try to load user's risk modifiers if user_id provided
    risk_modifiers = None
    if user_id_param:
        try:
            from bson import ObjectId
            from db import get_collection
            users = get_collection("users")
            user_doc = await users.find_one({"_id": ObjectId(user_id_param)})
            if user_doc and user_doc.get("risk_modifiers"):
                try:
                    from risk_profile import RiskModifiers
                    risk_modifiers = RiskModifiers.from_dict(user_doc["risk_modifiers"])
                except ImportError:
                    pass
        except Exception as e:
            print(f"[ws] Failed to load risk modifiers: {e}", flush=True)

    try:
        if mode == "video":
            await _handle_video_mode(websocket, session_id, cluster_threshold)
        else:
            await _handle_webcam_mode(websocket, cluster_threshold, target_fps, risk_modifiers, client_type, stream_id)
    finally:
        _active_streams.pop(stream_id, None)


async def _handle_video_mode(
    websocket: WebSocket, session_id: str, cluster_threshold: float
) -> None:
    """Process an uploaded video file and stream results."""
    video_path = _uploads.get(session_id)
    if not video_path or not video_path.exists():
        await websocket.send_json({"type": "error", "message": "Video not found. Upload first."})
        await websocket.close()
        return

    try:
        reid = _reid_extractor if (not DISABLE_REID and not isinstance(_reid_extractor, DummyExtractor)) else None
        clip_reid = _clip_reid_extractor if not DISABLE_REID else None
        await process_video(video_path, websocket, pipeline, cluster_threshold, reid, clip_reid)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def _build_vr_response(
    subjects_data: dict[str, dict[str, Any]],
    active_ids: list[int],
    frame_idx: int,
    video_time: float,
    vr_selected_subject: int | None,
    timing: dict[str, Any],
) -> dict[str, Any]:
    """Build VR-optimized response: minimal data for all, detailed for selected."""
    vr_subjects: dict[str, dict[str, Any]] = {}
    for sid_str, data in subjects_data.items():
        sid = int(sid_str)
        if sid == vr_selected_subject:
            # Selected: send analysis data but strip heavy visualization fields
            vr_sub: dict[str, Any] = {
                "label": data["label"],
                "bbox": data["bbox"],
                "phase": data["phase"],
                "selected": True,
                "n_segments": data["n_segments"],
                "n_clusters": data["n_clusters"],
                "cluster_id": data["cluster_id"],
                "consistency_score": data["consistency_score"],
                "is_anomaly": data["is_anomaly"],
                "cluster_summary": data["cluster_summary"],
                "identity_status": data.get("identity_status", "unknown"),
                "identity_confidence": data.get("identity_confidence", 0),
                "velocity": data.get("velocity", 0),
                "rolling_velocity": data.get("rolling_velocity", 0),
                "fatigue_index": data.get("fatigue_index", 0),
                "peak_velocity": data.get("peak_velocity", 0),
            }
            if "quality" in data:
                vr_sub["quality"] = data["quality"]
            vr_subjects[sid_str] = vr_sub
        else:
            # Unselected: bbox + label only (~80 bytes)
            vr_subjects[sid_str] = {
                "label": data["label"],
                "bbox": data["bbox"],
                "phase": data.get("phase", "calibrating"),
                "selected": False,
            }
    return {
        "frame_index": frame_idx,
        "video_time": video_time,
        "subjects": vr_subjects,
        "active_track_ids": active_ids,
        "timing": timing,
    }


async def _run_analysis_bg(analyzer: "StreamingAnalyzer", subject_id: str) -> None:
    """Run segmentation/clustering analysis in background thread."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, analyzer.run_analysis)
    finally:
        analyzer._analysis_pending = False


async def _handle_webcam_mode(
    websocket: WebSocket, cluster_threshold: float, target_fps: float,
    risk_modifiers: Any = None, client_type: str = "web", stream_id: str = "",
) -> None:
    """Process live webcam frames with multi-person tracking + pose estimation."""
    session_id = str(uuid.uuid4())  # unique per-session for VectorAI tracking
    manager = SubjectManager(fps=target_fps, cluster_threshold=cluster_threshold, risk_modifiers=risk_modifiers, vectorai_store=_vectorai_store)
    loop = asyncio.get_event_loop()
    frame_idx = 0
    vr_selected_subject: int | None = None  # VR client's selected subject

    # Create IdentityResolver for this session (skip if using DummyExtractor)
    use_reid = (
        not DISABLE_REID
        and _reid_extractor is not None
        and not isinstance(_reid_extractor, DummyExtractor)
    )
    resolver = IdentityResolver(
        _reid_extractor,
        cross_cut_extractor=_clip_reid_extractor,
        vectorai_store=_vectorai_store,
    ) if use_reid else None

    # Gemini activity classification
    gemini: GeminiActivityClassifier | None = None
    if _GEMINI_AVAILABLE:
        gemini = GeminiActivityClassifier()

    # Jersey detection for real-time mode
    jersey_detector: JerseyDetector | None = None
    if _JERSEY_DETECTOR_AVAILABLE and _GEMINI_AVAILABLE:
        jersey_detector = JerseyDetector()
    jersey_last_detect_time: float = 0.0
    _JERSEY_INITIAL_DELAY = 3.0   # seconds before first detection
    _JERSEY_REDETECT_INTERVAL = 30.0  # seconds between re-detections
    jersey_debug_cache: dict[int, dict] = {}  # subject_id -> {"crop_b64": str, "gemini_response": str}
    jersey_detect_pending: bool = False
    jersey_session_start: float = time.monotonic()
    team_clustering: TeamClustering | None = None

    # Ring buffer of recent RGB frames for Gemini classification
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 1800  # 60s at 30fps — retain frames for Gemini classification

    # Track video_time to detect loop boundaries (video_time jumps backward)
    prev_video_time: float = 0.0
    loop_cooldown: int = 0  # frames to skip loop detection after a loop

    # Equipment tracking state
    equipment_state = {
        "box": None,
        "momentum": 0.0,
        "held_by_id": None,
        "last_box": None,
        "last_time": 0.0,
        "pending": False,
    }
    analysis_tasks: set[asyncio.Task[Any]] = set()

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster_webcam(analyzer, cluster_id: int, gem: GeminiActivityClassifier):
        # Gemini vision classification
        indices = analyzer.get_cluster_frame_indices(cluster_id)
        if not indices:
            print(f"[gemini] cluster {cluster_id}: no frame indices, skipping", flush=True)
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        bbox = analyzer.get_cluster_bbox(cluster_id)
        if bbox is None:
            print(f"[gemini] cluster {cluster_id}: no bbox, skipping", flush=True)
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        frames = gem.get_representative_frames(indices, _get_buffered_frame, count=4)
        if not frames:
            print(
                f"[gemini] cluster {cluster_id}: no buffered frames "
                f"(indices {min(indices)}..{max(indices)}, buffer has {len(frame_buffer)} frames)",
                flush=True,
            )
            analyzer.set_activity_label(cluster_id, "unknown")
            return
        label = gem.classify_activity(frames, bbox)
        print(f"[gemini] cluster {cluster_id} -> '{label}' ({len(frames)} frames, {len(indices)} indices)", flush=True)
        analyzer.set_activity_label(cluster_id, label)

    try:
        while True:
            msg = await websocket.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            # Handle text messages (reset, config)
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "reset":
                        # Video loop detected — freeze analyzers, keep pipeline warm
                        manager.note_loop()
                        if resolver is not None:
                            resolver.on_scene_cut()
                        frame_idx = 0
                        frame_buffer.clear()
                        continue
                    if data.get("type") == "config":
                        if "cluster_threshold" in data:
                            manager.cluster_threshold = float(data["cluster_threshold"])
                        continue
                    if data.get("type") == "select_subject":
                        vr_selected_subject = data.get("subject_id")  # int or None
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass

            # Decode frame (binary messages include 8-byte timestamp prefix)
            video_time = 0.0
            t_decode_start = time.monotonic()
            try:
                if "bytes" in msg and msg["bytes"]:
                    rgb, video_time = decode_frame_with_timestamp(msg["bytes"])
                elif "text" in msg and msg["text"]:
                    rgb = decode_frame(msg["text"])
                else:
                    continue
            except (ValueError, Exception) as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue
            t_decode = time.monotonic() - t_decode_start

            h, w = rgb.shape[:2]

            # Detect video loop: video_time jumps backward by >1s (debounced)
            if loop_cooldown > 0:
                loop_cooldown -= 1
            elif video_time > 0 and prev_video_time > 0 and video_time < prev_video_time - 1.0:
                # Video looped — freeze analyzers and flush current state
                manager.note_loop()
                # Keep pipeline warm (no reset) — tracker naturally re-acquires
                if resolver is not None:
                    resolver.on_scene_cut()
                # Send analysis state to frontend immediately
                loop_subjects: dict[str, Any] = {}
                for tid in manager.get_active_track_ids():
                    a = manager.analyzers[tid]
                    if a._frozen and a._reps_version > 0:
                        resp = a._build_response(frame_idx, None, None)
                        resp["label"] = manager.get_label(tid)
                        loop_subjects[str(tid)] = resp
                if loop_subjects:
                    print(f"[loop] flushing {len(loop_subjects)} frozen subjects", flush=True)
                    await websocket.send_json({
                        "frame_index": frame_idx,
                        "subjects": loop_subjects,
                        "active_track_ids": manager.get_active_track_ids(),
                    })
                loop_cooldown = 30  # ignore loop detection for next 30 frames (~1s)
            prev_video_time = video_time

            # Store frame in ring buffer for Gemini classification
            frame_buffer[frame_idx] = rgb
            if len(frame_buffer) > _FRAME_BUFFER_MAX:
                oldest = min(frame_buffer)
                del frame_buffer[oldest]

            t0 = time.monotonic()

            # Step 1: Pipeline processes frame -> PipelineResults
            try:
                results = await loop.run_in_executor(_executor, pipeline.process_frame, rgb)
            except Exception as e:
                print(f"[pipeline] frame {frame_idx} error: {e}", flush=True)
                results = []

            t_infer = time.monotonic()

            # Step 1.5: Resolve identities (stable subject_ids across re-appearances)
            if resolver is not None and results:
                resolved = await loop.run_in_executor(
                    _executor, resolver.resolve_pipeline_results, results, rgb, w, h
                )
            else:
                resolved = None

            t_resolve = time.monotonic()

            # Step 1.6: Jersey detection (periodic, background)
            if (
                jersey_detector is not None
                and jersey_detector.available
                and not jersey_detect_pending
                and results
                and resolver is not None
            ):
                elapsed_since_start = time.monotonic() - jersey_session_start
                elapsed_since_detect = time.monotonic() - jersey_last_detect_time
                should_detect = (
                    (jersey_last_detect_time == 0.0 and elapsed_since_start >= _JERSEY_INITIAL_DELAY)
                    or (jersey_last_detect_time > 0.0 and elapsed_since_detect >= _JERSEY_REDETECT_INTERVAL)
                )
                if should_detect:
                    jersey_detect_pending = True
                    jersey_last_detect_time = time.monotonic()

                    # Collect crops for subjects without jersey info
                    crops_to_detect: dict[int, np.ndarray] = {}
                    for pr_item in results:
                        tid = pr_item.track_id
                        sid = resolver._track_to_subject.get(tid, tid) if resolver else tid
                        gallery = resolver._galleries.get(sid) if resolver else None
                        if gallery is not None and gallery.jersey_number is not None:
                            continue  # already detected
                        nb = pr_item.bbox_normalized
                        x1_px = max(0, int(nb[0] * w))
                        y1_px = max(0, int(nb[1] * h))
                        x2_px = min(w, int(nb[2] * w))
                        y2_px = min(h, int(nb[3] * h))
                        if x2_px > x1_px and y2_px > y1_px:
                            crops_to_detect[sid] = rgb[y1_px:y2_px, x1_px:x2_px].copy()

                    if crops_to_detect:
                        def _run_jersey_detection(
                            det=jersey_detector,
                            crops=crops_to_detect,
                            res=resolver,
                            mgr=manager,
                            cache=jersey_debug_cache,
                        ):
                            try:
                                results_j = det.detect_batch(crops)
                                for sid, info in results_j.items():
                                    res.set_jersey(sid, number=info.number, color=info.color)
                                    # Cache crop + response for debug
                                    crop = crops.get(sid)
                                    if crop is not None:
                                        from jersey_detector import _encode_crop_jpeg
                                        crop_b64 = base64.b64encode(_encode_crop_jpeg(crop)).decode()
                                        response_str = json.dumps({"number": info.number, "color": info.color})
                                        cache[sid] = {"crop_b64": crop_b64, "gemini_response": response_str}
                                # Merge subjects with same jersey
                                merges = res.merge_by_jersey()
                                for from_id, to_id in merges:
                                    mgr.merge_subject(from_id, to_id)
                                    print(f"[jersey] merged S{from_id} -> S{to_id}", flush=True)
                            except Exception as e:
                                print(f"[jersey] detection error: {e}", flush=True)

                        def _jersey_done(*_):
                            nonlocal jersey_detect_pending
                            jersey_detect_pending = False

                        fut = asyncio.ensure_future(loop.run_in_executor(_executor, _run_jersey_detection))
                        fut.add_done_callback(_jersey_done)
                    else:
                        jersey_detect_pending = False

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
                analyzer._session_id = session_id
                analyzer._person_id = label

                # Process through analyzer (pixel coordinates for SRP)
                response = analyzer.process_frame(landmarks_xyzv, img_wh=(w, h))

                # Normalize landmark coordinates to [0, 1] -- sparse, body joints only
                # Round to 4 decimal places to reduce JSON payload size
                response["landmarks"] = [
                    {
                        "i": i,
                        "x": round(float(landmarks_xyzv[i, 0] / w), 4),
                        "y": round(float(landmarks_xyzv[i, 1] / h), 4),
                        "v": round(float(landmarks_xyzv[i, 3]), 4),
                    }
                    for i in _SEND_INDICES
                ]
                # Use bbox from pipeline result (rounded to 4 decimals)
                nb = pr.bbox_normalized
                response["bbox"] = {
                    "x1": round(nb[0], 4),
                    "y1": round(nb[1], 4),
                    "x2": round(nb[2], 4),
                    "y2": round(nb[3], 4),
                }

                # Equipment Tracking Background Task
                if gemini is not None and gemini.available and not equipment_state["pending"]:
                    if frame_idx % 15 == 0:  # Check every ~500ms
                        equipment_state["pending"] = True
                        
                        def _track_equipment(f=rgb, t=video_time, w_w=w, h_h=h, snapshot_subjects=list(results)):
                            box = gemini.locate_object(f, "american football")
                            
                            if box is not None:
                                ymin, xmin, ymax, xmax = box
                                cx = (xmin + xmax) / 2.0
                                cy = (ymin + ymax) / 2.0
                                
                                # Calculate momentum if we have a previous box
                                momentum = 0.0
                                if equipment_state["last_box"] is not None and equipment_state["last_time"] > 0:
                                    dt = t - equipment_state["last_time"]
                                    if dt > 0 and dt < 2.0: # avoid huge gaps
                                        lymin, lxmin, lymax, lxmax = equipment_state["last_box"]
                                        lcx = (lxmin + lxmax) / 2.0
                                        lcy = (lymin + lymax) / 2.0
                                        # Distance in normalized space (0-1)
                                        dist = math.sqrt((cx - lcx)**2 + (cy - lcy)**2)
                                        # Momentum arbitrarily scaled for UI display (0-100)
                                        momentum = min(100.0, (dist / dt) * 50.0)
                                        
                                # Determine possession (closest player)
                                held_by = None
                                min_dist = 0.15  # Max normalized distance to consider "held"
                                for spr in snapshot_subjects:
                                    pxx1, pyy1, pxx2, pyy2 = spr.bbox_normalized
                                    pcx = (pxx1 + pxx2) / 2.0
                                    pcy = (pyy1 + pyy2) / 2.0
                                    p_dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                                    if p_dist < min_dist:
                                        min_dist = p_dist
                                        held_by = str(spr.track_id)
                                        
                                equipment_state["last_box"] = box
                                equipment_state["last_time"] = t
                                equipment_state["box"] = box
                                equipment_state["momentum"] = momentum
                                equipment_state["held_by_id"] = held_by
                            else:
                                equipment_state["box"] = None
                                equipment_state["held_by_id"] = None
                                equipment_state["momentum"] = 0.0
                                
                            equipment_state["pending"] = False

                        asyncio.ensure_future(loop.run_in_executor(_executor, _track_equipment))

                # Run re-analysis if needed
                if analyzer.needs_reanalysis():
                    if not getattr(analyzer, "_analysis_pending", False):
                        analyzer._analysis_pending = True
                        task = asyncio.create_task(_run_analysis_bg(analyzer, subject_id))
                        analysis_tasks.add(task)
                        task.add_done_callback(lambda t: analysis_tasks.discard(t))

                # Gemini activity classification for unlabeled clusters
                if gemini is not None and gemini.available:
                    for cid in analyzer.get_clusters_needing_classification():
                        analyzer.mark_classification_pending(cid)
                        asyncio.ensure_future(
                            loop.run_in_executor(
                                _executor,
                                _classify_cluster_webcam,
                                analyzer, cid, gemini,
                            )
                        )

                # Handle UMAP embedding (non-blocking: refit runs in background)
                embedding_update = None

                # Check for cached refit result first (from background task)
                if getattr(analyzer, '_umap_refit_result', None) is not None:
                    embedding_update = analyzer._umap_refit_result
                    analyzer._umap_refit_result = None
                elif analyzer.needs_umap_refit():
                    if not getattr(analyzer, '_umap_refit_pending', False):
                        analyzer._umap_refit_pending = True

                        async def _umap_refit_bg(a=analyzer):
                            try:
                                result = await loop.run_in_executor(None, a.run_umap_fit)
                                a._umap_refit_result = result
                            finally:
                                a._umap_refit_pending = False

                        asyncio.create_task(_umap_refit_bg())
                elif len(analyzer.features_list) > 0 and analyzer._umap_mapper is not None:
                    feat = analyzer.features_list[-1]
                    embedding_update = analyzer.run_umap_transform(feat)

                srp_joints = analyzer.get_srp_joints()
                joint_vis = analyzer.get_joint_visibility()
                # Round SRP joints to 4 decimal places to reduce JSON size
                if srp_joints is not None:
                    srp_joints = [[round(v, 4) for v in jt] for jt in srp_joints]

                rep_joints = response.get("representative_joints")
                if rep_joints is not None:
                    rep_joints = [[round(v, 4) for v in jt] for jt in rep_joints]

                subject_out: dict[str, Any] = {
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
                    "joint_visibility": joint_vis,
                    "representative_joints": rep_joints,
                    "identity_status": identity_status,
                    "identity_confidence": round(identity_confidence, 3),
                    "velocity": response["velocity"],
                    "rolling_velocity": response["rolling_velocity"],
                    "fatigue_index": response["fatigue_index"],
                    "peak_velocity": response["peak_velocity"],
                }
                if "quality" in response:
                    subject_out["quality"] = response["quality"]

                # Jersey detection results
                if resolver is not None:
                    gallery = resolver._galleries.get(subject_id)
                    if gallery is not None:
                        if gallery.jersey_number is not None:
                            subject_out["jersey_number"] = gallery.jersey_number
                        if gallery.jersey_color is not None:
                            subject_out["jersey_color"] = gallery.jersey_color
                # Jersey debug info (only when newly available, not every frame)
                debug_info = jersey_debug_cache.pop(subject_id, None)
                if debug_info is not None:
                    subject_out["jersey_crop_b64"] = debug_info["crop_b64"]
                    subject_out["jersey_gemini_response"] = debug_info["gemini_response"]

                if embedding_update is not None:
                    subject_out["embedding_update"] = embedding_update
                if "cluster_representatives" in response:
                    subject_out["cluster_representatives"] = response["cluster_representatives"]

                # Include SMPL params and UV texture if available
                if pr.smpl_params is not None:
                    subject_out["smpl_params"] = pr.smpl_params
                # UV texture JPEG encoding is expensive; throttle to reduce frame-loop stalls.
                if pr.smpl_texture_uv is not None and frame_idx % 10 == 0:
                    import base64 as b64
                    _, buf = cv2.imencode(".jpg", pr.smpl_texture_uv,
                                         [cv2.IMWRITE_JPEG_QUALITY, 60])
                    subject_out["uv_texture"] = b64.b64encode(buf.tobytes()).decode()

                subjects_data[str(subject_id)] = subject_out

            # Cleanup stale subjects and resolver tracks
            manager.cleanup_stale(frame_idx)
            if resolver is not None:
                active_track_ids = {pr.track_id for pr in results}
                resolver.cleanup_stale_tracks(active_track_ids)

            t_analyze = time.monotonic()

            # Send combined multi-subject response with timing
            active_ids = manager.get_active_track_ids()
            timing_dict = {
                "decode_ms": round(t_decode * 1000, 1),
                "pipeline_ms": round((t_infer - t0) * 1000, 1),
                "identity_ms": round((t_resolve - t_infer) * 1000, 1),
                "analyzer_ms": round((t_analyze - t_resolve) * 1000, 1),
                "total_ms": round((t_analyze - t0 + t_decode) * 1000, 1),
            }

            equipment_data = {
                "box": equipment_state["box"],
                "momentum": round(equipment_state["momentum"], 2),
                "held_by_id": equipment_state["held_by_id"],
            }

            if client_type == "vr":
                await websocket.send_json(_build_vr_response(
                    subjects_data, active_ids, frame_idx, video_time,
                    vr_selected_subject, timing_dict,
                ))
            else:
                await websocket.send_json({
                    "frame_index": frame_idx,
                    "video_time": video_time,
                    "subjects": subjects_data,
                    "active_track_ids": active_ids,
                    "equipment": equipment_data,
                    "timing": timing_dict,
                    "gemini_stats": gemini.get_stats() if gemini is not None else None,
                })

            # Update active stream registry (every 30 frames to avoid overhead)
            if stream_id and stream_id in _active_streams:
                info = _active_streams[stream_id]
                info["frame_count"] = frame_idx
                info["subject_count"] = len(active_ids)
                if info["resolution"] is None:
                    info["resolution"] = [w, h]
                # Store latest frame for high-res snapshot endpoint
                info["_last_rgb"] = rgb
                # Store latest response data for the debug data endpoint
                info["_last_subjects"] = subjects_data
                info["_last_response"] = {
                    "frame_index": frame_idx,
                    "video_time": round(video_time, 3),
                    "active_track_ids": active_ids,
                    "equipment": equipment_data,
                    "timing": timing_dict,
                    "subject_count": len(active_ids),
                }
                # Generate thumbnail every 30 frames (~1s)
                if frame_idx % 30 == 0:
                    try:
                        thumb = cv2.resize(rgb, (160, 120))
                        _, buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
                                              [cv2.IMWRITE_JPEG_QUALITY, 50])
                        info["last_thumbnail"] = base64.b64encode(buf.tobytes()).decode()
                    except Exception:
                        pass

            # Log timing every 30 frames
            if frame_idx % 30 == 0 and frame_idx > 0:
                print(
                    f"[perf] frame={frame_idx} "
                    f"decode={t_decode*1000:.0f}ms "
                    f"pipeline={(t_infer-t0)*1000:.0f}ms "
                    f"resolve={(t_resolve-t_infer)*1000:.0f}ms "
                    f"analyze={(t_analyze-t_resolve)*1000:.0f}ms "
                    f"total={(t_analyze-t0+t_decode)*1000:.0f}ms "
                    f"subjects={len(active_ids)}",
                    flush=True,
                )

            frame_idx += 1

    except WebSocketDisconnect:
        pass
    finally:
        for task in list(analysis_tasks):
            task.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
