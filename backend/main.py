"""FastAPI WebSocket server for real-time multi-person BRACE analysis.

Receives JPEG frames (webcam) or processes uploaded videos,
runs GPU inference for detection + pose estimation + motion
consistency analysis, and streams multi-subject results back to the frontend.
"""

import asyncio
import base64
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

import struct

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
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
except Exception:
    _nvml_available = False

SCRIPT_DIR = Path(__file__).parent
_default_upload = "/app/uploads" if Path("/app").exists() else str(SCRIPT_DIR / "uploads")
_default_demo = "/app/data/sports_videos" if Path("/app/data").exists() else str(SCRIPT_DIR.parent / "data" / "sports_videos")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", _default_upload))
DEMO_VIDEOS_DIR = Path(os.environ.get("DEMO_VIDEOS_DIR", _default_demo))

app = FastAPI(title="BRACE API")

if auth_router is not None:
    app.include_router(auth_router)

if chat_router is not None:
    app.include_router(chat_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=8)

# Only send joints used by frontend skeleton renderer (14 feature + hips/shoulders = 14 unique)
_SEND_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]

pipeline: PoseBackend | None = None
_reid_extractor: EmbeddingExtractor | None = None
_clip_reid_extractor: EmbeddingExtractor | None = None

# Store uploaded video paths by session ID
_uploads: dict[str, Path] = {}

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

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[startup] Upload dir: {UPLOAD_DIR}", flush=True)

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
    user_id_param = websocket.query_params.get("user_id", "")

    await websocket.accept()

    if pipeline is None:
        await websocket.send_json({"type": "error", "message": "Models not loaded"})
        await websocket.close()
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

    if mode == "video":
        await _handle_video_mode(websocket, session_id, cluster_threshold)
    else:
        await _handle_webcam_mode(websocket, cluster_threshold, risk_modifiers=risk_modifiers)


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


async def _handle_webcam_mode(
    websocket: WebSocket, cluster_threshold: float, risk_modifiers: Any = None,
) -> None:
    """Process live webcam frames with multi-person tracking + pose estimation."""
    manager = SubjectManager(fps=30.0, cluster_threshold=cluster_threshold, risk_modifiers=risk_modifiers)
    loop = asyncio.get_event_loop()
    frame_idx = 0

    # Create IdentityResolver for this session (skip if using DummyExtractor)
    use_reid = (
        not DISABLE_REID
        and _reid_extractor is not None
        and not isinstance(_reid_extractor, DummyExtractor)
    )
    resolver = IdentityResolver(_reid_extractor) if use_reid else None

    # Gemini activity classification
    gemini: GeminiActivityClassifier | None = None
    if _GEMINI_AVAILABLE:
        gemini = GeminiActivityClassifier()

    # Ring buffer of recent RGB frames for Gemini classification
    frame_buffer: dict[int, np.ndarray] = {}
    _FRAME_BUFFER_MAX = 1800  # 60s at 30fps — retain frames for Gemini classification

    # Track video_time to detect loop boundaries (video_time jumps backward)
    prev_video_time: float = 0.0
    loop_cooldown: int = 0  # frames to skip loop detection after a loop

    def _get_buffered_frame(idx: int) -> np.ndarray | None:
        return frame_buffer.get(idx)

    def _classify_cluster_webcam(analyzer, cluster_id: int, gem: GeminiActivityClassifier):
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

                # Run re-analysis if needed
                if analyzer.needs_reanalysis():
                    await loop.run_in_executor(None, analyzer.run_analysis)

                    # Trigger Gemini classification for stable clusters (background)
                    if gemini is not None and gemini.available:
                        for cid in analyzer.get_clusters_needing_classification(min_segments=1):
                            analyzer.mark_classification_pending(cid)
                            asyncio.ensure_future(
                                loop.run_in_executor(
                                    _executor, _classify_cluster_webcam,
                                    analyzer, cid, gemini,
                                )
                            )

                # Handle UMAP embedding (non-blocking: refit runs in background)
                embedding_update = None
                if analyzer.needs_umap_refit():
                    if not getattr(analyzer, '_umap_refit_pending', False):
                        analyzer._umap_refit_pending = True

                        async def _umap_refit_bg(a=analyzer):
                            try:
                                result = await loop.run_in_executor(None, a.run_umap_fit)
                                a._umap_refit_result = result
                            finally:
                                a._umap_refit_pending = False

                        asyncio.create_task(_umap_refit_bg())
                    # Serve cached refit result if available
                    if hasattr(analyzer, '_umap_refit_result') and analyzer._umap_refit_result is not None:
                        embedding_update = analyzer._umap_refit_result
                        analyzer._umap_refit_result = None
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
                if "alert_text" in response:
                    subject_out["alert_text"] = response["alert_text"]
                if embedding_update is not None:
                    subject_out["embedding_update"] = embedding_update
                if "cluster_representatives" in response:
                    subject_out["cluster_representatives"] = response["cluster_representatives"]

                # Include SMPL params and UV texture if available
                if pr.smpl_params is not None:
                    subject_out["smpl_params"] = pr.smpl_params
                if pr.smpl_texture_uv is not None:
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
            await websocket.send_json({
                "frame_index": frame_idx,
                "video_time": video_time,
                "subjects": subjects_data,
                "active_track_ids": active_ids,
                "timing": {
                    "decode_ms": round(t_decode * 1000, 1),
                    "pipeline_ms": round((t_infer - t0) * 1000, 1),
                    "identity_ms": round((t_resolve - t_infer) * 1000, 1),
                    "analyzer_ms": round((t_analyze - t_resolve) * 1000, 1),
                    "total_ms": round((t_analyze - t0 + t_decode) * 1000, 1),
                },
                "gemini_stats": gemini.get_stats() if gemini is not None else None,
            })

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
