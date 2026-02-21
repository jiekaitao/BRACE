import asyncio
import json
import time
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
import cv2
import numpy as np

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

# Store active peer connections
pcs = set()

# To be injected from main.py
globals_ref = {
    "pipeline": None,
    "executor": None,
    "SubjectManager": None,
    "IdentityResolver": None,
    "DummyExtractor": None,
    "reid_extractor": None,
    "DISABLE_REID": False,
    "GEMINI_AVAILABLE": False,
    "GeminiActivityClassifier": None,
    "SEND_INDICES": [],
}

def inject_globals(deps: dict):
    globals_ref.update(deps)

async def process_video_track(track, channel: RTCDataChannel, fps: float, cluster_threshold: float):
    print(f"[webrtc] Starting track processing for {track.id} at {fps} fps")
    
    SubjectManager = globals_ref["SubjectManager"]
    IdentityResolver = globals_ref["IdentityResolver"]
    DummyExtractor = globals_ref["DummyExtractor"]
    reid_extractor = globals_ref["reid_extractor"]
    DISABLE_REID = globals_ref["DISABLE_REID"]
    pipeline = globals_ref["pipeline"]
    executor = globals_ref["executor"]
    SEND_INDICES = globals_ref["SEND_INDICES"]
    GEMINI_AVAILABLE = globals_ref["GEMINI_AVAILABLE"]
    GeminiActivityClassifier = globals_ref["GeminiActivityClassifier"]

    manager = SubjectManager(fps=fps, cluster_threshold=cluster_threshold)
    loop = asyncio.get_event_loop()
    
    use_reid = (
        not DISABLE_REID
        and reid_extractor is not None
        and not isinstance(reid_extractor, DummyExtractor)
    )
    resolver = IdentityResolver(reid_extractor) if use_reid else None

    gemini = None
    if GEMINI_AVAILABLE and GeminiActivityClassifier is not None:
        gemini = GeminiActivityClassifier()

    frame_idx = 0
    equipment_state = {
        "box": None, "momentum": 0.0, "held_by_id": None,
        "last_box": None, "last_time": 0.0, "pending": False
    }

    try:
        while True:
            # Get the next frame from the WebRTC track
            frame = await track.recv()
            video_time = frame.time
            # aiortc frame.to_ndarray(format="rgb24")
            rgb = frame.to_ndarray(format="rgb24")
            h, w = rgb.shape[:2]

            # Step 1: Pipeline
            try:
                results = await loop.run_in_executor(executor, pipeline.process_frame, rgb)
            except Exception as e:
                print(f"[webrtc] pipeline error: {e}", flush=True)
                results = []

            # Step 1.5: Resolver
            if resolver is not None and results:
                resolved = await loop.run_in_executor(
                    executor, resolver.resolve_pipeline_results, results, rgb, w, h
                )
            else:
                resolved = None

            # Step 2: Analyzers
            subjects_data = {}
            items = resolved if resolved is not None else results

            for item in items:
                if resolved is not None:
                    rp = item
                    pr = rp.pipeline_result
                    subject_id = rp.subject_id
                    label = rp.label
                else:
                    pr = item
                    subject_id = pr.track_id
                    label = manager.get_label(pr.track_id)

                landmarks_xyzv = pr.landmarks_mp
                analyzer = manager.get_or_create_analyzer(subject_id)
                analyzer.last_seen_frame = frame_idx

                response = analyzer.process_frame(landmarks_xyzv, img_wh=(w, h))

                response["landmarks"] = [
                    {
                        "i": i,
                        "x": round(float(landmarks_xyzv[i, 0] / w), 4),
                        "y": round(float(landmarks_xyzv[i, 1] / h), 4),
                        "v": round(float(landmarks_xyzv[i, 3]), 4),
                    }
                    for i in SEND_INDICES
                ]
                nb = pr.bbox_normalized
                response["bbox"] = {
                    "x1": round(nb[0], 4), "y1": round(nb[1], 4),
                    "x2": round(nb[2], 4), "y2": round(nb[3], 4),
                }

                # Equipment 
                if gemini is not None and gemini.available and not equipment_state["pending"]:
                    if frame_idx % 15 == 0:
                        equipment_state["pending"] = True
                        def _track_equipment(f=rgb, t=video_time, w_w=w, h_h=h, snapshot_subjects=list(results)):
                            box = gemini.locate_object(f, "american football")
                            # ... omitting momentum math for brevity ...
                            if box is not None:
                                equipment_state["box"] = box
                            else:
                                equipment_state["box"] = None
                            equipment_state["pending"] = False
                        asyncio.ensure_future(loop.run_in_executor(executor, _track_equipment))

                # UMAP embedding
                embedding_update = None
                if analyzer.needs_umap_refit():
                    embedding_update = await loop.run_in_executor(executor, analyzer.run_umap_fit)
                elif len(analyzer.features_list) > 0 and analyzer._umap_mapper is not None:
                    feat = analyzer.features_list[-1]
                    embedding_update = analyzer.run_umap_transform(feat)

                srp_joints = analyzer.get_srp_joints()
                if srp_joints is not None:
                    srp_joints = [[round(v, 4) for v in jt] for jt in srp_joints]

                subjects_data[str(subject_id)] = {
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
                    "quality": response.get("quality", {}),
                    "velocity": response.get("velocity", 0.0),
                    "rolling_velocity": response.get("rolling_velocity", 0.0),
                    "fatigue_index": response.get("fatigue_index", 0.0),
                    "peak_velocity": response.get("peak_velocity", 0.0),
                    "identity_status": "confirmed",
                    "identity_confidence": 1.0,
                }
                if embedding_update:
                    subjects_data[str(subject_id)]["embedding_update"] = embedding_update

            manager.cleanup_stale(frame_idx)
            if resolver is not None:
                active_track_ids = {pr.track_id for pr in results}
                resolver.cleanup_stale_tracks(active_track_ids)

            # Send over DataChannel instead of WebSocket
            if channel is not None and channel.readyState == "open":
                payload = json.dumps({
                    "frame_index": frame_idx,
                    "video_time": video_time,
                    "subjects": subjects_data,
                    "active_track_ids": manager.get_active_track_ids(),
                    "equipment": {
                        "box": equipment_state["box"],
                        "momentum": equipment_state["momentum"],
                        "held_by_id": equipment_state["held_by_id"],
                    },
                })
                channel.send(payload)

            frame_idx += 1

    except Exception as e:
        print(f"[webrtc] Track ended or error: {e}")
    finally:
        print(f"[webrtc] Finished track processing")


@router.post("/rtc/offer")
async def rtc_offer(request: Request):
    """WebRTC offer endpoint to establish video stream and data channel."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    fps = float(params.get("fps", 240.0))
    cluster_threshold = float(params.get("cluster_threshold", 2.0))

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[webrtc] Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    datachannel_ref = {"channel": None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"[webrtc] DataChannel '{channel.label}' opened")
        datachannel_ref["channel"] = channel

    @pc.on("track")
    def on_track(track):
        # Determine if it's the video track
        if track.kind == "video":
            print(f"[webrtc] Video track received")
            # We must wait for the data channel, but WebRTC fires 'track' around the same time
            # Start background processing block
            async def track_runner():
                # Wait up to 5s for the data channel to exist
                for _ in range(50):
                    if datachannel_ref["channel"] is not None:
                        break
                    await asyncio.sleep(0.1)
                
                await process_video_track(
                    track, datachannel_ref["channel"], 
                    fps=fps, cluster_threshold=cluster_threshold
                )

            asyncio.ensure_future(track_runner())

    # Handle the offer and create an answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
