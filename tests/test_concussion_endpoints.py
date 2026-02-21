from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.concussion_pipeline as cp


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(cp.router)
    return app


def test_upload_clip_endpoint_contract(tmp_path: Path):
    cp.CLIP_UPLOAD_DIR = tmp_path
    calls: list[dict] = []

    def fake_analyze_clip(*args, **kwargs):
        return {
            "play_id": "play-101",
            "player_id": "player-22",
            "impact_detected": True,
            "risk_level": "HIGH",
            "linear_velocity_ms": 8.5,
            "rotational_velocity_degs": 3200.0,
            "impact_duration_ms": 18.0,
            "impact_location": "FRONT",
            "frame_of_impact": 42,
            "recommendation": "EVALUATE NOW",
        }

    original = cp._concussion_analyzer.analyze_clip
    original_notify = cp._coach_notifier.queue_report

    async def fake_queue_report(report, coach_email=None, coach_id=None):
        calls.append(
            {
                "report": report,
                "coach_email": coach_email,
                "coach_id": coach_id,
            }
        )

    cp._concussion_analyzer.analyze_clip = fake_analyze_clip
    cp._coach_notifier.queue_report = fake_queue_report
    try:
        client = TestClient(_app())
        resp = client.post(
            "/upload-clip",
            data={
                "play_id": "play-101",
                "player_id": "player-22",
                "coach_email": "coach@example.com",
                "coach_id": "coach-77",
            },
            files={"file": ("clip.mp4", b"\x00\x00\x00\x00", "video/mp4")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["play_id"] == "play-101"
        assert body["player_id"] == "player-22"
        assert body["impact_detected"] is True
        assert body["risk_level"] == "HIGH"
        assert body["recommendation"] == "EVALUATE NOW"
        assert len(calls) == 1
        assert calls[0]["coach_email"] == "coach@example.com"
        assert calls[0]["coach_id"] == "coach-77"
    finally:
        cp._concussion_analyzer.analyze_clip = original
        cp._coach_notifier.queue_report = original_notify


def test_live_stream_endpoint_emits_collision_start_and_end():
    client = TestClient(_app())
    with client.websocket_connect("/live-stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        ws.send_json({"timestamp_ms": 0.0, "head_velocity_ms": 0.0})
        ws.send_json(
            {
                "timestamp_ms": 16.6,
                "head_velocity_ms": 8.0,
                "play_id": "play-9",
                "player_id": "player-3",
            }
        )
        start = ws.receive_json()
        assert start["type"] == "collision_start"
        assert start["record_240fps"] is True

        ws.send_json({"timestamp_ms": 80.0, "head_velocity_ms": 2.0})
        ws.send_json({"timestamp_ms": 220.0, "head_velocity_ms": 1.5})
        end = ws.receive_json()
        assert end["type"] == "collision_end"
        assert end["stop_after_ms"] == 1000
