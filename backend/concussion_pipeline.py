"""Concussion-focused sideline analysis pipeline.

Implements:
1. POST /upload-clip
2. WebSocket /live-stream

The clip pipeline uses YOLOv8-Pose in 60-frame batches, computes impact
kinematics, and returns a structured risk report.
"""

from __future__ import annotations

import asyncio
import email.message
import json
import math
import os
import smtplib
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).resolve().parent
_default_upload_dir = (
    "/app/uploads/concussion"
    if Path("/app").exists()
    else str(SCRIPT_DIR.parent / "data" / "concussion_clips")
)
CLIP_UPLOAD_DIR = Path(os.environ.get("CLIP_UPLOAD_DIR", _default_upload_dir))
CLIP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

POSE_BATCH_SIZE = int(os.environ.get("POSE_BATCH_SIZE", "60"))
MAX_PROCESS_SECONDS = float(os.environ.get("CLIP_PROCESS_TIMEOUT_SECONDS", "20"))
ASSUMED_HEAD_NECK_METERS = float(os.environ.get("ASSUMED_HEAD_NECK_METERS", "0.24"))
YOLO_MODEL_NAME = os.environ.get("CONCUSSION_POSE_MODEL", "yolo11x-pose.pt")
YOLO_CONF = float(os.environ.get("CONCUSSION_POSE_CONF", "0.25"))
YOLO_IMGSZ = int(os.environ.get("CONCUSSION_POSE_IMGSZ", "640"))

HIGH_LINEAR_THRESHOLD_MS = float(os.environ.get("HIGH_LINEAR_THRESHOLD_MS", "7.0"))
HIGH_ROT_THRESHOLD_DEGS = float(os.environ.get("HIGH_ROT_THRESHOLD_DEGS", "3000.0"))
IMPACT_LINEAR_MIN_MS = float(os.environ.get("IMPACT_LINEAR_MIN_MS", "3.0"))
IMPACT_ROT_MIN_DEGS = float(os.environ.get("IMPACT_ROT_MIN_DEGS", "1200.0"))

LIVE_DEFAULT_THRESHOLD_MS = float(os.environ.get("LIVE_COLLISION_THRESHOLD_MS", "7.0"))
LIVE_DEFAULT_END_RATIO = float(os.environ.get("LIVE_COLLISION_END_RATIO", "0.6"))
LIVE_DEFAULT_END_HOLD_MS = float(os.environ.get("LIVE_COLLISION_END_HOLD_MS", "120"))

_default_outbox_dir = (
    "/app/data/coach_reports"
    if Path("/app").exists()
    else str(SCRIPT_DIR.parent / "data" / "coach_reports")
)
COACH_OUTBOX_DIR = Path(os.environ.get("COACH_OUTBOX_DIR", _default_outbox_dir))
COACH_WEBHOOK_URL = os.environ.get("COACH_REPORT_WEBHOOK_URL", "").strip()
COACH_NOTIFY_RETRY_SECONDS = float(os.environ.get("COACH_NOTIFY_RETRY_SECONDS", "15"))

SMTP_HOST = os.environ.get("COACH_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("COACH_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("COACH_SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("COACH_SMTP_PASSWORD", "").strip()
SMTP_FROM = os.environ.get("COACH_SMTP_FROM", "brace@localhost").strip()
SMTP_USE_TLS = os.environ.get("COACH_SMTP_USE_TLS", "1") == "1"
DEFAULT_COACH_EMAIL = os.environ.get("DEFAULT_COACH_EMAIL", "").strip()


class ConcussionReport(BaseModel):
    play_id: str
    player_id: str
    impact_detected: bool
    risk_level: str = Field(..., description="HIGH | MODERATE | LOW")
    linear_velocity_ms: float
    rotational_velocity_degs: float
    impact_duration_ms: float
    impact_location: str = Field(..., description="FRONT | SIDE | REAR | UNKNOWN")
    frame_of_impact: int
    recommendation: str = Field(..., description="EVALUATE NOW | MONITOR | CLEARED")


@dataclass
class PoseSample:
    head: tuple[float, float] | None = None
    neck: tuple[float, float] | None = None
    nose_conf: float = 0.0
    left_eye_conf: float = 0.0
    right_eye_conf: float = 0.0
    left_ear_conf: float = 0.0
    right_ear_conf: float = 0.0


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # pydantic v2
    return model.dict()  # pydantic v1


def _interpolate_series(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs and clamp edges to nearest valid value."""
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    if valid.sum() == 1:
        arr[~valid] = arr[valid][0]
        return arr
    idx = np.arange(arr.shape[0], dtype=np.float32)
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def _contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_true in enumerate(mask):
        if is_true and start is None:
            start = i
        elif not is_true and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions


def _extract_head_point(payload: dict[str, Any]) -> tuple[float, float] | None:
    """Extract a head point from several mobile payload schemas."""
    for key in ("head", "head_keypoint"):
        value = payload.get(key)
        if isinstance(value, dict):
            x = _safe_float(value.get("x"))
            y = _safe_float(value.get("y"))
            if x is not None and y is not None:
                return x, y
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x = _safe_float(value[0])
            y = _safe_float(value[1])
            if x is not None and y is not None:
                return x, y

    landmarks = payload.get("landmarks")
    if isinstance(landmarks, dict):
        for key in ("head", "nose"):
            value = landmarks.get(key)
            if isinstance(value, dict):
                x = _safe_float(value.get("x"))
                y = _safe_float(value.get("y"))
                if x is not None and y is not None:
                    return x, y
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                x = _safe_float(value[0])
                y = _safe_float(value[1])
                if x is not None and y is not None:
                    return x, y
    return None


def _estimate_impact_location(sample: PoseSample | None) -> str:
    """Approximate orientation class from facial keypoint visibility."""
    if sample is None:
        return "UNKNOWN"

    nose = sample.nose_conf
    left_profile = max(sample.left_eye_conf, sample.left_ear_conf)
    right_profile = max(sample.right_eye_conf, sample.right_ear_conf)
    asymmetry = abs(left_profile - right_profile)

    if nose >= 0.35 and min(left_profile, right_profile) >= 0.15:
        return "FRONT"
    if nose <= 0.12 and (left_profile + right_profile) <= 0.45:
        return "REAR"
    if asymmetry >= 0.25:
        return "SIDE"
    return "UNKNOWN"


class LiveCollisionDetector:
    """Detect collision start/end from 60 FPS head motion stream."""

    def __init__(
        self,
        threshold_ms: float = LIVE_DEFAULT_THRESHOLD_MS,
        end_ratio: float = LIVE_DEFAULT_END_RATIO,
        end_hold_ms: float = LIVE_DEFAULT_END_HOLD_MS,
    ) -> None:
        self.threshold_ms = max(threshold_ms, 0.1)
        self.end_ratio = max(min(end_ratio, 1.0), 0.05)
        self.end_hold_ms = max(end_hold_ms, 0.0)

        self.prev_head: tuple[float, float] | None = None
        self.prev_ts_ms: float | None = None
        self.active = False
        self.current_event_id: str | None = None
        self.event_start_ms: float | None = None
        self.end_candidate_ms: float | None = None

    def _compute_velocity(self, payload: dict[str, Any], ts_ms: float) -> float | None:
        direct_velocity = _safe_float(payload.get("head_velocity_ms"))
        if direct_velocity is not None:
            return max(0.0, direct_velocity)

        head = _extract_head_point(payload)
        if head is None:
            return None

        if self.prev_head is None or self.prev_ts_ms is None:
            self.prev_head = head
            self.prev_ts_ms = ts_ms
            return 0.0

        dt_s = max((ts_ms - self.prev_ts_ms) / 1000.0, 1e-3)
        dx = head[0] - self.prev_head[0]
        dy = head[1] - self.prev_head[1]
        dist_units = math.hypot(dx, dy)

        # Prefer explicit metric scale from client.
        scale_m_per_unit = _safe_float(payload.get("scale_m_per_pixel"))
        if scale_m_per_unit is None:
            max_abs = max(abs(head[0]), abs(head[1]), abs(self.prev_head[0]), abs(self.prev_head[1]))
            if max_abs <= 1.5:
                # Normalized coords. Convert using normalized shoulder width when available.
                shoulder_norm = _safe_float(payload.get("shoulder_width_norm"))
                shoulder_norm = shoulder_norm if shoulder_norm and shoulder_norm > 1e-4 else 0.18
                scale_m_per_unit = 0.40 / shoulder_norm
            else:
                shoulder_px = _safe_float(payload.get("shoulder_width_px"))
                shoulder_px = shoulder_px if shoulder_px and shoulder_px > 1.0 else 80.0
                scale_m_per_unit = 0.40 / shoulder_px

        velocity_ms = dist_units * scale_m_per_unit / dt_s
        self.prev_head = head
        self.prev_ts_ms = ts_ms
        return max(0.0, velocity_ms)

    def update(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        ts_ms = _safe_float(payload.get("timestamp_ms"))
        if ts_ms is None:
            ts_ms = time.time() * 1000.0

        velocity_ms = self._compute_velocity(payload, ts_ms)
        if velocity_ms is None:
            return None

        base = {
            "play_id": str(payload.get("play_id", "")),
            "player_id": str(payload.get("player_id", "")),
            "timestamp_ms": round(ts_ms, 3),
            "velocity_ms": round(float(velocity_ms), 4),
            "threshold_ms": self.threshold_ms,
        }

        if not self.active and velocity_ms >= self.threshold_ms:
            self.active = True
            self.current_event_id = str(uuid.uuid4())
            self.event_start_ms = ts_ms
            self.end_candidate_ms = None
            return {
                **base,
                "type": "collision_start",
                "event_id": self.current_event_id,
                "collision_active": True,
                "record_240fps": True,
                "stop_after_ms": 0,
            }

        if not self.active:
            return None

        if velocity_ms < self.threshold_ms * self.end_ratio:
            if self.end_candidate_ms is None:
                self.end_candidate_ms = ts_ms
            elif ts_ms - self.end_candidate_ms >= self.end_hold_ms:
                self.active = False
                duration = 0.0
                if self.event_start_ms is not None:
                    duration = ts_ms - self.event_start_ms
                finished_event_id = self.current_event_id
                self.current_event_id = None
                self.event_start_ms = None
                self.end_candidate_ms = None
                return {
                    **base,
                    "type": "collision_end",
                    "event_id": finished_event_id,
                    "collision_active": False,
                    "record_240fps": True,
                    "stop_after_ms": 1000,
                    "event_duration_ms": round(duration, 3),
                }
        else:
            self.end_candidate_ms = None

        return None


class ConcussionAnalyzer:
    """Batch pose inference + concussion kinematics for 240 FPS clips."""

    def __init__(self, batch_size: int = POSE_BATCH_SIZE) -> None:
        self.batch_size = max(1, batch_size)
        self._model: Any = None
        self._device = self._resolve_device()

    def _resolve_device(self) -> str:
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for /upload-clip processing") from exc
        self._model = YOLO(YOLO_MODEL_NAME)
        return self._model

    @staticmethod
    def _select_subject_index(result: Any) -> int | None:
        if getattr(result, "keypoints", None) is None:
            return None
        n = len(result.keypoints)
        if n <= 0:
            return None
        if getattr(result, "boxes", None) is None or result.boxes.xyxy is None:
            return 0

        boxes = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(n)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        scores = areas * conf
        return int(np.argmax(scores))

    @staticmethod
    def _compute_head_point(kpts: np.ndarray, conf: np.ndarray) -> tuple[float, float] | None:
        # COCO head points: nose, left_eye, right_eye, left_ear, right_ear
        idx = [0, 1, 2, 3, 4]
        valid = [i for i in idx if i < len(conf) and conf[i] >= 0.15]
        if not valid:
            return None
        return float(np.mean(kpts[valid, 0])), float(np.mean(kpts[valid, 1]))

    @staticmethod
    def _compute_neck_point(kpts: np.ndarray, conf: np.ndarray) -> tuple[float, float] | None:
        # Approximate neck from shoulders in COCO format.
        l_shoulder, r_shoulder = 5, 6
        valid = [i for i in (l_shoulder, r_shoulder) if i < len(conf) and conf[i] >= 0.15]
        if not valid:
            return None
        return float(np.mean(kpts[valid, 0])), float(np.mean(kpts[valid, 1]))

    def _extract_pose_sample(self, result: Any) -> PoseSample:
        idx = self._select_subject_index(result)
        if idx is None or getattr(result, "keypoints", None) is None:
            return PoseSample()

        key_xy = result.keypoints.xy[idx].cpu().numpy()  # (17,2)
        if result.keypoints.conf is not None:
            key_conf = result.keypoints.conf[idx].cpu().numpy()
        else:
            key_conf = np.ones((key_xy.shape[0],), dtype=np.float32)

        head = self._compute_head_point(key_xy, key_conf)
        neck = self._compute_neck_point(key_xy, key_conf)

        return PoseSample(
            head=head,
            neck=neck,
            nose_conf=float(key_conf[0]) if key_conf.shape[0] > 0 else 0.0,
            left_eye_conf=float(key_conf[1]) if key_conf.shape[0] > 1 else 0.0,
            right_eye_conf=float(key_conf[2]) if key_conf.shape[0] > 2 else 0.0,
            left_ear_conf=float(key_conf[3]) if key_conf.shape[0] > 3 else 0.0,
            right_ear_conf=float(key_conf[4]) if key_conf.shape[0] > 4 else 0.0,
        )

    def _infer_pose_batch(self, frames_bgr: list[np.ndarray]) -> list[PoseSample]:
        model = self._load_model()
        results = model.predict(
            source=frames_bgr,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            device=self._device,
            verbose=False,
        )
        return [self._extract_pose_sample(result) for result in results]

    def _read_pose_samples(self, clip_path: Path) -> tuple[list[PoseSample], float]:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open clip: {clip_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 1.0 else 240.0

        batch: list[np.ndarray] = []
        samples: list[PoseSample] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(frame)
                if len(batch) >= self.batch_size:
                    samples.extend(self._infer_pose_batch(batch))
                    batch.clear()
            if batch:
                samples.extend(self._infer_pose_batch(batch))
        finally:
            cap.release()

        return samples, fps

    def analyze_pose_samples(
        self,
        samples: list[PoseSample],
        fps: float,
        play_id: str,
        player_id: str,
        scale_m_per_pixel: float | None = None,
    ) -> dict[str, Any]:
        if not samples:
            return _model_dump(
                ConcussionReport(
                    play_id=play_id,
                    player_id=player_id,
                    impact_detected=False,
                    risk_level="LOW",
                    linear_velocity_ms=0.0,
                    rotational_velocity_degs=0.0,
                    impact_duration_ms=0.0,
                    impact_location="UNKNOWN",
                    frame_of_impact=-1,
                    recommendation="CLEARED",
                )
            )

        num_frames = len(samples)
        head = np.full((num_frames, 2), np.nan, dtype=np.float32)
        neck = np.full((num_frames, 2), np.nan, dtype=np.float32)
        for i, sample in enumerate(samples):
            if sample.head is not None:
                head[i, 0] = sample.head[0]
                head[i, 1] = sample.head[1]
            if sample.neck is not None:
                neck[i, 0] = sample.neck[0]
                neck[i, 1] = sample.neck[1]

        head[:, 0] = _interpolate_series(head[:, 0])
        head[:, 1] = _interpolate_series(head[:, 1])
        neck[:, 0] = _interpolate_series(neck[:, 0])
        neck[:, 1] = _interpolate_series(neck[:, 1])

        if scale_m_per_pixel is not None and scale_m_per_pixel > 0:
            meters_per_pixel = float(scale_m_per_pixel)
        else:
            head_neck_px = np.linalg.norm(head - neck, axis=1)
            valid = head_neck_px > 1.0
            if np.any(valid):
                median_px = float(np.median(head_neck_px[valid]))
                meters_per_pixel = ASSUMED_HEAD_NECK_METERS / max(median_px, 1e-3)
            else:
                meters_per_pixel = 0.004

        displacements = np.linalg.norm(np.diff(head, axis=0), axis=1)
        linear_velocity = np.zeros((num_frames,), dtype=np.float32)
        linear_velocity[1:] = displacements * meters_per_pixel * fps

        head_neck_vec = head - neck
        angles = np.unwrap(np.arctan2(head_neck_vec[:, 1], head_neck_vec[:, 0]))
        rot_velocity = np.zeros((num_frames,), dtype=np.float32)
        rot_velocity[1:] = np.abs(np.diff(angles)) * (180.0 / math.pi) * fps

        if num_frames >= 5:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            linear_velocity = np.convolve(linear_velocity, kernel, mode="same")
            rot_velocity = np.convolve(rot_velocity, kernel, mode="same")

        impact_mask = (linear_velocity >= IMPACT_LINEAR_MIN_MS) | (rot_velocity >= IMPACT_ROT_MIN_DEGS)
        severity = (linear_velocity / max(HIGH_LINEAR_THRESHOLD_MS, 1e-6)) + (
            rot_velocity / max(HIGH_ROT_THRESHOLD_DEGS, 1e-6)
        )

        impact_detected = bool(np.any(impact_mask))
        frame_of_impact = int(np.argmax(severity))
        segment_start = frame_of_impact
        segment_end = frame_of_impact
        impact_duration_ms = 0.0

        if impact_detected:
            regions = _contiguous_true_regions(impact_mask)
            if regions:
                segment_start, segment_end = max(
                    regions,
                    key=lambda region: float(np.max(severity[region[0] : region[1] + 1])),
                )
                frame_of_impact = int(segment_start + np.argmax(severity[segment_start : segment_end + 1]))
                impact_duration_ms = ((segment_end - segment_start + 1) / fps) * 1000.0

        metric_slice = slice(segment_start, segment_end + 1) if impact_detected else slice(0, num_frames)
        peak_linear = float(np.max(linear_velocity[metric_slice])) if num_frames > 0 else 0.0
        peak_rot = float(np.max(rot_velocity[metric_slice])) if num_frames > 0 else 0.0

        if peak_linear > HIGH_LINEAR_THRESHOLD_MS or peak_rot > HIGH_ROT_THRESHOLD_DEGS:
            risk_level = "HIGH"
            recommendation = "EVALUATE NOW"
        elif impact_detected and (peak_linear > 4.0 or peak_rot > 1500.0):
            risk_level = "MODERATE"
            recommendation = "MONITOR"
        else:
            risk_level = "LOW"
            recommendation = "CLEARED"

        impact_sample = samples[frame_of_impact] if 0 <= frame_of_impact < len(samples) else None
        impact_location = _estimate_impact_location(impact_sample) if impact_detected else "UNKNOWN"

        report = ConcussionReport(
            play_id=play_id,
            player_id=player_id,
            impact_detected=impact_detected,
            risk_level=risk_level,
            linear_velocity_ms=round(peak_linear, 4),
            rotational_velocity_degs=round(peak_rot, 4),
            impact_duration_ms=round(float(impact_duration_ms), 4),
            impact_location=impact_location,
            frame_of_impact=frame_of_impact if impact_detected else -1,
            recommendation=recommendation,
        )
        return _model_dump(report)

    def analyze_clip(
        self,
        clip_path: Path,
        play_id: str,
        player_id: str,
        scale_m_per_pixel: float | None = None,
    ) -> dict[str, Any]:
        samples, fps = self._read_pose_samples(clip_path)
        return self.analyze_pose_samples(
            samples=samples,
            fps=fps,
            play_id=play_id,
            player_id=player_id,
            scale_m_per_pixel=scale_m_per_pixel,
        )


class CoachReportNotifier:
    """Durable coach report dispatcher with webhook/SMTP delivery and retries."""

    def __init__(
        self,
        outbox_dir: Path | None = None,
        webhook_url: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        smtp_from: str | None = None,
        smtp_use_tls: bool | None = None,
        default_coach_email: str | None = None,
        retry_seconds: float | None = None,
    ) -> None:
        base_dir = outbox_dir or COACH_OUTBOX_DIR
        self.pending_dir = base_dir / "pending"
        self.sent_dir = base_dir / "sent"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.sent_dir.mkdir(parents=True, exist_ok=True)

        self.webhook_url = webhook_url if webhook_url is not None else COACH_WEBHOOK_URL
        self.smtp_host = smtp_host if smtp_host is not None else SMTP_HOST
        self.smtp_port = smtp_port if smtp_port is not None else SMTP_PORT
        self.smtp_user = smtp_user if smtp_user is not None else SMTP_USER
        self.smtp_password = smtp_password if smtp_password is not None else SMTP_PASSWORD
        self.smtp_from = smtp_from if smtp_from is not None else SMTP_FROM
        self.smtp_use_tls = smtp_use_tls if smtp_use_tls is not None else SMTP_USE_TLS
        self.default_coach_email = (
            default_coach_email if default_coach_email is not None else DEFAULT_COACH_EMAIL
        )
        self.retry_seconds = retry_seconds if retry_seconds is not None else COACH_NOTIFY_RETRY_SECONDS

        self._flush_lock = asyncio.Lock()
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._retry_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def queue_report(
        self,
        report: dict[str, Any],
        coach_email: str | None = None,
        coach_id: str | None = None,
    ) -> None:
        payload = {
            "id": str(uuid.uuid4()),
            "created_at_ms": round(time.time() * 1000.0, 3),
            "attempts": 0,
            "next_retry_ts": time.time(),
            "coach_email": coach_email or self.default_coach_email or None,
            "coach_id": coach_id or "",
            "play_id": report.get("play_id", ""),
            "player_id": report.get("player_id", ""),
            "report": report,
        }
        path = self.pending_dir / f"{payload['id']}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        await self.flush_once()

    async def flush_once(self) -> None:
        async with self._flush_lock:
            now = time.time()
            for path in sorted(self.pending_dir.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    # Corrupt file: quarantine in sent dir to unblock queue progression.
                    path.rename(self.sent_dir / path.name)
                    continue

                next_retry_ts = float(payload.get("next_retry_ts", 0.0))
                if next_retry_ts > now:
                    continue

                delivered = await asyncio.to_thread(self._deliver_payload, payload)
                if delivered:
                    path.rename(self.sent_dir / path.name)
                    continue

                attempts = int(payload.get("attempts", 0)) + 1
                backoff_s = min((2**attempts) * 5.0, 300.0)
                payload["attempts"] = attempts
                payload["next_retry_ts"] = time.time() + backoff_s
                path.write_text(json.dumps(payload), encoding="utf-8")

    async def _retry_loop(self) -> None:
        try:
            while True:
                await self.flush_once()
                await asyncio.sleep(self.retry_seconds)
        except asyncio.CancelledError:
            return

    def _deliver_payload(self, payload: dict[str, Any]) -> bool:
        delivered = False
        report = payload.get("report", {})
        coach_email = payload.get("coach_email")

        if self.webhook_url:
            delivered = self._send_webhook(payload) or delivered

        if self.smtp_host and coach_email:
            delivered = self._send_email(report=report, coach_email=str(coach_email)) or delivered

        # If no delivery channel is configured, keep local sent logs only.
        if not self.webhook_url and not self.smtp_host:
            return True
        return delivered

    def _send_webhook(self, payload: dict[str, Any]) -> bool:
        if not self.webhook_url:
            return False
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return 200 <= int(response.status) < 300
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    def _send_email(self, report: dict[str, Any], coach_email: str) -> bool:
        if not self.smtp_host:
            return False

        msg = email.message.EmailMessage()
        msg["Subject"] = f"BRACE Concussion Report: play {report.get('play_id', 'unknown')}"
        msg["From"] = self.smtp_from
        msg["To"] = coach_email
        msg.set_content(
            "Automated BRACE sideline report:\n\n"
            + json.dumps(report, indent=2)
        )

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=8) as server:
                if self.smtp_use_tls:
                    server.starttls()
                if self.smtp_user:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            return True
        except Exception:
            return False


router = APIRouter(tags=["concussion"])
_concussion_analyzer = ConcussionAnalyzer(batch_size=POSE_BATCH_SIZE)
_coach_notifier = CoachReportNotifier()


@router.on_event("startup")
async def concussion_startup() -> None:
    _coach_notifier.start()


@router.on_event("shutdown")
async def concussion_shutdown() -> None:
    await _coach_notifier.stop()


@router.post("/upload-clip", response_model=ConcussionReport)
async def upload_clip(
    play_id: str = Form(...),
    player_id: str = Form(...),
    file: UploadFile = File(...),
    scale_m_per_pixel: float | None = Form(None),
    coach_email: str | None = Form(None),
    coach_id: str | None = Form(None),
) -> dict[str, Any]:
    """Accept uploaded 240 FPS clip and return concussion risk report."""
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    clip_path = CLIP_UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    try:
        with clip_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()

    if not clip_path.exists() or clip_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded clip is empty")

    try:
        report = await asyncio.wait_for(
            asyncio.to_thread(
                _concussion_analyzer.analyze_clip,
                clip_path,
                play_id,
                player_id,
                scale_m_per_pixel,
            ),
            timeout=MAX_PROCESS_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Clip processing exceeded {MAX_PROCESS_SECONDS:.0f} seconds",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clip processing failed: {exc}") from exc

    try:
        await _coach_notifier.queue_report(report=report, coach_email=coach_email, coach_id=coach_id)
    except Exception:
        # Notification failures should not block clip risk response.
        pass

    return report


@router.websocket("/live-stream")
async def live_stream(websocket: WebSocket) -> None:
    """Consume 60 FPS landmark stream and emit collision-trigger recording flags."""
    threshold = _safe_float(websocket.query_params.get("velocity_threshold"))
    end_ratio = _safe_float(websocket.query_params.get("end_ratio"))
    end_hold_ms = _safe_float(websocket.query_params.get("end_hold_ms"))

    detector = LiveCollisionDetector(
        threshold_ms=threshold if threshold is not None else LIVE_DEFAULT_THRESHOLD_MS,
        end_ratio=end_ratio if end_ratio is not None else LIVE_DEFAULT_END_RATIO,
        end_hold_ms=end_hold_ms if end_hold_ms is not None else LIVE_DEFAULT_END_HOLD_MS,
    )

    await websocket.accept()
    await websocket.send_json(
        {
            "type": "ready",
            "collision_active": False,
            "record_240fps": False,
            "stop_after_ms": 0,
            "velocity_threshold_ms": detector.threshold_ms,
        }
    )

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            payload: dict[str, Any] | None = None
            text = msg.get("text")
            raw = msg.get("bytes")
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        payload = parsed
                except json.JSONDecodeError:
                    continue
            elif raw:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    if isinstance(parsed, dict):
                        payload = parsed
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

            if payload is None:
                continue

            if payload.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp_ms": round(time.time() * 1000.0, 3)})
                continue

            event = detector.update(payload)
            if event is not None:
                await websocket.send_json(event)
    except WebSocketDisconnect:
        return
