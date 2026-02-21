"""Classify physical activities in video clips using Google Gemini Vision API."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

GEMINI_API_KEY = os.environ.get(
    "GOOGLE_GEMINI_API_KEY",
    "AIzaSyDrfeCSYbecPiJ1aLWUlY9MdGDBA96__W8",
)

# Rate limit: minimum seconds between API calls
_MIN_CALL_INTERVAL = 2.0

_CLASSIFY_PROMPT = (
    "What physical activity is this person performing? "
    "Respond with ONLY a single word or short phrase. "
    "Prefer these specific labels when applicable: "
    "squat, front squat, back squat, goblet squat, "
    "lunge, forward lunge, reverse lunge, walking lunge, "
    "running, jogging, sprinting, "
    "walking, "
    "jump, box jump, jumping, "
    "deadlift, romanian deadlift, "
    "push-up, "
    "plank, side plank. "
    "Other valid labels: "
    "stretching, curl, bench-press, rowing, cycling, swimming, dancing, "
    "boxing, serving, swinging, throwing, catching, climbing, skating, skiing, "
    "dribbling, shooting, dunking, kicking, punching, sit-up."
)


def _crop_frame(frame: np.ndarray, bbox: tuple[float, float, float, float],
                padding: float = 0.15) -> np.ndarray:
    """Crop a frame to the bounding box region with padding.

    Args:
        frame: RGB image (H, W, 3).
        bbox: Normalized (x1, y1, x2, y2) in [0, 1].
        padding: Fractional padding around the bbox.

    Returns:
        Cropped RGB image.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0.0, x1 - bw * padding)
    y1 = max(0.0, y1 - bh * padding)
    x2 = min(1.0, x2 + bw * padding)
    y2 = min(1.0, y2 + bh * padding)

    px1 = int(x1 * w)
    py1 = int(y1 * h)
    px2 = int(x2 * w)
    py2 = int(y2 * h)

    px1 = max(0, min(px1, w - 1))
    py1 = max(0, min(py1, h - 1))
    px2 = max(px1 + 1, min(px2, w))
    py2 = max(py1 + 1, min(py2, h))

    return frame[py1:py2, px1:px2]


def _encode_frame_jpeg(frame: np.ndarray, max_dim: int = 384) -> bytes:
    """Resize and JPEG-encode a frame for the API.

    Args:
        frame: RGB crop.
        max_dim: Maximum dimension (width or height) to reduce payload size.

    Returns:
        JPEG bytes.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _frames_hash(crops: list[np.ndarray]) -> str:
    """Compute a hash from a list of cropped frames for cache keying."""
    hasher = hashlib.md5(usedforsecurity=False)
    for crop in crops:
        small = cv2.resize(crop, (32, 32))
        hasher.update(small.tobytes())
    return hasher.hexdigest()


class GeminiActivityClassifier:
    """Thread-safe classifier that sends cropped person frames to Gemini."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or GEMINI_API_KEY
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        self._model = None
        self._init_error: str | None = None
        self._api_call_count: int = 0
        self._cache_hit_count: int = 0

    def _ensure_model(self):
        """Lazy-init the Gemini client on first use."""
        if self._model is not None or self._init_error is not None:
            return
        try:
            from google import genai
            self._model = genai.Client(api_key=self._api_key)
        except Exception as e:
            self._init_error = str(e)

    @property
    def available(self) -> bool:
        """True if the Gemini SDK loaded and API key is set."""
        self._ensure_model()
        return self._model is not None

    def classify_activity(
        self,
        frames: list[np.ndarray],
        bbox: tuple[float, float, float, float],
    ) -> str:
        """Classify the activity of a person across representative frames.

        Args:
            frames: 3-5 representative RGB frames (full resolution).
            bbox: Normalized bounding box (x1, y1, x2, y2) in [0, 1].

        Returns:
            A single-word/short-phrase activity label, or "unknown" on failure.
        """
        if not frames:
            return "unknown"

        # Crop each frame to the person
        crops = [_crop_frame(f, bbox) for f in frames]

        # Check cache
        key = _frames_hash(crops)
        with self._lock:
            if key in self._cache:
                self._cache_hit_count += 1
                return self._cache[key]

        # Ensure model is available
        self._ensure_model()
        if self._model is None:
            return "unknown"

        # Rate limit
        with self._lock:
            now = time.monotonic()
            wait = _MIN_CALL_INTERVAL - (now - self._last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()

        # Build multimodal content: images + prompt
        try:
            from PIL import Image
            import io

            contents = []
            for crop in crops:
                jpeg_bytes = _encode_frame_jpeg(crop)
                img = Image.open(io.BytesIO(jpeg_bytes))
                contents.append(img)
            contents.append(_CLASSIFY_PROMPT)

            response = self._model.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
            )
            with self._lock:
                self._api_call_count += 1
            label = response.text.strip().lower().rstrip(".")

            # Clean up: take only the first line, first few words
            label = label.split("\n")[0].strip()
            words = label.split()
            if len(words) > 3:
                label = " ".join(words[:3])
            if not label:
                label = "unknown"
            print(f"[gemini] classified activity: {label}", flush=True)

        except Exception as e:
            print(f"[gemini] classification failed: {e}", flush=True)
            label = "unknown"

        # Cache result
        with self._lock:
            self._cache[key] = label

        return label

    def get_representative_frames(
        self,
        frame_indices: list[int],
        frame_getter,
        count: int = 4,
    ) -> list[np.ndarray]:
        """Select evenly-spaced representative frames.

        Args:
            frame_indices: All frame indices belonging to this cluster's segments.
            frame_getter: Callable(frame_index) -> np.ndarray (RGB) or None.
            count: Number of frames to select.

        Returns:
            List of RGB frames (may be fewer than count if some are unavailable).
        """
        if not frame_indices:
            return []

        n = len(frame_indices)
        if n <= count:
            selected = frame_indices
        else:
            step = n / count
            selected = [frame_indices[int(i * step)] for i in range(count)]

        frames = []
        for idx in selected:
            f = frame_getter(idx)
            if f is not None:
                frames.append(f)
        return frames

    def classify_cached(self, cluster_key: str) -> str | None:
        """Look up a cached classification by cluster key."""
        with self._lock:
            return self._cache.get(cluster_key)

    def cache_result(self, cluster_key: str, label: str) -> None:
        """Store a classification result under a custom key."""
        with self._lock:
            self._cache[cluster_key] = label

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict:
        """Return usage statistics for debug display."""
        with self._lock:
            return {
                "api_calls": self._api_call_count,
                "cache_hits": self._cache_hit_count,
                "estimated_cost_usd": round(self._api_call_count * 0.00011, 6),
            }
