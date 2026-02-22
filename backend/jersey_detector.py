"""Detect jersey numbers and colors from player crops using Gemini Vision.

Uses gemini-2.5-flash for fast, cost-effective jersey detection.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "")

# Rate limit: minimum seconds between API calls
_MIN_CALL_INTERVAL = 0.5  # flash model is faster / cheaper

_JERSEY_PROMPT = (
    "Look at this cropped image of a sports player. "
    "Identify their jersey number and the primary color of their jersey. "
    "Return ONLY a JSON object with two fields: "
    '"number" (integer or null if not visible) and '
    '"color" (string like "red", "blue", "white", or null if not visible). '
    "Example: {\"number\": 23, \"color\": \"red\"}"
)


@dataclass
class JerseyInfo:
    """Detected jersey information for a player."""
    number: int | None = None
    color: str | None = None


def parse_jersey_response(text: str) -> JerseyInfo | None:
    """Parse Gemini's JSON response into JerseyInfo.

    Handles various response formats including markdown-wrapped JSON.

    Returns:
        JerseyInfo if valid JSON parsed, None if unparseable.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (possibly with language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    number = data.get("number")
    color = data.get("color")

    # Normalize number
    if number is not None:
        try:
            number = int(number)
        except (ValueError, TypeError):
            number = None

    # Normalize color
    if color is not None:
        color = str(color).lower().strip()
        if color in ("null", "none", ""):
            color = None

    return JerseyInfo(number=number, color=color)


def _encode_crop_jpeg(crop: np.ndarray, max_dim: int = 256) -> bytes:
    """Resize and JPEG-encode a crop for the API."""
    h, w = crop.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def cluster_teams(jerseys: dict[int, JerseyInfo]) -> dict[str, list[int]]:
    """Cluster subjects into teams based on jersey color.

    Args:
        jerseys: Mapping of subject_id -> JerseyInfo.

    Returns:
        Mapping of color_group -> list of subject_ids.
    """
    color_groups: dict[str, list[int]] = defaultdict(list)

    for sid, info in jerseys.items():
        if info.color is None:
            continue
        # Normalize color to base color (first word)
        base_color = info.color.strip().split()[-1]  # "dark red" -> "red"
        color_groups[base_color].append(sid)

    return dict(color_groups)


class JerseyDetector:
    """Detect jersey numbers and colors using Gemini 2.5 Flash."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or GEMINI_API_KEY
        self._cache: dict[int, JerseyInfo] = {}  # subject_id -> JerseyInfo
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        self._model = None
        self._init_error: str | None = None
        self._api_call_count: int = 0

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

    def detect_jersey(
        self,
        crop: np.ndarray,
        subject_id: int | None = None,
    ) -> JerseyInfo | None:
        """Detect jersey number and color from a player crop.

        Args:
            crop: RGB image crop of the player.
            subject_id: Optional subject ID for caching.

        Returns:
            JerseyInfo or None on failure.
        """
        # Check cache
        if subject_id is not None:
            with self._lock:
                if subject_id in self._cache:
                    return self._cache[subject_id]

        self._ensure_model()
        if self._model is None:
            return None

        # Rate limit
        with self._lock:
            now = time.monotonic()
            wait = _MIN_CALL_INTERVAL - (now - self._last_call_time)
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()

        try:
            from PIL import Image
            import io

            jpeg_bytes = _encode_crop_jpeg(crop)
            img = Image.open(io.BytesIO(jpeg_bytes))

            response = self._model.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, _JERSEY_PROMPT],
            )
            with self._lock:
                self._api_call_count += 1

            info = parse_jersey_response(response.text)
            if info is not None and subject_id is not None:
                with self._lock:
                    self._cache[subject_id] = info
            return info

        except Exception as e:
            print(f"[jersey] detection failed: {e}", flush=True)
            return None

    def detect_batch(
        self,
        crops: dict[int, np.ndarray],
    ) -> dict[int, JerseyInfo]:
        """Detect jerseys for multiple subjects.

        Args:
            crops: Mapping of subject_id -> RGB crop.

        Returns:
            Mapping of subject_id -> JerseyInfo (only successful detections).
        """
        results: dict[int, JerseyInfo] = {}
        for sid, crop in crops.items():
            info = self.detect_jersey(crop, subject_id=sid)
            if info is not None:
                results[sid] = info
        return results

    def get_cached(self, subject_id: int) -> JerseyInfo | None:
        """Look up cached jersey info for a subject."""
        with self._lock:
            return self._cache.get(subject_id)

    def clear_cache(self) -> None:
        """Clear the jersey detection cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return usage statistics."""
        with self._lock:
            return {
                "api_calls": self._api_call_count,
                "cached": len(self._cache),
            }
