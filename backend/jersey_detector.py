"""Detect jersey numbers and team colors from player crops using Gemini Vision."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

GEMINI_PRO_MODEL = os.environ.get("GEMINI_PRO_MODEL", "gemini-2.0-flash")

GEMINI_API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "")

# Rate limit: minimum seconds between API calls
_MIN_CALL_INTERVAL = 5.0

_JERSEY_PROMPT = (
    "Look at this basketball player. Identify:\n"
    "1. Their jersey number (if visible)\n"
    "2. The dominant color of their jersey/uniform\n"
    "3. The RGB color value of their jersey\n\n"
    "Respond ONLY with valid JSON in this exact format:\n"
    '{"jersey_number": 23, "jersey_color_name": "red", "dominant_color": [255, 0, 0]}\n\n'
    "If the jersey number is not visible, use null for jersey_number.\n"
    "Always provide the color name and RGB values."
)


def _encode_crop_jpeg(crop: np.ndarray, max_dim: int = 512) -> bytes:
    """Resize and JPEG-encode a crop for the API."""
    h, w = crop.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    if len(crop.shape) == 3 and crop.shape[2] == 3:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    else:
        bgr = crop
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def parse_jersey_response(text: str) -> dict:
    """Parse the Gemini response text into a structured dict.

    Returns:
        Dict with keys: jersey_number (int|None), jersey_color_name (str),
        dominant_color (list[int] of 3 RGB values).
    """
    # Try to extract JSON from the response
    # Handle markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code block markers
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return _validate_jersey_data(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in the text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_jersey_data(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract fields manually
    number = None
    num_match = re.search(r'(?:number|jersey)[:\s]*(\d+)', text, re.IGNORECASE)
    if num_match:
        number = int(num_match.group(1))

    color_name = "unknown"
    color_match = re.search(
        r'(?:color|jersey)[:\s]*(red|blue|white|black|green|yellow|purple|orange|grey|gray|navy|gold|maroon|teal)',
        text, re.IGNORECASE,
    )
    if color_match:
        color_name = color_match.group(1).lower()

    return {
        "jersey_number": number,
        "jersey_color_name": color_name,
        "dominant_color": [128, 128, 128],
    }


def _validate_jersey_data(data: dict) -> dict:
    """Validate and normalize parsed jersey data."""
    number = data.get("jersey_number")
    if number is not None:
        number = int(number)
        if number < 0 or number > 99:
            number = None

    color_name = str(data.get("jersey_color_name", "unknown")).lower().strip()

    dominant = data.get("dominant_color", [128, 128, 128])
    if isinstance(dominant, list) and len(dominant) == 3:
        dominant = [max(0, min(255, int(c))) for c in dominant]
    else:
        dominant = [128, 128, 128]

    return {
        "jersey_number": number,
        "jersey_color_name": color_name,
        "dominant_color": dominant,
    }


class JerseyDetector:
    """Thread-safe jersey number and color detector using Gemini Pro Vision."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self._api_key = api_key or GEMINI_API_KEY
        self._model_name = model_name or GEMINI_PRO_MODEL
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        self._client = None
        self._init_error: str | None = None
        self._api_call_count: int = 0
        self._results: dict[int, dict] = {}  # subject_id -> detection result

    def _ensure_client(self):
        """Lazy-init the Gemini client on first use."""
        if self._client is not None or self._init_error is not None:
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        except Exception as e:
            self._init_error = str(e)

    @property
    def available(self) -> bool:
        """True if the Gemini SDK loaded and API key is set."""
        self._ensure_client()
        return self._client is not None

    def detect(self, crop_rgb: np.ndarray) -> dict:
        """Detect jersey number and color from a player crop.

        Args:
            crop_rgb: RGB image crop of a single player.

        Returns:
            Dict with jersey_number (int|None), jersey_color_name (str),
            dominant_color ([R, G, B]).
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return {"jersey_number": None, "jersey_color_name": "unknown",
                    "dominant_color": [128, 128, 128]}

        self._ensure_client()
        if self._client is None:
            return {"jersey_number": None, "jersey_color_name": "unknown",
                    "dominant_color": [128, 128, 128]}

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

            jpeg_bytes = _encode_crop_jpeg(crop_rgb)
            img = Image.open(io.BytesIO(jpeg_bytes))

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[img, _JERSEY_PROMPT],
            )
            with self._lock:
                self._api_call_count += 1

            result = parse_jersey_response(response.text)
            print(f"[jersey] detected: number={result['jersey_number']}, "
                  f"color={result['jersey_color_name']}", flush=True)
            return result

        except Exception as e:
            print(f"[jersey] detection failed: {e}", flush=True)
            return {"jersey_number": None, "jersey_color_name": "unknown",
                    "dominant_color": [128, 128, 128]}

    def get_result(self, subject_id: int) -> dict | None:
        """Get cached detection result for a subject."""
        with self._lock:
            return self._results.get(subject_id)

    def store_result(self, subject_id: int, result: dict) -> None:
        """Store a detection result for a subject."""
        with self._lock:
            self._results[subject_id] = result

    def has_result(self, subject_id: int) -> bool:
        """Check if a detection result exists for a subject."""
        with self._lock:
            return subject_id in self._results

    def get_all_results(self) -> dict[int, dict]:
        """Get all stored detection results."""
        with self._lock:
            return dict(self._results)

    def get_stats(self) -> dict:
        """Return usage statistics."""
        with self._lock:
            return {
                "api_calls": self._api_call_count,
                "players_detected": len(self._results),
                "estimated_cost_usd": round(self._api_call_count * 0.01, 4),
            }


def cluster_teams(jersey_results: dict[int, dict], n_teams: int = 2) -> dict[int, int]:
    """Cluster players into teams based on dominant jersey color.

    Args:
        jersey_results: {subject_id: {dominant_color: [R, G, B], ...}}
        n_teams: Number of teams (default 2).

    Returns:
        {subject_id: team_id} where team_id is 0 or 1.
    """
    if len(jersey_results) < 2:
        return {sid: 0 for sid in jersey_results}

    subject_ids = list(jersey_results.keys())
    colors = np.array([
        jersey_results[sid].get("dominant_color", [128, 128, 128])
        for sid in subject_ids
    ], dtype=np.float32)

    # Simple k-means clustering on RGB colors
    from sklearn.cluster import KMeans

    n_clusters = min(n_teams, len(subject_ids))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(colors)

    return {sid: int(label) for sid, label in zip(subject_ids, labels)}
