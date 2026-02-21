"""Inline scene cut detection for video frame streams.

Uses mean absolute pixel difference between consecutive frames to detect
scene cuts (camera changes). This is faster and simpler than full
PySceneDetect ContentDetector and works frame-by-frame without batching.

Typical cost: ~1ms per 640x480 frame on CPU.
"""

from __future__ import annotations

import cv2
import numpy as np


class InlineSceneDetector:
    """Detect scene cuts by comparing consecutive frames.

    Uses downscaled grayscale frame differencing to detect abrupt scene
    transitions. A scene cut is flagged when the mean absolute pixel
    difference exceeds the threshold and at least ``min_scene_len``
    frames have passed since the last cut.
    """

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_len: int = 15,
        downscale_width: int = 160,
    ):
        """
        Args:
            threshold: Mean absolute pixel difference (0-255) to trigger a cut.
                       30.0 works well for typical sports/broadcast footage.
            min_scene_len: Minimum number of frames between consecutive cuts
                           to avoid rapid false positives.
            downscale_width: Width to downscale frames to before comparison.
                             Smaller = faster but less sensitive.
        """
        self._threshold = threshold
        self._min_scene_len = min_scene_len
        self._downscale_width = downscale_width

        self._prev_gray: np.ndarray | None = None
        self._frames_since_cut = 0

    def _to_small_gray(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB frame to downscaled grayscale for fast comparison."""
        h, w = rgb.shape[:2]
        scale = self._downscale_width / w
        new_h = max(1, int(h * scale))
        small = cv2.resize(rgb, (self._downscale_width, new_h), interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        else:
            gray = small
        return gray

    def process_frame(self, rgb: np.ndarray) -> bool:
        """Process a single frame. Returns True if a scene cut is detected.

        Args:
            rgb: (H, W, 3) RGB numpy array.

        Returns:
            True if this frame represents a scene cut from the previous frame.
        """
        gray = self._to_small_gray(rgb)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._frames_since_cut = 0
            return False

        # Mean absolute difference
        diff = np.mean(np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32)))

        self._prev_gray = gray
        self._frames_since_cut += 1

        if diff >= self._threshold and self._frames_since_cut >= self._min_scene_len:
            self._frames_since_cut = 0
            return True

        return False

    def reset(self) -> None:
        """Reset detector state for a new video."""
        self._prev_gray = None
        self._frames_since_cut = 0
