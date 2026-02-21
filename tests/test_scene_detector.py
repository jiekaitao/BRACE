"""Tests for InlineSceneDetector."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from scene_detector import InlineSceneDetector


def _make_frame(r: int, g: int, b: int, h: int = 120, w: int = 160) -> np.ndarray:
    """Create a solid-color RGB frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = r
    frame[:, :, 1] = g
    frame[:, :, 2] = b
    return frame


class TestInlineSceneDetector:
    def test_identical_frames_no_cut(self):
        """Identical consecutive frames should never trigger a scene cut."""
        det = InlineSceneDetector(threshold=30.0, min_scene_len=1)
        frame = _make_frame(100, 100, 100)

        # First frame is never a cut
        assert det.process_frame(frame) is False

        # Repeated identical frames should not trigger
        for _ in range(20):
            assert det.process_frame(frame) is False

    def test_dramatically_different_frames_trigger_cut(self):
        """A sudden change from black to white should trigger a scene cut."""
        det = InlineSceneDetector(threshold=30.0, min_scene_len=1)
        black = _make_frame(0, 0, 0)
        white = _make_frame(255, 255, 255)

        # First frame (black) - no cut
        assert det.process_frame(black) is False
        # Second frame (white) - should detect cut
        assert det.process_frame(white) is True

    def test_gradual_change_no_cut(self):
        """Small incremental changes should not trigger a cut."""
        det = InlineSceneDetector(threshold=30.0, min_scene_len=1)

        for i in range(0, 50, 2):
            frame = _make_frame(i, i, i)
            result = det.process_frame(frame)
            # Gradual changes of ~2 per frame should stay well below threshold
            assert result is False, f"Unexpected cut at brightness {i}"

    def test_min_scene_len_prevents_rapid_cuts(self):
        """Scene cuts should be suppressed within min_scene_len frames."""
        det = InlineSceneDetector(threshold=30.0, min_scene_len=10)
        black = _make_frame(0, 0, 0)
        white = _make_frame(255, 255, 255)

        # Build up enough frames
        assert det.process_frame(black) is False
        for _ in range(10):
            det.process_frame(black)

        # First real cut
        assert det.process_frame(white) is True

        # Immediate switch back should be suppressed (within min_scene_len)
        assert det.process_frame(black) is False

    def test_reset_clears_state(self):
        """After reset, detector should behave as if freshly constructed."""
        det = InlineSceneDetector(threshold=30.0, min_scene_len=1)
        black = _make_frame(0, 0, 0)
        white = _make_frame(255, 255, 255)

        det.process_frame(black)
        det.process_frame(white)  # triggers cut

        det.reset()

        # After reset, first frame should not be a cut (no previous frame)
        assert det.process_frame(white) is False

    def test_noisy_frames_below_threshold(self):
        """Frames with small random noise should not trigger cuts."""
        rng = np.random.RandomState(42)
        det = InlineSceneDetector(threshold=30.0, min_scene_len=1)

        base = _make_frame(128, 128, 128)
        assert det.process_frame(base) is False

        for _ in range(20):
            noise = rng.randint(-5, 6, size=base.shape).astype(np.int16)
            noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            assert det.process_frame(noisy) is False

    def test_first_frame_never_cut(self):
        """The very first frame should never be detected as a scene cut."""
        det = InlineSceneDetector(threshold=1.0, min_scene_len=1)
        # Even with threshold=1.0 (very sensitive), first frame is not a cut
        frame = _make_frame(200, 50, 100)
        assert det.process_frame(frame) is False

    def test_custom_threshold(self):
        """Higher threshold should require bigger changes to trigger."""
        # With very high threshold, even black->white might not trigger
        det = InlineSceneDetector(threshold=200.0, min_scene_len=1)
        black = _make_frame(0, 0, 0)
        gray = _make_frame(128, 128, 128)

        det.process_frame(black)
        # Mean diff is ~128, below threshold of 200
        assert det.process_frame(gray) is False

        white = _make_frame(255, 255, 255)
        # Now mean diff from gray is ~127, still below 200
        assert det.process_frame(white) is False
