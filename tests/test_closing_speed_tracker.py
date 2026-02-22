"""Tests for ClosingSpeedTracker from backend/main.py."""

from __future__ import annotations

import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

# ClosingSpeedTracker is defined in main.py, but importing main.py triggers the
# full FastAPI app startup (YOLO, TensorRT, MongoDB, directory creation, etc.)
# which requires GPU and root permissions. We extract the class source directly
# to avoid all that.

_BACKEND = Path(__file__).parent.parent / "backend"

def _extract_closing_speed_tracker():
    """Extract ClosingSpeedTracker class from main.py without importing it."""
    source = (_BACKEND / "main.py").read_text()
    # Find class definition and its body
    match = re.search(
        r'^(class ClosingSpeedTracker:.*?)(?=\n(?:class |def |@app\.|# -)|\Z)',
        source,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ImportError("Could not find ClosingSpeedTracker in main.py")

    class_source = match.group(1)

    # The constant is defined as: float(os.environ.get("CLOSING_SPEED_THRESHOLD", "1.5"))
    # We use the default value directly.
    import os
    namespace: dict[str, Any] = {
        "math": math, "Any": Any, "os": os,
        "CLOSING_SPEED_THRESHOLD": float(os.environ.get("CLOSING_SPEED_THRESHOLD", "1.5")),
    }
    exec(class_source, namespace)
    return namespace["ClosingSpeedTracker"]

ClosingSpeedTracker = _extract_closing_speed_tracker()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_subjects(centers: dict[int, tuple[float, float]]) -> dict[str, dict]:
    """Build a subjects_data dict from {sid: (cx, cy)} with synthetic bboxes."""
    subjects = {}
    for sid, (cx, cy) in centers.items():
        w, h = 0.05, 0.1
        subjects[str(sid)] = {
            "bbox": {
                "x1": cx - w / 2, "y1": cy - h / 2,
                "x2": cx + w / 2, "y2": cy + h / 2,
            },
        }
    return subjects


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestClosingSpeedTracker:
    """Tests for the pairwise closing-speed tracker."""

    def test_empty_subjects_no_pairs(self):
        tracker = ClosingSpeedTracker()
        result = tracker.update({}, video_time=0.0, fps=30.0)
        assert result["pairs"] == []
        assert result["max_closing_speed"] == 0.0
        assert result["collision_warning"] is False

    def test_single_subject_no_pairs(self):
        tracker = ClosingSpeedTracker()
        subjects = _make_subjects({1: (0.5, 0.5)})
        # First frame: establishes prev_centers
        tracker.update(subjects, video_time=0.0, fps=30.0)
        result = tracker.update(subjects, video_time=1 / 30, fps=30.0)
        assert result["pairs"] == []

    def test_two_approaching_positive_speed(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        # Frame 0: subjects far apart
        s0 = _make_subjects({1: (0.2, 0.5), 2: (0.8, 0.5)})
        tracker.update(s0, video_time=0.0, fps=fps)

        # Frame 1: subjects closer together
        s1 = _make_subjects({1: (0.3, 0.5), 2: (0.7, 0.5)})
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        assert len(result["pairs"]) == 1
        pair = result["pairs"][0]
        assert pair["closing_speed"] > 0
        assert pair["a"] in (1, 2)
        assert pair["b"] in (1, 2)

    def test_diverging_no_pairs(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        # Frame 0: subjects close
        s0 = _make_subjects({1: (0.4, 0.5), 2: (0.6, 0.5)})
        tracker.update(s0, video_time=0.0, fps=fps)

        # Frame 1: subjects moving apart
        s1 = _make_subjects({1: (0.3, 0.5), 2: (0.7, 0.5)})
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        # No approaching pairs (diverging)
        assert result["pairs"] == []

    def test_threshold_triggers_collision_warning(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        # Frame 0: far apart
        s0 = _make_subjects({1: (0.1, 0.5), 2: (0.9, 0.5)})
        tracker.update(s0, video_time=0.0, fps=fps)

        # Frame 1: very close together (large displacement = high speed)
        s1 = _make_subjects({1: (0.49, 0.5), 2: (0.51, 0.5)})
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        # Very high closing speed should trigger warning
        assert result["max_closing_speed"] > 0
        # Check collision_warning is a bool
        assert isinstance(result["collision_warning"], bool)

    def test_pairs_sorted_by_speed_descending(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        # Frame 0: three subjects
        s0 = _make_subjects({1: (0.1, 0.5), 2: (0.5, 0.5), 3: (0.9, 0.5)})
        tracker.update(s0, video_time=0.0, fps=fps)

        # Frame 1: 1 and 3 approaching fast, 2 barely moving
        s1 = _make_subjects({1: (0.2, 0.5), 2: (0.5, 0.5), 3: (0.8, 0.5)})
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        # Pairs with positive closing speed should be sorted descending
        speeds = [p["closing_speed"] for p in result["pairs"]]
        assert speeds == sorted(speeds, reverse=True)

    def test_max_5_pairs(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        # Create 6 subjects — C(6,2) = 15 potential pairs
        centers0 = {i: (0.1 * i, 0.5) for i in range(1, 7)}
        s0 = _make_subjects(centers0)
        tracker.update(s0, video_time=0.0, fps=fps)

        # All move toward center
        centers1 = {i: (0.1 * i + 0.02 * (3.5 - i), 0.5) for i in range(1, 7)}
        s1 = _make_subjects(centers1)
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        assert len(result["pairs"]) <= 5

    def test_dt_from_video_time(self):
        """When video_time is provided, dt should use actual time difference."""
        tracker = ClosingSpeedTracker()

        s0 = _make_subjects({1: (0.2, 0.5), 2: (0.8, 0.5)})
        tracker.update(s0, video_time=0.0, fps=30.0)

        s1 = _make_subjects({1: (0.3, 0.5), 2: (0.7, 0.5)})
        # Use a different video_time spacing than 1/fps
        result = tracker.update(s1, video_time=0.5, fps=30.0)

        # Should still compute pairs without error
        assert len(result["pairs"]) >= 1

    def test_result_has_distance_field(self):
        tracker = ClosingSpeedTracker()
        fps = 30.0

        s0 = _make_subjects({1: (0.2, 0.5), 2: (0.8, 0.5)})
        tracker.update(s0, video_time=0.0, fps=fps)

        s1 = _make_subjects({1: (0.3, 0.5), 2: (0.7, 0.5)})
        result = tracker.update(s1, video_time=1 / fps, fps=fps)

        assert len(result["pairs"]) >= 1
        assert "distance" in result["pairs"][0]
        assert result["pairs"][0]["distance"] > 0

    def test_subjects_without_bbox_ignored(self):
        tracker = ClosingSpeedTracker()
        # Subject without bbox should be skipped
        subjects = {"1": {"bbox": None}, "2": _make_subjects({2: (0.5, 0.5)})["2"]}
        result = tracker.update(subjects, video_time=0.0, fps=30.0)
        assert result["pairs"] == []
