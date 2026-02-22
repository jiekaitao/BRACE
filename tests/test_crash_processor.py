"""Tests for crash_processor.py — the offline crash video analysis pipeline.

Uses synthetic skeleton data and mock objects (same pattern as test_movement_quality.py).
Tests are written TDD-style before the implementation exists.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ── Synthetic helpers ──────────────────────────────────────────────────────


def _make_coco17_skeleton(
    x_offset: float = 0.5,
    y_offset: float = 0.5,
    scale: float = 0.2,
    confidence: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic COCO-17 skeleton at (x_offset, y_offset) in pixel space.

    Returns (kpts, confidences) where kpts is (17, 2) and conf is (17,).
    """
    # Basic upright skeleton template in normalized [-1, 1] space
    template = np.array([
        [0.0, -0.9],   # 0: nose
        [-0.05, -0.95], # 1: l_eye
        [0.05, -0.95],  # 2: r_eye
        [-0.1, -0.9],  # 3: l_ear
        [0.1, -0.9],   # 4: r_ear
        [-0.2, -0.6],  # 5: l_shoulder
        [0.2, -0.6],   # 6: r_shoulder
        [-0.3, -0.3],  # 7: l_elbow
        [0.3, -0.3],   # 8: r_elbow
        [-0.35, 0.0],  # 9: l_wrist
        [0.35, 0.0],   # 10: r_wrist
        [-0.15, 0.0],  # 11: l_hip
        [0.15, 0.0],   # 12: r_hip
        [-0.2, 0.4],   # 13: l_knee
        [0.2, 0.4],    # 14: r_knee
        [-0.2, 0.8],   # 15: l_ankle
        [0.2, 0.8],    # 16: r_ankle
    ], dtype=np.float64)

    kpts = template * scale + np.array([x_offset, y_offset])
    conf = np.full(17, confidence, dtype=np.float64)
    return kpts, conf


class MockPipelineResult:
    """Minimal mock of PipelineResult from pipeline_interface.py.

    Matches the real PipelineResult dataclass: landmarks_mp is a (33, 4)
    numpy array with [x_px, y_px, z, visibility].
    """

    def __init__(self, track_id: int, kpts: np.ndarray, conf: np.ndarray, bbox: np.ndarray):
        self.track_id = track_id
        self.landmarks_mp = _coco17_to_mp33(kpts, conf)
        self.bbox_pixel = bbox


def _coco17_to_mp33(kpts: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Convert COCO-17 keypoints to a (33, 4) MediaPipe-33 landmark array.

    Returns ndarray matching PipelineResult.landmarks_mp format:
    [x_px, y_px, z, visibility] per landmark.
    """
    coco_to_mp = {
        0: 0,   # nose
        1: 2,   # l_eye
        2: 5,   # r_eye
        3: 7,   # l_ear
        4: 8,   # r_ear
        5: 11,  # l_shoulder
        6: 12,  # r_shoulder
        7: 13,  # l_elbow
        8: 14,  # r_elbow
        9: 15,  # l_wrist
        10: 16, # r_wrist
        11: 23, # l_hip
        12: 24, # r_hip
        13: 25, # l_knee
        14: 26, # r_knee
        15: 27, # l_ankle
        16: 28, # r_ankle
    }

    out = np.zeros((33, 4), dtype=np.float64)
    for coco_idx, mp_idx in coco_to_mp.items():
        if coco_idx < len(kpts):
            out[mp_idx, 0] = kpts[coco_idx, 0]   # x
            out[mp_idx, 1] = kpts[coco_idx, 1]   # y
            out[mp_idx, 2] = 0.0                  # z
            out[mp_idx, 3] = conf[coco_idx]       # visibility

    return out


class MockPipeline:
    """Mock PoseBackend that returns configurable skeletons per frame."""

    def __init__(self, frame_results: list[list[MockPipelineResult]] | None = None):
        self._frame_results = frame_results or []
        self._call_count = 0

    def process_frame(self, rgb: np.ndarray) -> list[MockPipelineResult]:
        if self._call_count < len(self._frame_results):
            results = self._frame_results[self._call_count]
        else:
            results = []
        self._call_count += 1
        return results


def _make_two_approaching_frames(
    n_frames: int = 30,
    fps: float = 30.0,
    collision_frame: int = 15,
) -> list[list[MockPipelineResult]]:
    """Generate frames with two people approaching each other, colliding at collision_frame.

    Subject 1 starts at x=100, Subject 2 starts at x=500.
    They meet near x=300 at the collision_frame.
    """
    all_frames = []
    start_a, start_b = 100.0, 500.0
    end_a, end_b = 300.0, 300.0

    for i in range(n_frames):
        t = min(i / max(collision_frame, 1), 1.0)
        x_a = start_a + (end_a - start_a) * t
        x_b = start_b + (end_b - start_b) * t
        y = 300.0

        kpts_a, conf_a = _make_coco17_skeleton(x_a, y, scale=50.0)
        kpts_b, conf_b = _make_coco17_skeleton(x_b, y, scale=50.0)

        bbox_a = np.array([x_a - 60, y - 90, x_a + 60, y + 90])
        bbox_b = np.array([x_b - 60, y - 90, x_b + 60, y + 90])

        frame_results = [
            MockPipelineResult(1, kpts_a, conf_a, bbox_a),
            MockPipelineResult(2, kpts_b, conf_b, bbox_b),
        ]
        all_frames.append(frame_results)

    return all_frames


def _make_single_person_frames(n_frames: int = 30) -> list[list[MockPipelineResult]]:
    """Generate frames with a single stationary person."""
    all_frames = []
    for _ in range(n_frames):
        kpts, conf = _make_coco17_skeleton(300.0, 300.0, scale=50.0)
        bbox = np.array([240.0, 210.0, 360.0, 390.0])
        all_frames.append([MockPipelineResult(1, kpts, conf, bbox)])
    return all_frames


def _make_empty_frames(n_frames: int = 10) -> list[list[MockPipelineResult]]:
    """Generate frames with no detections."""
    return [[] for _ in range(n_frames)]


def _make_distant_subjects_frames(n_frames: int = 30) -> list[list[MockPipelineResult]]:
    """Generate frames with two people far apart and not moving."""
    all_frames = []
    for _ in range(n_frames):
        kpts_a, conf_a = _make_coco17_skeleton(100.0, 300.0, scale=50.0)
        kpts_b, conf_b = _make_coco17_skeleton(800.0, 300.0, scale=50.0)
        bbox_a = np.array([40.0, 210.0, 160.0, 390.0])
        bbox_b = np.array([740.0, 210.0, 860.0, 390.0])
        all_frames.append([
            MockPipelineResult(1, kpts_a, conf_a, bbox_a),
            MockPipelineResult(2, kpts_b, conf_b, bbox_b),
        ])
    return all_frames


# ── Mock video capture ─────────────────────────────────────────────────────


class MockVideoCapture:
    """Mock cv2.VideoCapture that returns synthetic frames."""

    def __init__(self, n_frames: int = 30, w: int = 640, h: int = 480, fps: float = 30.0):
        self._n_frames = n_frames
        self._w = w
        self._h = h
        self._fps = fps
        self._frame_idx = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def get(self, prop_id):
        if prop_id == 7:  # CAP_PROP_FRAME_COUNT
            return float(self._n_frames)
        if prop_id == 5:  # CAP_PROP_FPS
            return self._fps
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return float(self._w)
        if prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return float(self._h)
        return 0.0

    def read(self):
        if self._frame_idx >= self._n_frames:
            return False, None
        self._frame_idx += 1
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self._opened = False


# ═══════════════════════════════════════════════════════════════════════════
# CrashAnalysisResult structure tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCrashAnalysisResult:
    """Tests for the structure and content of crash analysis results."""

    @pytest.fixture
    def crash_result(self):
        """Run a basic crash analysis and return the result."""
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)

        result = process_crash_video(
            pipeline=pipeline,
            n_frames=30,
            fps=30.0,
            img_wh=(640, 480),
        )
        return result

    def test_result_has_analysis_id(self, crash_result):
        assert "analysis_id" in crash_result
        assert isinstance(crash_result["analysis_id"], str)
        assert len(crash_result["analysis_id"]) > 0

    def test_result_has_collision_events(self, crash_result):
        assert "collision_events" in crash_result
        assert isinstance(crash_result["collision_events"], list)

    def test_result_has_subject_summaries(self, crash_result):
        assert "subject_summaries" in crash_result
        assert isinstance(crash_result["subject_summaries"], dict)

    def test_result_has_overall_risk(self, crash_result):
        assert "overall_risk" in crash_result
        assert crash_result["overall_risk"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_result_has_overall_recommendation(self, crash_result):
        assert "overall_recommendation" in crash_result
        assert isinstance(crash_result["overall_recommendation"], str)

    def test_result_has_status_complete(self, crash_result):
        assert crash_result.get("status") == "complete"

    def test_result_has_total_frames(self, crash_result):
        assert "total_frames" in crash_result
        assert crash_result["total_frames"] == 30

    def test_result_has_duration_sec(self, crash_result):
        assert "duration_sec" in crash_result
        assert crash_result["duration_sec"] == pytest.approx(1.0, abs=0.1)

    def test_result_has_fps(self, crash_result):
        assert "fps" in crash_result
        assert crash_result["fps"] == 30.0

    def test_result_has_subjects_tracked(self, crash_result):
        assert "subjects_tracked" in crash_result
        assert crash_result["subjects_tracked"] >= 1

    def test_result_serializable_to_json(self, crash_result):
        """Entire result must be JSON-serializable (no numpy types)."""
        json_str = json.dumps(crash_result, default=str)
        assert len(json_str) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Collision detection integration
# ═══════════════════════════════════════════════════════════════════════════


class TestCollisionDetectionIntegration:
    """Tests that the collision detection pipeline works end-to-end within crash analysis."""

    def test_two_approaching_subjects_trigger_collision(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        # Two people moving toward each other should produce at least one collision event
        assert len(result["collision_events"]) >= 1

    def test_distant_subjects_no_collision(self):
        from crash_processor import process_crash_video

        frames = _make_distant_subjects_frames(n_frames=30)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        # Two people far apart with no movement should have no collisions
        assert len(result["collision_events"]) == 0

    def test_collision_event_has_concussion_probability(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "concussion_probability" in event
            assert 0.0 <= event["concussion_probability"] <= 1.0

    def test_collision_event_has_risk_level(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "risk_level" in event
            assert event["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_collision_event_has_closing_speed(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "closing_speed_ms" in event
            assert event["closing_speed_ms"] >= 0.0

    def test_collision_event_has_hic(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "hic" in event
            assert event["hic"] >= 0.0

    def test_collision_event_has_contact_zone(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "contact_zone" in event
            assert isinstance(event["contact_zone"], str)

    def test_collision_event_has_frame_index(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "frame_index" in event
            assert 0 <= event["frame_index"] < 30

    def test_collision_event_has_video_time(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "video_time" in event
            assert event["video_time"] >= 0.0

    def test_collision_event_has_subjects(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "subject_a" in event
            assert "subject_b" in event

    def test_collision_event_has_peak_fields(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            event = result["collision_events"][0]
            assert "peak_linear_g" in event
            assert "peak_rotational_rads2" in event
            assert event["peak_linear_g"] >= 0.0
            assert event["peak_rotational_rads2"] >= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Head impact integration
# ═══════════════════════════════════════════════════════════════════════════


class TestHeadImpactIntegration:
    """Tests for head impact tracking within the crash pipeline."""

    def test_head_accel_tracked_per_subject(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        # Should have summaries for both subjects
        assert len(result["subject_summaries"]) >= 2

    def test_subject_summary_has_collision_count(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for sid, summary in result["subject_summaries"].items():
            assert "collision_count" in summary
            assert isinstance(summary["collision_count"], int)

    def test_subject_summary_has_max_concussion_probability(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for sid, summary in result["subject_summaries"].items():
            assert "max_concussion_probability" in summary

    def test_subject_summary_has_worst_risk_level(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for sid, summary in result["subject_summaries"].items():
            assert "worst_risk_level" in summary
            assert summary["worst_risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL", "NONE")

    def test_subject_summary_has_recommendation(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for sid, summary in result["subject_summaries"].items():
            assert "recommendation" in summary
            assert isinstance(summary["recommendation"], str)


# ═══════════════════════════════════════════════════════════════════════════
# Biomechanics scoring integration
# ═══════════════════════════════════════════════════════════════════════════


class TestBiomechanicsScoring:
    """Tests that score_collision is called for each detected collision."""

    def test_collision_events_have_biomechanics(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for event in result["collision_events"]:
            # score_collision fields
            assert "concussion_probability" in event
            assert "risk_level" in event
            assert "hic" in event
            assert "peak_linear_g" in event
            assert "peak_rotational_rads2" in event

    def test_risk_levels_are_valid(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        valid_levels = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        for event in result["collision_events"]:
            assert event["risk_level"] in valid_levels

    def test_hic_computed_for_each_collision(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for event in result["collision_events"]:
            assert isinstance(event["hic"], (int, float))
            assert event["hic"] >= 0.0

    def test_head_coupling_factor_present(self):
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        for event in result["collision_events"]:
            assert "head_coupling_factor" in event
            assert event["head_coupling_factor"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Progress callback tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProgressCallback:
    """Tests for progress reporting during crash analysis."""

    def test_progress_callback_called(self):
        from crash_processor import process_crash_video

        progress_values = []

        def on_progress(pct: float, data: dict):
            progress_values.append(pct)

        frames = _make_two_approaching_frames(n_frames=60, collision_frame=30)
        pipeline = MockPipeline(frames)
        process_crash_video(
            pipeline=pipeline, n_frames=60, fps=30.0, img_wh=(640, 480),
            progress_callback=on_progress,
        )
        assert len(progress_values) > 0

    def test_progress_monotonically_increases(self):
        from crash_processor import process_crash_video

        progress_values = []

        def on_progress(pct: float, data: dict):
            progress_values.append(pct)

        frames = _make_two_approaching_frames(n_frames=90, collision_frame=45)
        pipeline = MockPipeline(frames)
        process_crash_video(
            pipeline=pipeline, n_frames=90, fps=30.0, img_wh=(640, 480),
            progress_callback=on_progress,
        )
        if len(progress_values) >= 2:
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i - 1]

    def test_progress_reaches_100(self):
        from crash_processor import process_crash_video

        progress_values = []

        def on_progress(pct: float, data: dict):
            progress_values.append(pct)

        frames = _make_two_approaching_frames(n_frames=60, collision_frame=30)
        pipeline = MockPipeline(frames)
        process_crash_video(
            pipeline=pipeline, n_frames=60, fps=30.0, img_wh=(640, 480),
            progress_callback=on_progress,
        )
        # Final progress should be at or near 100
        if progress_values:
            assert progress_values[-1] >= 90.0

    def test_progress_data_has_frame_info(self):
        from crash_processor import process_crash_video

        progress_data_list = []

        def on_progress(pct: float, data: dict):
            progress_data_list.append(data)

        frames = _make_two_approaching_frames(n_frames=60, collision_frame=30)
        pipeline = MockPipeline(frames)
        process_crash_video(
            pipeline=pipeline, n_frames=60, fps=30.0, img_wh=(640, 480),
            progress_callback=on_progress,
        )
        if progress_data_list:
            data = progress_data_list[0]
            assert "frame_index" in data
            assert "total_frames" in data


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_video_no_impact(self):
        from crash_processor import process_crash_video

        frames = _make_empty_frames(n_frames=10)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=10, fps=30.0, img_wh=(640, 480),
        )
        assert result["status"] == "complete"
        assert len(result["collision_events"]) == 0
        assert result["overall_risk"] == "LOW"

    def test_single_person_no_collisions(self):
        from crash_processor import process_crash_video

        frames = _make_single_person_frames(n_frames=30)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        assert len(result["collision_events"]) == 0

    def test_no_visible_heads_handled(self):
        """Subjects with no visible head keypoints should not cause errors."""
        from crash_processor import process_crash_video

        # Create frames where head confidence is 0.0
        all_frames = []
        for _ in range(10):
            kpts_a, conf_a = _make_coco17_skeleton(200.0, 300.0, scale=50.0, confidence=0.9)
            kpts_b, conf_b = _make_coco17_skeleton(400.0, 300.0, scale=50.0, confidence=0.9)
            # Zero out head confidences
            conf_a[:5] = 0.0
            conf_b[:5] = 0.0
            bbox_a = np.array([140.0, 210.0, 260.0, 390.0])
            bbox_b = np.array([340.0, 210.0, 460.0, 390.0])
            all_frames.append([
                MockPipelineResult(1, kpts_a, conf_a, bbox_a),
                MockPipelineResult(2, kpts_b, conf_b, bbox_b),
            ])

        pipeline = MockPipeline(all_frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=10, fps=30.0, img_wh=(640, 480),
        )
        assert result["status"] == "complete"

    def test_zero_fps_handled(self):
        """Zero FPS should not cause division by zero."""
        from crash_processor import process_crash_video

        frames = _make_single_person_frames(n_frames=5)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=5, fps=0.0, img_wh=(640, 480),
        )
        assert result["status"] == "complete"

    def test_many_subjects_handled(self):
        """More than 2 subjects should be handled (all pairs checked)."""
        from crash_processor import process_crash_video

        all_frames = []
        for _ in range(20):
            frame = []
            for tid in range(1, 5):
                x = 100.0 + tid * 100.0
                kpts, conf = _make_coco17_skeleton(x, 300.0, scale=50.0)
                bbox = np.array([x - 60, 210.0, x + 60, 390.0])
                frame.append(MockPipelineResult(tid, kpts, conf, bbox))
            all_frames.append(frame)

        pipeline = MockPipeline(all_frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=20, fps=30.0, img_wh=(640, 480),
        )
        assert result["status"] == "complete"
        assert result["subjects_tracked"] == 4

    def test_pipeline_exception_handled(self):
        """If pipeline raises an exception on a frame, processing should continue."""
        from crash_processor import process_crash_video

        class FailingPipeline:
            def __init__(self):
                self._call_count = 0

            def process_frame(self, rgb):
                self._call_count += 1
                if self._call_count == 5:
                    raise RuntimeError("Simulated pipeline failure")
                kpts, conf = _make_coco17_skeleton(300.0, 300.0, scale=50.0)
                bbox = np.array([240.0, 210.0, 360.0, 390.0])
                return [MockPipelineResult(1, kpts, conf, bbox)]

        pipeline = FailingPipeline()
        result = process_crash_video(
            pipeline=pipeline, n_frames=10, fps=30.0, img_wh=(640, 480),
        )
        assert result["status"] == "complete"

    def test_collision_events_sorted_by_time(self):
        """Collision events should be sorted by frame_index."""
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=60, collision_frame=30)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=60, fps=30.0, img_wh=(640, 480),
        )
        frame_indices = [e["frame_index"] for e in result["collision_events"]]
        assert frame_indices == sorted(frame_indices)

    def test_overall_risk_is_worst_of_all_events(self):
        """overall_risk should reflect the worst collision event."""
        from crash_processor import process_crash_video

        frames = _make_two_approaching_frames(n_frames=30, collision_frame=15)
        pipeline = MockPipeline(frames)
        result = process_crash_video(
            pipeline=pipeline, n_frames=30, fps=30.0, img_wh=(640, 480),
        )
        if result["collision_events"]:
            risk_order = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
            worst_event_risk = max(
                result["collision_events"],
                key=lambda e: risk_order.get(e["risk_level"], 0),
            )["risk_level"]
            assert risk_order[result["overall_risk"]] >= risk_order[worst_event_risk]
