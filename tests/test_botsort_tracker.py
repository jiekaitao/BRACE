"""Tests for BoT-SORT tracker (backend/botsort_tracker.py).

Uses mocked YOLO + BotSort so tests run without GPU models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure backend/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from botsort_tracker import BoTSortTracker, TrackedDetection, _iou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_result(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    keypoints_data: np.ndarray | None = None,
):
    """Build a mock Ultralytics Results object."""
    import torch

    result = MagicMock()
    result.boxes = MagicMock()
    result.boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    result.boxes.conf = torch.tensor(confs, dtype=torch.float32)
    result.boxes.cls = torch.tensor(clss, dtype=torch.float32)
    result.boxes.__len__ = lambda self: len(boxes_xyxy)

    if keypoints_data is not None:
        kpts = MagicMock()
        kpts.data = torch.tensor(keypoints_data, dtype=torch.float32)
        kpts.data.__len__ = lambda self: len(keypoints_data)
        result.keypoints = kpts
    else:
        result.keypoints = None

    return result


def _random_keypoints(n_persons: int = 1) -> np.ndarray:
    """(N, 17, 3) random COCO keypoints."""
    return np.random.rand(n_persons, 17, 3).astype(np.float32) * 400


def _make_tracked_output(
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    det_inds: np.ndarray,
) -> np.ndarray:
    """Build a mock BoT-SORT tracker output: (M, 8)."""
    return np.column_stack([boxes_xyxy, track_ids, confs, clss, det_inds])


# ---------------------------------------------------------------------------
# Tests: _iou helper
# ---------------------------------------------------------------------------

class TestIoU:
    def test_identical_boxes(self):
        box = np.array([10, 10, 50, 50])
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 20, 20])
        b = np.array([10, 10, 30, 30])
        # intersection: 10x10=100, union: 400+400-100=700
        assert _iou(a, b) == pytest.approx(100.0 / 700.0)

    def test_zero_area_box(self):
        a = np.array([5, 5, 5, 5])
        b = np.array([0, 0, 10, 10])
        assert _iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: TrackedDetection dataclass
# ---------------------------------------------------------------------------

class TestTrackedDetection:
    def test_fields(self):
        kp = np.zeros((17, 3), dtype=np.float32)
        det = TrackedDetection(
            track_id=42,
            bbox_pixel=(10, 20, 100, 200),
            bbox_normalized=(0.01, 0.02, 0.1, 0.2),
            keypoints=kp,
            conf=0.85,
        )
        assert det.track_id == 42
        assert det.bbox_pixel == (10, 20, 100, 200)
        assert det.keypoints.shape == (17, 3)
        assert det.conf == 0.85
        assert det.crop_rgb.shape == (0,)  # default empty


# ---------------------------------------------------------------------------
# Tests: BoTSortTracker (mocked models)
# ---------------------------------------------------------------------------

class TestBoTSortTracker:
    """All YOLO and BotSort calls are mocked so tests run CPU-only."""

    @patch("botsort_tracker.BotSort")
    @patch("botsort_tracker.YOLO")
    def _make_tracker(self, mock_yolo_cls, mock_botsort_cls):
        """Create a BoTSortTracker with mocked dependencies."""
        mock_yolo_cls.return_value = MagicMock()
        mock_botsort_cls.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=False):
            tracker = BoTSortTracker(
                model_name="yolo11x-pose.pt",
                conf_threshold=0.3,
            )
        return tracker, mock_yolo_cls, mock_botsort_cls

    def test_init(self):
        tracker, mock_yolo, mock_botsort = self._make_tracker()
        assert tracker is not None
        assert tracker.conf_threshold == 0.3
        mock_yolo.assert_called_once_with("yolo11x-pose.pt")
        mock_botsort.assert_called_once()

    def test_process_frame_empty_results(self):
        tracker, _, _ = self._make_tracker()
        tracker.model.predict.return_value = []
        tracker._tracker.update.return_value = np.empty((0, 8))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = tracker.process_frame(frame)

        assert persons == []
        tracker._tracker.update.assert_called_once()

    def test_process_frame_no_boxes(self):
        tracker, _, _ = self._make_tracker()
        result = MagicMock()
        result.boxes = None
        tracker.model.predict.return_value = [result]
        tracker._tracker.update.return_value = np.empty((0, 8))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = tracker.process_frame(frame)

        assert persons == []

    def test_process_frame_single_person(self):
        tracker, _, _ = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # YOLO returns one detection
        boxes_xyxy = np.array([[100.0, 50.0, 300.0, 400.0]])
        confs = np.array([0.9])
        clss = np.array([0.0])
        kps = _random_keypoints(1)
        yolo_result = _make_yolo_result(boxes_xyxy, confs, clss, kps)
        tracker.model.predict.return_value = [yolo_result]

        # BoT-SORT returns one tracked person
        tracked = _make_tracked_output(
            boxes_xyxy=np.array([[100.0, 50.0, 300.0, 400.0]]),
            track_ids=np.array([1]),
            confs=np.array([0.9]),
            clss=np.array([0]),
            det_inds=np.array([0]),
        )
        tracker._tracker.update.return_value = tracked

        persons = tracker.process_frame(frame)

        assert len(persons) == 1
        assert isinstance(persons[0], TrackedDetection)
        assert persons[0].track_id == 1
        assert persons[0].keypoints.shape == (17, 3)
        assert persons[0].bbox_pixel == (100, 50, 300, 400)
        assert 0 <= persons[0].bbox_normalized[0] <= 1

    def test_process_frame_multiple_people(self):
        tracker, _, _ = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        boxes_xyxy = np.array([
            [100.0, 50.0, 300.0, 400.0],
            [350.0, 60.0, 500.0, 380.0],
        ])
        confs = np.array([0.9, 0.8])
        clss = np.array([0.0, 0.0])
        kps = _random_keypoints(2)
        yolo_result = _make_yolo_result(boxes_xyxy, confs, clss, kps)
        tracker.model.predict.return_value = [yolo_result]

        tracked = _make_tracked_output(
            boxes_xyxy=boxes_xyxy,
            track_ids=np.array([1, 2]),
            confs=np.array([0.9, 0.8]),
            clss=np.array([0, 0]),
            det_inds=np.array([0, 1]),
        )
        tracker._tracker.update.return_value = tracked

        persons = tracker.process_frame(frame)

        assert len(persons) == 2
        assert {p.track_id for p in persons} == {1, 2}

    def test_process_frame_filters_tiny_boxes(self):
        tracker, _, _ = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Detection is tiny: 5x5 = 25 pixels, frame_area = 640*480 = 307200
        # 25/307200 = 0.00008 < 0.01 threshold -> filtered
        boxes_xyxy = np.array([[10.0, 10.0, 15.0, 15.0]])
        confs = np.array([0.9])
        clss = np.array([0.0])
        kps = _random_keypoints(1)
        yolo_result = _make_yolo_result(boxes_xyxy, confs, clss, kps)
        tracker.model.predict.return_value = [yolo_result]

        tracked = _make_tracked_output(
            boxes_xyxy=boxes_xyxy,
            track_ids=np.array([1]),
            confs=np.array([0.9]),
            clss=np.array([0]),
            det_inds=np.array([0]),
        )
        tracker._tracker.update.return_value = tracked

        persons = tracker.process_frame(frame)
        assert len(persons) == 0

    def test_process_frame_iou_fallback(self):
        """When det_ind is out of range, falls back to IoU matching."""
        tracker, _, _ = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        boxes_xyxy = np.array([[100.0, 50.0, 300.0, 400.0]])
        confs = np.array([0.9])
        clss = np.array([0.0])
        kps = _random_keypoints(1)
        yolo_result = _make_yolo_result(boxes_xyxy, confs, clss, kps)
        tracker.model.predict.return_value = [yolo_result]

        # det_ind=99 is out of range -> should use IoU fallback
        tracked = _make_tracked_output(
            boxes_xyxy=np.array([[100.0, 50.0, 300.0, 400.0]]),
            track_ids=np.array([1]),
            confs=np.array([0.9]),
            clss=np.array([0]),
            det_inds=np.array([99]),
        )
        tracker._tracker.update.return_value = tracked

        persons = tracker.process_frame(frame)
        assert len(persons) == 1
        assert persons[0].keypoints.shape == (17, 3)

    def test_consistent_track_ids_across_frames(self):
        """BoT-SORT should maintain consistent IDs across frames."""
        tracker, _, _ = self._make_tracker()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        boxes_xyxy = np.array([[100.0, 50.0, 300.0, 400.0]])
        confs = np.array([0.9])
        clss = np.array([0.0])
        kps = _random_keypoints(1)
        yolo_result = _make_yolo_result(boxes_xyxy, confs, clss, kps)
        tracker.model.predict.return_value = [yolo_result]

        track_ids_over_frames = []
        for frame_idx in range(5):
            tracked = _make_tracked_output(
                boxes_xyxy=boxes_xyxy,
                track_ids=np.array([7]),  # same ID across frames
                confs=np.array([0.9]),
                clss=np.array([0]),
                det_inds=np.array([0]),
            )
            tracker._tracker.update.return_value = tracked
            persons = tracker.process_frame(frame)
            if persons:
                track_ids_over_frames.append(persons[0].track_id)

        # All frames should have the same track ID
        assert len(set(track_ids_over_frames)) == 1
        assert track_ids_over_frames[0] == 7

    def test_reset_clears_state(self):
        """reset() should re-create the BoT-SORT tracker."""
        tracker, _, mock_botsort_cls = self._make_tracker()
        old_tracker = tracker._tracker
        with patch("botsort_tracker.BotSort") as mock_bs:
            mock_bs.return_value = MagicMock()
            tracker.reset()
        # After reset, _tracker should be a new instance
        assert tracker._tracker is not old_tracker
