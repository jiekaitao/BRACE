"""YOLO-pose + ByteTrack wrapper for multi-person detection, tracking, and pose estimation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class TrackedPerson:
    track_id: int
    bbox_pixel: tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_normalized: tuple[float, float, float, float]  # 0-1
    keypoints: np.ndarray  # (17, 3) [x_px, y_px, conf] from YOLO-pose
    crop_rgb: np.ndarray = field(default_factory=lambda: np.empty(0))  # kept for compat
    crop_offset: tuple[int, int] = (0, 0)  # kept for compat


class MultiPersonTracker:
    """Detect, track, and estimate pose for multiple people using YOLO-pose."""

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        conf_threshold: float = 0.3,
        min_bbox_area_ratio: float = 0.01,
    ):
        from device_utils import get_best_device
        _best = get_best_device()
        self.model = YOLO(model_name)
        self._is_engine = model_name.endswith(".engine")
        self._use_half = (_best == "cuda")
        # TensorRT engines are already on GPU with FP16 baked in — skip .to()
        if not self._is_engine and _best != "cpu":
            self.model.to(_best)
        self.conf_threshold = conf_threshold
        self.min_bbox_area_ratio = min_bbox_area_ratio

    def reset_tracker(self):
        """Reset ByteTrack state for new session without reloading model/TRT engine."""
        if self.model.predictor is not None and hasattr(self.model.predictor, "trackers"):
            for tracker in self.model.predictor.trackers:
                # Clear tracker lists directly (BYTETracker in ultralytics 8.x)
                if hasattr(tracker, "tracked_stracks"):
                    tracker.tracked_stracks = []
                    tracker.lost_stracks = []
                    tracker.removed_stracks = []
                    tracker.frame_id = 0
                elif hasattr(tracker, "reset"):
                    tracker.reset()
        # Only destroy predictor as last resort (causes TRT engine reload)
        elif self.model.predictor is None:
            pass  # Nothing to reset

    def process_frame(self, rgb_frame: np.ndarray) -> list[TrackedPerson]:
        """Detect, track, and extract keypoints for all people in a frame.

        Args:
            rgb_frame: (H, W, 3) RGB numpy array.

        Returns:
            List of TrackedPerson with persistent track IDs and 17 COCO keypoints.
        """
        h, w = rgb_frame.shape[:2]
        frame_area = h * w

        results = self.model.track(
            rgb_frame,
            persist=True,
            conf=self.conf_threshold,
            imgsz=640,
            half=self._use_half,
            verbose=False,
        )

        persons: list[TrackedPerson] = []
        if not results or len(results) == 0:
            return persons

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return persons

        boxes = result.boxes
        kpts = result.keypoints  # Ultralytics Keypoints object

        for i in range(len(boxes)):
            # Skip if no track ID assigned yet
            if boxes.id is None:
                continue
            track_id = int(boxes.id[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # Filter tiny detections (distant people / false positives)
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area / frame_area < self.min_bbox_area_ratio:
                continue

            # Extract keypoints: (17, 3) [x, y, conf]
            if kpts is not None and kpts.data is not None and i < len(kpts.data):
                kp = kpts.data[i].cpu().numpy().astype(np.float32)  # (17, 3)
            else:
                kp = np.zeros((17, 3), dtype=np.float32)

            persons.append(TrackedPerson(
                track_id=track_id,
                bbox_pixel=(int(x1), int(y1), int(x2), int(y2)),
                bbox_normalized=(x1 / w, y1 / h, x2 / w, y2 / h),
                keypoints=kp,
            ))

        return persons


def denormalize_landmarks(
    landmarks_xyzv: np.ndarray,
    crop_offset: tuple[int, int],
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Shift crop-local landmarks to frame-global pixel coordinates.

    Kept for backward compatibility with Pipeline B / offline analysis.
    """
    result = landmarks_xyzv.copy()
    result[:, 0] = result[:, 0] + crop_offset[0]
    result[:, 1] = result[:, 1] + crop_offset[1]
    return result
