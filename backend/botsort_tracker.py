"""BoT-SORT tracker with GMC camera motion compensation via BoxMOT.

Decouples YOLO-pose detection from BoT-SORT tracking to get:
  - Global Motion Compensation (GMC) for camera pans/zooms
  - Built-in ReID for within-shot identity maintenance
  - Better occlusion handling than plain ByteTrack
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from boxmot import BotSort


@dataclass
class TrackedDetection:
    """A single person detection with BoT-SORT persistent track ID."""

    track_id: int
    bbox_pixel: tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_normalized: tuple[float, float, float, float]  # 0-1 range
    keypoints: np.ndarray  # (17, 3) COCO [x_px, y_px, conf]
    conf: float  # detection confidence
    crop_rgb: np.ndarray = field(default_factory=lambda: np.empty(0))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class BoTSortTracker:
    """YOLO11-Pose detection with BoT-SORT tracking via BoxMOT.

    YOLO handles detection + keypoint extraction.
    BoT-SORT handles tracking with GMC and optional ReID.
    We match tracked boxes back to YOLO keypoints via IoU overlap.
    """

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        conf_threshold: float = 0.3,
        min_bbox_area_ratio: float = 0.01,
        reid_weights: Path | str = Path("osnet_x0_25_msmt17.pt"),
        with_reid: bool = False,
        cmc_method: str = "sof",
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._half = torch.cuda.is_available()
        self.conf_threshold = conf_threshold
        self.min_bbox_area_ratio = min_bbox_area_ratio

        # YOLO for detection + keypoints only (no built-in tracker)
        self.model = YOLO(model_name)
        if self._half:
            self.model.to("cuda")

        # Store tracker init kwargs for reset()
        self._tracker_kwargs = dict(
            reid_weights=Path(reid_weights),
            device=self._device,
            half=self._half,
            with_reid=with_reid,
            cmc_method=cmc_method,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

        # BoT-SORT tracker (separate from detection)
        self._tracker = BotSort(**self._tracker_kwargs)

    def process_frame(self, rgb_frame: np.ndarray) -> list[TrackedDetection]:
        """Detect, track, and extract keypoints for all people in a frame.

        Pipeline:
          1. YOLO.predict() for detections + keypoints
          2. BoTSORT.update(dets, frame) for tracking with GMC
          3. Match tracked IDs back to YOLO keypoints via bbox IoU

        Args:
            rgb_frame: (H, W, 3) RGB numpy array.

        Returns:
            List of TrackedDetection with persistent track IDs and COCO keypoints.
        """
        h, w = rgb_frame.shape[:2]
        frame_area = h * w

        # --- Step 1: YOLO detection (no tracking) ---
        results = self.model.predict(
            rgb_frame,
            conf=self.conf_threshold,
            imgsz=640,
            half=self._half,
            verbose=False,
        )

        if not results or len(results) == 0:
            # Still update tracker with empty dets so it ages out old tracks
            self._tracker.update(np.empty((0, 6)), rgb_frame)
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self._tracker.update(np.empty((0, 6)), rgb_frame)
            return []

        boxes = result.boxes
        kpts_obj = result.keypoints

        # Build arrays: detections (N, 6) = [x1, y1, x2, y2, conf, cls]
        # and keypoints (N, 17, 3) for later matching
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = boxes.conf.cpu().numpy()  # (N,)
        clss = boxes.cls.cpu().numpy()  # (N,)
        dets = np.column_stack([xyxy, confs, clss])  # (N, 6)

        # Store YOLO keypoints indexed by detection order
        yolo_keypoints: list[np.ndarray] = []
        for i in range(len(boxes)):
            if (kpts_obj is not None
                    and kpts_obj.data is not None
                    and i < len(kpts_obj.data)):
                kp = kpts_obj.data[i].cpu().numpy().astype(np.float32)
            else:
                kp = np.zeros((17, 3), dtype=np.float32)
            yolo_keypoints.append(kp)

        # --- Step 2: BoT-SORT tracking ---
        try:
            tracked = self._tracker.update(dets, rgb_frame)
        except cv2.error:
            # CMC optical flow fails when frame size changes (e.g. video loop)
            # Reset CMC state and retry
            if hasattr(self._tracker, "cmc"):
                self._tracker.cmc.prev_img = None
            try:
                tracked = self._tracker.update(dets, rgb_frame)
            except cv2.error:
                # Still failing — skip this frame
                return []
        # tracked shape: (M, 8) = [x1, y1, x2, y2, track_id, conf, cls, det_ind]

        if len(tracked) == 0:
            return []

        # --- Step 3: Match tracked boxes to YOLO keypoints ---
        persons: list[TrackedDetection] = []

        for row in tracked:
            tx1, ty1, tx2, ty2 = row[0:4]
            track_id = int(row[4])
            track_conf = float(row[5])
            det_ind = int(row[7])

            # Filter tiny detections
            bbox_area = (tx2 - tx1) * (ty2 - ty1)
            if bbox_area / frame_area < self.min_bbox_area_ratio:
                continue

            # Use det_ind to directly index YOLO keypoints
            if 0 <= det_ind < len(yolo_keypoints):
                kp = yolo_keypoints[det_ind]
            else:
                # Fallback: match by IoU if det_ind is invalid
                tracked_box = np.array([tx1, ty1, tx2, ty2])
                best_iou, best_idx = 0.0, -1
                for j, yolo_kp in enumerate(yolo_keypoints):
                    iou_val = _iou(tracked_box, xyxy[j])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = j
                kp = yolo_keypoints[best_idx] if best_idx >= 0 else np.zeros(
                    (17, 3), dtype=np.float32
                )

            ix1, iy1, ix2, iy2 = int(tx1), int(ty1), int(tx2), int(ty2)
            persons.append(TrackedDetection(
                track_id=track_id,
                bbox_pixel=(ix1, iy1, ix2, iy2),
                bbox_normalized=(tx1 / w, ty1 / h, tx2 / w, ty2 / h),
                keypoints=kp,
                conf=track_conf,
            ))

        return persons

    def reset(self) -> None:
        """Reset BoT-SORT tracker state for a new session (reuses YOLO model)."""
        self._tracker = BotSort(**self._tracker_kwargs)
