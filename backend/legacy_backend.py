"""Legacy pose backend wrapping the existing MultiPersonTracker + COCO-to-MediaPipe mapping.

Produces PipelineResult with landmarks_3d=None and smpl_params=None,
providing exact behavioral parity with the pre-refactor pipeline.
"""

from __future__ import annotations

import numpy as np

from pipeline_interface import PipelineResult, PoseBackend
from multi_person_tracker import MultiPersonTracker
from brace.core.pose import coco_keypoints_to_landmarks


class LegacyPoseBackend(PoseBackend):
    """Wraps MultiPersonTracker to produce PipelineResult instances.

    This backend uses YOLO-pose (17 COCO keypoints, 2D) with built-in
    ByteTrack tracking. The COCO keypoints are mapped to 33 MediaPipe
    landmarks with z=0.
    """

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        conf_threshold: float = 0.3,
    ):
        self._tracker = MultiPersonTracker(
            model_name=model_name,
            conf_threshold=conf_threshold,
        )

    @property
    def tracker(self) -> MultiPersonTracker:
        """Access underlying tracker (for model device info etc.)."""
        return self._tracker

    @property
    def model(self):
        """Access the YOLO model (for startup logging)."""
        return self._tracker.model

    def process_frame(self, rgb: np.ndarray) -> list[PipelineResult]:
        h, w = rgb.shape[:2]
        persons = self._tracker.process_frame(rgb)

        results = []
        for person in persons:
            landmarks_mp = coco_keypoints_to_landmarks(person.keypoints, w, h)
            results.append(PipelineResult(
                track_id=person.track_id,
                bbox_pixel=person.bbox_pixel,
                bbox_normalized=person.bbox_normalized,
                landmarks_mp=landmarks_mp,
                landmarks_3d=None,
                smpl_params=None,
                smpl_texture_uv=None,
                reid_embedding=None,
                crop_rgb=person.crop_rgb,
            ))

        return results

    def reset(self) -> None:
        self._tracker.reset_tracker()

    def warmup(self) -> None:
        """Run a dummy inference to warm up the GPU pipeline."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._tracker.process_frame(dummy)
