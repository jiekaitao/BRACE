"""BoT-SORT pose backend wrapping BoTSortTracker + COCO-to-MediaPipe mapping.

Produces PipelineResult with landmarks_3d=None and smpl_params=None,
same as LegacyPoseBackend but using BoT-SORT tracking with GMC instead of ByteTrack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_interface import PipelineResult, PoseBackend
from botsort_tracker import BoTSortTracker
from brace.core.pose import coco_keypoints_to_landmarks


class BoTSortPoseBackend(PoseBackend):
    """BoT-SORT tracking backend producing PipelineResult instances.

    Uses YOLO-pose (17 COCO keypoints) with BoT-SORT tracking (GMC + optional ReID).
    COCO keypoints are mapped to 33 MediaPipe landmarks with z=0.
    """

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        conf_threshold: float = 0.3,
        reid_weights: Path | str = Path("osnet_x0_25_msmt17.pt"),
        with_reid: bool = False,
        cmc_method: str = "sof",
    ):
        self._tracker = BoTSortTracker(
            model_name=model_name,
            conf_threshold=conf_threshold,
            reid_weights=reid_weights,
            with_reid=with_reid,
            cmc_method=cmc_method,
        )

    @property
    def tracker(self) -> BoTSortTracker:
        """Access underlying tracker."""
        return self._tracker

    @property
    def model(self):
        """Access the YOLO model (for startup logging)."""
        return self._tracker.model

    def process_frame(self, rgb: np.ndarray) -> list[PipelineResult]:
        h, w = rgb.shape[:2]
        detections = self._tracker.process_frame(rgb)

        results = []
        for det in detections:
            landmarks_mp = coco_keypoints_to_landmarks(det.keypoints, w, h)
            results.append(PipelineResult(
                track_id=det.track_id,
                bbox_pixel=det.bbox_pixel,
                bbox_normalized=det.bbox_normalized,
                landmarks_mp=landmarks_mp,
                landmarks_3d=None,
                smpl_params=None,
                smpl_texture_uv=None,
                reid_embedding=None,
                crop_rgb=det.crop_rgb,
            ))

        return results

    def reset(self) -> None:
        self._tracker.reset()

    def warmup(self) -> None:
        """Run a dummy inference to warm up the GPU pipeline."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._tracker.process_frame(dummy)
