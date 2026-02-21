"""Advanced pose backend: YOLO11 detection + BoT-SORT tracking + RTMW3D 3D pose.

Falls back gracefully when optional dependencies are missing:
- No rtmlib: uses YOLO-pose keypoints (17 COCO, 2D) instead of RTMW3D
- No boxmot: uses built-in ByteTrack from YOLO instead of BoT-SORT
- No SMPL backend: mesh estimation disabled
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline_interface import PipelineResult, PoseBackend
from brace.core.pose import (
    coco_keypoints_to_landmarks,
    wholebody133_to_mediapipe33,
)

logger = logging.getLogger(__name__)

# Optional: RTMW3D estimator
try:
    from rtmw3d_estimator import RTMW3DEstimator, is_available as rtmw3d_available
except ImportError:
    rtmw3d_available = lambda: False
    RTMW3DEstimator = None

# Optional: BoT-SORT tracker
_BOTSORT_AVAILABLE = False
try:
    from botsort_tracker import BoTSortTracker
    _BOTSORT_AVAILABLE = True
except ImportError:
    BoTSortTracker = None

# Optional: SMPL estimator (ROMP or HybrIK)
try:
    from hybrik_estimator import HybrIKEstimator
except ImportError:
    HybrIKEstimator = None

from uv_texture_projector import UVTextureProjector


class AdvancedPoseBackend(PoseBackend):
    """Advanced multi-stage pose pipeline.

    Stages:
    1. YOLO11-pose detection + BoT-SORT tracking (fallback: ByteTrack)
    2. RTMW3D top-down 3D pose estimation per tracked person (if available)
    3. Keypoint mapping to MediaPipe 33-landmark format
    4. SMPL mesh estimation (optional, every Nth frame)

    Falls back to legacy YOLO-pose keypoints when RTMW3D is unavailable.
    Falls back to ByteTrack when BoT-SORT is unavailable.
    """

    def __init__(
        self,
        model_name: str = "yolo11x-pose.pt",
        conf_threshold: float = 0.3,
        use_botsort: bool = True,
        use_rtmw3d: bool = True,
        device: str = "cuda",
    ):
        self._device = device
        self._use_botsort = False

        # Try BoT-SORT first, fall back to MultiPersonTracker (ByteTrack)
        if use_botsort and _BOTSORT_AVAILABLE:
            try:
                self._botsort_tracker = BoTSortTracker(
                    model_name=model_name,
                    conf_threshold=conf_threshold,
                )
                self._use_botsort = True
                logger.info("BoT-SORT tracker initialized (GMC + ReID)")
            except Exception as exc:
                logger.warning("BoT-SORT init failed: %s, falling back to ByteTrack", exc)

        if not self._use_botsort:
            from multi_person_tracker import MultiPersonTracker
            self._legacy_tracker = MultiPersonTracker(
                model_name=model_name,
                conf_threshold=conf_threshold,
            )
            logger.info("Using legacy ByteTrack tracker (BoT-SORT unavailable)")

        # Try to initialize RTMW3D
        self._rtmw3d = None
        if use_rtmw3d and rtmw3d_available():
            try:
                self._rtmw3d = RTMW3DEstimator(device=device)
                logger.info("RTMW3D-x initialized for 133-keypoint 3D pose (via rtmlib)")
            except Exception as exc:
                logger.warning("RTMW3D init failed, falling back to YOLO keypoints: %s", exc)

        # SMPL estimation (runs every 3rd frame)
        self._smpl = None
        if HybrIKEstimator is not None:
            try:
                est = HybrIKEstimator(device=device)
                if est.available:
                    self._smpl = est
                    logger.info("SMPL estimator initialized (%s)", est.backend_name)
            except Exception as exc:
                logger.warning("SMPL estimator init failed: %s", exc)

        # UV texture projectors per track_id
        self._uv_projectors: dict[int, UVTextureProjector] = {}

        # Frame counter for amortized SMPL (every 3rd frame) and UV texture (every 5th)
        self._frame_count = 0
        self._smpl_interval = 3
        self._uv_interval = 5

        # Cache last SMPL params per track_id for inter-frame interpolation
        self._smpl_cache: dict[int, dict] = {}

        logger.info(
            "AdvancedPoseBackend: botsort=%s, rtmw3d=%s, smpl=%s",
            self._use_botsort,
            self._rtmw3d is not None,
            self._smpl is not None,
        )

    @property
    def tracker(self):
        if self._use_botsort:
            return self._botsort_tracker
        return self._legacy_tracker

    @property
    def model(self):
        if self._use_botsort:
            return self._botsort_tracker.model
        return self._legacy_tracker.model

    @property
    def has_rtmw3d(self) -> bool:
        return self._rtmw3d is not None

    @property
    def has_botsort(self) -> bool:
        return self._use_botsort

    def process_frame(self, rgb: np.ndarray) -> list[PipelineResult]:
        h, w = rgb.shape[:2]

        # Stage 1: Detect + track (BoT-SORT or ByteTrack)
        if self._use_botsort:
            detections = self._botsort_tracker.process_frame(rgb)
        else:
            detections = self._legacy_tracker.process_frame(rgb)

        if not detections:
            return []

        results = []

        # Stage 2: RTMW3D 3D pose estimation (if available)
        wholebody_kpts = None
        if self._rtmw3d is not None:
            bboxes = [d.bbox_pixel for d in detections]
            try:
                wholebody_kpts = self._rtmw3d.estimate(rgb, bboxes)
            except Exception as exc:
                logger.warning("RTMW3D inference failed: %s", exc)

        # Stage 3: SMPL estimation (every Nth frame)
        smpl_results = None
        if self._smpl is not None and self._frame_count % self._smpl_interval == 0:
            bboxes_for_smpl = [d.bbox_pixel for d in detections]
            try:
                smpl_results = self._smpl.estimate(rgb, bboxes_for_smpl)
            except Exception as exc:
                logger.warning("SMPL inference failed: %s", exc)

        self._frame_count += 1

        # Stage 4: Build PipelineResult for each person
        for i, det in enumerate(detections):
            if wholebody_kpts is not None and i < len(wholebody_kpts):
                wb_kpts = wholebody_kpts[i]  # (133, 4)
                # Check if we got valid keypoints (not all zeros)
                if np.any(wb_kpts[:, 3] > 0):
                    landmarks_mp = wholebody133_to_mediapipe33(wb_kpts)
                    landmarks_3d = wb_kpts
                else:
                    landmarks_mp = coco_keypoints_to_landmarks(det.keypoints, w, h)
                    landmarks_3d = None
            else:
                landmarks_mp = coco_keypoints_to_landmarks(det.keypoints, w, h)
                landmarks_3d = None

            # SMPL params: use fresh result or cached
            smpl_params = None
            if smpl_results is not None and i < len(smpl_results):
                smpl_params = smpl_results[i]
                if smpl_params is not None:
                    self._smpl_cache[det.track_id] = smpl_params
            if smpl_params is None:
                smpl_params = self._smpl_cache.get(det.track_id)

            # UV texture projection (requires SMPL params + crop, every 5th frame)
            uv_texture = None
            crop = getattr(det, "crop_rgb", None)
            if (smpl_params is not None and crop is not None and crop.size > 0
                    and (self._frame_count - 1) % self._uv_interval == 0):
                if det.track_id not in self._uv_projectors:
                    self._uv_projectors[det.track_id] = UVTextureProjector()
                uv_texture = self._uv_projectors[det.track_id].project(crop)

            results.append(PipelineResult(
                track_id=det.track_id,
                bbox_pixel=det.bbox_pixel,
                bbox_normalized=det.bbox_normalized,
                landmarks_mp=landmarks_mp,
                landmarks_3d=landmarks_3d,
                smpl_params=smpl_params,
                smpl_texture_uv=uv_texture,
                reid_embedding=None,
                crop_rgb=getattr(det, "crop_rgb", np.empty(0)),
            ))

        return results

    def reset(self) -> None:
        if self._use_botsort:
            self._botsort_tracker.reset()
        else:
            self._legacy_tracker.reset_tracker()
        self._frame_count = 0
        self._smpl_cache.clear()
        self._uv_projectors.clear()

    def on_scene_cut(self) -> None:
        """Reset tracker state on scene cut."""
        self.reset()

    def warmup(self) -> None:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.process_frame(dummy)
