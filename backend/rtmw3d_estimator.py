"""RTMW3D whole-body 3D pose estimation via rtmlib (ONNX Runtime).

Top-down estimator: takes bounding boxes from a detector, crops each person,
and runs RTMW3D-x to produce 133 whole-body keypoints with real 3D depth.

Uses rtmlib for inference (no mmpose/mmcv dependency), with onnxruntime-gpu
for CUDA acceleration.

Gracefully returns None if rtmlib is not installed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import rtmlib; mark availability
_RTMLIB_AVAILABLE = False
try:
    from rtmlib import Wholebody3d
    _RTMLIB_AVAILABLE = True
except ImportError:
    Wholebody3d = None

# Number of whole-body keypoints: 17 body + 6 feet + 68 face + 42 hands = 133
NUM_WHOLEBODY_KEYPOINTS = 133


def is_available() -> bool:
    """Return True if rtmlib is installed and RTMW3D can be used."""
    return _RTMLIB_AVAILABLE


class RTMW3DEstimator:
    """RTMW3D-x 133-keypoint 3D whole-body pose estimator.

    Uses rtmlib + ONNX Runtime for inference: given bounding boxes from a
    detector, runs the RTMW3D-x model to produce (133, 4) keypoints
    with [x_pixel, y_pixel, z_metric, confidence].
    """

    def __init__(
        self,
        device: str = "cuda",
        backend: str = "onnxruntime",
    ):
        if not _RTMLIB_AVAILABLE:
            raise RuntimeError(
                "rtmlib is not installed. Install with: "
                "pip install rtmlib onnxruntime-gpu"
            )

        self._device = device
        self._backend = backend
        self._model = None

        try:
            # Wholebody3d handles its own model download from HuggingFace
            self._model = Wholebody3d(
                to_openpose=False,
                mode="balanced",
                backend=backend,
                device=device,
            )
            logger.info("RTMW3D-x loaded via rtmlib (backend=%s, device=%s)", backend, device)
        except Exception as exc:
            logger.warning("RTMW3D-x init failed: %s", exc)
            raise

    def estimate(
        self,
        rgb: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[np.ndarray]:
        """Run top-down 3D pose estimation on person crops.

        Args:
            rgb: Full frame (H, W, 3) RGB uint8.
            bboxes: List of (x1, y1, x2, y2) pixel bounding boxes.

        Returns:
            List of (133, 4) arrays per person: [x_pixel, y_pixel, z, confidence].
            Length matches len(bboxes). If estimation fails for a person,
            the corresponding array is all zeros.
        """
        if not bboxes or self._model is None:
            return []

        outputs = []

        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = bbox
                # rtmlib Wholebody3d expects the full frame and runs its own
                # detection internally. We crop to the bbox region and run
                # on the crop, then offset coordinates back.
                h, w = rgb.shape[:2]
                x1c = max(0, int(x1))
                y1c = max(0, int(y1))
                x2c = min(w, int(x2))
                y2c = min(h, int(y2))

                if x2c <= x1c or y2c <= y1c:
                    outputs.append(np.zeros((NUM_WHOLEBODY_KEYPOINTS, 4), dtype=np.float32))
                    continue

                # Pad crop slightly for better pose estimation
                pad_w = int((x2c - x1c) * 0.1)
                pad_h = int((y2c - y1c) * 0.1)
                cx1 = max(0, x1c - pad_w)
                cy1 = max(0, y1c - pad_h)
                cx2 = min(w, x2c + pad_w)
                cy2 = min(h, y2c + pad_h)

                crop = rgb[cy1:cy2, cx1:cx2]

                # rtmlib returns: keypoints_3d (N,133,3), scores (N,133), ...
                keypoints_3d, scores, _, keypoints_2d = self._model(crop)

                if keypoints_3d is not None and len(keypoints_3d) > 0:
                    kpts_3d = keypoints_3d[0]  # (133, 3) - take first person in crop
                    kpts_2d = keypoints_2d[0] if keypoints_2d is not None and len(keypoints_2d) > 0 else None
                    scr = scores[0]  # (133,)

                    out = np.zeros((NUM_WHOLEBODY_KEYPOINTS, 4), dtype=np.float32)
                    n_kpts = min(kpts_3d.shape[0], NUM_WHOLEBODY_KEYPOINTS)

                    if kpts_2d is not None and kpts_2d.shape[0] >= n_kpts:
                        # Use 2D keypoints for pixel coords (offset to full frame)
                        out[:n_kpts, 0] = kpts_2d[:n_kpts, 0] + cx1  # x pixel
                        out[:n_kpts, 1] = kpts_2d[:n_kpts, 1] + cy1  # y pixel
                    else:
                        # Fallback: use 3D x,y as pixel coords
                        out[:n_kpts, 0] = kpts_3d[:n_kpts, 0] + cx1
                        out[:n_kpts, 1] = kpts_3d[:n_kpts, 1] + cy1

                    # Z depth from 3D keypoints
                    out[:n_kpts, 2] = kpts_3d[:n_kpts, 2]
                    out[:n_kpts, 3] = scr[:n_kpts]

                    outputs.append(out)
                else:
                    outputs.append(np.zeros((NUM_WHOLEBODY_KEYPOINTS, 4), dtype=np.float32))

            except Exception as exc:
                logger.warning("RTMW3D failed for bbox %s: %s", bbox, exc)
                outputs.append(np.zeros((NUM_WHOLEBODY_KEYPOINTS, 4), dtype=np.float32))

        return outputs
