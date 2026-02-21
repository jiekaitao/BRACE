"""SMPL mesh estimation from person crops.

Estimates SMPL body model parameters (shape, pose, translation) from
cropped person images. Supports multiple backends:

1. simple-romp (ROMP) - pip installable, multi-person
2. HybrIK - research code, needs manual installation

Falls back gracefully to None when no backend is available.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SMPLEstimator:
    """Estimates SMPL body model parameters from person crops.

    Produces:
    - betas: (10,) shape parameters
    - pose: (72,) joint rotations (24 joints * 3 axis-angle)
    - trans: (3,) global translation

    If no SMPL backend is installed, all calls return None gracefully.
    """

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._backend = None
        self._available = False
        self._backend_name = "none"
        self._load_backend()

    def _load_backend(self) -> None:
        """Try to load an SMPL estimation backend."""
        # Try 1: simple-romp (ROMP)
        if self._try_romp():
            return

        # Try 2: HybrIK
        if self._try_hybrik():
            return

        logger.info("No SMPL estimation backend available. SMPL estimation disabled.")

    def _try_romp(self) -> bool:
        """Try to load ROMP backend."""
        try:
            import romp

            settings = romp.main.default_settings
            settings.render_mesh = False
            settings.show_largest = False

            self._romp_model = romp.ROMP(settings)
            self._available = True
            self._backend = "romp"
            self._backend_name = "ROMP"
            logger.info("SMPL backend: ROMP (simple-romp)")
            return True
        except ImportError:
            logger.debug("simple-romp not installed")
        except Exception as e:
            logger.warning("ROMP init failed: %s", e)
        return False

    def _try_hybrik(self) -> bool:
        """Try to load HybrIK backend."""
        try:
            from hybrik.models import builder as hybrik_builder
            from hybrik.utils.config import update_config
            import torch

            cfg = update_config("configs/hybrik_hrnet.yaml")
            model = hybrik_builder.build_model(cfg)
            model = model.to(self._device)
            model.eval()
            self._hybrik_model = model
            self._available = True
            self._backend = "hybrik"
            self._backend_name = "HybrIK"
            logger.info("SMPL backend: HybrIK")
            return True
        except ImportError:
            logger.debug("hybrik package not installed")
        except FileNotFoundError:
            logger.debug("HybrIK model weights not found")
        except Exception as e:
            logger.warning("HybrIK init failed: %s", e)
        return False

    @property
    def available(self) -> bool:
        """Whether an SMPL estimation backend loaded successfully."""
        return self._available

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend_name

    def estimate(self, rgb: np.ndarray, bboxes: list[tuple]) -> list[dict | None]:
        """Estimate SMPL params for each person crop.

        Args:
            rgb: Full frame as (H, W, 3) uint8 RGB array.
            bboxes: List of (x1, y1, x2, y2) bounding boxes in pixel coords.

        Returns:
            List of dicts with keys {"betas": list[10], "pose": list[72],
            "trans": list[3]} or None per person if estimation failed.
        """
        if not self._available:
            return [None] * len(bboxes)

        if self._backend == "romp":
            return self._estimate_romp(rgb, bboxes)
        elif self._backend == "hybrik":
            return self._estimate_hybrik(rgb, bboxes)
        return [None] * len(bboxes)

    def _estimate_romp(self, rgb: np.ndarray, bboxes: list[tuple]) -> list[dict | None]:
        """Estimate SMPL params using ROMP on individual crops."""
        results = []
        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = rgb.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    results.append(None)
                    continue

                crop = rgb[y1:y2, x1:x2]
                # ROMP expects BGR
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

                outputs = self._romp_model(crop_bgr)

                if outputs is not None and len(outputs) > 0:
                    # Extract first person's params
                    out = outputs[0] if isinstance(outputs, list) else outputs
                    smpl_thetas = out.get("smpl_thetas", out.get("body_pose", None))
                    smpl_betas = out.get("smpl_betas", out.get("betas", None))
                    cam_trans = out.get("cam_trans", out.get("trans", None))

                    if smpl_thetas is not None and smpl_betas is not None:
                        pose = np.array(smpl_thetas).flatten()[:72].tolist()
                        betas = np.array(smpl_betas).flatten()[:10].tolist()
                        trans = np.array(cam_trans).flatten()[:3].tolist() if cam_trans is not None else [0, 0, 0]
                        results.append({"betas": betas, "pose": pose, "trans": trans})
                    else:
                        results.append(None)
                else:
                    results.append(None)
            except Exception as e:
                logger.debug("ROMP estimation failed for crop: %s", e)
                results.append(None)
        return results

    def _estimate_hybrik(self, rgb: np.ndarray, bboxes: list[tuple]) -> list[dict | None]:
        """Estimate SMPL params using HybrIK on individual crops."""
        import torch

        results = []
        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = rgb.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    results.append(None)
                    continue

                crop = rgb[y1:y2, x1:x2]
                crop_resized = cv2.resize(crop, (256, 256))

                inp = torch.from_numpy(crop_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                inp = inp.to(self._device)

                with torch.no_grad():
                    output = self._hybrik_model(inp)

                betas = output.pred_shape.cpu().numpy().flatten()[:10].tolist()
                pose = output.pred_theta_mats.cpu().numpy().flatten()[:72].tolist()
                trans = output.pred_cam_root.cpu().numpy().flatten()[:3].tolist()

                results.append({"betas": betas, "pose": pose, "trans": trans})
            except Exception as e:
                logger.debug("HybrIK estimation failed for crop: %s", e)
                results.append(None)
        return results


# Backward compatibility alias
HybrIKEstimator = SMPLEstimator
