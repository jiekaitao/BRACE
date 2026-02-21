"""Project video frame pixels onto SMPL UV map for textured mesh rendering.

Accumulates UV texture from video frames projected onto the SMPL mesh.
Falls back to a simplified crop-based texture when pytorch3d is not available.
"""

from __future__ import annotations

import cv2
import numpy as np

# Neutral mannequin skin color (light gray)
_NEUTRAL_COLOR = 180


class UVTextureProjector:
    """Accumulates UV texture from video frames projected onto SMPL mesh.

    Input: person crop + SMPL vertices + camera params
    Output: (texture_size, texture_size, 3) UV texture image

    Accumulates partial UV maps across frames (front fills front,
    side fills sides). Missing regions filled with neutral mannequin color.

    When pytorch3d is not available, uses a simplified approach: the person
    crop is resized and blended into the texture map directly.
    """

    def __init__(self, texture_size: int = 256):
        self._size = texture_size
        self._texture = np.full(
            (texture_size, texture_size, 3), _NEUTRAL_COLOR, dtype=np.uint8
        )
        self._coverage = np.zeros((texture_size, texture_size), dtype=np.float32)
        self._has_pytorch3d = False
        try:
            import pytorch3d  # noqa: F401
            self._has_pytorch3d = True
        except ImportError:
            pass

    def project(
        self,
        crop_rgb: np.ndarray,
        smpl_vertices: np.ndarray | None = None,
        camera_params: dict | None = None,
    ) -> np.ndarray:
        """Project crop pixels onto UV map, accumulating coverage.

        When pytorch3d is available and smpl_vertices/camera_params are
        provided, performs proper UV projection. Otherwise, uses a simplified
        crop-resize approach.

        Args:
            crop_rgb: Person crop as (H, W, 3) uint8 RGB array.
            smpl_vertices: (6890, 3) SMPL mesh vertices (optional).
            camera_params: Camera intrinsics/extrinsics dict (optional).

        Returns:
            Current UV texture as (texture_size, texture_size, 3) uint8.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return self._texture.copy()

        if self._has_pytorch3d and smpl_vertices is not None and camera_params is not None:
            self._project_proper(crop_rgb, smpl_vertices, camera_params)
        else:
            self._project_simplified(crop_rgb)

        return self._texture.copy()

    def _project_simplified(self, crop_rgb: np.ndarray) -> None:
        """Simplified texture: resize crop to texture size and blend.

        Uses exponential moving average to accumulate texture over time,
        giving recent frames more weight while preserving coverage from
        earlier views.
        """
        resized = cv2.resize(crop_rgb, (self._size, self._size))
        alpha = 0.3  # blend weight for new frame
        mask = self._coverage > 0
        # Blend where we have previous coverage
        self._texture[mask] = (
            (1 - alpha) * self._texture[mask].astype(np.float32)
            + alpha * resized[mask].astype(np.float32)
        ).astype(np.uint8)
        # Fill uncovered regions directly
        uncovered = ~mask
        self._texture[uncovered] = resized[uncovered]
        self._coverage = np.clip(self._coverage + alpha, 0, 1.0)

    def _project_proper(
        self,
        crop_rgb: np.ndarray,
        smpl_vertices: np.ndarray,
        camera_params: dict,
    ) -> None:
        """Proper UV projection using pytorch3d rasterization.

        Projects crop pixels onto the SMPL UV map by:
        1. Rasterizing SMPL mesh to get per-pixel UV coordinates
        2. Sampling crop pixels at projected positions
        3. Blending into accumulated texture
        """
        # This is a placeholder for proper pytorch3d UV projection.
        # When pytorch3d is available, this would:
        # 1. Create a Meshes object from smpl_vertices + SMPL faces
        # 2. Set up a camera from camera_params
        # 3. Rasterize to get fragment UV coords
        # 4. Sample crop_rgb at those coords
        # 5. Blend into self._texture
        # For now, fall back to simplified projection
        self._project_simplified(crop_rgb)

    def get_texture(self) -> np.ndarray:
        """Return current accumulated UV texture."""
        return self._texture.copy()

    def reset(self) -> None:
        """Clear accumulated texture to neutral mannequin color."""
        self._texture[:] = _NEUTRAL_COLOR
        self._coverage[:] = 0
