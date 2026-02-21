"""Abstract pipeline interface for pose estimation backends.

Decouples main.py and video_processor.py from specific model implementations,
allowing both webcam and video paths to use the same backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PipelineResult:
    """Result from a pose estimation backend for one detected person.

    Contains all per-person data that downstream code needs:
    tracking ID, bounding boxes, landmarks in both MediaPipe and
    full 3D formats, optional SMPL params, re-ID embeddings, and crop.
    """

    track_id: int
    bbox_pixel: tuple[int, int, int, int]  # x1, y1, x2, y2
    bbox_normalized: tuple[float, float, float, float]  # 0-1
    landmarks_mp: np.ndarray  # (33, 4) MediaPipe format [x_px, y_px, z, vis]
    landmarks_3d: np.ndarray | None = None  # (133, 4) wholebody or None
    smpl_params: dict | None = None  # {betas, pose, trans} or None
    smpl_texture_uv: np.ndarray | None = None  # (256, 256, 3) UV texture or None
    reid_embedding: np.ndarray | None = None  # (D,) appearance embedding
    crop_rgb: np.ndarray = field(default_factory=lambda: np.empty(0))


class PoseBackend(ABC):
    """Abstract base class for pose estimation pipelines.

    Implementations process a single RGB frame and return a list of
    PipelineResult instances, one per detected person.
    """

    @abstractmethod
    def process_frame(self, rgb: np.ndarray) -> list[PipelineResult]:
        """Process a single RGB frame.

        Args:
            rgb: (H, W, 3) RGB numpy array.

        Returns:
            List of PipelineResult, one per detected person.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state for a new session (reuses loaded models)."""
        ...

    def on_scene_cut(self) -> None:
        """Handle a detected scene cut (camera change).

        Default implementation does nothing. Override in backends
        that support cross-cut re-identification.
        """
        pass
